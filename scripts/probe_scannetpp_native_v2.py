"""Bounded v2 renderer/CPU validation. Never admits labels or starts training.

Confirmation images are sealed: this executable only handles the nineteen known
quarantines and sixteen development controls. Original cache payloads are read
for comparison and remain immutable. Each failed/rejected candidate is retained.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

from cache_scannetpp_geometry import camera_for_row, colmap_poses, conservative_anonymization
from indoor_download import write_json
from probe_scannetpp_software_clip_render import render as software_render
from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import publish_json, read_geometry_member, sha_file
from radio_moge.mesh_clip import MeshClipContract, clipped_cpu_depth, verify_clipped_cpu_intersections
from radio_moge.mesh_depth import CudaMeshDepth


def implementation():
    root = Path(__file__).resolve().parents[1]
    return {name: sha_file(root/name) for name in (
        'scripts/probe_scannetpp_native_v2.py', 'scripts/cache_scannetpp_geometry.py',
        'scripts/probe_scannetpp_software_clip_render.py', 'src/radio_moge/mesh_clip.py',
        'src/radio_moge/mesh_depth.py', 'src/radio_moge/geometry_sources.py',
        'src/radio_moge/geometry_cache.py')}


def software_backend():
    required = dict(PYOPENGL_PLATFORM='osmesa', LIBGL_ALWAYS_SOFTWARE='1',
                    GALLIUM_DRIVER='llvmpipe', LP_NUM_THREADS='1')
    if any(os.environ.get(k) != v for k, v in required.items()):
        raise ValueError('Explicit llvmpipe software reference required alongside CUDA')
    import pyglet
    pyglet.options['headless'], pyglet.options['shadow_window'] = True, False
    import pyrender
    from OpenGL import GL
    renderer = pyrender.OffscreenRenderer(64, 48)
    identity = {key: GL.glGetString(value).decode() for key, value in
        [('renderer', GL.GL_RENDERER), ('vendor', GL.GL_VENDOR), ('version', GL.GL_VERSION)]}
    if 'llvmpipe' not in identity['renderer'].lower():
        renderer.delete()
        raise ValueError('CPU software reference backend not verified')
    return pyrender, renderer, identity


def synthetic_gate():
    contract = MeshClipContract()
    h, w = 32, 48
    k = np.array([[.8, 0, .43], [0, 1.1, .57], [0, 0, 1.]])
    pose = np.eye(4)
    a = .4
    pose[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    pose[:3, 3] = [3, -2, 1]
    xy = np.array([[-5., -5], [5, -5], [5, 5], [-5, 5]])
    cases = []
    for front, rear, expected in [(.02, 2., 2.), (.05, 2., None), (.08, 2., .08),
                                  (30., None, None)]:
        planes = [front] if rear is None else [front, rear]
        camera_vertices = np.concatenate([np.column_stack([xy, np.full(4, z)]) for z in planes])
        vertices = camera_vertices @ pose[:3, :3].T+pose[:3, 3]
        faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
        if rear is None:
            faces = faces[:2]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        engine = RayMeshIntersector(mesh)
        renderer = CudaMeshDepth(vertices, faces, clip_contract=contract)
        raw, valid, _ = renderer.render(k, pose, h, w)
        depth, support, reason, stats = verify_clipped_cpu_intersections(
            engine, k, pose, raw, valid, contract)
        if expected is None:
            if support.any():
                raise ValueError('Boundary or beyond-far fixture was incorrectly admitted')
        elif not support.all() or np.max(abs(depth-expected)) > 1e-5:
            raise ValueError('Camera-Z clipped foreground/rear fixture failed')
        if front == .02:
            # Face order and chunk boundaries must not alter visibility.
            alternative = CudaMeshDepth(vertices, faces[::-1].copy(), clip_contract=contract,
                                        max_faces_per_pass=1)
            other, mask, _ = alternative.render(k, pose, h, w)
            if not np.array_equal(valid, mask) or np.max(abs(raw-other)) > 1e-5:
                raise ValueError('Chunked nearest visibility differs')
            del alternative
        cases.append(dict(front_m=front, expected_m=expected, stats=stats,
                          supported=int(support.sum()), reason_codes=np.unique(reason).tolist()))
        del renderer, engine, mesh
    # One slanted triangle crosses the near plane: keep its surviving polygon.
    vertices = np.array([[-1, -1, -.02], [1, -1, .14], [0, 1, .06]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2]], process=False)
    renderer = CudaMeshDepth(vertices, mesh.faces, clip_contract=contract)
    raw, valid, _ = renderer.render(k, np.eye(4), h, w)
    cpu, reason = clipped_cpu_depth(RayMeshIntersector(mesh), k, np.eye(4),
                                    np.ones((h, w), bool), contract)
    common = valid & (reason == 4)
    if common.sum() < 30 or np.max(abs(raw[common]-cpu[common])) > 1e-4:
        raise ValueError('Cross-near-plane polygon/ray interpolation fixture failed')
    cases.append(dict(cross_near_triangle=True, common=int(common.sum()),
                      max_error_m=float(np.max(abs(raw[common]-cpu[common])))))
    del renderer
    return dict(passed=True, clip_contract=contract.as_dict(), cases=cases,
                implementation=implementation(),
                gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                optimizer_updates=0, physical_accuracy_established=False)


def distribution(values):
    values = np.asarray(values)
    return dict(zip(('min', 'p50', 'p95', 'p99', 'max'),
                    np.quantile(values, [0, .5, .95, .99, 1]).tolist())) if len(values) else None


def depth_panel(depth, support):
    value = np.clip(np.log(np.maximum(depth, .01)/.01)/np.log(2000), 0, 1)
    color = np.stack([value, 1-abs(2*value-1), 1-value], -1)
    color[~support] = 0
    return Image.fromarray((color*255).astype(np.uint8))


def save_page(row, old, old_valid, raw, reference, final, support, target):
    with Image.open(row['image']) as f:
        rgb = f.convert('RGB')
    panels = [(rgb, 'Original perspective DSLR RGB'),
        (depth_panel(old, old_valid), 'OLD cached CANDIDATE (.01--100); NOT admitted'),
        (depth_panel(raw, raw > 0), 'NEW raw CUDA (.05--20)'),
        (depth_panel(reference, reference > 0), 'NEW raw CPU PyRender (.05--20)'),
        (depth_panel(final, support), 'NEW final candidate depth; NOT admitted'),
        (Image.fromarray((support*255).astype(np.uint8)).convert('RGB'), 'NEW final support')]
    page = Image.new('RGB', (1440, 850), 'white')
    draw = ImageDraw.Draw(page)
    draw.text((8, 8), f"{row['scene']} / {row['view']} | fixed log palette .01--20m; black missing", fill='black')
    for i, (panel, label) in enumerate(panels):
        x, y = (i % 3)*480, 35+(i//3)*400
        draw.text((x+5, y), label, fill='black')
        panel = ImageOps.contain(panel, (470, 360),
            Image.Resampling.LANCZOS if i == 0 else Image.Resampling.NEAREST)
        page.paste(panel, (x+5, y+25))
    temp = target.with_suffix('.png.incoming')
    page.save(temp, format='PNG')
    os.rename(temp, target)


def development(selection_path, fixture_path, output):
    contract = MeshClipContract()
    selection_ref, fixture_ref = file_ref(selection_path), file_ref(fixture_path)
    selected, fixture = checked_json(selection_ref), checked_json(fixture_ref)
    if (selected['stage'] != 'v2_validation_identities_frozen_not_rendered'
            or selected['clip_contract'] != contract.as_dict()
            or fixture['passed'] is not True or fixture['clip_contract'] != contract.as_dict()
            or fixture['implementation'] != implementation()
            or selected['confirmation_execution_allowed'] is not False):
        raise ValueError('Frozen selection, exact new contract and actual fixture pass required')
    rows = [r for r in checked_json(selected['selection']) if r['role'] != 'confirmation']
    if len(rows) != 35 or any(r['role'] not in ('known_quarantine', 'ordinary_control') for r in rows):
        raise ValueError('Exactly35 development images; confirmation remains sealed')
    output.mkdir(exist_ok=False)
    status = dict(stage='running', completed=0, expected=35, started_at=time.time(),
                  selection=selection_ref, fixture=fixture_ref, optimizer_updates=0,
                  training_admission=False, clip_contract=contract.as_dict())
    write_json(output/'status.json', status)
    pyrender, software, backend = software_backend()
    active, mesh, renderer, intersector = None, None, None, None
    records = []
    try:
        for chosen in sorted(rows, key=lambda r: (r['group'], r['corpus_ordinal'])):
            row, ordinal = chosen['image'], chosen['corpus_ordinal']
            receipt_path = checked_path(chosen['old_receipt'])
            receipt = json.loads(receipt_path.read_text())
            entries = [e for e in receipt['entries'] if e['key'] == chosen['old_entry_key']]
            if len(entries) != 1:
                raise ValueError('Unique original payload identity required')
            entry = entries[0]
            actual_row = dict(entry['image'])
            if actual_row.pop('corpus_ordinal', ordinal) != ordinal or actual_row != row:
                raise ValueError('Frozen image and old receipt differ')
            old_depth, old_valid, old_k, meta = read_geometry_member(receipt_path.parent/receipt['tar'], entry)
            if meta['policy_sha256'] != chosen['old_policy_sha256']:
                raise ValueError('Original candidate policy differs')
            proofs = {p['source']: p['sha256'] for p in meta['source_proofs']}
            scene = Path(row['image']).parents[2]
            if active != row['scene']:
                del renderer, mesh, intersector
                renderer = mesh = intersector = None
                gc.collect()
                mesh_path = scene/'scans/mesh_aligned_0.05.ply'
                pose_path = scene/'dslr/colmap/images.txt'
                camera_path = scene/'dslr/nerfstudio/transforms_undistorted.json'
                source_refs = [file_ref(p) for p in (mesh_path, pose_path, camera_path)]
                if any(proofs.get(p['path']) != p['sha256'] for p in source_refs):
                    raise ValueError('Original source mesh/pose/camera changed')
                mesh = trimesh.load_mesh(mesh_path, process=False)
                renderer = CudaMeshDepth(mesh.vertices, mesh.faces, clip_contract=contract)
                intersector = RayMeshIntersector(mesh)
                calibration = json.loads(camera_path.read_text())
                frames = {Path(f['file_path']).name: f for f in calibration['frames']}
                poses = colmap_poses(pose_path)
                active = row['scene']
            if (sha_file(row['image']) != row['sha256']
                    or sha_file(row['mask']) != proofs.get(row['mask'])
                    or any(proofs.get(p['path']) != p['sha256'] for p in source_refs)):
                raise ValueError('Original RGB/mask/source identity differs')
            with Image.open(row['image']) as image:
                if image.size != (row['width'], row['height']) or image.getexif().get(274, 1) != 1:
                    raise ValueError('Original RGB grid/orientation differs')
            k, pose = camera_for_row(row, calibration, frames, poses)
            h, w = round(row['height']*640/row['width']), 640
            if old_depth.shape != (h, w) or not np.array_equal(k, old_k):
                raise ValueError('New and old pinhole grids differ')
            raw, raw_valid, _ = renderer.render(k, pose, h, w)
            stable_depth, stable, _, raster_stats = renderer.render_conservative(k, pose, h, w)
            anonymity = conservative_anonymization(row['mask'], (row['height'], row['width']), (h, w))
            selected_mask = stable & anonymity
            depth, support, reason, stats = verify_clipped_cpu_intersections(
                intersector, k, pose, stable_depth, selected_mask, contract)
            reference = software_render(pyrender, software, mesh, k, pose, h, w,
                                         contract.near_m, contract.far_m)
            reference_valid = contract.interior(reference)
            common = support & reference_valid
            public_errors = abs(depth[common].astype(float)-reference[common])
            raw_common = raw_valid & (reference > 0)
            record = dict(corpus_ordinal=ordinal, role=chosen['role'], group=chosen['group'],
                image=dict(path=row['image'], sha256=row['sha256']), mask=file_ref(Path(row['mask'])),
                source_refs=source_refs, old_receipt=chosen['old_receipt'], old_entry_key=chosen['old_entry_key'],
                old_policy_sha256=chosen['old_policy_sha256'], intrinsics=k.tolist(), c2w=pose.tolist(),
                grid=[h, w], raster_stats=raster_stats, verification=stats,
                old_support_pixels=int(old_valid.sum()), final_support_pixels=int(support.sum()),
                common_old_new_support=int((old_valid & support).sum()),
                union_old_new_support=int((old_valid | support).sum()),
                new_only_support=int((~old_valid & support).sum()),
                lost_old_support=int((old_valid & ~support).sum()),
                raw_public_support_disagreement=int((raw_valid != (reference > 0)).sum()),
                raw_public_common_error_m=distribution(abs(raw[raw_common]-reference[raw_common])),
                final_public_missing=int((support & ~reference_valid).sum()),
                final_public_error_m=distribution(public_errors),
                final_public_over_2mm=int((public_errors > .002).sum()),
                minimum_support_passed=bool(support.mean() >= .05),
                public_numerical_parity_passed=bool(support.any() and common.sum() == support.sum()
                    and np.all(public_errors <= .002)),
                training_admission=False, source_quality_accepted=False, optimizer_updates=0)
            arrays = output/f'{ordinal}.npz'
            temp = arrays.with_suffix('.npz.incoming')
            with temp.open('xb') as f:
                np.savez_compressed(f, old_depth=old_depth, old_valid=old_valid,
                    raw_cuda_depth=raw, raw_cuda_valid=raw_valid, raw_pyrender_depth=reference,
                    stable_depth=stable_depth, stable_mask=stable, anonymity_mask=anonymity,
                    selected_mask=selected_mask, final_depth=depth, final_support=support,
                    cpu_reason=reason, intrinsics=k, c2w=pose)
            os.rename(temp, arrays)
            page = output/f'{ordinal}.png'
            save_page(row, old_depth, old_valid, raw, reference, depth, support, page)
            record.update(arrays=file_ref(arrays), page=file_ref(page))
            publish_json(output/f'{ordinal}.json', record)
            records.append(file_ref(output/f'{ordinal}.json'))
            status.update(completed=len(records), last_ordinal=ordinal, updated_at=time.time())
            write_json(output/'status.json', status)
            print(json.dumps({k: record[k] for k in ('corpus_ordinal', 'role', 'final_support_pixels',
                'final_public_missing', 'final_public_over_2mm', 'public_numerical_parity_passed')}), flush=True)
        publish_json(output/'report.json', dict(stage='development_rendered_not_reviewed',
            selection=selection_ref, fixture=fixture_ref, clip_contract=contract.as_dict(),
            records=records, completed=len(records), confirmation_images_read=0,
            software_reference=backend, code=file_ref(Path(__file__)),
            implementation=implementation(), gpu_used=True,
            training_admission=False, source_quality_accepted=False, optimizer_updates=0))
        status.update(stage='complete', completed_at=time.time())
        write_json(output/'status.json', status)
    except BaseException as error:
        status.update(stage='failed', error=type(error).__name__, message=str(error)[:1500])
        write_json(output/'status.json', status)
        raise
    finally:
        software.delete()


if __name__ == '__main__':
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['fixtures', 'development'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--fixture', type=Path)
    args = parser.parse_args()
    if args.mode == 'fixtures':
        args.output.mkdir(exist_ok=False)
        result = synthetic_gate()
        result['code'] = file_ref(Path(__file__))
        publish_json(args.output/'report.json', result)
        print(json.dumps(dict(passed=result['passed'], cases=len(result['cases']))))
    else:
        if args.selection is None or args.fixture is None:
            parser.error('Development requires frozen selection and actual fixture report')
        development(args.selection, args.fixture, args.output)
