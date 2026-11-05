"""One sealed 16-view conditional-source confirmation; no training or admission.

The fixed sparse grid never creates dense source support. All raw source gaps,
unsupported slots and both prospectively selected scanners remain in evidence.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from indoor_download import write_json
from probe_scannetpp_exact_target_visibility import query_targets
from probe_scannetpp_triangle_scan_support import directions, scanner_pose_diagnostics
from probe_scannetpp_v2_source_pilot import bound_path, compare_slot, read, target_slots
from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import publish_json, sha_file
from radio_moge.mesh_clip import MeshClipContract, select_public_consensus
from radio_moge.projected_footprint import projected_footprint
from radio_moge.saved_angle_match import match_saved_angles
from radio_moge.scanner_support import ScannerSupportContract, paired_support
from radio_moge.source_footprint import extract_topology

PLAN_SHA = '8365609ec741092d3543e640afd77374ee219a6b7786953266346919a4be59d0'
POLICY_SHA = '3216cac67ac2a03854858b174a6362f8bddbb73940c8018b610719de0ac23d75'
ORIGINAL_SHA = '37b957e897013bf3e58d9636b279d70ad1c1598c4f90f60bd7a89cae8d87c322'
EXCLUSIONS_SHA = '58a1b02acfe087db2c28e4ae334c762359d0efc6ab16a6d99dd3deba7ab3d86b'


def implementation():
    root = Path(__file__).resolve().parents[1]
    names = ['scripts/run_scannetpp_source_confirmation.py',
        'scripts/probe_scannetpp_native_v2.py', 'scripts/cache_scannetpp_geometry.py',
        'scripts/probe_scannetpp_software_clip_render.py',
        'scripts/probe_scannetpp_exact_target_visibility.py',
        'scripts/probe_scannetpp_v2_source_pilot.py',
        'scripts/probe_scannetpp_triangle_scan_support.py', 'scripts/indoor_download.py',
        'scripts/probe_scannetpp_native_near_surfaces.py',
        'scripts/audit_scannetpp_saved_clip_intervals.py',
        'src/radio_moge/mesh_clip.py', 'src/radio_moge/mesh_depth.py',
        'src/radio_moge/geometry_sources.py', 'src/radio_moge/geometry_cache.py',
        'src/radio_moge/corpus_admission.py', 'src/radio_moge/source_footprint.py',
        'src/radio_moge/projected_footprint.py', 'src/radio_moge/saved_angle_match.py',
        'src/radio_moge/scanner_support.py']
    return {name: sha_file(root/name) for name in names}


def validate_scope(rows, originals):
    expected = [r for r in originals if r['role'] == 'confirmation']
    other = {r['group'] for r in originals if r['role'] != 'confirmation'}
    if (len(rows) != 16 or len(expected) != 16
            or [r['original_selection'] for r in rows] != expected
            or len({r['corpus_ordinal'] for r in rows}) != 16
            or len({r['scene'] for r in rows}) != 16):
        raise ValueError('Exactly the original sixteen disjoint confirmation identities required')
    for row in rows:
        chosen = row['original_selection']
        rgb = chosen['image']
        h, w = round(rgb['height']*640/rgb['width']), 640
        pixels = [[int((y+.5)*h/8), int((x+.5)*w/12)] for y in range(8) for x in range(12)]
        if (row['role'] != 'confirmation' or row['corpus_ordinal'] != chosen['corpus_ordinal']
                or row['scene'] != rgb['scene'] or chosen['group'] != 'scannetpp:'+rgb['scene']
                or chosen['group'] in other or row['grid'] != [h, w] or row['pixel_yx'] != pixels
                or rgb['dataset'] != 'scannetpp_dslr' or rgb['split'] != 'train'
                or rgb.get('image_rotation_cw', 0) != 0
                or [s['scan'] for s in row['scanners']] != ['0', '1']):
            raise ValueError('Perspective RGB, original grid and fixed scanner identities required')
        for scan in row['scanners']:
            scanner_pose_diagnostics(np.asarray(scan['pose']))
            if set(scan['inputs']) != {'range', 'mask', 'azim', 'elev'}:
                raise ValueError('All four scanner diagnostic sources required')
        if np.array_equal(row['scanners'][0]['pose'], row['scanners'][1]['pose']):
            raise ValueError('Distinct fixed scanner stations required')


def validate_fixture(fixture):
    if (fixture.get('stage') != 'current_confirmation_fixtures_passed'
            or fixture.get('passed') is not True or fixture.get('implementation') != implementation()
            or fixture.get('policy') != asdict(ScannerSupportContract())
            or fixture.get('clip_contract') != MeshClipContract().as_dict()
            or fixture.get('confirmation_images_read') != 0):
        raise ValueError('Fresh actual fixtures for exact current implementation required')


def fixtures(output):
    import torch
    import trimesh
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    from probe_scannetpp_native_v2 import software_backend, synthetic_gate
    from probe_scannetpp_software_clip_render import preflight, render
    from radio_moge.mesh_depth import CudaMeshDepth
    from radio_moge.mesh_clip import verify_clipped_cpu_intersections

    output.mkdir(exist_ok=False)
    base = synthetic_gate()
    pyrender, software, backend = software_backend()
    cases = []
    contract = MeshClipContract()
    k, pose, h, w = np.array([[.8, 0, .43], [0, 1.1, .57], [0, 0, 1.]]), np.eye(4), 32, 48
    try:
        public_camera_fixture = preflight(pyrender, software)
        # Actual CUDA + public software + CPU, including float32 near/far guards.
        for z, keep in [(2., True), (.08, True), (.0501, False), (19.9999, False), (21., False)]:
            vertices = np.array([[-40., -40, z], [40, -40, z], [40, 40, z], [-40, 40, z]])
            mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [0, 2, 3]], process=False)
            gpu = CudaMeshDepth(vertices, mesh.faces, clip_contract=contract)
            raw, valid, _ = gpu.render(k, pose, h, w)
            depth, supported, _, _ = verify_clipped_cpu_intersections(
                RayMeshIntersector(mesh), k, pose, raw, valid, contract)
            public = render(pyrender, software, mesh, k, pose, h, w, contract.near_m, contract.far_m)
            final, mask, _, stats = select_public_consensus(depth, supported, public, contract)
            if (keep and not mask.all()) or (not keep and mask.any()):
                raise ValueError('Actual public-consensus/float32 boundary fixture failed')
            if not np.array_equal(final[mask], depth[mask]) or (mask & ~supported).any():
                raise ValueError('Consensus replaced a CPU value or created support')
            cases.append(dict(z_m=z, expected_support=keep, stats=stats))
            del gpu, mesh
        result = dict(stage='current_confirmation_fixtures_passed', passed=True,
            base_actual_CUDA_fixtures=base, actual_consensus_cases=cases,
            actual_public_camera_fixture=public_camera_fixture,
            implementation=implementation(), software_reference=backend,
            policy=asdict(ScannerSupportContract()), clip_contract=contract.as_dict(),
            confirmation_images_read=0, optimizer_updates=0, training_admission=False,
            GPU=torch.cuda.get_device_name(), physical_accuracy_established=False)
        publish_json(output/'report.json', result)
        return result
    finally:
        software.delete()


def load_scope(root, fixture_path):
    plan_ref = file_ref(root/'scannetpp-native-v2-confirmation-policy-plan-01/report.json')
    if plan_ref['sha256'] != PLAN_SHA:
        raise ValueError('Frozen confirmation input plan changed')
    plan = read(plan_ref, root)
    if (plan['source_policy']['sha256'] != POLICY_SHA or plan['original_selection']['sha256'] != ORIGINAL_SHA
            or plan['exclusions']['sha256'] != EXCLUSIONS_SHA
            or plan['policy'] != asdict(ScannerSupportContract())
            or plan['clip_contract'] != MeshClipContract().as_dict()):
        raise ValueError('Original selection, exclusions and frozen policy required')
    policy = read(plan['source_policy'], root)
    if (policy['policy'] != plan['policy'] or policy['code'][1]['sha256'] !=
            implementation()['src/radio_moge/scanner_support.py']):
        raise ValueError('Frozen source decision implementation changed')
    original = read(plan['original_selection'], root)
    rows = read(plan['selection'], root)
    validate_scope(rows, read(original['selection'], root))
    fixture_ref = file_ref(fixture_path)
    fixture = read(fixture_ref, root)
    validate_fixture(fixture)
    excluded = [json.loads(s)['corpus_ordinal'] for s in bound_path(plan['exclusions'], root).read_text().splitlines()]
    if len(excluded) != 4190 or len(set(excluded)) != 4190:
        raise ValueError('All prior exact dispositions required')
    return plan_ref, plan, rows, fixture_ref, fixture, set(excluded)


def numerical_case(chosen, contract, pyrender, software):
    import trimesh
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    from cache_scannetpp_geometry import camera_for_row, colmap_poses, conservative_anonymization
    from probe_scannetpp_software_clip_render import render
    from radio_moge.corpus_admission import checked_path
    from radio_moge.geometry_cache import read_geometry_member
    from radio_moge.mesh_depth import CudaMeshDepth
    from radio_moge.mesh_clip import verify_clipped_cpu_intersections

    row, ordinal = chosen['image'], chosen['corpus_ordinal']
    receipt_path = checked_path(chosen['old_receipt'])
    receipt = json.loads(receipt_path.read_text())
    entries = [e for e in receipt['entries'] if e['key'] == chosen['old_entry_key']]
    if len(entries) != 1:
        raise ValueError('Unique original payload identity required')
    entry = entries[0]
    actual_row = dict(entry['image'])
    if actual_row.pop('corpus_ordinal', ordinal) != ordinal or actual_row != row:
        raise ValueError('Original image and receipt identity mismatch')
    old_depth, old_valid, old_k, meta = read_geometry_member(receipt_path.parent/receipt['tar'], entry)
    if meta['policy_sha256'] != chosen['old_policy_sha256']:
        raise ValueError('Original native policy changed')
    proofs = {p['source']: p['sha256'] for p in meta['source_proofs']}
    base = Path(row['image']).parents[2]
    paths = [base/'scans/mesh_aligned_0.05.ply', base/'dslr/colmap/images.txt',
             base/'dslr/nerfstudio/transforms_undistorted.json']
    source_refs = [file_ref(p) for p in paths]
    if (any(proofs.get(p['path']) != p['sha256'] for p in source_refs)
            or sha_file(row['image']) != row['sha256'] or sha_file(row['mask']) != proofs.get(row['mask'])):
        raise ValueError('Mesh, camera, RGB or mask bytes differ')
    with Image.open(row['image']) as im:
        if im.size != (row['width'], row['height']) or im.getexif().get(274, 1) != 1:
            raise ValueError('Original perspective RGB grid/orientation changed')
    calibration = json.loads(paths[2].read_text())
    frames = {Path(f['file_path']).name: f for f in calibration['frames']}
    k, pose = camera_for_row(row, calibration, frames, colmap_poses(paths[1]))
    h, w = round(row['height']*640/row['width']), 640
    if old_depth.shape != (h, w) or not np.array_equal(k, old_k):
        raise ValueError('Original/native pinhole grid mismatch')
    mesh = trimesh.load_mesh(paths[0], process=False)
    engine = RayMeshIntersector(mesh)
    gpu = CudaMeshDepth(mesh.vertices, mesh.faces, clip_contract=contract)
    raw, raw_mask, _ = gpu.render(k, pose, h, w)
    stable_depth, stable, _, raster_stats = gpu.render_conservative(k, pose, h, w)
    anonymity = conservative_anonymization(row['mask'], (row['height'], row['width']), (h, w))
    selected = stable & anonymity
    cpu, support, reason, stats = verify_clipped_cpu_intersections(engine, k, pose, stable_depth, selected, contract)
    public = render(pyrender, software, mesh, k, pose, h, w, contract.near_m, contract.far_m)
    depth, valid, public_reason, consensus_stats = select_public_consensus(cpu, support, public, contract)
    arrays = dict(old_depth=old_depth, old_valid=old_valid, raw_cuda_depth=raw, raw_cuda_valid=raw_mask,
        stable_depth=stable_depth, stable_mask=stable, anonymity_mask=anonymity, selected_mask=selected,
        cpu_depth=cpu, cpu_support=support, cpu_reason=reason, raw_pyrender_depth=public,
        final_depth=depth, final_support=valid, consensus_reason=public_reason, intrinsics=k, c2w=pose)
    record = dict(corpus_ordinal=ordinal, role='confirmation', scene=row['scene'], group=row['group'],
        image=dict(path=row['image'], sha256=row['sha256']), mask=file_ref(Path(row['mask'])),
        old_receipt=chosen['old_receipt'], old_entry_key=chosen['old_entry_key'],
        old_policy_sha256=chosen['old_policy_sha256'], source_refs=source_refs,
        source_flags=meta.get('source_review', meta.get('source_review_flags')),
        original_metadata=meta, grid=[h, w], intrinsics=k.tolist(), c2w=pose.tolist(),
        raster_stats=raster_stats, verification=stats, public_consensus=consensus_stats,
        minimum_numerical_support_passed=bool(valid.mean() >= .05),
        training_admission=False, source_quality_accepted=False, optimizer_updates=0)
    del gpu
    return record, arrays, mesh, engine


def source_case(binding, record, arrays, mesh, engine, excluded, root, output):
    ordinal = binding['corpus_ordinal']
    targets = target_slots(mesh, engine, record, binding['pixel_yx'], arrays['final_depth'],
                           arrays['final_support'], MeshClipContract())
    target_path = output/f'{ordinal}-targets.json'
    publish_json(target_path, dict(corpus_ordinal=ordinal, scene=binding['scene'], slots=targets,
                                  source_refs=record['source_refs']))
    poses = np.asarray(read(binding['scanner_poses'], root))
    stations, refs = [], []
    selected_faces = {t['face_id'] for t in targets if t['numerical_support']}
    for scan in binding['scanners']:
        pose = np.asarray(scan['pose'])
        if not np.array_equal(poses[int(scan['scan'])], pose):
            raise ValueError('Bound published scanner transform changed')
        raw = {}
        for kind, proof in scan['inputs'].items():
            with Image.open(bound_path(proof, root)) as im:
                raw[kind] = np.asarray(im).copy()
        if len({a.shape for a in raw.values()}) != 1 or any(a.ndim != 2 for a in raw.values()):
            raise ValueError('Original scanner grids differ')
        measures = [dict(compare_slot(t, raw, pose, engine), scanner_identity=[binding['scene'], scan['scan']])
                    for t in targets]
        exact = query_targets(targets, pose, engine)
        for m in measures:
            for s in m.get('discrete_angle_sensitivity', {}).get('samples', []):
                hit = s['nearest_whole_mesh_hit']
                if hit is not None:
                    selected_faces.add(hit['face_id'])
        station = dict(scan=scan['scan'], scene=binding['scene'], scanner_pose=pose.tolist(),
            inputs=scan['inputs'], targets=file_ref(target_path), measurements=measures, exact_target=exact)
        path = output/f'{ordinal}-scan{scan["scan"]}.json'
        publish_json(path, station)
        refs.append(file_ref(path))
        stations.append(station)
        del raw
    topology = (extract_topology(mesh.vertices, mesh.faces, selected_faces) if selected_faces
                else dict(faces=[], edge_incidence={}, global_faces=len(mesh.faces), selected_faces=0))
    topology_path = output/f'{ordinal}-topology.json'
    publish_json(topology_path, topology)
    points = []
    for target in targets:
        slot = target['slot']
        evidence = []
        for station, ref in zip(stations, refs):
            m = station['measurements'][slot]
            if target['numerical_support']:
                samples = m['discrete_angle_sensitivity']['samples']
                rays = directions(np.array([m['stored_azimuth_millirad']+s['azimuth_offset_millirad'] for s in samples]),
                                  np.array([m['stored_polar_millirad']+s['polar_offset_millirad'] for s in samples]))
                footprint = projected_footprint(target, [s['nearest_whole_mesh_hit'] for s in samples],
                    np.asarray(station['scanner_pose']), rays, topology)
                matches = match_saved_angles(m)
            else:
                footprint = dict(status='DSLR_numerically_unsupported', sampled_C0_correspondence=False)
                matches = dict(status='DSLR_numerically_unsupported')
            evidence.append(dict(corpus_ordinal=ordinal, scene=binding['scene'], slot=slot,
                pixel_yx=target['pixel_yx'], scan=station['scan'], station=ref,
                projected_footprint=footprint, angle_matches=matches,
                exact_target_visibility=station['exact_target'][slot]['visibility']))
        points.append(dict(corpus_ordinal=ordinal, scene=binding['scene'], slot=slot,
            pixel_yx=target['pixel_yx'], numerical_support=target['numerical_support'],
            decision=paired_support(evidence, ['0', '1'], target['numerical_support'], ordinal in excluded),
            station_evidence=evidence))
    path = output/f'{ordinal}-decisions.json'
    publish_json(path, points)
    return points, dict(targets=file_ref(target_path), stations=refs,
                       topology=file_ref(topology_path), decisions=file_ref(path))


def coverage(points, rows):
    expected = {(r['corpus_ordinal'], slot): tuple(yx) for r in rows for slot, yx in enumerate(r['pixel_yx'])}
    actual = {(p['corpus_ordinal'], p['slot']): tuple(p['pixel_yx']) for p in points}
    if len(points) != 1536 or len(actual) != len(points) or actual != expected:
        raise ValueError('All1536 original confirmation slots required without replacement')
    for p in points:
        evidence = p['station_evidence']
        if (len(evidence) != 2 or [s['scan'] for s in evidence] != ['0', '1']
                or any((s['corpus_ordinal'], s['slot'], tuple(s['pixel_yx'])) !=
                       (p['corpus_ordinal'], p['slot'], tuple(p['pixel_yx'])) for s in evidence)):
            raise ValueError('Both original3072station identities required')
    return dict(DSLR_slots=len(points), station_slots=2*len(points),
        conditional_sample_candidates=sum(p['decision']['conditional_sample_candidate'] for p in points),
        per_image=[dict(corpus_ordinal=r['corpus_ordinal'], sampled_slots=96,
            conditional_sample_candidates=sum(p['decision']['conditional_sample_candidate'] for p in points
                                               if p['corpus_ordinal'] == r['corpus_ordinal'])) for r in rows],
        point_reason_counts=dict(Counter(reason for p in points for reason in p['decision']['reasons'])),
        source_accepted_images=0, dense_source_mask_constructed=False, sparse_coverage_extrapolated=False)


def page(chosen, arrays, points, path):
    from probe_scannetpp_native_v2 import depth_panel
    with Image.open(chosen['image']['image']) as im:
        rgb = im.convert('RGB')
    h, w = arrays['final_depth'].shape
    marked = rgb.resize((w, h), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(marked)
    for p in points:
        y, x = p['pixel_yx']
        color = '#00ff00' if p['decision']['conditional_sample_candidate'] else ('#ff3030' if p['numerical_support'] else '#ffcc00')
        draw.ellipse((x-3, y-3, x+3, y+3), outline=color, width=2)
    panels = [(rgb, 'Perspective DSLR RGB'),
        (depth_panel(arrays['old_depth'], arrays['old_valid']), 'Old native candidate; NOT accepted'),
        (depth_panel(arrays['final_depth'], arrays['final_support']), 'Current numerical candidate; NOT source-accepted'),
        (Image.fromarray((arrays['final_support']*255).astype(np.uint8)).convert('RGB'), 'Numerical support ONLY; not source support'),
        (marked, '96 probes: green conditional / red unresolved / yellow numerical missing')]
    canvas = Image.new('RGB', (1440, 850), 'white')
    d = ImageDraw.Draw(canvas)
    d.text((5, 5), f"{chosen['corpus_ordinal']} {chosen['group']} | fixed depth palette .01--20m", fill='black')
    for i, (im, label) in enumerate(panels):
        x, y = (i % 3)*480, 30+(i//3)*400
        d.text((x+5, y), label, fill='black')
        tile = ImageOps.contain(im, (470, 360), Image.Resampling.LANCZOS if i in (0, 4) else Image.Resampling.NEAREST)
        canvas.paste(tile, (x+5, y+24))
    d.text((965, 455), 'Sparse probes do not create a dense mask.\nTwo fixed stations; no substitutions.\n1 cm engineering budget, not physical bound.\nNo training admission.', fill='black')
    temp = path.with_suffix('.png.incoming')
    canvas.save(temp, format='PNG')
    os.rename(temp, path)


def run(root, fixture_path, output):
    from probe_scannetpp_native_v2 import software_backend
    plan_ref, plan, rows, fixture_ref, fixture, excluded = load_scope(root, fixture_path)
    output.mkdir(exist_ok=False)
    publish_json(output/'execution-binding.json', dict(plan=plan_ref, fixture=fixture_ref,
        implementation=implementation(), policy=plan['policy'], selection=plan['selection'],
        execution_scope='16 confirmation views only; no optimizer or bulk generation'))
    status = dict(stage='confirmation_running', completed_images=0, expected_images=16,
        started_at=time.time(), optimizer_updates=0, training_admission=False)
    write_json(output/'status.json', status)
    pyrender, software, backend = software_backend()
    records, all_points = [], []
    try:
        if backend != fixture['software_reference']:
            raise ValueError('Actual software backend differs from current fixture')
        for binding in rows:
            ordinal = binding['corpus_ordinal']
            record, arrays, mesh, engine = numerical_case(binding['original_selection'], MeshClipContract(), pyrender, software)
            array_path = output/f'{ordinal}.npz'
            temp = array_path.with_suffix('.npz.incoming')
            with temp.open('xb') as f:
                np.savez_compressed(f, **arrays)
            os.rename(temp, array_path)
            points, sources = source_case(binding, record, arrays, mesh, engine, excluded, root, output)
            png = output/f'{ordinal}.png'
            page(binding['original_selection'], arrays, points, png)
            record.update(arrays=file_ref(array_path), page=file_ref(png), source_evidence=sources,
                conditional_samples=sum(p['decision']['conditional_sample_candidate'] for p in points),
                full_dense_source_support_evaluated=False, independent_physical_calibration=False)
            target = output/f'{ordinal}.json'
            publish_json(target, record)
            records.append(file_ref(target))
            all_points.extend(points)
            status.update(completed_images=len(records), last_ordinal=ordinal, updated_at=time.time())
            write_json(output/'status.json', status)
            print(json.dumps(dict(ordinal=ordinal, completed=len(records), conditional_samples=record['conditional_samples'])), flush=True)
            del record, arrays, mesh, engine, points
            gc.collect()
        result = dict(stage='confirmation_rendered_and_compared_not_visually_reviewed',
            plan=plan_ref, fixture=fixture_ref, records=records, implementation=implementation(),
            software_reference=backend, policy=plan['policy'], clip_contract=plan['clip_contract'],
            coverage=coverage(all_points, rows), images=16, distinct_scenes=16,
            confirmation_images_read=16, panorama_RGB_used=False, source_flags_retained=True,
            source_quality_accepted=False, training_admission=False, optimizer_updates=0,
            labels_changed=False, new_exclusions=0, engineering_budget_not_physical_bound=True,
            independent_physical_calibration=False, continuous_uncertainty_bound=False)
        publish_json(output/'report.json', result)
        write_json(output/'status.json', dict(status, stage='complete', report=file_ref(output/'report.json')))
        return result
    except BaseException as error:
        write_json(output/'status.json', dict(status, stage='failed', error=type(error).__name__, message=str(error)))
        raise
    finally:
        software.delete()


if __name__ == '__main__':
    import torch
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['fixtures', 'confirmation'], required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', type=Path)
    args = parser.parse_args()
    if args.mode == 'fixtures':
        fixtures(args.output)
    else:
        if args.fixture is None:
            parser.error('Current actual fixture report required')
        run(args.root.resolve(), args.fixture.resolve(), args.output.resolve())
