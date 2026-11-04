"""CPU-only PyRender clipping comparison on three previously diagnosed views.

Uses the public DUSt3R mesh/camera/render core on our fixed pinhole ray grid.
This is not an end-to-end reproduction of its input selection/undistortion, nor
of the ScanNet++ renderpy backend. Never produces admitted training labels.
"""
import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from audit_scannetpp_saved_clip_intervals import check_ray, interval_hit
from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import publish_json


def backend():
    required = dict(PYOPENGL_PLATFORM='osmesa', LIBGL_ALWAYS_SOFTWARE='1',
                    GALLIUM_DRIVER='llvmpipe', LP_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    if any(os.environ.get(k) != v for k, v in required.items()):
        raise ValueError('Explicit isolated OSMesa software-only environment required')
    import pyglet
    pyglet.options['headless'] = True
    pyglet.options['shadow_window'] = False
    import pyrender
    from OpenGL import GL
    renderer = pyrender.OffscreenRenderer(64, 48)
    info = {k: GL.glGetString(v).decode() for k, v in (
        ('renderer', GL.GL_RENDERER), ('vendor', GL.GL_VENDOR), ('version', GL.GL_VERSION))}
    if 'llvmpipe' not in info['renderer'].lower():
        renderer.delete()
        raise ValueError('Software renderer identity not verified')
    info['pyrender'] = pyrender.__version__
    info['source_files'] = [file_ref(Path(pyrender.__file__).parent / p) for p in
                            ('camera.py', 'renderer.py', 'platforms/osmesa.py')]
    return pyrender, renderer, info


def render(pyrender, renderer, mesh, k, pose, height, width, near, far):
    if not np.allclose(np.asarray(k)[[0, 1], [1, 0]], 0, rtol=0, atol=1e-12):
        raise ValueError('This public renderer comparison requires zero-skew camera')
    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    camera = pyrender.camera.IntrinsicsCamera(
        k[0, 0]*width, k[1, 1]*height, k[0, 2]*width, k[1, 2]*height,
        znear=near, zfar=far)
    scene.add(camera, pose=pose @ np.diag([1., -1., -1., 1.]))
    renderer.viewport_width, renderer.viewport_height = width, height
    _, depth = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
    if depth.shape != (height, width) or not np.isfinite(depth).all() or (depth < 0).any():
        raise ValueError('Invalid raw rendered depth')
    return depth


def preflight(pyrender, renderer):
    import trimesh
    h, w = 48, 64
    k = np.array([[.8, 0, .43], [0, 1.1, .57], [0, 0, 1.]])
    pose = np.eye(4)
    a = .4
    pose[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    pose[:3, 3] = [3, -2, 1]
    xy = np.array([[-5, -5], [5, -5], [5, 5], [-5, 5.]])
    front = np.column_stack([xy, np.full(4, .02)])
    rear = np.column_stack([xy, 2+.1*xy[:, 0]+.08*xy[:, 1]])
    vertices = np.concatenate([front, rear]) @ pose[:3, :3].T+pose[:3, 3]
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    x, y = np.meshgrid((np.arange(w)+.5)/w, (np.arange(h)+.5)/h)
    dx, dy = (x-k[0, 2])/k[0, 0], (y-k[1, 2])/k[1, 1]
    expected = 2/(1-.1*dx-.08*dy)
    old = render(pyrender, renderer, mesh, k, pose, h, w, .01, 100.)
    clipped = render(pyrender, renderer, mesh, k, pose, h, w, .05, 20.)
    reverse = trimesh.Trimesh(vertices=vertices, faces=faces[:, ::-1], process=False)
    reversed_depth = render(pyrender, renderer, reverse, k, pose, h, w, .05, 20.)
    far_mesh = trimesh.Trimesh(vertices=(rear+[0, 0, 28]) @ pose[:3, :3].T+pose[:3, 3],
                              faces=faces[:2], process=False)
    far = render(pyrender, renderer, far_mesh, k, pose, h, w, .05, 20.)
    errors = dict(front_max_error_m=float(np.max(abs(old-.02))),
                  rear_max_error_m=float(np.max(abs(clipped-expected))),
                  reversed_winding_max_error_m=float(np.max(abs(reversed_depth-clipped))),
                  far_clipped_nonzero_pixels=int(np.count_nonzero(far)))
    if (errors['front_max_error_m'] > 2e-5 or errors['rear_max_error_m'] > 2e-4
            or errors['reversed_winding_max_error_m'] > 2e-4 or errors['far_clipped_nonzero_pixels']):
        raise ValueError('Software rendering ray-grid/clipping fixture failed: '+json.dumps(errors))
    return dict(passed=True, errors=errors, offcenter_camera=True, slanted_rear_surface=True,
                rotated_translated_camera=True, culling_disabled=True,
                normalized_half_center_ray_grid=True, fixture_only=True)


def page(case, depths, target):
    data = checked_path(case['image'])
    with Image.open(data) as original:
        rgb = original.convert('RGB')
    width = 400
    height = round(width*case['grid'][0]/case['grid'][1])
    result = Image.new('RGB', (width*4, height+75), 'white')
    draw = ImageDraw.Draw(result)
    draw.text((5, 5), f"{case['corpus_ordinal']} | RGB + RAW CPU renders | fixed log palette .01--5m; black=missing", fill='black')
    panels = [rgb.resize((width, height), Image.Resampling.LANCZOS)]
    labels = ['Original RGB', 'Near .01 / far 100 m', 'Near .05 / far 100 m', 'Near .05 / far 20 m']
    for depth in depths:
        valid = depth > 0
        q = np.clip(np.log(np.maximum(depth, .01)/.01)/np.log(500), 0, 1)
        color = np.stack([q, 1-abs(2*q-1), 1-q], -1)
        color[~valid] = 0
        panel = Image.fromarray((255*color).astype(np.uint8)).resize((width, height), Image.Resampling.NEAREST)
        panels.append(panel)
    for i, panel in enumerate(panels):
        draw.text((5+i*width, 23), labels[i], fill='black')
        result.paste(panel, (i*width, 45))
    draw.text((5, height+50), 'Diagnostic only: no conservative masks, CPU parity mask, label admission or repair.', fill='black')
    incoming = target.with_suffix('.png.incoming')
    with incoming.open('xb') as f:
        result.save(f, format='PNG')
    incoming.replace(target)


def run(source_path, output, ordinals):
    import trimesh
    source_ref = file_ref(source_path)
    source = checked_json(source_ref)
    if (source['stage'] != 'bounded_raw_mesh_ray_diagnosis_complete'
            or source['training_admission'] is not False or source['optimizer_updates'] != 0
            or not 1 <= len(set(ordinals)) == len(ordinals) <= 3):
        raise ValueError('Completed saved diagnosis and at most three unique explicit views required')
    chosen = []
    for ref in source['cases']:
        if int(Path(ref['path']).stem) in ordinals:
            chosen.append((ref, checked_json(ref)))
    if len(chosen) != len(ordinals):
        raise ValueError('Missing selected case')
    output.mkdir(exist_ok=False)
    pyrender, renderer, renderer_info = backend()
    try:
        proof = preflight(pyrender, renderer)
        publish_json(output/'preflight.json', dict(renderer=renderer_info, **proof))
        results = []
        for ref, case in chosen:
            meshes = [p for p in case['source_refs'] if Path(p['path']).name == 'mesh_aligned_0.05.ply']
            if len(meshes) != 1:
                raise ValueError('Exact mesh source required')
            mesh_path = checked_path(meshes[0])
            mesh = trimesh.load_mesh(mesh_path, process=False)
            k, pose = np.asarray(case['intrinsics']), np.asarray(case['c2w'])
            h, w = case['grid']
            intervals = [(.01, 100.), (.05, 100.), (.05, 20.)]
            depths = [render(pyrender, renderer, mesh, k, pose, h, w, *bounds) for bounds in intervals]
            samples = []
            for ray in case['rays']:
                check_ray(case, ray)
                y, x = ray['pixel_yx']
                comparison = []
                for depth, bounds in zip(depths, intervals):
                    ideal = interval_hit(ray['analytical'], *bounds)
                    hit = ideal['hit']
                    expected = hit['camera_z_m'] if hit else 0.
                    comparison.append(dict(clip_m=list(bounds), rendered_z_m=float(depth[y, x]),
                        ideal_outcome=ideal['outcome'], ideal_z_m=expected,
                        abs_difference_m=abs(float(depth[y, x])-expected)))
                samples.append(dict(pixel_yx=[y, x], original_cached_valid=ray['cached_valid'],
                                    original_cached_z_m=ray['cached_z_m'], comparisons=comparison))
            target = output/f"{case['corpus_ordinal']}.npz"
            with target.with_suffix('.npz.incoming').open('xb') as f:
                np.savez_compressed(f, near01_far100=depths[0], near05_far100=depths[1], near05_far20=depths[2])
            target.with_suffix('.npz.incoming').replace(target)
            visual = output/f"{case['corpus_ordinal']}.png"
            page(case, depths, visual)
            record = dict(corpus_ordinal=case['corpus_ordinal'], source_case=ref, mesh=meshes[0],
                image=case['image'], renderer_grid=[h, w], saved_rays=samples,
                arrays=file_ref(target), page=file_ref(visual),
                positive_pixels=[int((d>0).sum()) for d in depths],
                near_change_over_2mm_pixels=int((abs(depths[0]-depths[1])>.002).sum()),
                far_change_over_2mm_pixels=int((abs(depths[1]-depths[2])>.002).sum()))
            publish_json(output/f"{case['corpus_ordinal']}.json", record)
            results.append(record)
            print(json.dumps(dict(completed=len(results), ordinal=case['corpus_ordinal'])), flush=True)
            del mesh, depths
            gc.collect()
        report = dict(stage='software_clip_render_complete_not_visually_reviewed', source=source_ref,
            code=file_ref(Path(__file__)), preflight=file_ref(output/'preflight.json'), cases=results,
            comparison='DUSt3R PyRender mesh/camera/render core on fixed existing pinhole grid',
            full_upstream_preprocessing_reproduced=False, official_renderpy_executed=False,
            gpu_used=False, software_renderer_verified=True, optimizer_updates=0,
            source_quality_accepted=False, geometry_edited=False, source_flags_cleared=False,
            training_admission=False, new_trainable_labels=False,
            limitations=['Raw diagnostic renders omit producer visibility/anonymity/CPU parity masks.',
                         'Farther visible mesh surfaces are not independent physical ground truth.',
                         'Three selected views cannot establish whole-corpus error prevalence.'])
        checked_json(source_ref)
        publish_json(output/'report.json', report)
        return report
    finally:
        renderer.delete()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--ordinals', required=True, nargs='+', type=int)
    a = p.parse_args()
    run(a.source, a.output, a.ordinals)
