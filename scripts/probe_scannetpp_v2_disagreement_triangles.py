"""CPU-only bounded attribution of saved renderer disagreements; no label edits.

Re-render only the exact first-hit triangle, retaining world vertices and camera.
This distinguishes single-triangle rasterization from whole-mesh visibility;
it does not certify either renderer or the physical surface.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np

from probe_scannetpp_software_clip_render import backend, render
from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import publish_json


def remaining_disagreements(source_ref, source, analysis, previous):
    """Bind the saved32 pixels and subtract exactly the five already diagnosed."""
    if (analysis['source']['sha256'] != source_ref['sha256']
            or previous['source']['sha256'] != source_ref['sha256']
            or previous['stage'] != 'bounded_single_triangle_diagnosis_complete'
            or previous['training_admission'] is not False
            or analysis['training_admission'] is not False):
        raise ValueError('Exact unadmitted source and completed previous diagnosis required')
    bound = {int(Path(r['path']).stem): r for r in source['records']}
    rows = {}
    for row in analysis['disagreements']:
        ordinal, pixel = row['ordinal'], row['pixel_yx']
        if (type(ordinal) is not int or ordinal not in bound or len(pixel) != 2
                or any(type(n) is not int or n < 0 for n in pixel)):
            raise ValueError('Invalid saved disagreement identity')
        key = (ordinal, *pixel)
        if key in rows:
            raise ValueError('Duplicate saved disagreement')
        rows[key] = row
    done = set()
    for case in previous['cases']:
        key = (case['corpus_ordinal'], *case['pixel_yx'])
        if key not in rows or key in done or case['record'] != bound[key[0]]:
            raise ValueError('Unknown, duplicate or changed previous case')
        row = rows[key]
        if (case['arrays']['sha256'] != row['source']['sha256']
                or any(case[a] != row[b] for a, b in (
                    ('saved_cpu_z_m', 'depth'), ('saved_cuda_z_m', 'gpu'),
                    ('saved_public_z_m', 'software')))):
            raise ValueError('Previous saved depths or arrays changed')
        done.add(key)
    if len(rows) != 32 or len(done) != 5:
        raise ValueError('Frozen32 disagreements minus completed5 required')
    selected = {}
    for ordinal, y, x in sorted(set(rows)-done):
        selected.setdefault(ordinal, []).append((y, x))
    return selected, rows


def investigate(directory, output, analysis_path=None, previous_path=None):
    import trimesh
    from trimesh.ray.ray_pyembree import RayMeshIntersector

    source_ref = file_ref(directory/'report.json')
    source = checked_json(source_ref)
    if source['completed'] != 35 or source['training_admission'] is not False:
        raise ValueError('Closed unadmitted development evidence required')
    # Worst saved discrepancy, a clean control discrepancy, and the guard case.
    selected = {853213: [(39, 541)], 851268: [(274, 135)],
                1181343: [(149, 208)], 1563104: [(75, 610)],
                1461784: [(309, 518)]}
    expected, incremental = {}, None
    if (analysis_path is None) != (previous_path is None):
        raise ValueError('Both analysis and previous diagnosis required')
    if analysis_path is not None:
        analysis_ref, previous_ref = file_ref(analysis_path), file_ref(previous_path)
        selected, expected = remaining_disagreements(source_ref, source,
            checked_json(analysis_ref), checked_json(previous_ref))
        incremental = dict(analysis=analysis_ref, previous=previous_ref,
            selected=[dict(corpus_ordinal=i, pixel_yx=list(p)) for i, ps in selected.items() for p in ps],
            completed_cases_repeated=0)
    output.mkdir(exist_ok=False)
    if incremental:
        publish_json(output/'selection.json', incremental)
    bound = {Path(r['path']).stem: r for r in source['records']}
    pyrender, renderer, identity = backend()
    from OpenGL import GL
    identity.update({name: int(GL.glGetIntegerv(value)) for name, value in (
        ('sample_buffers', GL.GL_SAMPLE_BUFFERS),
        ('samples', GL.GL_SAMPLES), ('subpixel_bits', GL.GL_SUBPIXEL_BITS))})
    # GL_DEPTH_BITS is removed from this core profile; query the actual default
    # framebuffer attachment rather than assuming the requested OSMesa format.
    identity['depth_bits'] = int(GL.glGetFramebufferAttachmentParameteriv(
        GL.GL_FRAMEBUFFER, GL.GL_DEPTH, GL.GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE))
    identity['supports_framebuffers'] = renderer._platform.supports_framebuffers()
    identity['additional_sources'] = [file_ref(Path(pyrender.__file__).parent/n)
                                      for n in ('offscreen.py', 'mesh.py', 'primitive.py')]
    cases = []
    try:
        for ordinal, pixels in selected.items():
            record = checked_json(bound[str(ordinal)])
            refs = [p for p in record['source_refs'] if Path(p['path']).suffix == '.ply']
            if len(refs) != 1:
                raise ValueError('One exact original mesh required')
            mesh = trimesh.load_mesh(checked_path(refs[0]), process=False)
            engine = RayMeshIntersector(mesh)
            pose, k = np.array(record['c2w']), np.array(record['intrinsics'])
            h, w = record['grid']
            arrays = checked_path(record['arrays'])
            with np.load(arrays) as saved:
                for y, x in pixels:
                    if not 0 <= y < h or not 0 <= x < w:
                        raise ValueError('Saved disagreement outside original grid')
                    if expected:
                        row = expected[(ordinal, y, x)]
                        if (record['arrays']['sha256'] != row['source']['sha256']
                                or any(float(saved[a][y, x]) != row[b] for a, b in (
                                    ('final_depth', 'depth'), ('raw_cuda_depth', 'gpu'),
                                    ('raw_pyrender_depth', 'software')))):
                            raise ValueError('Saved disagreement payload changed')
                    ray = np.linalg.solve(k, [(x+.5)/w, (y+.5)/h, 1])
                    world_ray = pose[:3, :3] @ ray
                    origin = pose[:3, 3]+.0498*world_ray
                    points, hits, faces = engine.intersects_location(
                        origin[None], world_ray[None], multiple_hits=False)
                    if len(points) != 1 or hits.tolist() != [0]:
                        raise ValueError('Selected supported ray lost its closest hit')
                    point, face = points[0], int(faces[0])
                    triangle = np.asarray(mesh.vertices)[np.asarray(mesh.faces)[face]]
                    z = float(((point-pose[:3, 3]) @ np.linalg.inv(pose[:3, :3]).T)[2])
                    if not saved['final_support'][y, x] or abs(z-saved['final_depth'][y, x]) > 1e-5:
                        raise ValueError('Exact saved CPU depth not reproduced')
                    camera = (triangle-pose[:3, 3]) @ np.linalg.inv(pose[:3, :3]).T
                    uvh = camera @ k.T
                    screen = uvh[:, :2]/uvh[:, 2:3]*[w, h]
                    target = np.array([x+.5, y+.5])
                    edges = np.roll(screen, -1, axis=0)-screen
                    distances = abs(edges[:, 0]*(target-screen)[:, 1]
                                    - edges[:, 1]*(target-screen)[:, 0]) / np.linalg.norm(edges, axis=1)
                    matrix = np.column_stack([triangle[1]-triangle[0], triangle[2]-triangle[0]])
                    uv = np.linalg.lstsq(matrix, point-triangle[0], rcond=None)[0]
                    barycentric = [float(1-uv.sum()), *uv.tolist()]
                    one = trimesh.Trimesh(vertices=triangle, faces=[[0, 1, 2]], process=False)
                    isolated = render(pyrender, renderer, one, k, pose, h, w, .05, 20.)
                    raw_buffer = GL.glReadPixels(x, h-1-y, 1, 1, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)
                    raw_window_depth = float(np.asarray(raw_buffer).reshape(-1)[0])
                    linearized_float64 = .05*20/(20-raw_window_depth*(20-.05))
                    # Nearby rays diagnose sensitivity only; they never replace the center ray.
                    shifts = [(0., 0.), (-1/256, 0.), (1/256, 0.), (0., -1/256), (0., 1/256)]
                    q = np.array([np.linalg.solve(k, [(x+.5+dx)/w, (y+.5+dy)/h, 1])
                                  for dx, dy in shifts]) @ pose[:3, :3].T
                    pts, ids, fs = engine.intersects_location(pose[:3, 3]+.0498*q, q, multiple_hits=False)
                    nearby = [dict(offset_pixel=list(shifts[i]), face_id=int(f),
                                   camera_z_m=float(((p-pose[:3, 3]) @ np.linalg.inv(pose[:3, :3]).T)[2]))
                              for p, i, f in zip(pts, ids, fs)]
                    cases.append(dict(corpus_ordinal=ordinal, pixel_yx=[y, x], record=bound[str(ordinal)],
                        arrays=record['arrays'], mesh=refs[0], face_id=face, triangle_world=triangle.tolist(),
                        triangle_camera=camera.tolist(), triangle_screen_xy=screen.tolist(),
                        barycentric=barycentric, distance_to_edges_pixels=distances.tolist(),
                        exact_cpu_z_m=z, saved_cpu_z_m=float(saved['final_depth'][y, x]),
                        saved_cuda_z_m=float(saved['raw_cuda_depth'][y, x]),
                        saved_public_z_m=float(saved['raw_pyrender_depth'][y, x]),
                        isolated_first_triangle_public_z_m=float(isolated[y, x]),
                        isolated_window_depth=raw_window_depth,
                        isolated_window_linearized_float64_m=linearized_float64, nearby_rays=nearby))
                    print(json.dumps(dict(ordinal=ordinal, pixel=[y, x], cpu=z,
                        public=float(saved['raw_pyrender_depth'][y, x]), isolated=float(isolated[y, x]))), flush=True)
            del mesh, engine
            gc.collect()
        report = dict(stage='bounded_single_triangle_diagnosis_complete',
            source=file_ref(directory/'report.json'), code=file_ref(Path(__file__)), renderer=identity,
            incremental=incremental,
            cases=cases, original_render_semantics_unchanged=True, full_source_rerendered=False,
            gpu_used=False, labels_changed=False, physical_accuracy_established=False,
            confirmation_images_read=0, training_admission=False, optimizer_updates=0)
        publish_json(output/'report.json', report)
        return report
    finally:
        renderer.delete()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--development', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--analysis', type=Path)
    parser.add_argument('--previous', type=Path)
    args = parser.parse_args()
    investigate(args.development, args.output, args.analysis, args.previous)
