"""Identify the nearby face behind the eleven extra foreground raster samples.

Only isolated neighboring triangles are rendered. Completed center-triangle
diagnoses, whole scenes, source labels and confirmation views are untouched.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import trimesh

from probe_scannetpp_software_clip_render import backend, render
from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import publish_json


def run(reports, output):
    refs = [file_ref(p) for p in reports]
    sources = [checked_json(p) for p in refs]
    if len(sources) != 2 or sources[0]['source'] != sources[1]['source']:
        raise ValueError('Two exact completed parts of the same development diagnosis required')
    all_cases = {}
    for d in sources:
        if d['stage'] != 'bounded_single_triangle_diagnosis_complete' or d['training_admission'] is not False:
            raise ValueError('Completed unadmitted diagnosis required')
        for c in d['cases']:
            key = (c['corpus_ordinal'], *c['pixel_yx'])
            if key in all_cases:
                raise ValueError('Duplicate diagnosed pixel')
            all_cases[key] = c
    selected = [c for c in all_cases.values() if c['isolated_first_triangle_public_z_m'] > 0
                and c['saved_public_z_m'] < c['exact_cpu_z_m']-.002]
    if len(all_cases) != 32 or len(selected) != 11:
        raise ValueError('Frozen32 cases and exact11 extra foreground samples required')
    output.mkdir(exist_ok=False)
    publish_json(output/'selection.json', dict(reports=refs,
        pixels=[dict(ordinal=c['corpus_ordinal'], pixel_yx=c['pixel_yx']) for c in selected]))
    pyrender, renderer, identity = backend()
    records, active, mesh = [], None, None
    try:
        for c in sorted(selected, key=lambda c: (c['mesh']['path'], c['corpus_ordinal'])):
            original = checked_json(c['record'])
            if c['mesh'] not in original['source_refs'] or c['arrays'] != original['arrays']:
                raise ValueError('Original source record differs')
            if active != c['mesh']:
                del mesh
                gc.collect()
                mesh = trimesh.load_mesh(checked_path(c['mesh']), process=False)
                active = c['mesh']
            k, pose = np.array(original['intrinsics']), np.array(original['c2w'])
            h, w = original['grid']
            y, x = c['pixel_yx']
            ray = np.linalg.solve(k, [(x+.5)/w, (y+.5)/h, 1]) @ pose[:3, :3].T
            candidates = sorted({n['face_id'] for n in c['nearby_rays'] if n['face_id'] != c['face_id']})
            results = []
            for face in candidates:
                triangle = np.asarray(mesh.vertices)[np.asarray(mesh.faces)[face]]
                a, b = triangle[1]-triangle[0], triangle[2]-triangle[0]
                # Independent plane equation: camera origin + t*ray = v0+u*a+v*b.
                t, u, v = np.linalg.solve(np.column_stack((ray, -a, -b)), triangle[0]-pose[:3, 3])
                one = trimesh.Trimesh(vertices=triangle, faces=[[0, 1, 2]], process=False)
                depth = render(pyrender, renderer, one, k, pose, h, w, .05, 20.)
                value = float(depth[y, x])
                camera = (triangle-pose[:3, 3]) @ np.linalg.inv(pose[:3, :3]).T
                uvh = camera @ k.T
                screen = uvh[:, :2]/uvh[:, 2:3]*[w, h]
                edges = np.roll(screen, -1, axis=0)-screen
                target = np.array([x+.5, y+.5])
                distances = abs(edges[:, 0]*(target-screen)[:, 1]
                                - edges[:, 1]*(target-screen)[:, 0])/np.linalg.norm(edges, axis=1)
                results.append(dict(face_id=face, triangle_world=triangle.tolist(),
                    triangle_screen_xy=screen.tolist(), barycentric=[float(1-u-v), float(u), float(v)],
                    center_plane_camera_z_m=float(t), isolated_public_z_m=value,
                    distance_to_edges_pixels=distances.tolist(),
                    matches_saved_public_float32_exactly=value == c['saved_public_z_m'],
                    matches_saved_public_within_one_micrometer=abs(value-c['saved_public_z_m']) <= 1e-6))
            record = dict(ordinal=c['corpus_ordinal'], pixel_yx=c['pixel_yx'],
                original_record=c['record'], mesh=c['mesh'], saved_cpu_z_m=c['saved_cpu_z_m'],
                saved_public_z_m=c['saved_public_z_m'], neighboring_triangles=results)
            records.append(record)
            print(json.dumps(dict(ordinal=record['ordinal'], pixel=record['pixel_yx'],
                matches=sum(x['matches_saved_public_float32_exactly'] for x in results))), flush=True)
        publish_json(output/'report.json', dict(stage='neighbor_coverage_diagnosis_complete',
            reports=refs, selection=file_ref(output/'selection.json'), code=file_ref(Path(__file__)),
            renderer=identity, records=records, original_center_triangles_rerendered=False,
            full_scenes_rerendered=False, labels_changed=False, physical_accuracy_established=False,
            gpu_used=False, confirmation_images_read=0, training_admission=False, optimizer_updates=0))
    finally:
        renderer.delete()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports', nargs=2, type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.reports, args.output)
