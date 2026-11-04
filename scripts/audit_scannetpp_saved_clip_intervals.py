"""Compare clip intervals using complete saved mesh intersections, CPU only.

This is an ideal camera-Z clipping diagnostic on existing ray selections, not
an upstream renderer reproduction or repaired training labels. Truncated saved
intersections fail closed. Near/far boundary cases remain explicitly ambiguous.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from radio_moge.corpus_admission import checked_json, file_ref
from radio_moge.geometry_cache import publish_json, sha_file


def interval_hit(analytical, near=.05, far=20., boundary_tolerance=1e-9):
    hits = analytical['first_hits']
    count = analytical['positive_triangle_intersections']
    if type(count) is not int or count < 0 or count != len(hits):
        raise ValueError('Complete saved positive intersections required; prefix is insufficient')
    if not (0 < boundary_tolerance < near < far and np.isfinite(far)):
        raise ValueError('Finite ordered clip bounds and explicit boundary tolerance required')
    if len({h['face_id'] for h in hits}) != count:
        raise ValueError('Repeated triangle identity')
    order = [(h['t'], h['face_id']) for h in hits]
    depths = [h['camera_z_m'] for h in hits]
    if (order != sorted(order) or not np.isfinite(depths).all()
            or any(not np.isfinite(t) or t <= 0 for t, _ in order)
            or any(z <= 0 for z in depths) or depths != sorted(depths)):
        raise ValueError('Finite positive ordered intersections required')
    if any(min(abs(z-near), abs(z-far)) <= boundary_tolerance for z in depths):
        return dict(outcome='clip_boundary_ambiguous', hit=None)
    eligible = [h for h in hits if near < h['camera_z_m'] < far]
    hit = eligible[0] if eligible else None
    if not hits:
        outcome = 'no_positive_intersection'
    elif hit is None:
        outcome = 'no_intersection_inside_clip_interval'
    else:
        outcome = 'unchanged_first_hit' if hit['face_id'] == hits[0]['face_id'] else 'different_first_hit'
    return dict(outcome=outcome, hit=hit)


def check_ray(case, ray):
    pose = np.asarray(case['c2w'], np.float64)
    origin = np.asarray(ray['ray_origin_world'], np.float64)
    world = np.asarray(ray['ray_direction_world'], np.float64)
    camera = np.asarray(ray['camera_ray'], np.float64)
    if (pose.shape != (4, 4) or origin.shape != (3,) or world.shape != (3,)
            or camera.shape != (3,) or not all(np.isfinite(x).all() for x in (pose, origin, world, camera))
            or camera[2] != 1 or not np.allclose(pose[3], [0, 0, 0, 1], rtol=0, atol=1e-12)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), rtol=0, atol=1e-10)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, rtol=0, atol=1e-10)
            or not np.array_equal(origin, pose[:3, 3])
            or not np.allclose(world, camera @ pose[:3, :3].T, rtol=0, atol=1e-12)):
        raise ValueError('Saved calibrated rigid camera rays differ')
    for hit in ray['analytical']['first_hits']:
        point = np.asarray(hit['point_world'], np.float64)
        z = ((point-origin) @ pose[:3, :3])[2]
        if (not np.allclose(point, origin+hit['t']*world, rtol=0, atol=1e-9)
                or not np.isclose(z, hit['camera_z_m'], rtol=0, atol=1e-9)):
            raise ValueError('Saved point/parameter/camera-Z disagree')


def audit(source_path, output, case_directory=None):
    source_ref = file_ref(source_path)
    source = checked_json(source_ref)
    producer = Path(__file__).with_name('probe_scannetpp_native_near_surfaces.py')
    if (source['stage'] != 'bounded_raw_mesh_ray_diagnosis_complete'
            or source['training_admission'] is not False or source['optimizer_updates'] != 0
            or sha_file(producer) != source['code']['sha256']):
        raise ValueError('Matching completed all-face diagnostic and producer code required')
    records, cases, identities = [], [], set()
    for original_ref in source['cases']:
        path = (case_directory / Path(original_ref['path']).name) if case_directory else Path(original_ref['path'])
        actual_ref = file_ref(path)
        if actual_ref['sha256'] != original_ref['sha256']:
            raise ValueError('Saved case content changed')
        case = checked_json(actual_ref)
        if (Path(original_ref['path']).name != f"{case['corpus_ordinal']}.json"
                or case['corpus_ordinal'] in identities):
            raise ValueError('Case ordinal duplicated or path identity differs')
        identities.add(case['corpus_ordinal'])
        pixels = set()
        for ray in case['rays']:
            pixel = tuple(ray['pixel_yx'])
            if pixel in pixels:
                raise ValueError('Duplicate saved ray')
            pixels.add(pixel)
            check_ray(case, ray)
            hits = ray['analytical']['first_hits']
            before = hits[0] if hits else None
            clipped = interval_hit(ray['analytical'])
            after = clipped['hit']
            records.append(dict(corpus_ordinal=case['corpus_ordinal'], group=case['group'],
                pixel_yx=ray['pixel_yx'], selection=ray['selection'], cached_valid=ray['cached_valid'],
                cached_z_m=ray['cached_z_m'], positive_intersections=len(hits),
                original_first_hit=before, clipped_first_hit=after, outcome=clipped['outcome'],
                original_hit_removed_by_near_clip=bool(before and before['camera_z_m'] < .05),
                original_first_z_m=before['camera_z_m'] if before else None,
                clipped_first_z_m=after['camera_z_m'] if after else None))
        cases.append(dict(original=original_ref, read_from=actual_ref, image=case['image'],
                          source_refs=case['source_refs'], rays=len(pixels)))
    if not records:
        raise ValueError('No saved rays')
    near = [r for r in records if r['original_hit_removed_by_near_clip']]
    supported = [r for r in records if r['cached_valid']]
    report = dict(stage='saved_complete_ray_clip_interval_diagnostic_complete', source=source_ref,
        producer_code=file_ref(producer), code=file_ref(Path(__file__)), cases=cases, records=records,
        images=len(cases), rays=len(records), cached_supported_rays=len(supported),
        clip_interval_camera_z_m=[.05, 20.], boundary_tolerance_m=1e-9,
        outcomes=dict(Counter(r['outcome'] for r in records)),
        supported_outcomes=dict(Counter(r['outcome'] for r in supported)),
        original_first_hit_below_near=len(near),
        below_near_exposes_farther_hit=sum(r['outcome']=='different_first_hit' for r in near),
        below_near_leaves_no_hit=sum(r['outcome']=='no_intersection_inside_clip_interval' for r in near),
        all_saved_positive_intersections_complete=True, original_selections_unchanged=True,
        raw_images_reread=False, meshes_reread=False, gpu_used=False, optimizer_updates=0,
        official_renderer_executed=False, full_image_depth_produced=False,
        physical_accuracy_established=False, source_quality_accepted=False,
        geometry_edited=False, source_flags_cleared=False, training_admission=False,
        limitations=['Ideal double-sided point-ray intersection visibility only; not OpenGL rasterizer parity.',
                     'Farther intersections are not proven physical surfaces or usable replacement labels.',
                     'Selected abnormal views and rays do not estimate corpus error prevalence.'])
    checked_json(source_ref)
    for case in cases:
        checked_json(case['read_from'])
    output.mkdir(exist_ok=False)
    publish_json(output/'report.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--case-directory', type=Path)
    a = p.parse_args()
    report = audit(a.source, a.output, a.case_directory)
    print(json.dumps({k: report[k] for k in ('stage', 'images', 'rays', 'outcomes', 'supported_outcomes')}))
