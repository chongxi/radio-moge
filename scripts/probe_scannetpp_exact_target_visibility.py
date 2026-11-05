"""Freeze and execute bounded NEW exact-target queries on the four pilot scenes.

Only meshes are read anew. Saved source patches, nine-angle probes and image
payloads are not regenerated. No confirmation view or optimizer is opened.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import gc
from pathlib import Path

import numpy as np

from probe_scannetpp_triangle_scan_support import scanner_pose_diagnostics
from probe_scannetpp_v2_source_pilot import bound_path, nearest_world_hits, read
from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import publish_json
from radio_moge.scanner_support import ScannerSupportContract, exact_target_visibility


NOMINAL_SHA = '33d0a185c1fc1418a8dcd395e641f4c608b40179bf6bfd3287eb8d3263947faa'
EXCLUSIONS_SHA = '58a1b02acfe087db2c28e4ae334c762359d0efc6ab16a6d99dd3deba7ab3d86b'


def implementation():
    here = Path(__file__)
    return [file_ref(here), file_ref(here.parents[1]/'src/radio_moge/scanner_support.py'),
            file_ref(here.with_name('probe_scannetpp_v2_source_pilot.py')),
            file_ref(here.with_name('probe_scannetpp_triangle_scan_support.py'))]


def freeze(root, output):
    nominal_ref = file_ref(root/'scannetpp-native-v2-saved-angle-match-01/report.json')
    exclusions = file_ref(root/'scannetpp-native-candidate-decisions-01/known-geometry-exclusions.jsonl')
    if nominal_ref['sha256'] != NOMINAL_SHA or exclusions['sha256'] != EXCLUSIONS_SHA:
        raise ValueError('Frozen complete nominal comparison/exclusions required')
    report = read(nominal_ref, root)
    images = []
    for ref in report['observations']:
        rows = read(ref, root)
        if len(rows) != 192 or len({r['corpus_ordinal'] for r in rows}) != 1:
            raise ValueError('Two fixed stations and96slots required')
        row = rows[0]
        topology = read(row['topology'], root)
        targets = read(topology['targets'], root)
        stations = [read(s, root) for s in topology['stations']]
        if (len(stations) != 2 or {s['scan'] for s in stations} != {'0', '1'}
                or len(targets['slots']) != 96
                or np.array_equal(stations[0]['scanner_pose'], stations[1]['scanner_pose'])):
            raise ValueError('Original distinct stations required without substitution')
        images.append(dict(corpus_ordinal=row['corpus_ordinal'], scene=row['scene'], observations=ref,
            topology=row['topology'], targets=topology['targets'], stations=topology['stations'],
            source_mesh=topology['source_mesh'], expected_queries=2*sum(t['numerical_support'] for t in targets['slots'])))
    if len(images) != 4 or sum(i['expected_queries'] for i in images) != 598:
        raise ValueError('Bounded existing four-scene scope only')
    output.mkdir(exist_ok=False)
    plan = dict(stage='conditional_source_policy_and_exact_target_probe_frozen',
        nominal_report=nominal_ref, exclusions=exclusions, images=images,
        policy=asdict(ScannerSupportContract()), code=implementation(),
        budget_rationale='Prospective conservative centimeter-scale engineering filter for corroborated mesh labels; not derived as a physical bound from pilot quantiles or2mmrenderer tolerance.',
        label_role='laser_mesh_derived with conditional released-source corroboration; not direct per-DSLR measured GT',
        per_pixel='Both preselected distinct scanner stations: C0patch, complete supported9nominalmatches, everyabsresidual<=1cm, exacttarget returnedfirstface/interior/radialagreement. Prior exclusions veto.',
        coverage='Full384/768denominators. Sparse slots describe sampled coverage only. Dense eligibility later requires every grid slot evaluated,source mask subset numerical mask,>=5%fullimage and>=32points.',
        uncertainty='Physical pose/angle covariance unavailable; nominal quantization, shared acquisition, untested between-ray occluders and unreturned coincident faces remain unbounded.',
        exact_query='Unshifted original origin; solve fullpublished3x3; unit localray but unnormalized worldray; no clipping/nearhit skipping/fitting.',
        maximum_new_mesh_reads=4, maximum_new_rays=598, new_range_reads=0,new_RGB_reads=0,
        scanner_replacement_allowed=False, threshold_sweep_allowed=False,
        confirmation_images_read=0, confirmation_execution_allowed=False,
        physical_uncertainty_budget_calibrated=False, physical_accuracy_pass_defined=False,
        GPU_launch=False, bulk_regeneration=False, training_admission=False, optimizer_updates=0)
    publish_json(output/'report.json', plan)
    print(file_ref(output/'report.json'), flush=True)
    return plan


def query_targets(targets, pose, engine):
    pose = np.asarray(pose, np.float64)
    scanner_pose_diagnostics(pose)
    selected = [t for t in targets if t['numerical_support']]
    directions = []
    for target in selected:
        local = np.linalg.solve(pose[:3, :3], np.asarray(target['point_world'])-pose[:3, 3])
        directions.append(local/np.linalg.norm(local))
    hits = nearest_world_hits(engine, pose, np.asarray(directions)) if selected else []
    results = {t['slot']: exact_target_visibility(t, pose, h) for t, h in zip(selected, hits)}
    return [dict(slot=t['slot'], pixel_yx=t['pixel_yx'],
        visibility=results.get(t['slot'], dict(status='DSLR_numerically_unsupported',
            exact_target_first_surface=False, physical_accuracy_accepted=False))) for t in targets]


def run(root, plan_path, output):
    import trimesh
    from trimesh.ray.ray_pyembree import RayMeshIntersector

    plan_ref = file_ref(plan_path)
    plan = read(plan_ref, root)
    if (plan['stage'] != 'conditional_source_policy_and_exact_target_probe_frozen'
            or plan['policy'] != asdict(ScannerSupportContract()) or plan['nominal_report']['sha256'] != NOMINAL_SHA
            or plan['maximum_new_rays'] != 598 or plan['confirmation_execution_allowed'] is not False
            or plan['training_admission'] is not False
            or [r['sha256'] for r in plan['code']] != [r['sha256'] for r in implementation()]):
        raise ValueError('Exact frozen policy, source scope and code required')
    nominal = read(plan['nominal_report'], root)
    if ({i['corpus_ordinal'] for i in plan['images']} != {851268, 1318289, 1415712, 1412812}
            or len(plan['images']) != 4
            or {i['observations']['sha256'] for i in plan['images']}
               != {r['sha256'] for r in nominal['observations']}):
        raise ValueError('Only the frozen four development images are allowed')
    for image in plan['images']:
        rows = read(image['observations'], root)
        if any(r['corpus_ordinal'] != image['corpus_ordinal'] or r['scene'] != image['scene'] for r in rows):
            raise ValueError('Frozen scene identity differs')
        if image['topology']['sha256'] != rows[0]['topology']['sha256']:
            raise ValueError('Frozen topology binding differs')
        topology = read(image['topology'], root)
        if any(image[k] != topology[k] for k in ('targets', 'stations', 'source_mesh')):
            raise ValueError('Frozen target, scanner or mesh binding differs')
    output.mkdir(exist_ok=False)
    artifacts, queries, statuses = [], 0, Counter()
    for image in plan['images']:
        targets = read(image['targets'], root)
        if [t['slot'] for t in targets['slots']] != list(range(96)):
            raise ValueError('All96original slots required')
        mesh = trimesh.load(bound_path(image['source_mesh'], root), process=False)
        for target in targets['slots']:
            if target['numerical_support'] and not np.array_equal(
                    np.asarray(mesh.vertices)[np.asarray(mesh.faces)[target['face_id']]], target['vertices_world']):
                raise ValueError('Original target face/vertices differ from bound mesh')
        engine = RayMeshIntersector(mesh)
        records = []
        for station_ref in image['stations']:
            station = read(station_ref, root)
            if station['targets']['sha256'] != image['targets']['sha256']:
                raise ValueError('Station target binding differs')
            values = query_targets(targets['slots'], station['scanner_pose'], engine)
            records.extend(dict(corpus_ordinal=image['corpus_ordinal'], scene=image['scene'],
                scan=station['scan'], station=station_ref, targets=image['targets'], **v) for v in values)
            queries += sum(t['numerical_support'] for t in targets['slots'])
        path = output/f"{image['corpus_ordinal']}-exact-targets.json"
        publish_json(path, records)
        artifacts.append(file_ref(path))
        counts = Counter(r['visibility']['status'] for r in records)
        statuses.update(counts)
        print(image['corpus_ordinal'], dict(counts), flush=True)
        del engine, mesh
        gc.collect()
    if queries != 598 or sum(statuses.values()) != 768:
        raise ValueError('Bounded source denominator changed')
    result = dict(stage='new_exact_target_visibility_complete_not_accepted', plan=plan_ref,
        observations=artifacts, station_slots=768, DSLR_slots=384, new_rays=queries,
        new_mesh_reads=4, new_range_reads=0, new_RGB_reads=0, status_counts=dict(statuses),
        unique_face_ownership_proven=False, physical_accuracy_accepted=False,
        confirmation_images_read=0, GPU_launch=False, optimizer_updates=0,
        source_quality_accepted=False, training_admission=False, labels_changed=False)
    publish_json(output/'report.json', result)
    print(file_ref(output/'report.json'), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['freeze', 'run'], required=True)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--plan', type=Path)
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze(args.root.resolve(), args.output.resolve())
    else:
        run(args.root.resolve(), args.plan.resolve(), args.output.resolve())
