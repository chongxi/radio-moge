"""Bounded topology audit of saved pilot rays; no new ranges, images or rays.

The four original meshes are read sequentially for exact indexed triangles and
global edge incidence only. All 768 fixed station slots remain in the report.
This diagnostic contract cannot publish source or training acceptance.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import gc
from pathlib import Path
import time

import numpy as np

from indoor_download import write_json
from probe_scannetpp_triangle_scan_support import directions, scanner_pose_diagnostics
from probe_scannetpp_v2_source_pilot import ORDINALS, bound_path, read
from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import publish_json
from radio_moge.source_footprint import (
    FootprintContract, extract_topology, planar_footprint, sampled_range_evidence, validate_saved_hits,
)

PILOT_SHA = '341dd02d09f93a2fb487b689f1c17ba238166f94f1e6552947a738c41a731fe8'
TRANSFER_SHA = '2d8895debfc9b84ca85bbf7e283a72bf16d950914e9bf560cf374e51ca864193'


def coverage(rows):
    if len(rows) != 768 or len({(r['corpus_ordinal'], r['scan'], r['slot']) for r in rows}) != 768:
        raise ValueError('All 768 distinct station slots required')
    groups = {}
    for r in rows:
        groups.setdefault((r['corpus_ordinal'], r['slot']), []).append(r)
    if len(groups) != 384 or any({r['scan'] for r in v} != {'0', '1'} for v in groups.values()):
        raise ValueError('Both frozen stations per DSLR slot required')
    def key(row):
        return row['footprint'].get('diagnostic_planar_footprint', False)
    return dict(DSLR_slots=len(groups), station_slots=len(rows),
                planar_station_observations=sum(key(r) for r in rows),
                planar_at_least_one_station=sum(any(key(r) for r in v) for v in groups.values()),
                planar_both_stations=sum(all(key(r) for r in v) for v in groups.values()),
                footprint_status_counts=dict(Counter(r['footprint']['status'] for r in rows)),
                source_status_counts=dict(Counter(r.get('range_evidence', {}).get('status',
                    'DSLR_numerically_unsupported') for r in rows)),
                connected_nonplanar_observations=sum(
                    r['footprint'].get('indexed_component_connected', False)
                    and not r['footprint'].get('fixed_seed_coplanar', False) for r in rows),
                multiface_planar_observations=sum(key(r) and r['footprint'].get('distinct_hit_faces', 0) > 1
                                                   for r in rows),
                physical_accepted_pixels=0, source_accepted_images=0)


def run(root, output, saved_topology=None):
    pilot_ref = file_ref(root/'scannetpp-native-v2-source-pilot-01/report.json')
    transfer_ref = file_ref(root/'scannetpp-native-v2-scanner-transfer-01/report.json')
    if pilot_ref['sha256'] != PILOT_SHA or transfer_ref['sha256'] != TRANSFER_SHA:
        raise ValueError('Frozen pilot/counterexample identity changed')
    pilot, transfer = read(pilot_ref, root), read(transfer_ref, root)
    prior_ref, prior = None, None
    if saved_topology is not None:
        prior_ref = file_ref(saved_topology)
        prior = read(prior_ref, root)
        if (prior['stage'] != 'saved_source_footprint_diagnostics_complete_not_accepted'
                or prior['pilot']['sha256'] != PILOT_SHA):
            raise ValueError('Completed exact-pilot topology extraction required')
    contract = FootprintContract()
    output.mkdir(exist_ok=False)
    publish_json(output/'contract.json', dict(parameters=asdict(contract),
        scope='topology and finite-sample source diagnostics only', pilot=pilot_ref,
        topology='original vertex indices; only manifold opposite-side shared edges; no welding',
        normals='all compared to fixed seed triangle, no fitted or chained planes',
        first_hit='retained unshifted whole-mesh first hit, never replaced',
        boundaries='exposed or unretained incident face unresolved',
        nonplanar='unresolved, not classified as bad depth',
        missing_bridge='unresolved, not proof of physical disconnection',
        source_gaps='reported independently of footprint status; never hidden by coverage selection',
        source_patch='exact full3x3 with all original mask/range support required; border crops incomplete',
        triangle_membership='per-coordinate opposite-edge altitude numerical tolerance',
        interval_overlap_is_acceptance=False, numerical_tolerances_are_physical=False,
        incidence_threshold='prospective diagnostic exclusion for >10x plane-intersection sensitivity',
        continuous_uncertainty_bound=False, independent_physical_calibration=False,
        physical_pose_covariance_available=False, confirmation_images_read=0,
        training_admission=False))
    status = dict(stage='running_saved_topology_audit', started_at=time.time(), completed_meshes=0,
                  new_rays=0, raw_range_reads=0, optimizer_updates=0, training_admission=False)
    write_json(output/'status.json', status)
    rows, artifacts = [], []
    try:
        for ordinal in ORDINALS:
            target_refs = [p for p in pilot['targets'] if Path(p['path']).name == f'{ordinal}-targets.json']
            if len(target_refs) != 1:
                raise ValueError('Unique frozen target reference required')
            target_ref = target_refs[0]
            targets = read(target_ref, root)
            station_refs = [p for p in pilot['stations'] if Path(p['path']).name.startswith(f'{ordinal}-scan')]
            stations = [(p, read(p, root)) for p in station_refs]
            if len(stations) != 2 or {s['scan'] for _, s in stations} != {'0', '1'}:
                raise ValueError('Two original stations required')
            wanted = {t['face_id'] for t in targets['slots'] if t['numerical_support']}
            for _, station in stations:
                if station['targets']['sha256'] != target_ref['sha256']:
                    raise ValueError('Station target binding changed')
                for m in station['measurements']:
                    wanted.update(s['nearest_whole_mesh_hit']['face_id'] for s in
                        m.get('discrete_angle_sensitivity', {}).get('samples', [])
                        if s['nearest_whole_mesh_hit'] is not None)
            mesh_refs = [p for p in targets['source_refs'] if p['path'].endswith('.ply')]
            if len(mesh_refs) != 1:
                raise ValueError('One original source mesh required')
            if prior is None:
                import trimesh
                path = bound_path(mesh_refs[0], root)
                mesh = trimesh.load_mesh(path, process=False)
                topology = extract_topology(mesh.vertices, mesh.faces, wanted)
                del mesh
                gc.collect()
                topology.update(source_mesh=mesh_refs[0], targets=target_ref, stations=station_refs)
                target_path = output/f'{ordinal}-topology.json'
                publish_json(target_path, topology)
                topology_ref = file_ref(target_path)
            else:
                refs = [p for p in prior['artifacts'] if Path(p['path']).name == f'{ordinal}-topology.json']
                if len(refs) != 1:
                    raise ValueError('Unique saved topology required')
                topology_ref = refs[0]
                topology = read(topology_ref, root)
                if (topology['source_mesh'] != mesh_refs[0]
                        or topology['targets'] != target_ref or topology['stations'] != station_refs
                        or {f['face_id'] for f in topology['faces']} != wanted):
                    raise ValueError('Saved topology binding changed')
            artifacts.append(topology_ref)
            image_rows = []
            for station_ref, station in stations:
                pose = np.asarray(station['scanner_pose'])
                scanner_pose_diagnostics(pose)
                if len(station['measurements']) != 96 or len(targets['slots']) != 96:
                    raise ValueError('All ninety-six frozen slots required')
                for target, m in zip(targets['slots'], station['measurements']):
                    if target['slot'] != m['slot']:
                        raise ValueError('Saved slot order changed')
                    row = dict(corpus_ordinal=ordinal, scene=targets['scene'], scan=station['scan'],
                        slot=m['slot'], pixel_yx=target['pixel_yx'], station=station_ref,
                        topology=topology_ref, original_status=m['status'],
                        original_center_comparable=m.get('center_diagnostic_comparable', False),
                        original_nine_comparable=m.get('all_nine_diagnostic_comparable', False),
                        original_center_signed_residual_m=m.get('center_signed_residual_m'))
                    if not target['numerical_support']:
                        row['footprint'] = dict(status='DSLR_numerically_unsupported',
                            diagnostic_planar_footprint=False, physical_accuracy_accepted=False)
                    else:
                        samples = m['discrete_angle_sensitivity']['samples']
                        offsets = [(s['azimuth_offset_millirad'], s['polar_offset_millirad']) for s in samples]
                        if offsets != [(a, p) for a in (-1, 0, 1) for p in (-1, 0, 1)]:
                            raise ValueError('Original nine angular probes required')
                        rays = directions(np.array([m['stored_azimuth_millirad']+a for a, _ in offsets]),
                                          np.array([m['stored_polar_millirad']+p for _, p in offsets]))
                        hits = [s['nearest_whole_mesh_hit'] for s in samples]
                        validate_saved_hits(hits, pose[:3, 3], rays @ pose[:3, :3].T)
                        row['footprint'] = planar_footprint(target, hits, pose[:3, 3], topology, contract)
                        row['range_evidence'] = sampled_range_evidence(hits, m['raw_patch'])
                    image_rows.append(row)
            record_path = output/f'{ordinal}-observations.json'
            publish_json(record_path, image_rows)
            artifacts.append(file_ref(record_path))
            rows.extend(image_rows)
            status.update(completed_meshes=status['completed_meshes']+1, updated_at=time.time())
            write_json(output/'status.json', status)
            print(ordinal, dict(Counter(r['footprint']['status'] for r in image_rows)), flush=True)
        # These development conflicts are independent of the footprint selector.
        cases = [(1318289, 68, '1', .071529850001061),
                 (1412812, 71, '0', -.0680844373751856),
                 (1412812, 35, '0', -.0005623877740315)]
        counterexamples = []
        for ordinal, slot, scan, expected in cases:
            row = next(r for r in rows if (r['corpus_ordinal'], r['slot'], r['scan']) == (ordinal, slot, scan))
            if abs(row['range_evidence']['signed_interval_separation_m']-expected) > 1e-12:
                raise ValueError('Saved source gap changed or disappeared')
            counterexamples.append(row)
        previous = [r for r in transfer['records'] if r['existing_conditional_source_contradiction_transfers']]
        if (len(previous) != 1 or previous[0]['corpus_ordinal'] != 1284721
                or previous[0]['new_z_m'] != .08369004726409912):
            raise ValueError('Previously established counterexample missing')
        result = dict(stage='saved_source_footprint_diagnostics_complete_not_accepted',
            pilot=pilot_ref, contract=file_ref(output/'contract.json'), artifacts=artifacts,
            coverage=coverage(rows), counterexamples_preserved=counterexamples,
            prior_counterexample=dict(report=transfer_ref, record=previous[0]),
            new_mesh_topology_reads=4 if prior is None else 0, saved_topology=prior_ref,
            new_rays=0, new_range_reads=0, new_RGB_reads=0,
            confirmation_images_read=0, gpu_used=False, optimizer_updates=0,
            physical_accuracy_accepted=False, source_quality_accepted=False,
            training_admission=False, labels_changed=False,
            code=[file_ref(Path(__file__)), file_ref(Path(__file__).parents[1]/'src/radio_moge/source_footprint.py')])
        publish_json(output/'report.json', result)
        write_json(output/'status.json', dict(status, stage='complete', report=file_ref(output/'report.json')))
        return result
    except BaseException as error:
        write_json(output/'status.json', dict(status, stage='failed', error=type(error).__name__, message=str(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--saved-topology', type=Path)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve(), args.saved_topology)
