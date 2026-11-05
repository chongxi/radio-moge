"""Frozen CPU-only source-consistency pilot; no label generation or admission.

All fixed DSLR slots remain in the denominator. Scanner ranges come from the
same acquisition as the mesh: this is conditional internal consistency, not
independent physical calibration. No poses, labels or correspondence rays fit.
"""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from indoor_download import write_json
from probe_scannetpp_triangle_scan_support import (
    angle_sensitivity, directions, nearest_angle_pixel, range_neighborhood,
    scanner_local_target, scanner_pose_diagnostics,
)
from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import publish_json, sha_file
from radio_moge.mesh_clip import MeshClipContract, clipped_cpu_depth

PLAN_SHA = 'a9efd96a83ca590a382d27d86b8f8867ac13a3920eb15dc522fbe21f92bd57e7'
ORDINALS = [851268, 1318289, 1415712, 1412812]
MIRRORS = ('/shared/radio-moge/chongxi/geometry-corpus-20261102',
           '/home/buildhub/shared/radio-moge/chongxi/geometry-corpus-20261102')
VISIBILITY_NUMERICAL_TOLERANCE_M = 1e-5


def bound_path(ref, root):
    path = Path(ref['path'])
    for prefix in MIRRORS:
        if path.is_relative_to(prefix):
            relative = path.relative_to(prefix)
            if '..' in relative.parts:
                raise ValueError('Unsafe mirrored reference')
            path = root/relative
            break
    if sha_file(path) != ref['sha256']:
        raise ValueError(f'Changed evidence: {path}')
    return path


def read(ref, root):
    return json.loads(bound_path(ref, root).read_text())


def validate_slots(rows):
    if ([r['corpus_ordinal'] for r in rows] != ORDINALS
            or len({r['group'] for r in rows}) != 4):
        raise ValueError('Frozen four ordinary controls required')
    for row in rows:
        h, w = row['grid']
        expected = [[int((y+.5)*h/8), int((x+.5)*w/12)] for y in range(8) for x in range(12)]
        if row['role'] != 'ordinary_control' or row['pixel_yx'] != expected:
            raise ValueError('All ninety-six frozen slots per image required without replacement')


class CaptureFirstQuery:
    def __init__(self, engine):
        self.engine, self.first = engine, None

    def intersects_location(self, origins, rays, multiple_hits):
        result = self.engine.intersects_location(origins, rays, multiple_hits=multiple_hits)
        if self.first is None:
            self.first = tuple(np.asarray(x).copy() for x in result)
        return result


def target_slots(mesh, engine, case, pixels, depth, support, contract):
    h, w = case['grid']
    selected = np.zeros((h, w), bool)
    for y, x in pixels:
        selected[y, x] = support[y, x]
    capture = CaptureFirstQuery(engine)
    reproduced, reason = clipped_cpu_depth(capture, case['intrinsics'], case['c2w'], selected, contract)
    if not (reason[selected] == 4).all() or not np.array_equal(reproduced[selected], depth[selected]):
        raise ValueError('Frozen supported depths not exactly reproduced by clipped first hit')
    hits = {}
    if selected.any():
        points, indexes, faces = capture.first
        flat = np.flatnonzero(selected)
        for point, index, face in zip(points, indexes, faces):
            hits[int(flat[index])] = dict(point_world=point.tolist(), face_id=int(face),
                vertices_world=np.asarray(mesh.vertices)[np.asarray(mesh.faces)[face]].tolist())
    result = []
    for slot, (y, x) in enumerate(pixels):
        value = dict(slot=slot, pixel_yx=[y, x], numerical_support=bool(selected[y, x]),
                     saved_z_m=float(depth[y, x]))
        if selected[y, x]:
            value.update(hits[y*w+x])
        result.append(value)
    return result


def nearest_world_hits(engine, pose, local_rays):
    """Unshifted whole mesh visibility; t stays scanner-local radial range."""
    scanner_pose_diagnostics(pose)
    local = np.asarray(local_rays, np.float64)
    if local.ndim != 2 or local.shape[1] != 3 or not np.isfinite(local).all():
        raise ValueError('Finite scanner rays required')
    if not np.allclose(np.linalg.norm(local, axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError('Unit local rays required')
    rays = local @ pose[:3, :3].T
    origins = np.broadcast_to(pose[:3, 3], rays.shape).copy()
    points, indexes, faces = engine.intersects_location(origins, rays, multiple_hits=False)
    points, indexes, faces = np.asarray(points), np.asarray(indexes), np.asarray(faces)
    if (points.shape != (len(indexes), 3) or faces.shape != indexes.shape
            or indexes.ndim != 1 or indexes.dtype.kind not in 'iu'
            or len(set(indexes.tolist())) != len(indexes)
            or (len(indexes) and (indexes.min() < 0 or indexes.max() >= len(rays)))):
        raise ValueError('Malformed nearest visibility query')
    result = [None]*len(local)
    for point, i, face in zip(points, indexes, faces):
        t = float((point-origins[i]) @ rays[i] / (rays[i] @ rays[i]))
        if (not np.isfinite(t) or t < 0
                or np.max(abs(point-origins[i]-t*rays[i])) > 1e-7):
            raise ValueError('Nearest mesh point not on original scanner ray')
        result[i] = dict(face_id=int(face), radial_range_m=t, point_world=point.tolist())
    return result


def visibility(comparison, first, target_face):
    if not comparison['intersects_exact_triangle']:
        return 'target_triangle_missed'
    if first is None:
        return 'whole_mesh_hit_missing'
    delta = first['radial_range_m']-comparison['triangle_range_m']
    if delta < -VISIBILITY_NUMERICAL_TOLERANCE_M:
        return 'occluded_by_earlier_mesh_surface'
    if delta > VISIBILITY_NUMERICAL_TOLERANCE_M:
        return 'target_first_hit_disagrees'
    if first['face_id'] != target_face:
        return 'coincident_different_face_ambiguous'
    if comparison['boundary_hit']:
        return 'target_triangle_boundary'
    return 'same_target_is_first_mesh_surface'


def raw_patch(arrays, y, x):
    h, w = arrays['range'].shape
    y0, y1, x0, x1 = max(0, y-1), min(h, y+2), max(0, x-1), min(w, x+2)
    return dict(bounds_yxyx=[y0, x0, y1, x1],
                values={k: a[y0:y1, x0:x1].tolist() for k, a in arrays.items()})


def compare_slot(target, arrays, pose, engine):
    if not target['numerical_support']:
        return dict(slot=target['slot'], status='DSLR_numerically_unsupported')
    local = scanner_local_target(pose, target['point_world'])
    y, x, _, angle = nearest_angle_pixel(arrays['azim'], arrays['elev'], local)
    az, pol = int(arrays['azim'][y, x]), int(arrays['elev'][y, x])
    measure = range_neighborhood(arrays['range'], arrays['mask'], y, x)
    sensitivity = angle_sensitivity(target['vertices_world'], pose, az, pol, measure)
    offsets = [(a, p) for a in (-1, 0, 1) for p in (-1, 0, 1)]
    first = nearest_world_hits(engine, pose, directions(
        np.array([az+a for a, _ in offsets]), np.array([pol+p for _, p in offsets])))
    for sample, hit in zip(sensitivity['samples'], first):
        sample['nearest_whole_mesh_hit'] = hit
        sample['mesh_visibility'] = visibility(sample['comparison'], hit, target['face_id'])
    center = sensitivity['samples'][4]
    all_visible = all(s['mesh_visibility'] == 'same_target_is_first_mesh_surface'
                      for s in sensitivity['samples'])
    qualified = (center['mesh_visibility'] == 'same_target_is_first_mesh_surface'
                 and measure['stable_supported'])
    return dict(slot=target['slot'], status=center['mesh_visibility'],
        scanner_pixel_yx=[y, x], angular_search_error_rad=angle,
        stored_azimuth_millirad=az, stored_polar_millirad=pol,
        target_scanner_radial_m=float(np.linalg.norm(local)), raw_patch=raw_patch(arrays, y, x),
        discrete_angle_sensitivity=sensitivity, center_diagnostic_comparable=qualified,
        all_nine_diagnostic_comparable=bool(all_visible and measure['stable_supported']),
        center_signed_residual_m=center['comparison'].get('measured_minus_triangle_range_m'),
        physical_accuracy_accepted=False)


def summarize(records):
    counts = Counter(x['status'] for x in records)
    residuals = [x['center_signed_residual_m'] for x in records
                 if x.get('center_diagnostic_comparable')]
    return dict(slots=len(records), status_counts=dict(counts),
        center_diagnostic_comparable=len(residuals),
        all_nine_diagnostic_comparable=sum(x.get('all_nine_diagnostic_comparable', False) for x in records),
        signed_residual_quantiles_m=dict(zip(('min', 'p05', 'p50', 'p95', 'max'),
            np.quantile(residuals, [0, .05, .5, .95, 1]).tolist())) if residuals else None,
        unique_matched_scanner_pixels=len({tuple(x.get('scanner_identity', []))+tuple(x['scanner_pixel_yx']) for x in records
                                          if 'scanner_pixel_yx' in x}),
        observations_are_independent=False)


def run(root, output):
    import trimesh
    from trimesh.ray.ray_pyembree import RayMeshIntersector

    plan_ref = file_ref(root/'scannetpp-native-v2-source-pilot-plan-01/report.json')
    if plan_ref['sha256'] != PLAN_SHA:
        raise ValueError('Frozen pilot plan changed')
    plan = read(plan_ref, root)
    rows = read(plan['selection'], root)
    validate_slots(rows)
    contract = MeshClipContract(**plan['clip_contract'])
    source_bindings = []
    # Lock all eight stations before reading mesh, saved arrays, or raw ranges.
    for row in rows:
        scene = row['group'].split(':')[1]
        audit_ref = file_ref(root/'scannetpp-source-audit-01'/f'{scene}.json')
        audit = read(audit_ref, root)
        pose_ref = file_ref(root/'scanner-poses/data'/scene/'scans/scanner_poses.json')
        poses = np.asarray(read(pose_ref, root), np.float64)
        scans = sorted(audit['scanner_vs_mesh'], key=lambda x: int(x['scan']))
        if len(scans) < 2 or len({int(x['scan']) for x in scans}) != len(scans):
            raise ValueError('Two distinct retained scanners required without substitution')
        chosen = scans[:2]
        for scan in chosen:
            if scan['scanner_poses_sha256'] != pose_ref['sha256']:
                raise ValueError('Published scanner pose version differs')
            scanner_pose_diagnostics(poses[int(scan['scan'])])
        source_bindings.append(dict(corpus_ordinal=row['corpus_ordinal'], scene=scene,
            audit=audit_ref, scanner_poses=pose_ref, scanners=chosen))
    output.mkdir(exist_ok=False)
    publish_json(output/'source-bindings.json', source_bindings)
    status = dict(stage='running_CPU_source_pilot', started_at=time.time(), completed_scans=0,
                  expected_scans=8, training_admission=False, optimizer_updates=0)
    write_json(output/'status.json', status)
    reports, targets_refs, all_records = [], [], []
    try:
        for row, binding in zip(rows, source_bindings):
            case = read(row['saved_v2_record'], root)
            ordinal = row['corpus_ordinal']
            if (case['corpus_ordinal'] != ordinal or case['role'] != 'ordinary_control'
                    or case['grid'] != row['grid']):
                raise ValueError('Original saved development identity differs')
            source_refs = case['source_refs']
            paths = [bound_path(p, root) for p in source_refs]
            meshes = [p for p in paths if p.suffix == '.ply']
            if len(meshes) != 1 or sha_file(meshes[0]) != read(binding['audit'], root)['mesh_sha256']:
                raise ValueError('Native and scanner source mesh bytes differ')
            with np.load(bound_path(case['arrays'], root)) as a, np.load(
                    bound_path(row['saved_numerical_mask'], root)) as m:
                depth, support = a['final_depth'], m['consensus_support']
                if (not np.array_equal(a['intrinsics'], case['intrinsics'])
                        or not np.array_equal(a['c2w'], case['c2w'])
                        or support.shape != depth.shape or support.dtype != bool
                        or (support & ~a['final_support']).any()):
                    raise ValueError('Saved camera or support differs')
            mesh = trimesh.load_mesh(meshes[0], process=False)
            engine = RayMeshIntersector(mesh)
            targets = target_slots(mesh, engine, case, row['pixel_yx'], depth, support, contract)
            target_path = output/f'{ordinal}-targets.json'
            publish_json(target_path, dict(corpus_ordinal=ordinal, scene=binding['scene'],
                source_refs=source_refs, saved_record=row['saved_v2_record'],
                saved_mask=row['saved_numerical_mask'], slots=targets,
                numerical_supported_slots=sum(t['numerical_support'] for t in targets),
                face_identity='current exact clipped first query reproducing saved float32 depth',
                independent_physical_calibration=False))
            target_ref = file_ref(target_path)
            targets_refs.append(target_ref)
            poses = np.asarray(read(binding['scanner_poses'], root), np.float64)
            for scan in binding['scanners']:
                arrays = {}
                for key, proof in scan['inputs'].items():
                    with Image.open(bound_path(proof, root)) as im:
                        arrays[key] = np.asarray(im).copy()
                if (set(arrays) != {'range', 'mask', 'azim', 'elev'}
                        or len({a.shape for a in arrays.values()}) != 1):
                    raise ValueError('Raw scanner angle/range/support grids differ')
                pose = poses[int(scan['scan'])]
                measurements = [dict(compare_slot(t, arrays, pose, engine),
                    scanner_identity=[binding['scene'], str(scan['scan'])]) for t in targets]
                path = output/f'{ordinal}-scan{scan["scan"]}.json'
                summary = summarize(measurements)
                publish_json(path, dict(targets=target_ref, scanner=binding,
                    scan=scan['scan'], scanner_pose=pose.tolist(), inputs=scan['inputs'],
                    measurements=measurements, summary=summary))
                reports.append(file_ref(path))
                all_records.extend(measurements)
                del arrays
                status.update(completed_scans=len(reports), updated_at=time.time(),
                              last_summary=summary)
                write_json(output/'status.json', status)
                print(json.dumps(dict(ordinal=ordinal, scan=scan['scan'], **summary)), flush=True)
            del mesh, engine, depth, support
            gc.collect()
        if len(reports) != 8 or len(all_records) != 768:
            raise ValueError('Frozen full slot and station accounting incomplete')
        result = dict(stage='bounded_source_consistency_pilot_complete_not_accepted',
            plan=plan_ref, source_bindings=file_ref(output/'source-bindings.json'),
            targets=targets_refs, stations=reports, summary=summarize(all_records),
            distinct_DSLR_slots=384, scanner_stations=8, images=4,
            numerical_visibility_tolerance_m=VISIBILITY_NUMERICAL_TOLERANCE_M,
            residual_tolerance_is_physical=False, relative_patch_stability_is_diagnostic=True,
            no_global_nearest_angle_claim=True, no_continuous_uncertainty_bound=True,
            same_acquisition=True, independent_physical_calibration=False,
            source_quality_accepted=False, labels_changed=False, poses_fitted=False,
            panorama_RGB_used=False, confirmation_images_read=0, gpu_used=False,
            training_admission=False, optimizer_updates=0,
            code=[file_ref(Path(__file__)), file_ref(Path(__file__).with_name(
                'probe_scannetpp_triangle_scan_support.py')),
                file_ref(Path(__file__).parents[1]/'src/radio_moge/mesh_clip.py')])
        publish_json(output/'report.json', result)
        write_json(output/'status.json', dict(status, stage='complete', report=file_ref(output/'report.json')))
        return result
    except BaseException as error:
        write_json(output/'status.json', dict(status, stage='failed', error=type(error).__name__, message=str(error)))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    run(args.root.resolve(), args.output.resolve())
