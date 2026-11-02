"""Laser-mesh-derived DSLR cache with native calibration and all-ray verification.

Not direct per-DSLR-pixel laser measurements. Every candidate ray must match an
independent CPU mesh intersection; source-review scenes are quarantined, never
silently repaired. This producer never accepts the corpus or starts training.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

from indoor_download import write_json
from radio_moge.cache_recovery import recovered_rows
from radio_moge.geometry_cache import (GeometryPackWriter, canonical_json, publish_json,
                                       read_geometry_member, sha_file)
from radio_moge.geometry_sources import (normalized_opencv_intrinsics,
                                         scannetpp_nerfstudio_to_colmap_c2w)
from radio_moge.mesh_depth import CudaMeshDepth, verify_cpu_intersections


PREFLIGHT_SCENES = ('00777c41d4', '00a231a370', '0271889ec0', '033d0b9343', '04c23ccd1d',
                    '0fe8539661')


def colmap_poses(path):
    result = {}
    with Path(path).open() as file:
        for line in file:
            if not line.strip() or line.startswith('#'):
                continue
            fields = line.split()
            if len(fields) != 10 or fields[9] in result:
                raise ValueError('Malformed or duplicate COLMAP pose')
            q, t = np.array(fields[1:5], float), np.array(fields[5:8], float)
            if not np.isfinite(q).all() or not np.isfinite(t).all() or abs(np.linalg.norm(q) - 1) > 1e-5:
                raise ValueError('Invalid metric COLMAP pose')
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
            pose[:3, 3] = t
            result[fields[9].split('/')[-1]] = np.linalg.inv(pose)
            if next(file, None) is None:
                raise ValueError('Missing COLMAP observation line')
    return result


def camera_for_row(row, calibration, frames, poses):
    if (calibration['camera_model'] != 'PINHOLE'
            or any(calibration.get(k, 0) != 0 for k in ('k1', 'k2', 'k3', 'k4', 'p1', 'p2'))):
        raise ValueError('Expected the publisher undistorted pinhole camera')
    values = [calibration[k] for k in ('w', 'h', 'fl_x', 'fl_y', 'cx', 'cy')]
    if (not np.allclose(values, row['camera_width_height_fx_fy_cx_cy'], rtol=0, atol=1e-8)
            or values[:2] != [row['width'], row['height']]
            or row['view'] not in frames or row['view'] not in poses
            or not np.allclose(frames[row['view']]['transform_matrix'], row['transform_matrix'],
                               rtol=0, atol=1e-10)):
        raise ValueError('Frozen RGB calibration/pose differs from native publisher metadata')
    converted = scannetpp_nerfstudio_to_colmap_c2w(row['transform_matrix'])
    if not np.allclose(converted, poses[row['view']], rtol=0, atol=1e-5):
        raise ValueError('Publisher Nerfstudio and metric COLMAP poses disagree')
    return normalized_opencv_intrinsics(*values), poses[row['view']]


def conservative_anonymization(path, original_size, render_size):
    h, w = render_size
    with Image.open(path) as file:
        source = np.asarray(file)
    if source.shape != tuple(original_size) or source.ndim != 2 or min(source.shape) < min(h, w):
        raise ValueError('Native DSLR anonymity mask dimensions differ')
    # Float area support prevents a single unsupported native pixel being rounded
    # back into a valid render pixel. Erode again on the derived ray grid.
    averaged = np.asarray(Image.fromarray((source == 255).astype(np.float32)).resize(
        (w, h), Image.Resampling.BOX))
    return binary_erosion(averaged >= 1 - 1e-7, iterations=2)


def rows(db, preflight):
    if not preflight:
        query = ("SELECT row_json,image FROM images WHERE dataset='scannetpp_dslr' "
                 "ORDER BY scene_group,ordinal")
        for encoded, image in db.execute(query):
            row = json.loads(encoded)
            row['image'] = image
            yield row
    else:
        for scene in PREFLIGHT_SCENES:
            values = db.execute("SELECT row_json,image FROM images WHERE scene_group=? ORDER BY ordinal",
                                ('scannetpp:' + scene,)).fetchall()
            for index in sorted({0, len(values) - 1}):
                encoded, image = values[index]
                row = json.loads(encoded)
                row['image'] = image
                yield row


def run(config, mode, attempt, preflight_name, visibility_name='mesh-visibility-probe-02', recovery_name=None):
    root, data = Path(config['root']), Path(config['dataset_root'])
    inventory = json.loads((root / 'inventory.json').read_text())
    audit_root = root / 'scannetpp-source-audit-01'
    if Path(visibility_name).name != visibility_name:
        raise ValueError('Visibility probe must be an attempt basename')
    visibility_path = root / visibility_name / 'report.json'
    visibility = json.loads(visibility_path.read_text())
    if (inventory['manifest_sha256'] != config['manifest_sha256']
            or sha_file(root / 'inventory.sqlite') != inventory['inventory_sha256']
            or visibility['stage'] != 'complete' or not visibility['passed']
            or not visibility['cpu_verification_of_every_candidate']
            or visibility['manifest_sha256'] != config['manifest_sha256']
            or visibility['code_sha256'] != sha_file(Path(__file__).parents[1] / 'src/radio_moge/mesh_depth.py')):
        raise ValueError('Frozen inventory and exact all-ray verification preflight required')
    source_root = Path(__file__).parents[1]
    recovered, recovery_ref = recovered_rows(root, recovery_name, 'scannetpp_dslr', config['manifest_sha256'])
    if mode == 'preflight' and recovered:
        raise ValueError('Preflight must inspect its complete selection')
    policy = dict(dataset='scannetpp_dslr', manifest_sha256=config['manifest_sha256'],
        label_kind='laser_mesh_derived', not_direct_per_dslr_pixel_measurements=True,
        conversion='CPU nearest-hit camera Z meters on 640-wide pinhole ray grid, no fitting/filling',
        source_audit_policy_sha256=sha_file(audit_root / 'policy.json'),
        visibility_probe_sha256=sha_file(visibility_path), render_width=640,
        source_review_policy='quarantine entire scene pending explicit source review; no automatic repairs',
        anonymization='native mask==255, float area support, 2-pixel erosion on output grid',
        visibility='GPU 5x5/quarter-pixel stability AND EVERY candidate verified by CPU closest hit',
        gpu_cpu_max_difference_m=.002, minimum_image_valid_fraction=.05,
        large_mesh_handling='all faces in <=1048576-face passes, nearest-Z merge; full CPU parity unchanged',
        script_sha256=sha_file(Path(__file__)), code_sha256={name: sha_file(source_root / name) for name in (
            'src/radio_moge/mesh_depth.py', 'src/radio_moge/geometry_sources.py',
            'src/radio_moge/geometry_cache.py', 'src/radio_moge/cache_recovery.py',
            'scripts/indoor_download.py')})
    policy_hash = hashlib.sha256(canonical_json(policy)).hexdigest()
    if mode == 'generate':
        previous = json.loads((root / preflight_name / 'report.json').read_text())
        audit = json.loads((audit_root / 'report.json').read_text())
        if (not previous['passed'] or previous['policy_sha256'] != policy_hash
                or audit['stage'] != 'complete' or audit['scenes_completed'] != 820
                or audit['manifest_sha256'] != config['manifest_sha256']):
            raise ValueError('Matching cache preflight and completed full source accounting required')
    output = root / attempt
    output.mkdir(exist_ok=False)
    (output / 'rejected').mkdir()
    status = dict(stage='generating_cache', mode=mode, completed=0, rejected=0,
        scenes_seen=0, started_at=time.time(), training_admission=False, training_started=False,
        optimizer_updates=0, policy_sha256=policy_hash, recovered_subset=recovery_ref, recovered_skipped=0)
    write_json(output / 'status.json', status)
    writer = GeometryPackWriter(output / 'packs', policy, max_entries=256)
    active, renderer, intersector, mesh = None, None, None, None
    examples, rejections = [], []
    try:
        with sqlite3.connect(f'file:{root / "inventory.sqlite"}?mode=ro', uri=True) as db:
            db.execute('PRAGMA cache_size=-32768')
            for row in rows(db, mode == 'preflight'):
                if row['sha256'] in recovered:
                    status['recovered_skipped'] += 1
                    continue
                if (row['dataset'] != 'scannetpp_dslr' or row['split'] != 'train'
                        or row['official_split'] != 'nvs_sem_train' or row['group'] != 'scannetpp:' + row['scene']):
                    raise ValueError('Only frozen eligible training scenes allowed')
                if active != row['scene']:
                    del renderer, intersector, mesh
                    renderer, intersector, mesh = None, None, None
                    active = row['scene']
                    scene = data / 'scannetpp-v2-full/data' / active
                    audit_path = audit_root / (active + '.json')
                    source_audit = json.loads(audit_path.read_text())
                    proofs = [dict(source=str(audit_path), sha256=sha_file(audit_path))]
                    if source_audit['scene'] != active:
                        raise ValueError('Source scene identity differs')
                    status['scenes_seen'] += 1
                    if source_audit['stage'] == 'source_checks_passed':
                        mesh_path = scene / 'scans/mesh_aligned_0.05.ply'
                        colmap_path = scene / 'dslr/colmap/images.txt'
                        calibration_path = scene / 'dslr/nerfstudio/transforms_undistorted.json'
                        if (sha_file(mesh_path) != source_audit['mesh_sha256']
                                or sha_file(colmap_path) != source_audit['dslr']['colmap_images_sha256']):
                            raise ValueError('Audited source mesh or metric poses changed')
                        for path in (mesh_path, colmap_path, calibration_path):
                            proofs.append(dict(source=str(path), sha256=sha_file(path)))
                        calibration = json.loads(calibration_path.read_text())
                        frames = {Path(f['file_path']).name: f for f in calibration['frames']}
                        poses = colmap_poses(colmap_path)
                        mesh = trimesh.load_mesh(mesh_path, process=False)
                        renderer = CudaMeshDepth(mesh.vertices, mesh.faces)
                        intersector = RayMeshIntersector(mesh)
                reason, stats = None, {}
                if source_audit['stage'] != 'source_checks_passed':
                    reason = 'source_scene_pending_review'
                    stats = dict(source_audit_stage=source_audit['stage'],
                                 review_scans=source_audit.get('review_scans'))
                else:
                    if sha_file(row['image']) != row['sha256']:
                        raise ValueError('Frozen original RGB changed')
                    with Image.open(row['image']) as file:
                        if file.size != (row['width'], row['height']) or file.getexif().get(274, 1) != 1:
                            raise ValueError('Unexpected native RGB orientation or dimensions')
                    k, pose = camera_for_row(row, calibration, frames, poses)
                    h, w = round(row['height'] * 640 / row['width']), 640
                    depth, valid, _, raster_stats = renderer.render_conservative(k, pose, h, w)
                    valid &= conservative_anonymization(row['mask'], (row['height'], row['width']), (h, w))
                    depth, valid, verification = verify_cpu_intersections(intersector, k, pose, depth, valid)
                    stats = dict(**raster_stats, **verification, valid_fraction=float(valid.mean()))
                    if valid.mean() < .05:
                        reason = 'insufficient_verified_visible_support'
                if reason:
                    record = dict(image=row, reason=reason, stats=stats, source_proofs=proofs)
                    rejections.append(record)
                    publish_json(output / 'rejected' / (row['sha256'] + '.json'), record)
                    status['rejected'] += 1
                else:
                    provenance = dict(image_sha256=row['sha256'], dataset='scannetpp_dslr',
                        source_id=active + '/' + row['view'], label_kind='laser_mesh_derived',
                        camera_kind='calibrated_sensor', metric_calibrated=True,
                        conversion=policy['conversion'], policy_sha256=policy_hash,
                        source_proofs=proofs + [dict(source=row['mask'], sha256=sha_file(row['mask']))],
                        source_quality=stats, source_needs_review=False,
                        rgb_shape=[row['height'], row['width']], rgb_intrinsics_normalized=k.tolist(),
                        depth_to_rgb_uv_transform=np.eye(3).tolist(),
                        not_direct_per_dslr_pixel_measurements=True)
                    writer.add(row, depth, valid, k, provenance)
                    status['completed'] += 1
                if mode == 'preflight':
                    examples.append(dict(scene=active, view=row['view'], rejected=bool(reason),
                                         reason=reason, **stats))
                if (status['completed'] + status['rejected']) % 8 == 0:
                    status['updated_at'] = time.time()
                    write_json(output / 'status.json', status)
                    print(json.dumps(status), flush=True)
        writer.close()
        publish_json(output / 'rejections.json', rejections)
        verified = 0
        for item in writer.packs:
            receipt_path = output / 'packs' / item['receipt']
            receipt = json.loads(receipt_path.read_text())
            tar = receipt_path.parent / receipt['tar']
            if sha_file(receipt_path) != item['sha256'] or sha_file(tar) != receipt['sha256']:
                raise ValueError('Published laser-derived pack changed')
            for entry in receipt['entries']:
                read_geometry_member(tar, entry)
                verified += 1
        expected = 12 if mode == 'preflight' else config['expected_counts']['scannetpp_dslr'] - len(recovered)
        if (status['completed'] + status['rejected'] != expected or verified != status['completed']
                or status['recovered_skipped'] != len(recovered)):
            raise ValueError('Candidate cache did not account for full frozen scope')
        status.update(stage='complete', completed_at=time.time(), expected_scope=expected,
            examples=examples, verified_cache_members=verified,
            passed=bool(mode == 'preflight' and status['completed'] == 8
                        and len(rejections) == 4 and all(r['reason'] == 'source_scene_pending_review'
                                                       for r in rejections)))
        write_json(output / 'status.json', status)
        publish_json(output / 'report.json', status)
    except BaseException as error:
        status.update(stage='failed', error=type(error).__name__, message=str(error)[:1000])
        write_json(output / 'status.json', status)
        raise


if __name__ == '__main__':
    if hasattr(os, 'sched_getaffinity'):
        # One native CPU core; do not fan out Embree/TBB into unrelated jobs.
        allowed = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {allowed[min(1, len(allowed) - 1)]})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'generate'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--preflight', default='scannetpp-cache-preflight-01')
    parser.add_argument('--visibility-probe', default='mesh-visibility-probe-02')
    parser.add_argument('--recovered', help='Explicit verified subset to skip; only for generation')
    args = parser.parse_args()
    if any(Path(name).name != name for name in (args.attempt, args.preflight)):
        parser.error('Use a new attempt basename')
    run(json.loads(args.config.read_text()), args.mode, args.attempt, args.preflight,
        args.visibility_probe, args.recovered)
