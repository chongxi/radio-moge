"""Cross-check scanner/mesh/DSLR frames without fitting geometry or training."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

from radio_moge.geometry_sources import (normalized_opencv_intrinsics, pinhole_rays,
                                         scannetpp_nerfstudio_to_colmap_c2w)
from indoor_download import atomic_write, write_json
from probe_native_depth import quantiles, selected_rows


def sha_file(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def source_rows(config, limit=3):
    root = Path(config['root'])
    if (root / 'inventory.sqlite').is_file():
        return selected_rows(root, 'scannetpp_dslr', limit)
    # A bounded physical audit may run while the full leakage audit is pending.
    # It cannot approve training or publish a corpus. Verify the complete input
    # hash and never read the in-progress SQLite database.
    manifest = Path(config['dataset_root']) / config['manifest']
    if sha_file(manifest) != config['manifest_sha256']:
        raise ValueError('Authorized source manifest changed')
    rows, scenes = [], set()
    with manifest.open() as file:
        for line in file:
            row = json.loads(line)
            if row['dataset'] != 'scannetpp_dslr' or row['scene'] in scenes:
                continue
            if row.get('split') != 'train' or row.get('official_split') != 'nvs_sem_train':
                raise ValueError('Non-training scene in bounded source probe')
            scenes.add(row['scene'])
            rows.append(row)
            if len(rows) == limit:
                break
    if len(rows) != limit:
        raise ValueError('Insufficient eligible DSLR probe scenes')
    return rows


def colmap_pose(path, name):
    with path.open() as file:
        for line in file:
            if not line.strip() or line.startswith('#'):
                continue
            fields = line.split()
            if len(fields) < 10:
                raise ValueError('Malformed COLMAP images.txt pose line')
            if fields[9].split('/')[-1] == name:
                q = np.array(fields[1:5], float)
                w2c = np.eye(4)
                w2c[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
                w2c[:3, 3] = np.array(fields[5:8], float)
                return np.linalg.inv(w2c)
            next(file, None)  # Skip that image's independent 2D point observations.
    raise ValueError('DSLR frame missing from metric COLMAP poses')


def scanner_mesh_check(scene, pose_root, intersector):
    pose_path = pose_root / 'data' / scene.name / 'scans/scanner_poses.json'
    poses = np.asarray(json.loads(pose_path.read_text()), float)
    scans = sorted((scene / 'panocam/depth').glob('*.png'), key=lambda p: int(p.stem))
    result = []
    for path in scans[:2]:
        arrays, proofs = {}, {}
        for key, folder in (('range', 'depth'), ('azim', 'azim'), ('elev', 'elev'), ('mask', 'anon_mask')):
            asset = scene / 'panocam' / folder / path.name
            with Image.open(asset) as file:
                arrays[key] = np.array(file)
            proofs[key] = dict(path=str(asset), sha256=sha_file(asset))
        h, w = arrays['range'].shape
        ys, xs = np.meshgrid(np.arange(2, h - 2, 64), np.arange(2, w - 2, 64), indexing='ij')
        ys, xs = ys.ravel(), xs.ravel()
        radial = arrays['range'][ys, xs].astype(float) / 1000
        valid = (radial > .05) & (radial < 65.535) & (arrays['mask'][ys, xs] == 255)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                neighbor = arrays['range'][ys + dy, xs + dx].astype(float) / 1000
                valid &= ((neighbor > 0) & (np.abs(neighbor - radial) <= .03 * radial)
                          & (arrays['mask'][ys + dy, xs + dx] == 255))
        azimuth = arrays['azim'][ys[valid], xs[valid]].astype(float) / 1000
        polar = arrays['elev'][ys[valid], xs[valid]].astype(float) / 1000
        directions = np.stack((np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth),
                               np.cos(polar)), -1)
        pose = poses[int(path.stem)]
        directions = directions @ pose[:3, :3].T
        origins = np.broadcast_to(pose[:3, 3], directions.shape).copy()
        locations, ray_indices, _ = intersector.intersects_location(origins, directions,
                                                                    multiple_hits=False)
        distance = np.linalg.norm(locations - origins[ray_indices], axis=-1)
        truth = radial[valid][ray_indices]
        result.append(dict(scan=path.stem, eligible_sampled_rays=int(valid.sum()),
            mesh_hit_fraction=len(ray_indices) / max(1, int(valid.sum())),
            absolute_range_error_m=quantiles(np.abs(distance - truth)),
            relative_range_error=quantiles(np.abs(distance - truth) / truth),
            within_5cm_fraction=float((np.abs(distance - truth) < .05).mean()) if len(truth) else None,
            inputs=proofs, scanner_poses_sha256=sha_file(pose_path)))
    return result


def dslr_check(row, scene, mesh, intersector, output):
    if sha_file(Path(row['image'])) != row['sha256']:
        raise ValueError('DSLR RGB bytes differ from the frozen training manifest')
    known = colmap_pose(scene / 'dslr/colmap/images.txt', row['view'])
    converted = scannetpp_nerfstudio_to_colmap_c2w(row['transform_matrix'])
    pose_error = float(np.max(np.abs(known - converted)))
    if pose_error > 1e-5:
        raise ValueError('Manifest-to-metric-COLMAP pose conversion does not match')
    intrinsics = normalized_opencv_intrinsics(*row['camera_width_height_fx_fy_cx_cy'])
    w, h = 320, round(row['height'] * 320 / row['width'])
    rays = pinhole_rays(intrinsics, h, w).reshape(-1, 3)
    directions = rays @ known[:3, :3].T
    origins = np.broadcast_to(known[:3, 3], directions.shape).copy()
    locations, indices, triangles = intersector.intersects_location(origins, directions,
                                                                    multiple_hits=False)
    cam = (locations - known[:3, 3]) @ known[:3, :3]
    depth, colors = np.zeros(h * w, np.float32), np.zeros((h * w, 3), np.uint8)
    depth[indices] = cam[:, 2]
    colors[indices] = mesh.visual.vertex_colors[mesh.faces[triangles], :3].mean(1).astype(np.uint8)
    depth, colors = depth.reshape(h, w), colors.reshape(h, w, 3)
    with Image.open(row['mask']) as file:
        mask = np.array(file.resize((w, h), Image.Resampling.NEAREST)) == 255
    valid = mask & np.isfinite(depth) & (depth > .05) & (depth < 65.535)
    with Image.open(row['image']) as file:
        rgb = file.convert('RGB').resize((w, h), Image.Resampling.BILINEAR)
    color_depth = np.repeat((np.log1p(np.where(valid, depth, 0)).clip(0, np.log(11)) /
                            np.log(11) * 255).astype(np.uint8)[..., None], 3, axis=-1)
    panels = [rgb, Image.fromarray(colors), Image.fromarray(color_depth),
              Image.fromarray((valid * 255).astype(np.uint8)).convert('RGB')]
    sheet = Image.new('RGB', (w * 4, h + 32), 'white')
    ImageDraw.Draw(sheet).text((6, 6), f"{scene.name}/{row['view']}: RGB | mesh color | log Z 0-10m | support", fill='black')
    for i, image in enumerate(panels):
        sheet.paste(image, (i * w, 32))
    buffer = io.BytesIO()
    sheet.save(buffer, format='JPEG', quality=90)
    atomic_write(output / f'{scene.name}-{Path(row["view"]).stem}.jpg', buffer.getvalue())
    return dict(view=row['view'], metric_colmap_pose_max_abs_difference=pose_error,
        render_size=[h, w], valid_fraction=float(valid.mean()), depth_m=quantiles(depth[valid]),
        known_camera=True, training_admission=False,
        rgb_sha256=row['sha256'], mask_sha256=sha_file(Path(row['mask'])),
        colmap_images_sha256=sha_file(scene / 'dslr/colmap/images.txt'),
        caution='Mesh reconstruction diagnostic, not direct measured-per-DSLR-pixel truth')


def probe(config, attempt):
    root, data = Path(config['root']), Path(config['dataset_root'])
    output = root / attempt
    if output.exists():
        raise FileExistsError('Keep previous mesh probe evidence; use a new attempt')
    output.mkdir(parents=True)
    result = dict(stage='probing', scenes=[], training_started=False, training_admission=False,
                  no_alignment_fitting=True, sample_not_full_acceptance=True, started_at=time.time())
    write_json(output / 'report.json', result)
    for row in source_rows(config, 3):
        scene = data / 'scannetpp-v2-full/data' / row['scene']
        try:
            mesh_path = scene / 'scans/mesh_aligned_0.05.ply'
            mesh = trimesh.load_mesh(mesh_path, process=False)
            intersector = RayMeshIntersector(mesh)
            record = dict(scene=scene.name, mesh_sha256=sha_file(mesh_path),
                          vertices=len(mesh.vertices), faces=len(mesh.faces))
            record['scanner_vs_mesh'] = scanner_mesh_check(scene, root / 'scanner-poses', intersector)
            record['dslr'] = dslr_check(row, scene, mesh, intersector, output)
            result['scenes'].append(record)
        except Exception as error:
            result['scenes'].append(dict(scene=scene.name, stage='needs_review',
                                         error=type(error).__name__, message=str(error)))
        write_json(output / 'report.json', result)
    result.update(stage='complete', completed_at=time.time())
    write_json(output / 'report.json', result)
    print(json.dumps(dict(stage='complete', scenes=len(result['scenes']), training_admission=False)))


if __name__ == '__main__':
    # Also constrain native Embree/TBB work; OpenMP env vars alone are insufficient.
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--attempt', default='scannetpp-mesh-probe-01')
    args = parser.parse_args()
    if Path(args.attempt).name != args.attempt:
        parser.error('attempt must be a directory basename')
    probe(json.loads(args.config.read_text()), args.attempt)
