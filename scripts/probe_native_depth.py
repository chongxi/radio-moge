"""Bounded native-label probes in a NEW evidence directory, never training.

Probe bytes are quarantined evidence, not an accepted training cache. Missing
metadata or uncertain formats are reported, never replaced by guessed GT.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sqlite3
import tarfile
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import requests
from scipy.ndimage import binary_erosion

from indoor_download import RemoteZip, atomic_write, write_json


def unpack_image(data):
    with Image.open(io.BytesIO(data)) as image:
        return np.array(image)


def quantiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return None if not len(values) else dict(zip(('min', 'p01', 'p50', 'p99', 'max'),
                np.quantile(values, [0, .01, .5, .99, 1]).tolist(), strict=True))


class Evidence:
    def __init__(self, root):
        if root.exists():
            raise FileExistsError('Probe exists; use a new attempt, never overwrite')
        root.mkdir(parents=True)
        self.root, self.receipts = root, []
        self.session = requests.Session()
        self.archives = {}

    def save(self, name, data, source, member=None):
        path = self.root / name
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError('Probe path collision')
            return path
        atomic_write(path, data)
        self.receipts.append(dict(path=str(path), source=source, member=member,
            bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        write_json(self.root / 'receipts.json', self.receipts)
        return path

    def get(self, name, url):
        response = self.session.get(url, timeout=(15, 60))
        response.raise_for_status()
        if len(response.content) > 32 * 1024**2:
            raise ValueError('Metadata fetch exceeds bound')
        return self.save(name, response.content, url)

    def zip_member(self, name, archive, member):
        key = str(archive)
        if key not in self.archives:
            self.archives[key] = (RemoteZip(key, self.session) if key.startswith('https://')
                                  else zipfile.ZipFile(archive))
        data = self.archives[key].read(member)
        return self.save(name, data, key, member)


def selected_rows(root, dataset, limit=4):
    with sqlite3.connect(f'file:{root / "inventory.sqlite"}?mode=ro', uri=True) as db:
        rows = db.execute('''SELECT row_json FROM images WHERE ordinal IN
            (SELECT MIN(ordinal) FROM images WHERE dataset=? GROUP BY scene_group)
            ORDER BY ordinal LIMIT ?''', (dataset, limit)).fetchall()
    return [json.loads(row[0]) for row in rows]


def structured3d_probe(evidence, data, rows):
    result = []
    for row in rows:
        prefix = str(PurePosixPath(row['source_member']).parent)
        archive = data / 'structured3d-perspective-v1-20261022/archives' / row['source_archive']
        name = f"structured3d/{row['scene']}-{row['room']}-{row['configuration']}-{row['view']}"
        assets = {key: evidence.zip_member(f'{name}/{key}', archive, f'{prefix}/{key}')
                  for key in ('depth.png', 'camera_pose.txt', 'layout.json', 'normal.png')}
        depth_raw = unpack_image(assets['depth.png'].read_bytes())
        if depth_raw.dtype != np.uint16:
            raise ValueError('Structured3D depth encoding changed')
        depth = depth_raw.astype(np.float64) / 1000
        valid = (depth_raw > 0) & (depth_raw < 65535)
        h, w = depth.shape
        camera = np.loadtxt(assets['camera_pose.txt'])
        fx, fy = w / 2 / np.tan(camera[9]), h / 2 / np.tan(camera[10])
        layout = json.loads(assets['layout.json'].read_text())
        residuals = {key: [] for key in ('z_pixel0', 'range_pixel0', 'z_pixelhalf', 'range_pixelhalf')}
        for offset, suffix in ((0, 'pixel0'), (.5, 'pixelhalf')):
            x, y = np.meshgrid(np.arange(w) + offset, np.arange(h) + offset)
            # Layout planes use x-right/y-up/z-forward; not OpenCV y-down.
            rays = np.stack(((x - w / 2) / fx, -(y - h / 2) / fy, np.ones_like(x)), -1)
            for plane in layout['planes']:
                mask = Image.new('1', (w, h))
                draw = ImageDraw.Draw(mask)
                for polygon in plane['visible_mask']:
                    coordinates = [tuple(layout['junctions'][i]['coordinate']) for i in polygon]
                    if len(coordinates) >= 3:
                        draw.polygon(coordinates, fill=1)
                support = binary_erosion(np.array(mask), iterations=5) & valid
                if support.sum() < 100:
                    continue
                for convention in ('z', 'range'):
                    points = rays * depth[..., None]
                    if convention == 'range':
                        points /= np.linalg.norm(rays, axis=-1, keepdims=True)
                    error = np.abs(points @ np.array(plane['normal']) + plane['offset'] / 1000)
                    residuals[f'{convention}_{suffix}'].append(error[support])
        result.append(dict(scene=row['scene'], configuration=row['configuration'],
            camera_pose=camera.tolist(), shape=[h, w], valid_fraction=float(valid.mean()),
            depth_m=quantiles(depth[valid]), plane_residual_m={k: quantiles(np.concatenate(v))
                if v else None for k, v in residuals.items()},
            interpretation='Empty-layout plane check; full scenes may contain occluding furniture',
            training_admission=False))
    return result


def hdf5_array(path):
    import h5py
    with h5py.File(path, 'r') as file:
        return np.asarray(file['dataset'])


def hypersim_probe(evidence, rows):
    url = ('https://raw.githubusercontent.com/apple/ml-hypersim/main/'
           'contrib/mikeroberts3000/metadata_camera_parameters.csv')
    metadata = evidence.get('hypersim/metadata_camera_parameters.csv', url)
    cameras = {row['scene_name']: row for row in csv.DictReader(metadata.open())}
    result = []
    for row in rows:
        scene, camera, frame = row['group'].split(':')[1], row['camera'], row['frame_id']
        name = f'hypersim/{scene}-{camera}-{frame:04d}'
        geometry = f'{scene}/images/scene_{camera}_geometry_hdf5/frame.{frame:04d}'
        depth_path = evidence.zip_member(f'{name}/range.hdf5', row['source_url'],
                                         geometry + '.depth_meters.hdf5')
        position_path = evidence.zip_member(f'{name}/position.hdf5', row['source_url'],
                                            geometry + '.position.hdf5')
        assets = {key: evidence.zip_member(f'{name}/{key}', row['source_url'],
                  f'{scene}/_detail/{camera}/{key}') for key in (
                    'camera_keyframe_positions.hdf5', 'camera_keyframe_orientations.hdf5',
                    'camera_keyframe_frame_indices.hdf5')}
        scene_meta = evidence.zip_member(f'{name}/metadata_scene.csv', row['source_url'],
                                         f'{scene}/_detail/metadata_scene.csv')
        with scene_meta.open() as file:
            scale = float(next(x for x in csv.DictReader(file)
                              if x['parameter_name'] == 'meters_per_asset_unit')['parameter_value'])
        indices = hdf5_array(assets['camera_keyframe_frame_indices.hdf5']).reshape(-1)
        matches = np.flatnonzero(indices == frame)
        if len(matches) != 1:
            raise ValueError('Ambiguous Hypersim camera frame index')
        index = matches.item()
        r = hdf5_array(assets['camera_keyframe_orientations.hdf5'])[index]
        t = hdf5_array(assets['camera_keyframe_positions.hdf5'])[index]
        radial, world = hdf5_array(depth_path), hdf5_array(position_path)
        cam = (world - t) @ r * scale
        h, w = radial.shape
        uvx, uvy = np.meshgrid(2 * (np.arange(w) + .5) / w - 1,
                               1 - 2 * (np.arange(h) + .5) / h)
        matrix = np.array([[float(cameras[scene][f'M_cam_from_uv_{i}{j}'])
                           for j in range(3)] for i in range(3)])
        rays = np.stack((uvx, uvy, np.ones_like(uvx)), -1) @ matrix.T
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        z = -rays[..., 2] * radial
        valid = (np.isfinite(radial) & (radial > 0) & np.isfinite(cam).all(-1)
                 & (cam[..., 2] < 0) & (z > 0))
        result.append(dict(scene=scene, shape=[h, w], meters_per_asset_unit=scale,
            ray_matrix=matrix.tolist(), valid_fraction=float(valid.mean()),
            range_vs_world_m=quantiles(np.abs(np.linalg.norm(cam, axis=-1) - radial)[valid]),
            converted_z_vs_world_m=quantiles(np.abs(z + cam[..., 2])[valid]),
            converted_z_m=quantiles(z[valid]), training_admission=False))
    return result


def tartanair_probe(evidence, rows):
    result = []
    for row in rows:
        archive = row['source_url'].replace('/image_left.zip', '/depth_left.zip')
        member = row['source_member'].replace('/image_left/', '/depth_left/').replace(
            '_left.png', '_left_depth.npy')
        path = evidence.zip_member('tartanair/' + member, archive, member)
        depth = np.load(path, allow_pickle=False)
        result.append(dict(environment=row['environment'], trajectory=row['trajectory'],
            shape=list(depth.shape), dtype=str(depth.dtype), finite_fraction=float(np.isfinite(depth).mean()),
            positive_depth_m=quantiles(depth[np.isfinite(depth) & (depth > 0)]),
            sky_candidate_fraction_gt_1000m=float((depth > 1000).mean()),
            calibration=dict(width=640, height=480, fx=320, fy=320, cx=320, cy=240),
            training_admission=False, sky_mask_policy='PENDING source-specific review; 1000m is a diagnostic'))
    return result


def arkit_probe(evidence, rows):
    result = []
    for row in rows:
        base = row['source_url'].rsplit('/', 1)[0]
        video, timestamp = str(row['video_id']), float(row['timestamp'])
        archives = {}
        for kind in ('lowres_depth', 'confidence', 'lowres_wide_intrinsics'):
            archive = RemoteZip(base + '/' + kind + '.zip', evidence.session)
            archives[kind] = archive
        candidates = []
        for member in archives['lowres_depth'].names():
            if member.endswith('.png'):
                value = float(PurePosixPath(member).stem.split('_')[-1])
                candidates.append((abs(value - timestamp), value, member))
        delta, depth_timestamp, member = min(candidates)
        stem = PurePosixPath(member).stem
        path = evidence.save(f'arkitscenes/{video}/{stem}.depth.png',
            archives['lowres_depth'].read(member), base + '/lowres_depth.zip', member)
        matched_assets = {}
        for kind, suffix in (('confidence', '.png'), ('lowres_wide_intrinsics', '.pincam')):
            matches = [x for x in archives[kind].names() if PurePosixPath(x).name == stem + suffix]
            if len(matches) == 1:
                matched_assets[kind] = evidence.save(f'arkitscenes/{video}/{stem}.{kind}{suffix}',
                    archives[kind].read(matches[0]), base + '/' + kind + '.zip', matches[0])
        depth = unpack_image(path.read_bytes())
        record = dict(video=video, rgb_timestamp=timestamp, nearest_depth_timestamp=depth_timestamp,
            time_delta_s=delta, exact_timestamp_match=delta < 1e-8,
            shape=list(depth.shape), dtype=str(depth.dtype), native_rgb_camera=row.get(
                'native_camera_width_height_fx_fy_cx_cy'), image_rotation_cw=row.get('image_rotation_cw'),
            positive_depth_m=quantiles(depth[depth > 0].astype(float) / 1000),
            confidence_and_intrinsics_available=len(matched_assets) == 2, training_admission=False,
            warning='Nearest timestamp is diagnostic ONLY; no nearest-frame GT pairing is admitted')
        if 'lowres_wide_intrinsics' in matched_assets:
            record['lowres_camera'] = np.loadtxt(matched_assets['lowres_wide_intrinsics']).tolist()
        if 'confidence' in matched_assets:
            confidence = unpack_image(matched_assets['confidence'].read_bytes())
            keys, counts = np.unique(confidence, return_counts=True)
            record['confidence_counts'] = {str(k): int(v) for k, v in zip(keys, counts, strict=True)}
        result.append(record)
    return result


def irs_probe(evidence, data, row):
    member = str(PurePosixPath(row['source_member']).with_name('d_' + row['frame'] + '.exr'))
    archive = data / 'irs-full-v1-20261022/archives' / row['source_archive']
    with tarfile.open(archive, 'r|gz') as file:
        seen = 0
        for entry in file:
            seen += entry.size
            if entry.name == member:
                contents = file.extractfile(entry).read()
                break
            if seen > 256 * 1024**2:
                raise ValueError('IRS diagnostic scan bound reached; do not read entire archive here')
        else:
            raise ValueError('IRS disparity member not found')
    path = evidence.save('irs/' + member, contents, str(archive), member)
    import OpenEXR
    with OpenEXR.File(str(path), separate_channels=True) as exr:
        channels = {key: quantiles(value.pixels) for key, value in exr.channels().items()}
        shapes = {key: list(value.pixels.shape) for key, value in exr.channels().items()}
    return dict(member=member, rgb_eye=row['eye'], channels=channels, shapes=shapes,
        focal_px=480, baseline_m=None, training_admission=False,
        decision='Metric conversion blocked until publisher baseline and eye association are verified')


def nyu_probe(evidence, data, manifest, rows):
    info_path = data / 'nyuv2-labeled-s42/dataset_info.json'
    info = json.loads(info_path.read_text())
    evidence.save('nyuv2/original-dataset-info.json', info_path.read_bytes(), str(info_path))
    if info['depth_key'] != 'rawDepths' or info['depth_units'] != 'meters':
        raise ValueError('NYU candidate is not unfilled projected sensor depth')
    toolbox = evidence.get('nyuv2/toolbox.zip',
        'https://cs.nyu.edu/~fergus/datasets/toolbox_nyu_depth_v2.zip')
    with zipfile.ZipFile(toolbox) as archive:
        for member in archive.namelist():
            if PurePosixPath(member).name in ('camera_params.m', 'project_depth_map.m',
                                            'depth_rel2depth_abs.m', 'get_rgb_depth_overlay.m'):
                evidence.save('nyuv2/' + PurePosixPath(member).name, archive.read(member),
                              str(toolbox), member)
    result = []
    for row in rows:
        path = manifest.parent / row['depth']
        depth = np.load(path, allow_pickle=False)
        valid = np.isfinite(depth) & (depth > 0)
        result.append(dict(id=row['id'], scene=row['scene'], shape=list(depth.shape),
            valid_fraction=float(valid.mean()), depth_m=quantiles(depth[valid]),
            raw_depth_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), training_admission=False))
    return result


def probe(config, name):
    root, data = Path(config['root']), Path(config['dataset_root'])
    evidence = Evidence(root / name)
    result = dict(stage='probing', training_started=False, training_admission=False,
                  bounded_format_probe_not_full_acceptance=True, datasets={}, started_at=time.time())
    functions = {
        'structured3d': lambda rows: structured3d_probe(evidence, data, rows),
        'hypersim': lambda rows: hypersim_probe(evidence, rows),
        'tartanair': lambda rows: tartanair_probe(evidence, rows),
        'arkitscenes': lambda rows: arkit_probe(evidence, rows),
        'irs': lambda rows: irs_probe(evidence, data, rows[0]),
        'nyuv2': lambda rows: nyu_probe(evidence, data, data / config['manifest'], rows),
    }
    for dataset, function in functions.items():
        result['active'] = dataset
        write_json(evidence.root / 'report.json', result)
        try:
            rows = selected_rows(root, dataset, 4 if dataset == 'structured3d' else 2)
            write_json(evidence.root / f'{dataset}-rgb-records.json', rows)
            result['datasets'][dataset] = dict(stage='probe_complete', evidence=function(rows))
        except Exception as error:
            # Public-source diagnostic only. No signed/private URLs are used.
            result['datasets'][dataset] = dict(stage='needs_review', error=type(error).__name__,
                                               message=str(error)[:2000])
        write_json(evidence.root / 'report.json', result)
        print(json.dumps(dict(dataset=dataset, stage=result['datasets'][dataset]['stage'])), flush=True)
    result.update(stage='complete', completed_at=time.time(), active=None)
    result['datasets']['coco2017'] = dict(stage='pseudo_only', native_depth=False,
                                         teacher_generation_started=False)
    result['datasets']['scannetpp_dslr'] = dict(stage='renderer_and_scanner_registration_pending',
        note='Existing raw-laser pilot integrity audit is separate; no DSLR GT cache accepted yet')
    write_json(evidence.root / 'report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--attempt', default='native-probe-01')
    args = parser.parse_args()
    if Path(args.attempt).name != args.attempt:
        parser.error('attempt must be a directory basename')
    probe(json.loads(args.config.read_text()), args.attempt)
