"""Train-only native LiDAR versus FARO-derived depth diagnostic, not admission.

The publisher's high-resolution subset was filtered using sensor/laser error;
it is not an unbiased certificate for every raw frame. Missing high-res camera
files are not invented: report both explicit pixel-origin mappings, with a
conservative high-res stability interval, and keep exact calibration unresolved.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import time

import numpy as np
import requests
from scipy.ndimage import maximum_filter, minimum_filter

from cache_arkit_geometry import NativeBundle, MissingNativeAsset, convert, load_image
from indoor_download import RemoteZip, atomic_write, write_json
from radio_moge.arkit_depth import rotate_camera
from radio_moge.geometry_cache import publish_json, sha_file
from radio_moge.geometry_sources import normalized_opencv_intrinsics, pinhole_rays


METADATA_URL = 'https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/metadata.csv'


def compare(depth, valid, kd, faro, rgb_camera, turn, half_center_scaling):
    """No geometry fit: two documented-as-hypotheses pixel-origin transforms."""
    if faro.dtype != np.uint16 or faro.shape != (1440, 1920):
        raise ValueError('Unexpected FARO-derived native encoding')
    w, h, fx, fy, cx, cy = rgb_camera
    sx, sy = faro.shape[1] / w, faro.shape[0] / h
    if sx != sy:
        raise ValueError('High-res camera is not the same sensor aspect')
    shift = (sx - 1) / 2 if half_center_scaling else 0
    camera = rotate_camera([1920, 1440, fx * sx, fy * sy, cx * sx + shift, cy * sy + shift], turn)
    kf = normalized_opencv_intrinsics(*camera)
    faro = np.rot90(faro, -turn // 90).copy()
    valid_faro = (faro > 0) & (faro < 65535)
    z = faro.astype(np.float64) / 1000
    low = minimum_filter(np.where(valid_faro, z, -np.inf), 5, mode='constant', cval=-np.inf)
    high = maximum_filter(np.where(valid_faro, z, np.inf), 5, mode='constant', cval=np.inf)
    stable = valid_faro & np.isfinite(low) & np.isfinite(high) & (high - low <= .03 + .01 * z)
    rays = pinhole_rays(kd, *depth.shape)
    uv = rays @ kf.T
    uv = uv[..., :2] / uv[..., 2:]
    ij = np.rint(uv * [faro.shape[1], faro.shape[0]] - .5).astype(int)
    inside = ((ij[..., 0] >= 2) & (ij[..., 0] < faro.shape[1] - 2)
              & (ij[..., 1] >= 2) & (ij[..., 1] < faro.shape[0] - 2))
    yy, xx = ij[..., 1].clip(0, faro.shape[0] - 1), ij[..., 0].clip(0, faro.shape[1] - 1)
    mask = valid & inside & stable[yy, xx]
    if not mask.any():
        return dict(compared_pixels=0)
    a, b = depth[mask], z[yy, xx][mask]
    interval_error = np.maximum(np.maximum(low[yy, xx][mask] - a, a - high[yy, xx][mask]), 0)
    error = np.abs(a - b)
    return dict(compared_pixels=int(mask.sum()), candidate_pixels=int(valid.sum()),
        mapping='half_center_scaled' if half_center_scaling else 'opencv_integer_scaled',
        median_signed_difference_m=float(np.median(a - b)),
        abs_error_m_quantiles=np.quantile(error, [.5, .9, .95, .99, 1]).tolist(),
        interval_distance_m_quantiles=np.quantile(interval_error, [.5, .9, .95, .99, 1]).tolist(),
        fraction_within_5cm=float((error <= .05).mean()),
        fraction_interval_distance_within_5cm=float((interval_error <= .05).mean()),
        absrel=float(np.mean(error / b)), registration_calibration_accepted=False)


def probe(config, attempt):
    root = Path(config['root'])
    output = root / attempt
    output.mkdir(exist_ok=False)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    session = requests.Session()
    response = session.get(METADATA_URL, timeout=(15, 60))
    response.raise_for_status()
    payload = response.content
    atomic_write(output / 'raw-metadata.csv', payload)
    metadata = {row['video_id']: row for row in csv.DictReader(io.StringIO(payload.decode()))}
    inventory = json.loads((root / 'inventory.json').read_text())
    if (inventory['manifest_sha256'] != config['manifest_sha256']
            or sha_file(root / 'inventory.sqlite') != inventory['inventory_sha256']):
        raise ValueError('Frozen inventory changed')
    selected, groups, counts = {}, set(), {turn: 0 for turn in (0, 90, 180, 270)}
    all_frames, all_videos = 0, set()
    with sqlite3.connect(f'file:{root / "inventory.sqlite"}?mode=ro', uri=True) as db:
        for encoded, path in db.execute("SELECT row_json,image FROM images WHERE dataset='arkitscenes' ORDER BY ordinal"):
            row = json.loads(encoded)
            video, turn = row['video_id'], row['image_rotation_cw']
            if row['split'] != 'train' or metadata[video]['fold'] != 'Training':
                raise ValueError('Only frozen primary training rows are eligible')
            if metadata[video]['is_in_upsampling'] != 'True':
                continue
            all_frames += 1
            all_videos.add(video)
            if video not in selected and counts[turn] < 3 and row['group'] not in groups:
                selected[video] = []
                counts[turn] += 1
                groups.add(row['group'])
            if video in selected:
                row['image'] = path
                selected[video].append(row)
    status = dict(stage='probing', videos_completed=0, images_compared=0, training_admission=False,
                  optimizer_updates=0, started_at=time.time())
    write_json(output / 'status.json', status)
    records, availability = [], []
    assets = NativeBundle(output / 'native-source')
    try:
        for video, rows in selected.items():
            base = rows[0]['source_url'].rsplit('/', 1)[0]
            archive = RemoteZip(base + '/highres_depth.zip', session)
            names = set(archive.zip.namelist())
            eligible = [row for row in rows if 'highres_depth/' + PurePosixPath(row['source_member']).name in names]
            archive.zip.close()
            archive.file.close()
            availability.append(dict(video=video, group=rows[0]['group'],
                frozen_frames=len(rows), exact_highres_frames=len(eligible), rotation_cw=rows[0]['image_rotation_cw']))
            for index in sorted({0, len(eligible) // 2, len(eligible) - 1}) if eligible else []:
                row = eligible[index]
                stem = PurePosixPath(row['source_member']).stem
                try:
                    depth, valid, kd, _, proofs, stats = convert(row, assets)
                    high_path, high_proof = assets.get(base, 'highres_depth', stem)
                except MissingNativeAsset as error:
                    records.append(dict(video=video, image_sha256=row['sha256'], stage='missing_exact_native', error=str(error)))
                    continue
                comparisons = [compare(depth, valid, kd, load_image(high_path),
                    row['native_camera_width_height_fx_fy_cx_cy'], row['image_rotation_cw'], flag)
                    for flag in (False, True)]
                records.append(dict(video=video, group=row['group'], image_sha256=row['sha256'],
                    timestamp=row['timestamp'], stage='diagnostic_complete', native_quality=stats,
                    comparisons=comparisons, source_proofs=[*proofs, high_proof]))
                status['images_compared'] += 1
            status['videos_completed'] += 1
            status['updated_at'] = time.time()
            write_json(output / 'status.json', status)
            print(json.dumps(status), flush=True)
        publish_json(output / 'report.json', dict(stage='diagnostic_complete', training_admission=False,
            optimizer_updates=0, manifest_sha256=config['manifest_sha256'],
            metadata_source=METADATA_URL, metadata_sha256=hashlib.sha256(payload).hexdigest(),
            inventory_sha256=inventory['inventory_sha256'], all_upsampling_candidate_frames=all_frames,
            all_upsampling_candidate_videos=len(all_videos), availability=availability, records=records,
            code_sha256=sha_file(Path(__file__)),
            caveats=['FARO-derived mesh depth, not a direct per-pixel scanner measurement',
                     'Publisher filtered highres subset by native/highres error; selection-biased',
                     'Exact highres K is not released here: both pixel-origin mappings reported, neither fitted',
                     'This bounded comparison does not accept every raw native measurement']))
        status['stage'] = 'complete'
        write_json(output / 'status.json', status)
    except BaseException as error:
        status.update(stage='failed', error=repr(error))
        write_json(output / 'status.json', status)
        publish_json(output / 'failure.json', dict(status=status, records=records, availability=availability))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if Path(args.attempt).name != args.attempt:
        parser.error('Use a new attempt basename')
    probe(json.loads(args.config.read_text()), args.attempt)
