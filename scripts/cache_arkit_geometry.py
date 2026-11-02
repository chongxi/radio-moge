"""Offline ARKit measured native-grid labels, with per-frame RGB registration.

Only exact timestamps, high confidence, and photometrically consistent native
pixels. No nearest-time matching, depth upsampling, pose interpolation, or hole
filling. Native labels are still candidates pending corpus/loss acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sqlite3
import time

import numpy as np
from PIL import Image
import requests
from scipy.ndimage import map_coordinates

from indoor_download import RemoteZip, atomic_write, write_json
from radio_moge.arkit_depth import native_depth, registration_uv, rotate_camera
from radio_moge.geometry_cache import (GeometryPackWriter, canonical_json, publish_json,
                                       read_geometry_member, sha_file)


class MissingNativeAsset(Exception):
    pass


class NativeBundle:
    """One video's five ZIP indices, one HTTP request at a time, retained sources."""
    def __init__(self, root):
        self.root, self.session, self.archives, self.video = root, requests.Session(), {}, None

    def get(self, base, kind, stem):
        if base != self.video:
            for archive in self.archives.values():
                archive.zip.close()
                archive.file.close()
            self.archives, self.video = {}, base
        suffix = '.pincam' if kind.endswith('intrinsics') else '.png'
        member, url = f'{kind}/{stem}{suffix}', base + '/' + kind + '.zip'
        if not url.startswith('https://docs-assets.developer.apple.com/'):
            raise ValueError('Only the documented publisher source is allowed')
        path = self.root / hashlib.sha256(url.encode()).hexdigest()[:16] / member
        receipt_path = path.with_name(path.name + '.receipt.json')
        if path.exists() or receipt_path.exists():
            proof = json.loads(receipt_path.read_text())
            if proof['source'] != url or proof['member'] != member or sha_file(path) != proof['sha256']:
                raise ValueError('Native source differs from immutable receipt')
            return path, proof
        if kind not in self.archives:
            self.archives[kind] = RemoteZip(url, self.session)
        archive = self.archives[kind]
        try:
            contents = archive.read(member)
        except KeyError as error:
            raise MissingNativeAsset(member) from error
        proof = dict(source=url, member=member, sha256=hashlib.sha256(contents).hexdigest(),
                     crc32=archive.zip.getinfo(member).CRC, bytes=len(contents))
        atomic_write(path, contents)
        publish_json(receipt_path, proof)
        return path, proof


def rows(db, probe, preflight):
    video_sql = "json_extract(row_json,'$.video_id')"
    videos = [r[0] for r in db.execute(f'SELECT DISTINCT {video_sql} FROM images '
                                      "WHERE dataset='arkitscenes' ORDER BY 1")]
    if preflight:
        selected = {r['video'] for r in probe['records'] if r['stage'] == 'diagnostic_complete'}
        videos = [v for v in videos if v in selected]
        if len(videos) != 12:
            raise ValueError('Require the twelve four-orientation registered camera probes')
    for video in videos:
        values = db.execute(f'SELECT row_json,image FROM images WHERE dataset=? '
                             f'AND {video_sql}=? ORDER BY ordinal', ('arkitscenes', video)).fetchall()
        if preflight:
            values = [values[index] for index in sorted({0, len(values) - 1})]
        for encoded, path in values:
            row = json.loads(encoded)
            if row['split'] != 'train' or row['dataset'] != 'arkitscenes':
                raise ValueError('Only frozen training frames allowed')
            row['image'] = path
            yield row


def load_image(path):
    with Image.open(path) as file:
        if file.getexif().get(274, 1) != 1:
            raise ValueError('Unexpected native image EXIF orientation')
        return np.asarray(file)


def rejected_native_image(missing, valid):
    # Return a Python bool, including when valid.mean() returns a NumPy scalar.
    return bool(missing or valid.mean() < .05)


def convert(row, assets):
    video, turn = str(row['video_id']), row['image_rotation_cw']
    stem = PurePosixPath(row['source_member']).stem
    if stem.split('_')[0] != video or float(stem.split('_')[1]) != row['timestamp']:
        raise ValueError('Native filename differs from frozen timestamp')
    if sha_file(row['image']) != row['sha256']:
        raise ValueError('Original RGB bytes changed')
    base = row['source_url'].rsplit('/', 1)[0]
    paths, proofs = {}, []
    for kind in ('lowres_depth', 'confidence', 'lowres_wide', 'lowres_wide_intrinsics',
                 'vga_wide_intrinsics'):
        paths[kind], proof = assets.get(base, kind, stem)
        proofs.append(proof)
    low, rgb_camera = (np.loadtxt(paths[kind]).tolist() for kind in
                       ('lowres_wide_intrinsics', 'vga_wide_intrinsics'))
    if (not np.allclose(rgb_camera, row['native_camera_width_height_fx_fy_cx_cy'], rtol=0, atol=1e-8)
            or not np.allclose(rotate_camera(rgb_camera, turn),
                               row['camera_width_height_fx_fy_cx_cy'], rtol=0, atol=1e-8)):
        raise ValueError('Native/upright camera differs from frozen image metadata')
    depth, valid, kd = native_depth(load_image(paths['lowres_depth']),
                                    load_image(paths['confidence']), low, turn)
    uv, _, _ = registration_uv(low, rgb_camera)
    rgb, low_rgb = load_image(row['image']), load_image(paths['lowres_wide'])
    if rgb.shape != (int(rgb_camera[1]), int(rgb_camera[0]), 3) or low_rgb.shape != (*uv.shape[:2], 3):
        raise ValueError('Native RGB stream dimensions differ from cameras')
    xy = uv * [rgb.shape[1], rgb.shape[0]] - .5
    sampled = np.stack([map_coordinates(rgb[..., c].astype(float) / 255,
                       [xy[..., 1], xy[..., 0]], order=1, mode='nearest') for c in range(3)], -1)
    error = np.abs(sampled - low_rgb.astype(float) / 255).mean(-1)
    in_frame = (uv[..., 0] >= 0) & (uv[..., 0] <= 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= 1)
    support = np.rot90(in_frame & (error <= .08), -turn // 90)
    initial_count = int(valid.sum())
    valid &= support
    image_registration_pass = bool(error.mean() <= .04 and np.quantile(error, .9) <= .08)
    if not image_registration_pass:
        valid[:] = False
    depth[~valid] = 0
    _, k_depth, k_rgb = registration_uv(kd, rotate_camera(rgb_camera, turn))
    quality = dict(native_confident_stable_pixels=initial_count,
        paired_rgb_mae=float(error.mean()), paired_rgb_error_p90=float(np.quantile(error, .9)),
        valid_fraction=float(valid.mean()), image_registration_pass=image_registration_pass,
        independent_physical_depth_accuracy_checked=False)
    return depth, valid, k_depth, k_rgb, proofs, quality


def run(config, mode, attempt, preflight_name):
    root = Path(config['root'])
    inventory = json.loads((root / 'inventory.json').read_text())
    probe_path = root / 'arkit-registration-probe-01/report.json'
    probe = json.loads(probe_path.read_text())
    if (inventory['manifest_sha256'] != config['manifest_sha256']
            or sha_file(root / 'inventory.sqlite') != inventory['inventory_sha256']
            or probe['stage'] != 'complete' or probe['manifest_sha256'] != config['manifest_sha256']):
        raise ValueError('Inventory and source-registration probe must be complete')
    output = root / attempt
    output.mkdir(exist_ok=False)
    source_root = Path(__file__).parents[1]
    policy = dict(dataset='arkitscenes', manifest_sha256=config['manifest_sha256'],
        registration_probe_sha256=sha_file(probe_path), label_kind='measured_sensor',
        source='raw AppleDepth LiDAR, not FARO mesh and not 3DOD reprocessing',
        conversion='native integer-pixel camera Z mm to m, exact quarter turns, no depth resizing',
        valid_support='confidence2; finite positive nonsaturated; 3x3 stable; calibrated RGB match',
        timestamp_policy='exact filename timestamp only; missing assets explicitly quarantined',
        photometric_gate=dict(pixel_mae_max=.08, image_mae_max=.04, image_p90_max=.08),
        minimum_image_valid_fraction=.05, native_holes_filled=False,
        script_sha256=sha_file(Path(__file__)), code_sha256={name: sha_file(source_root / name) for name in (
            'src/radio_moge/arkit_depth.py', 'src/radio_moge/geometry_sources.py',
            'src/radio_moge/geometry_cache.py', 'scripts/indoor_download.py')})
    policy_hash = hashlib.sha256(canonical_json(policy)).hexdigest()
    if mode == 'generate':
        previous = json.loads((root / preflight_name / 'report.json').read_text())
        if not previous['passed'] or previous['policy_sha256'] != policy_hash:
            raise ValueError('Matching native cache/pack preflight is required')
    status = dict(stage='generating_cache', mode=mode, completed=0, rejected=0,
                  started_at=time.time(), training_admission=False, training_started=False,
                  optimizer_updates=0, policy_sha256=policy_hash)
    write_json(output / 'status.json', status)
    writer, assets = GeometryPackWriter(output / 'packs', policy, max_entries=256), NativeBundle(
        root / 'native-source/arkitscenes')
    rejections, examples, videos = [], [], set()
    try:
        with sqlite3.connect(f'file:{root / "inventory.sqlite"}?mode=ro', uri=True) as db:
            db.execute('PRAGMA cache_size=-32768')
            for row in rows(db, probe, mode == 'preflight'):
                videos.add(row['video_id'])
                missing = False
                try:
                    depth, valid, kd, kr, proofs, stats = convert(row, assets)
                except MissingNativeAsset as error:
                    missing = True
                    stats = dict(reason='missing_exact_timestamp_native_member', detail=str(error))
                except requests.HTTPError as error:
                    if error.response.status_code != 404:
                        raise
                    missing = True
                    stats = dict(reason='publisher_asset_not_available_404')
                reject_image = rejected_native_image(missing, None if missing else valid)
                if reject_image:
                    rejections.append(dict(image=row, reason=stats.get('reason',
                        'insufficient_registered_native_support'), stats=stats))
                    status['rejected'] += 1
                else:
                    provenance = dict(image_sha256=row['sha256'], dataset='arkitscenes',
                        source_id=row['source_member'], label_kind='measured_sensor',
                        camera_kind='calibrated_sensor', metric_calibrated=True,
                        conversion=policy['conversion'], policy_sha256=policy_hash,
                        source_proofs=proofs, source_quality=stats, image_rotation_cw=row['image_rotation_cw'],
                        rgb_intrinsics_normalized=kr.tolist(), rgb_shape=[row['height'], row['width']],
                        depth_to_rgb_uv_transform=(kr @ np.linalg.inv(kd)).tolist(),
                        native_depth_not_resized=True, source_needs_review=False)
                    writer.add(row, depth, valid, kd, provenance)
                    status['completed'] += 1
                if mode == 'preflight':
                    examples.append(dict(video=row['video_id'], timestamp=row['timestamp'],
                        rotation_cw=row['image_rotation_cw'], rejected=reject_image, **stats))
                if (status['completed'] + status['rejected']) % 16 == 0:
                    status.update(updated_at=time.time(), videos_seen=len(videos))
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
                raise ValueError('Published native pack or receipt changed')
            for entry in receipt['entries']:
                read_geometry_member(tar, entry)
                verified += 1
        expected = 24 if mode == 'preflight' else config['expected_counts']['arkitscenes']
        if status['completed'] + status['rejected'] != expected or verified != status['completed']:
            raise ValueError('Native cache did not account for full frozen scope')
        status.update(stage='complete', completed_at=time.time(), expected_scope=expected,
            examples=examples, verified_cache_members=verified, videos_seen=len(videos),
            passed=bool(mode == 'preflight' and status['completed'] >= 20 and not rejections))
        write_json(output / 'status.json', status)
        publish_json(output / 'report.json', status)
        print(json.dumps({k: v for k, v in status.items() if k != 'examples'}), flush=True)
    except BaseException as error:
        status.update(stage='failed', error=type(error).__name__, message=str(error)[:1000])
        write_json(output / 'status.json', status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'generate'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--preflight', default='arkit-cache-preflight-02')
    args = parser.parse_args()
    if any(Path(name).name != name for name in (args.attempt, args.preflight)):
        parser.error('Use a new attempt basename')
    run(json.loads(args.config.read_text()), args.mode, args.attempt, args.preflight)
