"""Offline native Hypersim labels, with every pixel checked against world geometry.

One network/download process, immutable source receipts, no training admission.
All downloaded native assets are retained and reused only after SHA verification.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sqlite3
import time

import h5py
import numpy as np
from PIL import Image
import requests

from indoor_download import RemoteZip, atomic_write, write_json
from radio_moge.geometry_cache import (GeometryPackWriter, canonical_json, publish_json,
                                       read_geometry_member, sha_file)
from radio_moge.hypersim_depth import convert_verified_range


class NativeAssets:
    def __init__(self, root):
        self.root, self.session = root, requests.Session()
        self.archive_url, self.archive = None, None

    def get(self, url, member):
        name = PurePosixPath(member)
        if name.is_absolute() or '..' in name.parts or not url.startswith('https://'):
            raise ValueError('Unsafe public source reference')
        path = self.root / hashlib.sha256(url.encode()).hexdigest()[:16] / member
        receipt_path = path.with_name(path.name + '.receipt.json')
        if path.exists() or receipt_path.exists():
            if not path.is_file() or not receipt_path.is_file():
                raise ValueError('Unreceipted native asset; inspect before recovery')
            receipt = json.loads(receipt_path.read_text())
            if (receipt['source'] != url or receipt['member'] != member
                    or sha_file(path) != receipt['sha256']):
                raise ValueError('Native asset receipt differs')
            return path, receipt
        if url != self.archive_url:
            if self.archive is not None:
                self.archive.zip.close()
                self.archive.file.close()
            self.archive_url, self.archive = url, RemoteZip(url, self.session)
        data = self.archive.read(member)  # Bounded HTTP Range and ZIP CRC verification.
        info = self.archive.zip.getinfo(member)
        receipt = dict(source=url, member=member, sha256=hashlib.sha256(data).hexdigest(),
                       crc32=info.CRC, bytes=len(data), archive_bytes=self.archive.file.size)
        atomic_write(path, data)
        publish_json(receipt_path, receipt)
        return path, receipt


def hdf5(path):
    with h5py.File(path, 'r') as file:
        return np.asarray(file['dataset'])


def source_rows(db, groups, preflight=False):
    for group in groups:
        rows = db.execute('SELECT row_json,image FROM images WHERE scene_group=? ORDER BY ordinal',
                          (group,)).fetchall()
        if preflight:
            rows = [rows[index] for index in sorted({0, len(rows) // 2, len(rows) - 1})]
        for encoded, path in rows:
            row = json.loads(encoded)
            if row['dataset'] != 'hypersim' or row['split'] != 'train':
                raise ValueError('Native cache is restricted to frozen training images')
            row['image'] = path
            yield row


def camera_assets(assets, row):
    scene = row['group'].split(':')[1]
    files, proofs = {}, []
    for key in ('camera_keyframe_positions.hdf5', 'camera_keyframe_orientations.hdf5',
                'camera_keyframe_frame_indices.hdf5', 'metadata_scene.csv'):
        member = f'{scene}/_detail/' + ('' if key == 'metadata_scene.csv' else row['camera'] + '/') + key
        path, proof = assets.get(row['source_url'], member)
        proofs.append(proof)
        if key.endswith('.hdf5'):
            files[key] = hdf5(path)
        else:
            with path.open() as file:
                files['scale'] = float(next(x for x in csv.DictReader(file)
                    if x['parameter_name'] == 'meters_per_asset_unit')['parameter_value'])
    return files, proofs


def run(config, mode, attempt, preflight_name):
    root = Path(config['root'])
    inventory = json.loads((root / 'inventory.json').read_text())
    tilt = json.loads((root / 'hypersim-tilt-probe-02/report.json').read_text())
    if (inventory['manifest_sha256'] != config['manifest_sha256'] or not tilt['passes']
            or sha_file(root / 'inventory.sqlite') != inventory['inventory_sha256']):
        raise ValueError('Native source/camera gates have not passed')
    output = root / attempt
    if output.exists():
        raise FileExistsError('Cache attempt exists; no automatic restart/overwrite')
    output.mkdir()
    metadata_path = root / 'native-probe-01/hypersim/metadata_camera_parameters.csv'
    with metadata_path.open() as file:
        matrices = {row['scene_name']: np.array([[float(row[f'M_cam_from_uv_{i}{j}'])
                     for j in range(3)] for i in range(3)]) for row in csv.DictReader(file)}
    script_root = Path(__file__).parents[1]
    policy = dict(dataset='hypersim', label_kind='synthetic_renderer',
        manifest_sha256=config['manifest_sha256'], camera_metadata_sha256=sha_file(metadata_path),
        conversion='range_to_Z_in_exact_RQ_pinhole_frame_no_RGB_warp',
        pixel_gate='XYZ agrees with independent world positions within 2x storage quantization + 0.1mm',
        minimum_image_valid_fraction=.05, native_holes_filled=False, original_RGB_preserved=True,
        source_dtypes_preserved_for_quantization_bounds=True,
        script_sha256=sha_file(Path(__file__)), code_sha256={name: sha_file(script_root / name)
            for name in ('src/radio_moge/hypersim_depth.py', 'src/radio_moge/geometry_sources.py',
                         'src/radio_moge/geometry_cache.py', 'scripts/indoor_download.py')})
    policy_hash = hashlib.sha256(canonical_json(policy)).hexdigest()
    if mode == 'generate':
        previous = json.loads((root / preflight_name / 'report.json').read_text())
        if not previous['passed'] or previous['policy_sha256'] != policy_hash:
            raise ValueError('Matching full conversion/pack preflight required')
    groups = sorted(x['group'] for x in inventory['source_groups'] if x['dataset'] == 'hypersim')
    if mode == 'preflight':
        chosen = {'ai_001_001', 'ai_001_002'} | {x['scene'] for x in tilt['probe_scenes']}
        groups = [x for x in groups if x.split(':')[1] in chosen]
        if len(groups) != 5:
            raise ValueError('Expected default and extreme-tilt training scenes')
    writer = GeometryPackWriter(output / 'packs', policy, max_entries=128)
    assets = NativeAssets(root / 'native-source/hypersim')
    cameras, rejections, example_stats = {}, [], []
    status = dict(stage='generating_cache', mode=mode, completed=0, rejected=0,
                  source_review_images=0, started_at=time.time(), training_admission=False,
                  training_started=False, optimizer_updates=0, policy_sha256=policy_hash)
    write_json(output / 'status.json', status)
    try:
        with sqlite3.connect(f'file:{root / "inventory.sqlite"}?mode=ro', uri=True) as db:
            db.execute('PRAGMA cache_size=-32768')
            for row in source_rows(db, groups, mode == 'preflight'):
                scene, camera = row['group'].split(':')[1], row['camera']
                pair = (scene, camera)
                if pair not in cameras:
                    cameras[pair] = camera_assets(assets, row)
                camera_data, camera_proofs = cameras[pair]
                indices = camera_data['camera_keyframe_frame_indices.hdf5'].reshape(-1)
                match = np.flatnonzero(indices == row['frame_id'])
                if len(match) != 1:
                    raise ValueError('Non-unique source camera frame')
                index = match.item()
                stem = f'{scene}/images/scene_{camera}_geometry_hdf5/frame.{row["frame_id"]:04d}'
                range_path, range_proof = assets.get(row['source_url'], stem + '.depth_meters.hdf5')
                world_path, world_proof = assets.get(row['source_url'], stem + '.position.hdf5')
                radial, world = hdf5(range_path), hdf5(world_path)
                rgb_bytes = Path(row['image']).read_bytes()
                if hashlib.sha256(rgb_bytes).hexdigest() != row['sha256']:
                    raise ValueError('Hypersim RGB bytes changed')
                with Image.open(io.BytesIO(rgb_bytes)) as file:
                    if file.size[::-1] != radial.shape or file.getexif().get(274, 1) != 1:
                        raise ValueError('Hypersim RGB/range size or orientation mismatch')
                depth, valid, k, stats = convert_verified_range(radial, world,
                    camera_data['camera_keyframe_orientations.hdf5'][index],
                    camera_data['camera_keyframe_positions.hdf5'][index],
                    camera_data['scale'], matrices[scene])
                needs_review = stats['inconsistent_supported_pixels'] > .01 * stats['supported_pixels']
                status['source_review_images'] += int(needs_review)
                if valid.mean() < .05:
                    rejections.append(dict(image=row, reason='insufficient_world_verified_support', stats=stats))
                    status['rejected'] += 1
                else:
                    provenance = dict(image_sha256=row['sha256'], dataset='hypersim',
                        source_id=stem, label_kind='synthetic_renderer', camera_kind='synthetic_camera',
                        metric_calibrated=True, conversion=policy['conversion'], policy_sha256=policy_hash,
                        source_proofs=[range_proof, world_proof, *camera_proofs,
                            dict(source=str(metadata_path), sha256=policy['camera_metadata_sha256'])],
                        source_quality=stats, source_needs_review=needs_review)
                    writer.add(row, depth, valid, k, provenance)
                    status['completed'] += 1
                if mode == 'preflight':
                    example_stats.append(dict(scene=scene, camera=camera, frame=row['frame_id'], **stats))
                if (status['completed'] + status['rejected']) % 16 == 0:
                    status['updated_at'] = time.time()
                    write_json(output / 'status.json', status)
                    print(json.dumps(status), flush=True)
        writer.close()
        publish_json(output / 'rejections.json', rejections)
        verified = 0
        for item in writer.packs:
            receipt_path = output / 'packs' / item['receipt']
            receipt = json.loads(receipt_path.read_text())
            if sha_file(receipt_path) != item['sha256']:
                raise ValueError('Geometry receipt changed after publication')
            tar = output / 'packs' / receipt['tar']
            if sha_file(tar) != receipt['sha256']:
                raise ValueError('Native geometry pack changed after publication')
            for entry in receipt['entries']:
                read_geometry_member(tar, entry)
                verified += 1
        expected = 15 if mode == 'preflight' else config['expected_counts']['hypersim']
        if status['completed'] + status['rejected'] != expected or verified != status['completed']:
            raise ValueError('Native cache did not account for the expected scope')
        status.update(stage='complete', completed_at=time.time(), examples=example_stats,
                      verified_cache_members=verified,
                      passed=mode == 'preflight' and not rejections and not status['source_review_images'])
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
    parser.add_argument('--preflight', default='hypersim-cache-preflight-02')
    args = parser.parse_args()
    if Path(args.attempt).name != args.attempt or Path(args.preflight).name != args.preflight:
        parser.error('attempt must be a new directory basename')
    run(json.loads(args.config.read_text()), args.mode, args.attempt, args.preflight)
