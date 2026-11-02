"""Recheck every authorized NYU training pair against the original rawDepths MAT.

Never opens validation/test frame values and never starts model inference/training.
This proves cache/source identity, not absolute Kinect accuracy or final camera QC.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
from PIL import Image

from indoor_download import write_json


def sha_file(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def verify(config):
    root, data = Path(config['root']), Path(config['dataset_root'])
    output = root / 'nyuv2-source-verification.json'
    if output.exists():
        raise FileExistsError('Source verification already exists; do not overwrite')
    manifest = data / config['manifest']
    source = root / 'native-source/nyuv2/nyu_depth_v2_labeled.mat'
    source_info = json.loads((data / 'nyuv2-labeled-s42/dataset_info.json').read_text())
    if source_info['depth_key'] != 'rawDepths' or source_info['depth_units'] != 'meters':
        raise ValueError('Cached NYU data not identified as raw projected sensor depth')
    result = dict(stage='verifying_source', started_at=time.time(), verified=0,
                  training_started=False, training_admission=False,
                  scope='Source-byte equality, NOT absolute sensor accuracy or final camera acceptance')
    write_json(output, result)
    try:
        rows, checksum = [], hashlib.sha256()
        with manifest.open('rb') as file:
            for line in file:
                checksum.update(line)
                if line.strip():
                    row = json.loads(line)
                    if row['dataset'] == 'nyuv2':
                        if row.get('split') != 'train':
                            raise ValueError('NYU source recheck accepts training frames only')
                        rows.append(row)
        if checksum.hexdigest() != config['manifest_sha256']:
            raise ValueError('Authorized training manifest changed')
        if len(rows) != config['expected_counts']['nyuv2']:
            raise ValueError('Authorized NYU training count changed')
        source_hash = sha_file(source)
        if source_hash != source_info['source_sha256']:
            raise ValueError('Original MAT differs from original preparation receipt')
        records = []
        with h5py.File(source, 'r') as mat:
            for row in rows:
                index = int(row['id'].split('_')[-1]) - 1
                raw_rgb = np.ascontiguousarray(mat['images'][index].transpose(2, 1, 0), dtype=np.uint8)
                raw_depth = np.ascontiguousarray(mat['rawDepths'][index].T, dtype=np.float32)
                valid = np.isfinite(raw_depth) & (raw_depth > 0)
                raw_depth[~valid] = 0
                rgb_path, depth_path = manifest.parent / row['image'], manifest.parent / row['depth']
                with Image.open(rgb_path) as file:
                    cached_rgb = np.array(file.convert('RGB'))
                cached_depth = np.load(depth_path, allow_pickle=False)
                if (not np.array_equal(raw_rgb, cached_rgb)
                        or not np.array_equal(raw_depth, cached_depth)
                        or sha_file(rgb_path) != row['sha256']):
                    raise ValueError(f'NYU cached source mismatch at {row["id"]}')
                records.append(dict(id=row['id'], scene=row['group'],
                    rgb_sha256=row['sha256'], depth_sha256=sha_file(depth_path),
                    measured_support_fraction=float(valid.mean()),
                    at_or_above_10m_fraction=float((cached_depth >= 10).mean())))
                result['verified'] += 1
                if result['verified'] % 100 == 0:
                    write_json(output, result)
        result.update(stage='source_identity_verified', source=str(source), source_sha256=source_hash,
            manifest_sha256=checksum.hexdigest(), records=records, completed_at=time.time(),
            independent_scene_groups=len({row['group'] for row in rows}),
            invalid_pixels_filled=False, depth_key='rawDepths', camera_acceptance='PENDING')
        write_json(output, result)
    except BaseException as error:
        result.update(stage='failed', error=type(error).__name__, message=str(error))
        write_json(output, result)
        raise
    print(json.dumps({key: result[key] for key in ('stage', 'verified', 'independent_scene_groups',
                                                 'training_admission')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    verify(json.loads(args.config.read_text()))
