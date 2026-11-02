"""Census frozen perspective camera records; no depth edits or training admission."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sqlite3
import time
import zipfile

import numpy as np

from indoor_download import write_json
from radio_moge.geometry_cache import publish_json, sha_file
from radio_moge.structured3d_depth import perspective_camera


def audit(config, attempt):
    root = Path(config['root'])
    output = root / attempt
    output.mkdir(exist_ok=False)
    inventory = json.loads((root / 'inventory.json').read_text())
    if sha_file(root / 'inventory.sqlite') != inventory['inventory_sha256']:
        raise ValueError('Frozen inventory changed')
    status = dict(stage='auditing', completed=0, rejected=0, old_orthonormal_check_failures=0,
        maximum_abs_direction_cosine=0., manifest_sha256=config['manifest_sha256'],
        training_admission=False, optimizer_updates=0, started_at=time.time(),
        parser_sha256=sha_file(Path(__file__).parents[1] / 'src/radio_moge/structured3d_depth.py'),
        original_archive_hashes_rechecked=False)
    active, archive = None, None
    worst, failures = [], []
    try:
        with sqlite3.connect(f'file:{root / "inventory.sqlite"}?mode=ro', uri=True) as db, \
                (output / 'cameras.jsonl.incoming').open('x') as log:
            query = ("SELECT row_json FROM images WHERE dataset='structured3d' "
                     "ORDER BY json_extract(row_json,'$.source_archive'),ordinal")
            for (encoded,) in db.execute(query):
                row = json.loads(encoded)
                if row['split'] != 'train' or '/perspective/' not in row['source_member']:
                    raise ValueError('Only frozen perspective training images allowed')
                if active != row['source_archive']:
                    if archive is not None:
                        archive.close()
                    active = row['source_archive']
                    path = Path(config['dataset_root']) / 'structured3d-perspective-v1-20261022/archives' / active
                    archive = zipfile.ZipFile(path)
                member = str(PurePosixPath(row['source_member']).parent / 'camera_pose.txt')
                payload = archive.read(member)
                camera = np.fromstring(payload.decode(), sep=' ')
                record = dict(image_sha256=row['sha256'], archive=active, member=member,
                              camera_sha256=hashlib.sha256(payload).hexdigest())
                try:
                    _, stats = perspective_camera(camera, row['height'], row['width'])
                    dot = abs(stats['normalized_look_up_dot'])
                    old_failed = bool(abs(stats['look_norm'] - 1) > 1e-4
                        or abs(stats['up_norm'] - 1) > 1e-4 or abs(camera[3:6] @ camera[6:9]) > 1e-4)
                    record.update(stage='camera_encoding_valid', old_check_failed=old_failed, **stats)
                    status['old_orthonormal_check_failures'] += int(old_failed)
                    status['maximum_abs_direction_cosine'] = max(status['maximum_abs_direction_cosine'], dot)
                    worst.append(record)
                    worst = sorted(worst, key=lambda r: abs(r['normalized_look_up_dot']), reverse=True)[:16]
                except ValueError as error:
                    record.update(stage='invalid_camera', error=str(error), camera=camera.tolist())
                    failures.append(record)
                    status['rejected'] += 1
                log.write(json.dumps(record, allow_nan=False) + '\n')
                status['completed'] += 1
                if status['completed'] % 4096 == 0:
                    status['updated_at'] = time.time()
                    write_json(output / 'status.json', status)
                    print(json.dumps(status), flush=True)
        if status['completed'] != config['expected_counts']['structured3d']:
            raise ValueError('Camera census omitted frozen images')
        (output / 'cameras.jsonl.incoming').rename(output / 'cameras.jsonl')
        status.update(stage='complete', completed_at=time.time(),
            records_sha256=sha_file(output / 'cameras.jsonl'), worst=worst, failures=failures,
            depth_geometry_acceptance=False)
        write_json(output / 'status.json', status)
        publish_json(output / 'report.json', status)
    except BaseException as error:
        status.update(stage='failed', error=type(error).__name__, message=str(error))
        write_json(output / 'status.json', status)
        raise
    finally:
        if archive is not None:
            archive.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if Path(args.attempt).name != args.attempt:
        parser.error('Use a new attempt basename')
    audit(json.loads(args.config.read_text()), args.attempt)
