"""Cache frozen TartanAir native left Z with both-eye depth/RGB evidence.

One source bundle at a time, one HTTP request at a time, no optimizer. A
completed candidate cache is not global data/loss/training acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
from PIL import Image
import requests

from indoor_download import RemoteZip, atomic_write, write_json
from radio_moge.geometry_cache import (GeometryPackWriter, canonical_json, publish_json,
                                       read_geometry_member, sha_file)
from radio_moge.tartanair_depth import convert_tartanair


class NativeBundle:
    def __init__(self, root):
        self.root, self.session = root, requests.Session()
        self.base, self.archives = None, {}

    def get(self, row, kind):
        source = row['source_url']
        if (not source.startswith('https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/')
                or not source.endswith('/image_left.zip') or kind not in (
                    'depth_left', 'depth_right', 'image_right')):
            raise ValueError('Only frozen V1 publisher sources and corresponding eyes allowed')
        base = source.rsplit('/', 1)[0]
        if self.base != base:
            for archive in self.archives.values():
                archive.zip.close()
                archive.file.close()
            self.base, self.archives = base, {}
        member = row['source_member'].replace('/image_left/', '/' + kind + '/')
        member = member.replace('_left.png', dict(depth_left='_left_depth.npy',
            depth_right='_right_depth.npy', image_right='_right.png')[kind])
        url = base + '/' + kind + '.zip'
        path = self.root / hashlib.sha256(url.encode()).hexdigest()[:16] / member
        receipt = path.with_name(path.name + '.receipt.json')
        if path.exists() or receipt.exists():
            proof = json.loads(receipt.read_text())
            if proof['source'] != url or proof['member'] != member or sha_file(path) != proof['sha256']:
                raise ValueError('Retained native source changed')
        else:
            if kind not in self.archives:
                self.archives[kind] = RemoteZip(url, self.session)
            archive = self.archives[kind]
            contents = archive.read(member)
            proof = dict(source=url, member=member, sha256=hashlib.sha256(contents).hexdigest(),
                crc32=archive.zip.getinfo(member).CRC, bytes=len(contents))
            atomic_write(path, contents)
            publish_json(receipt, proof)
        return path, proof


def rows(db, preflight):
    query = ("SELECT row_json,image FROM images WHERE dataset='tartanair' "
             "ORDER BY json_extract(row_json,'$.source_url'),ordinal")
    previous, first, last = None, None, None
    for encoded, image in db.execute(query):
        row = json.loads(encoded)
        if row['split'] != 'train' or row['dataset'] != 'tartanair':
            raise ValueError('Only frozen training frames permitted')
        row['image'] = image
        if not preflight:
            yield row
            continue
        if row['source_url'] != previous:
            if last is not None and last['sha256'] != first['sha256']:
                yield last
            previous, first = row['source_url'], row
            yield row
        last = row
    if preflight and last is not None and last['sha256'] != first['sha256']:
        yield last


def rgb(path):
    with Image.open(path) as file:
        if file.getexif().get(274, 1) != 1 or file.mode != 'RGB':
            raise ValueError('Unexpected source RGB encoding/orientation')
        return np.asarray(file)


def run(config, mode, attempt, preflight_name):
    root = Path(config['root'])
    inventory = json.loads((root / 'inventory.json').read_text())
    if (inventory['manifest_sha256'] != config['manifest_sha256']
            or sha_file(root / 'inventory.sqlite') != inventory['inventory_sha256']):
        raise ValueError('Frozen inventory changed')
    output = root / attempt
    output.mkdir(exist_ok=False)
    (output / 'rejected').mkdir()
    source_root = Path(__file__).parents[1]
    policy = dict(dataset='tartanair', manifest_sha256=config['manifest_sha256'],
        source_documentation='https://github.com/castacks/tartanair_tools/blob/master/data_type.md',
        label_kind='synthetic_renderer', conversion='native V1 camera Z float meters, original RGB grid',
        calibration=dict(width=640, height=480, fx=320, fy=320, cx=320, cy=240, baseline_m=.25),
        task_range_m=[.1, 200.], task_range_is_not_publisher_invalid_mask=True,
        native_support='finite positive; separately exclude outside task range including finite sky sphere',
        stereo_support='both eyes 3x3 stable (.05+.02Z m), x_right=x_left-80/Z, no y shift',
        stereo_gates=dict(z_difference_m='.01+.005Z', rgb_pixel_mae_max=.08),
        minimum_image_valid_fraction=.05, native_holes_filled=False,
        independent_physical_depth_accuracy_checked=False,
        script_sha256=sha_file(Path(__file__)), code_sha256={name: sha_file(source_root / name) for name in (
            'src/radio_moge/tartanair_depth.py', 'src/radio_moge/geometry_sources.py',
            'src/radio_moge/geometry_cache.py', 'scripts/indoor_download.py')})
    policy_hash = hashlib.sha256(canonical_json(policy)).hexdigest()
    if mode == 'generate':
        previous = json.loads((root / preflight_name / 'report.json').read_text())
        if not previous['passed'] or previous['policy_sha256'] != policy_hash:
            raise ValueError('Matching stratified conversion/pack preflight required')
    status = dict(stage='generating_cache', mode=mode, completed=0, rejected=0,
        started_at=time.time(), training_admission=False, training_started=False,
        optimizer_updates=0, policy_sha256=policy_hash)
    write_json(output / 'status.json', status)
    assets = NativeBundle(root / 'native-source/tartanair')
    writer = GeometryPackWriter(output / 'packs', policy, max_entries=256)
    examples, rejections, sources, environments = [], [], set(), set()
    try:
        with sqlite3.connect(f'file:{root / "inventory.sqlite"}?mode=ro', uri=True) as db:
            db.execute('PRAGMA cache_size=-32768')
            for row in rows(db, mode == 'preflight'):
                if sha_file(row['image']) != row['sha256'] or (row['width'], row['height']) != (640, 480):
                    raise ValueError('Frozen original RGB changed')
                paths, proofs = {}, []
                for kind in ('depth_left', 'depth_right', 'image_right'):
                    paths[kind], proof = assets.get(row, kind)
                    proofs.append(proof)
                depth, valid, k, quality = convert_tartanair(
                    np.load(paths['depth_left'], allow_pickle=False),
                    np.load(paths['depth_right'], allow_pickle=False),
                    rgb(row['image']), rgb(paths['image_right']))
                reject = bool(valid.mean() < .05)
                if reject:
                    record = dict(image=row, reason='insufficient_stereo_verified_support', stats=quality,
                                  source_proofs=proofs)
                    rejections.append(record)
                    # Persist rejections immediately, even if a later source download fails.
                    publish_json(output / 'rejected' / (row['sha256'] + '.json'), record)
                    status['rejected'] += 1
                else:
                    provenance = dict(image_sha256=row['sha256'], dataset='tartanair',
                        source_id=row['source_member'], label_kind='synthetic_renderer',
                        camera_kind='synthetic_camera', metric_calibrated=True,
                        conversion=policy['conversion'], policy_sha256=policy_hash,
                        source_proofs=proofs, source_quality=quality, source_needs_review=False)
                    writer.add(row, depth, valid, k, provenance)
                    status['completed'] += 1
                sources.add(row['source_url'])
                environments.add(row['environment'])
                if mode == 'preflight':
                    examples.append(dict(source_id=row['source_member'], source_url=row['source_url'],
                                         rejected=reject, **quality))
                if (status['completed'] + status['rejected']) % 8 == 0:
                    status.update(updated_at=time.time(), source_bundles_seen=len(sources),
                                  environment_names_seen=len(environments))
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
                raise ValueError('Published native pack changed')
            for entry in receipt['entries']:
                read_geometry_member(tar, entry)
                verified += 1
        expected = 48 if mode == 'preflight' else config['expected_counts']['tartanair']
        if status['completed'] + status['rejected'] != expected or verified != status['completed']:
            raise ValueError('Native cache did not account for frozen scope')
        status.update(stage='complete', completed_at=time.time(), expected_scope=expected,
            examples=examples, verified_cache_members=verified, source_bundles_seen=len(sources),
            environment_names_seen=len(environments),
            passed=bool(mode == 'preflight' and len(sources) == 24 and status['completed'] >= 32))
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
    parser.add_argument('--preflight', default='tartanair-cache-preflight-01')
    args = parser.parse_args()
    if any(Path(name).name != name for name in (args.attempt, args.preflight)):
        parser.error('Use a new attempt basename')
    run(json.loads(args.config.read_text()), args.mode, args.attempt, args.preflight)
