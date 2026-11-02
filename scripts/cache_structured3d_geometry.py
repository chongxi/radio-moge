"""Prepare native Structured3D depth packs from already downloaded ZIPs.

Checks upstream errata, each archive SHA, each member CRC/SHA, frozen RGB bytes,
native units/K, local normals, semantic support, and visible structural planes.
Publication is only a candidate cache: global quality/loss acceptance is separate.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sqlite3
import time
import zipfile

import numpy as np
from PIL import Image
import requests

from indoor_download import atomic_write, write_json
from radio_moge.geometry_cache import (GeometryPackWriter, canonical_json, publish_json,
                                       read_geometry_member, sha_file)
from radio_moge.structured3d_depth import convert_structured3d, is_erratum, parse_errata


def metadata(root, name):
    url = 'https://raw.githubusercontent.com/bertjiazheng/Structured3D/master/metadata/' + name
    path = root / 'native-source/structured3d-metadata' / name
    receipt = path.with_name(path.name + '.receipt.json')
    if path.exists() or receipt.exists():
        proof = json.loads(receipt.read_text())
        if proof['source'] != url or sha_file(path) != proof['sha256']:
            raise ValueError('Stored upstream metadata changed')
    else:
        response = requests.get(url, timeout=(15, 60))
        response.raise_for_status()
        if not 0 < len(response.content) < 256 * 1024:
            raise ValueError('Upstream metadata exceeds bound')
        proof = dict(source=url, sha256=hashlib.sha256(response.content).hexdigest())
        atomic_write(path, response.content)
        publish_json(receipt, proof)
    return path.read_text(), proof


def rows(db, preflight):
    archive_sql = "json_extract(row_json,'$.source_archive')"
    archives = [r[0] for r in db.execute(f'SELECT DISTINCT {archive_sql} FROM images '
                                        'WHERE dataset=? ORDER BY 1', ('structured3d',))]
    for archive in archives:
        query = f'SELECT row_json,image FROM images WHERE dataset=? AND {archive_sql}=? ORDER BY ordinal'
        values = db.execute(query, ('structured3d', archive)).fetchall()
        if preflight:
            values = [values[index] for index in sorted({0, len(values) - 1})]
        for encoded, image in values:
            row = json.loads(encoded)
            if row['dataset'] != 'structured3d' or row['split'] != 'train':
                raise ValueError('Only frozen training images are permitted')
            row['image'] = image
            yield row


def image_array(data):
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image)


def run(config, mode, attempt, preflight_name):
    root, data = Path(config['root']), Path(config['dataset_root'])
    inventory = json.loads((root / 'inventory.json').read_text())
    if (inventory['manifest_sha256'] != config['manifest_sha256']
            or sha_file(root / 'inventory.sqlite') != inventory['inventory_sha256']):
        raise ValueError('Frozen inventory changed')
    output = root / attempt
    output.mkdir(exist_ok=False)
    errata_text, errata_proof = metadata(root, 'errata.txt')
    errata = parse_errata(errata_text)
    source_root = Path(__file__).parents[1]
    policy = dict(dataset='structured3d', manifest_sha256=config['manifest_sha256'],
        label_kind='synthetic_renderer', upstream_errata=errata_proof,
        conversion='native uint16 camera Z millimeters to float32 meters; original integer pixel K',
        normal_convention='native camera x-right/y-up/z-forward to OpenCV Y-down; fixed, not fitted',
        mask='native nonzero/nonsaturated/opaque; 5x5 stable; normal agreement 15deg; semantics; layout',
        conservative_exclusions='all upstream errata scenes/rooms, window/mirror, border/occlusion uncertainty',
        layout_gate='only matching semantic class and normal; metric residual <=5mm; >=100 verified pixels',
        minimum_image_valid_fraction=.05, native_holes_filled=False, original_RGB_preserved=True,
        script_sha256=sha_file(Path(__file__)), code_sha256={name: sha_file(source_root / name) for name in (
            'src/radio_moge/structured3d_depth.py', 'src/radio_moge/geometry_sources.py',
            'src/radio_moge/geometry_cache.py', 'scripts/indoor_download.py')})
    policy_hash = hashlib.sha256(canonical_json(policy)).hexdigest()
    if mode == 'generate':
        previous = json.loads((root / preflight_name / 'report.json').read_text())
        if not previous['passed'] or previous['policy_sha256'] != policy_hash:
            raise ValueError('Matching full conversion/pack preflight required')
    status = dict(stage='generating_cache', mode=mode, completed=0, rejected=0,
                  source_review_images=0, archives_verified=0, started_at=time.time(),
                  training_admission=False, training_started=False, optimizer_updates=0,
                  policy_sha256=policy_hash)
    writer = GeometryPackWriter(output / 'packs', policy, max_entries=128)
    archive, active, archive_stats = None, None, {}
    rejections, examples = [], []
    try:
        with sqlite3.connect(f'file:{root / "inventory.sqlite"}?mode=ro', uri=True) as db:
            db.execute('PRAGMA cache_size=-32768')
            for row in rows(db, mode == 'preflight'):
                if is_erratum(row, errata):
                    rejections.append(dict(image=row, reason='upstream_errata_scene_or_room',
                                           source_proofs=[errata_proof]))
                    status['rejected'] += 1
                    continue
                name = row['source_archive']
                if name != active:
                    if archive is not None:
                        archive.close()
                    path = data / 'structured3d-perspective-v1-20261022/archives' / name
                    original = json.loads((path.parent.parent / 'receipts' / (name + '.json')).read_text())
                    if path.stat().st_size != original['bytes'] or sha_file(path) != original['sha256']:
                        raise ValueError('Native archive differs from completed download receipt')
                    archive, active = zipfile.ZipFile(path), name
                    archive_stats[name] = dict(sha256=original['sha256'], bytes=original['bytes'],
                                               source=original['url'])
                    status['archives_verified'] += 1
                    write_json(output / 'status.json', status)
                prefix = str(PurePosixPath(row['source_member']).parent)
                native, proofs = {}, [errata_proof]
                for key in ('rgb_rawlight.png', 'depth.png', 'normal.png', 'semantic.png',
                            'camera_pose.txt', 'layout.json'):
                    member = prefix + '/' + key
                    contents = archive.read(member)  # zipfile checks decoded CRC.
                    native[key] = contents
                    proofs.append(dict(source=str(path), member=member,
                        sha256=hashlib.sha256(contents).hexdigest(), crc32=archive.getinfo(member).CRC,
                        archive_sha256=archive_stats[name]['sha256']))
                if (hashlib.sha256(native['rgb_rawlight.png']).hexdigest() != row['sha256']
                        or sha_file(row['image']) != row['sha256']):
                    raise ValueError('Native RGB is not the frozen training image')
                with Image.open(io.BytesIO(native['rgb_rawlight.png'])) as image:
                    if image.getexif().get(274, 1) != 1:
                        raise ValueError('Unexpected native RGB orientation')
                    rgba = np.asarray(image.convert('RGBA'))
                raw = image_array(native['depth.png'])
                if raw.shape != rgba.shape[:2]:
                    raise ValueError('Native RGB and depth dimensions differ')
                depth, valid, k, stats = convert_structured3d(raw,
                    image_array(native['normal.png'])[..., :3], image_array(native['semantic.png']),
                    np.loadtxt(io.StringIO(native['camera_pose.txt'].decode())),
                    json.loads(native['layout.json']), rgba[..., 3])
                status['source_review_images'] += int(stats['source_needs_review'])
                if valid.mean() < .05:
                    rejections.append(dict(image=row, reason='insufficient_verified_support', stats=stats))
                    status['rejected'] += 1
                else:
                    provenance = dict(image_sha256=row['sha256'], dataset='structured3d',
                        source_id=prefix, label_kind='synthetic_renderer', camera_kind='synthetic_camera',
                        metric_calibrated=True, conversion=policy['conversion'], policy_sha256=policy_hash,
                        source_proofs=proofs, source_quality=stats,
                        source_needs_review=stats['source_needs_review'])
                    writer.add(row, depth, valid, k, provenance)
                    status['completed'] += 1
                if mode == 'preflight':
                    examples.append(dict(scene=row['scene'], room=row['room'],
                        configuration=row['configuration'], view=row['view'], **stats))
                if (status['completed'] + status['rejected']) % 16 == 0:
                    status['updated_at'] = time.time()
                    write_json(output / 'status.json', status)
                    print(json.dumps(status), flush=True)
        writer.close()
        if archive is not None:
            archive.close()
        publish_json(output / 'rejections.json', rejections)
        verified = 0
        for item in writer.packs:
            receipt_path = output / 'packs' / item['receipt']
            receipt = json.loads(receipt_path.read_text())
            if sha_file(receipt_path) != item['sha256']:
                raise ValueError('Pack receipt changed')
            tar = output / 'packs' / receipt['tar']
            if sha_file(tar) != receipt['sha256']:
                raise ValueError('Native geometry pack changed')
            for entry in receipt['entries']:
                read_geometry_member(tar, entry)
                verified += 1
        expected = 58 if mode == 'preflight' else config['expected_counts']['structured3d']
        if status['completed'] + status['rejected'] != expected or verified != status['completed']:
            raise ValueError('Native cache does not account for the frozen scope')
        status.update(stage='complete', completed_at=time.time(), expected_scope=expected,
            verified_cache_members=verified, examples=examples, archives=archive_stats,
            passed=bool(mode == 'preflight' and status['completed'] >= 50
                and not status['source_review_images']
                and all(x['reason'] == 'upstream_errata_scene_or_room' for x in rejections)))
        write_json(output / 'status.json', status)
        publish_json(output / 'report.json', status)
        print(json.dumps({k: v for k, v in status.items() if k not in ('examples', 'archives')}), flush=True)
    except BaseException as error:
        status.update(stage='failed', error=type(error).__name__, message=str(error)[:1000])
        write_json(output / 'status.json', status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'generate'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--preflight', default='structured3d-cache-preflight-01')
    args = parser.parse_args()
    if any(Path(name).name != name for name in (args.attempt, args.preflight)):
        parser.error('Use a new attempt basename')
    run(json.loads(args.config.read_text()), args.mode, args.attempt, args.preflight)
