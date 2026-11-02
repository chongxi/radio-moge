"""Canonical heldout RGB census and exact train comparison, without reading depth."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

import PIL

from indoor_download import write_json
from radio_moge.corpus_admission import checked_json, checked_path, file_ref, readonly_db
from radio_moge.geometry_cache import canonical_json, publish_json
from radio_moge.heldout_rgb import identity, inspect_heldout_rgb, manifest_protocol


def records(ref, protocol, preflight=False):
    path = checked_path({k: ref[k] for k in ('path', 'sha256')})
    selected, count = {}, 0
    with path.open('rb') as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            item = (number, row, hashlib.sha256(line).hexdigest())
            count += 1
            if preflight:
                info = identity(row, path, protocol)
                variant = (info['dataset'], info['original_sha256_kind'],
                           row.get('image_rotation_cw', 0), 'width' in row)
                if variant not in selected:
                    selected[variant] = [item, item, item]
                values = selected[variant]
                values[1] = item
                if row['sha256'] < values[2][1]['sha256']:
                    values[2] = item
            else:
                yield item
    if count != ref['rows']:
        raise ValueError('Heldout manifest row count differs')
    checked_path({k: ref[k] for k in ('path', 'sha256')})
    if preflight:
        unique = {item[0]: item for values in selected.values() for item in values}
        yield from sorted(unique.values())


def run(config, mode, attempt, preflight):
    root = Path(config['root'])
    source_ref = file_ref(root / 'inventory.json')
    source = checked_json(source_ref)
    if (source['stage'] != 'inventory_complete' or source['manifest_sha256'] != config['manifest_sha256']
            or source['dataset_counts'] != config['expected_counts']):
        raise ValueError('Matching frozen training inventory required')
    inventory_ref = dict(path=str(root / 'inventory.sqlite'), sha256=source['inventory_sha256'])
    checked_path(inventory_ref)
    manifests = []
    for ref in source['heldout_checks']:
        protocol = manifest_protocol(ref['path'])
        if protocol['scene_disjoint_required'] is not ref['scene_disjoint_required']:
            raise ValueError('Heldout scene protocol differs from the frozen inventory')
        checked_path({k: ref[k] for k in ('path', 'sha256')})
        manifests.append(dict(reference=ref, protocol=protocol))
    if not manifests or len({m['reference']['path'] for m in manifests}) != len(manifests):
        raise ValueError('Nonempty unique heldout manifest scope required')
    code = Path(__file__).parents[1]
    policy = dict(schema_version=1, manifest_sha256=config['manifest_sha256'], source_report=source_ref,
        inventory=inventory_ref, manifests=manifests,
        canonical_hash='SHA256(width x height :RGB: prefix plus contiguous RGB uint8 bytes)',
        input='RGB only; original split and SHA meaning preserved; no native depth or geometry opened',
        sampling='First, last, and low-hash row per dataset/hash-kind/rotation/dimension-schema variant',
        automatic_exclusions=False, evaluation_depth_read=False, pillow=PIL.__version__,
        code_sha256={n: file_ref(code / n)['sha256'] for n in (
            'scripts/audit_heldout_rgb.py', 'src/radio_moge/heldout_rgb.py', 'src/radio_moge/rgb_audit.py',
            'src/radio_moge/nyuv2.py', 'scripts/prepare_scannetpp_laser_gt.py',
            'scripts/prepare_scannetpp_depth_validation.py')})
    policy_hash = hashlib.sha256(canonical_json(policy)).hexdigest()
    if mode == 'census':
        previous = json.loads((root / preflight / 'report.json').read_text())
        if (previous['stage'] != 'heldout_rgb_preflight_complete_not_reviewed'
                or previous['failed_images'] != 0 or previous['policy_sha256'] != policy_hash):
            raise ValueError('Matching successful heldout RGB preflight required')
    output = root / attempt
    output.mkdir(exist_ok=False)
    publish_json(output / 'policy.json', policy)
    temporary = output / 'heldout.sqlite.incoming'
    db = sqlite3.connect(temporary)
    db.execute('PRAGMA cache_size=-16384')
    db.execute('PRAGMA temp_store=FILE')
    db.execute('CREATE TABLE images(id INTEGER PRIMARY KEY,manifest TEXT,line INTEGER,row_sha256 TEXT,'
        'dataset TEXT,scene_group TEXT,image TEXT,original_split TEXT,original_sha256 TEXT,'
        'original_sha256_kind TEXT,encoded_sha256 TEXT,raw_canonical_rgb_sha256 TEXT,'
        'canonical_rgb_sha256 TEXT,width INTEGER,height INTEGER,rotation INTEGER,'
        'scene_disjoint_required INTEGER,error TEXT,UNIQUE(manifest,line))')
    status = dict(stage='reading_heldout_rgb', mode=mode, started_at=time.time(), processed=0,
        failed_images=0, policy_sha256=policy_hash, training_admission=False, optimizer_updates=0,
        evaluation_depth_read=False, automatic_exclusions=0)
    write_json(output / 'status.json', status)
    counts, manifest_counts = Counter(), Counter()
    try:
        for binding in manifests:
            ref, protocol = binding['reference'], binding['protocol']
            for number, row, row_hash in records(ref, protocol, mode == 'preflight'):
                info, error = {}, None
                try:
                    info = inspect_heldout_rgb(row, ref['path'], protocol)
                except Exception as caught:
                    error = type(caught).__name__ + ': ' + str(caught)[:900]
                    status['failed_images'] += 1
                db.execute('INSERT INTO images VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (status['processed'], ref['path'], number, row_hash, info.get('dataset'),
                     info.get('group'), info.get('image'), row.get('split'), row.get('sha256'),
                     info.get('original_sha256_kind'), info.get('encoded_sha256'),
                     info.get('raw_canonical_rgb_sha256'), info.get('canonical_rgb_sha256'),
                     info.get('width'), info.get('height'), info.get('image_rotation_cw'),
                     protocol['scene_disjoint_required'], error))
                status['processed'] += 1
                counts[info.get('dataset', 'unresolved')] += 1
                manifest_counts[ref['path']] += 1
                if status['processed'] % 128 == 0:
                    db.commit()
                    status.update(updated_at=time.time(), dataset_row_counts=dict(counts))
                    write_json(output / 'status.json', status)
                if status['processed'] % 2048 == 0:
                    print(json.dumps(status), flush=True)
        expected = sum(m['reference']['rows'] for m in manifests)
        if mode == 'census' and status['processed'] != expected:
            raise ValueError('Full heldout scope not covered')
        if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('Heldout database integrity check failed')
        unique_paths = db.execute('SELECT COUNT(DISTINCT image) FROM images WHERE error IS NULL').fetchone()[0]
        db.commit()
        db.close()
        checked_path(source_ref)
        checked_path(inventory_ref)
        os.rename(temporary, output / 'heldout.sqlite')
        status.update(stage=f'heldout_rgb_{mode}_complete_not_reviewed', completed_at=time.time(),
            expected_full_manifest_rows=expected, dataset_row_counts=dict(counts),
            manifest_row_counts=dict(manifest_counts), distinct_verified_paths=unique_paths,
            source_report=source_ref, inventory=inventory_ref, index=file_ref(output / 'heldout.sqlite'),
            train_comparison_complete=False, exact_pixel_uniqueness_proven=False)
        publish_json(output / 'report.json', status)
        write_json(output / 'status.json', status)
        print(json.dumps({k: status[k] for k in ('stage', 'processed', 'failed_images')}), flush=True)
        return status
    except BaseException as error:
        db.close()
        status.update(stage='failed', error=type(error).__name__, message=str(error)[:1000])
        write_json(output / 'status.json', status)
        raise


def compare(training_report, heldout_report, output):
    """Join all raw/oriented exact hashes in a separate disk-backed work index."""
    refs = [file_ref(p) for p in (training_report, heldout_report)]
    train, heldout = [checked_json(ref) for ref in refs]
    if (train['stage'] != 'rgb_census_complete_not_reviewed'
            or heldout['stage'] != 'heldout_rgb_census_complete_not_reviewed'
            or train['failed_images'] or heldout['failed_images']
            or train['source_report'] != heldout['source_report']
            or train['inventory'] != heldout['inventory']):
        raise ValueError('Complete, error-free censuses of the identical inventory required')
    source = checked_json(train['source_report'])
    if (train['processed'] != source['images'] or train['source_counts'] != source['dataset_counts']
            or heldout['processed'] != sum(m['rows'] for m in source['heldout_checks'])
            or heldout['manifest_row_counts'] != {m['path']: m['rows'] for m in source['heldout_checks']}):
        raise ValueError('Censuses do not cover the full train and heldout scope')
    indexes = [checked_path(report['index']) for report in (train, heldout)]
    output = Path(output)
    output.mkdir(exist_ok=False)
    temporary = output / 'comparison.sqlite.incoming'
    db = sqlite3.connect(temporary)
    db.execute('PRAGMA cache_size=-16384')
    db.execute('PRAGMA temp_store=FILE')
    db.execute('CREATE TABLE training_hashes(checksum TEXT,ordinal INTEGER,kind TEXT,'
               'PRIMARY KEY(checksum,ordinal,kind)) WITHOUT ROWID')
    db.execute('CREATE TABLE training_groups(scene_group TEXT PRIMARY KEY) WITHOUT ROWID')
    db.execute('CREATE TABLE overlaps(train_ordinal INTEGER,heldout_id INTEGER,'
               'PRIMARY KEY(train_ordinal,heldout_id)) WITHOUT ROWID')
    db.execute('CREATE TABLE group_overlaps(heldout_id INTEGER PRIMARY KEY,required INTEGER)')
    status = dict(stage='indexing_train_hashes', started_at=time.time(), training_rows=0, heldout_rows=0,
        training_admission=False, optimizer_updates=0, evaluation_depth_read=False, automatic_exclusions=0)
    write_json(output / 'status.json', status)
    try:
        with readonly_db(indexes[0]) as incoming:
            for ordinal, group, raw, oriented, error in incoming.execute('SELECT ordinal,scene_group,'
                    'raw_canonical_rgb_sha256,canonical_rgb_sha256,error FROM images ORDER BY ordinal'):
                if error or not raw or not oriented:
                    raise ValueError('Training index contains failed or missing hashes')
                db.executemany('INSERT INTO training_hashes VALUES (?,?,?)',
                    [(raw, ordinal, 'raw'), (oriented, ordinal, 'oriented')])
                db.execute('INSERT OR IGNORE INTO training_groups VALUES (?)', (group,))
                status['training_rows'] += 1
                if status['training_rows'] % 10000 == 0:
                    db.commit()
                    write_json(output / 'status.json', status)
        db.commit()
        status['stage'] = 'comparing_heldout_hashes'
        counts = defaultdict(Counter)
        with readonly_db(indexes[1]) as incoming:
            for value in incoming.execute('SELECT id,manifest,raw_canonical_rgb_sha256,'
                    'canonical_rgb_sha256,scene_group,scene_disjoint_required,error FROM images ORDER BY id'):
                row_id, manifest, raw, oriented, group, required, error = value
                if error or not raw or not oriented:
                    raise ValueError('Heldout index contains failed or missing hashes')
                matches = db.execute('SELECT DISTINCT ordinal FROM training_hashes WHERE checksum IN (?,?)',
                                     (raw, oriented)).fetchall()
                db.executemany('INSERT INTO overlaps VALUES (?,?)', [(n, row_id) for n, in matches])
                shared = db.execute('SELECT 1 FROM training_groups WHERE scene_group=?', (group,)).fetchone()
                if shared:
                    db.execute('INSERT INTO group_overlaps VALUES (?,?)', (row_id, required))
                counts[manifest].update(rows=1, exact_pixel_overlap_rows=int(bool(matches)),
                    exact_pixel_pairs=len(matches), shared_scene_rows=int(bool(shared)),
                    forbidden_shared_scene_rows=int(bool(shared) and required))
                status['heldout_rows'] += 1
                if status['heldout_rows'] % 2048 == 0:
                    db.commit()
                    write_json(output / 'status.json', status)
        if status['training_rows'] != train['processed'] or status['heldout_rows'] != heldout['processed']:
            raise ValueError('Comparison did not read both complete indexes')
        db.commit()
        if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('Comparison database integrity check failed')
        db.close()
        for ref in refs + [train['index'], heldout['index'], train['source_report']]:
            checked_path(ref)
        os.rename(temporary, output / 'comparison.sqlite')
        status.update(stage='heldout_exact_comparison_complete_not_reviewed', completed_at=time.time(),
            reports=refs, index=file_ref(output / 'comparison.sqlite'), per_manifest={k: dict(v) for k, v in counts.items()},
            comparison_code=file_ref(Path(__file__)),
            exact_pixel_overlap_rows=sum(v['exact_pixel_overlap_rows'] for v in counts.values()),
            forbidden_shared_scene_rows=sum(v['forbidden_shared_scene_rows'] for v in counts.values()),
            near_duplicates_checked=False, independent_scene_count_proven=False,
            counts_include_repeated_heldout_memberships=True)
        publish_json(output / 'report.json', status)
        write_json(output / 'status.json', status)
        return status
    except BaseException as error:
        db.close()
        status.update(stage='failed', error=type(error).__name__, message=str(error)[:1000])
        write_json(output / 'status.json', status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'census', 'compare'))
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--preflight', default='heldout-rgb-preflight-01')
    parser.add_argument('--training-report', type=Path)
    parser.add_argument('--heldout-report', type=Path)
    args = parser.parse_args()
    if any(Path(n).name != n for n in (args.attempt, args.preflight)):
        parser.error('Use a new attempt basename')
    config = json.loads(args.config.read_text())
    if args.mode == 'compare':
        if not args.training_report or not args.heldout_report:
            parser.error('Both complete census reports are required')
        compare(args.training_report, args.heldout_report, Path(config['root']) / args.attempt)
    else:
        run(config, args.mode, args.attempt, args.preflight)
