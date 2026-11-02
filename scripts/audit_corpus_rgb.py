"""Canonical RGB census and bounded near-duplicate diagnostics; never admits training."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time

import PIL
from PIL import Image, ImageDraw, ImageOps

from indoor_download import write_json
from radio_moge.corpus_admission import checked_path, file_ref, readonly_db
from radio_moge.geometry_cache import canonical_json, publish_json
from radio_moge.rgb_audit import choose_groups, choose_rows, inspect_rgb, pair_diagnostics


def inventory_rows(db):
    for ordinal, image, encoded in db.execute('SELECT ordinal,image,row_json FROM images ORDER BY ordinal'):
        row = json.loads(encoded)
        row.update(image=image, corpus_ordinal=ordinal)
        yield row


def probe_rows(db, groups):
    result = []
    for group in choose_groups(groups):
        values = []
        for ordinal, image, encoded in db.execute('SELECT ordinal,image,row_json FROM images '
                'WHERE scene_group=? ORDER BY ordinal', (group['group'],)):
            row = json.loads(encoded)
            if row['dataset'] != group['dataset'] or row['group'] != group['group']:
                raise ValueError('Frozen source grouping changed')
            row.update(image=image, corpus_ordinal=ordinal)
            values.append(row)
        if len(values) != group['images']:
            raise ValueError('Frozen group image count differs')
        result.extend(choose_rows(values))
    return sorted(result, key=lambda r: r['corpus_ordinal'])


def sheet(records, output, columns=4):
    canvas = Image.new('RGB', (columns * 252, math.ceil(len(records) / columns) * 192), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(records):
        image, _ = inspect_rgb(row)
        thumbnail = ImageOps.contain(image, (240, 160), Image.Resampling.BILINEAR)
        x, y = i % columns * 252, i // columns * 192
        canvas.paste(thumbnail, (x, y))
        draw.text((x, y + 161), str(row['corpus_ordinal']) + ' ' + row['dataset'], fill='black')
        draw.text((x, y + 176), row['group'][-39:], fill='black')
        image.close()
    temporary = output.with_suffix('.jpg.incoming')
    canvas.save(temporary, format='JPEG', quality=90)
    os.rename(temporary, output)
    return file_ref(output)


def sample_visuals(records, pairs, output):
    output.mkdir()
    bins, by_id = defaultdict(list), {}
    for row in records:
        bins[row['dataset']].append(row)
        by_id[row['corpus_ordinal']] = row
    evidence = []
    for dataset, rows in sorted(bins.items()):
        groups = {}
        for row in rows:
            groups.setdefault(row['group'], row)
        selected = sorted(groups.values(), key=lambda r: r['sha256'])[:16]
        evidence.append(dict(kind='source_overview', dataset=dataset,
            ordinals=[r['corpus_ordinal'] for r in selected], file=sheet(selected, output / (dataset + '.jpg'))))
    for category, examples in sorted(pairs['examples'].items()):
        for i, pair in enumerate(examples[:3]):
            name = category.replace(':', '-') + f'-{i:02d}.jpg'
            evidence.append(dict(kind='retrieved_pair', category=category, **pair,
                file=sheet([by_id[pair['a']], by_id[pair['b']]], output / name, columns=2)))
    return evidence


def duplicate_census(db, output):
    results = {}
    temporary = output / 'exact-duplicates.jsonl.incoming'
    with temporary.open('x') as file:
        for field in ('raw_canonical_rgb_sha256', 'canonical_rgb_sha256'):
            summary = dict(clusters=0, excess_images=0, cross_dataset_clusters=0, cross_group_clusters=0)
            for checksum, count, datasets, groups in db.execute(f'SELECT {field},COUNT(*),'
                    f'COUNT(DISTINCT dataset),COUNT(DISTINCT scene_group) FROM images '
                    f'WHERE error IS NULL GROUP BY {field} HAVING COUNT(*)>1 ORDER BY 2 DESC,1'):
                summary['clusters'] += 1
                summary['excess_images'] += count - 1
                summary['cross_dataset_clusters'] += datasets > 1
                summary['cross_group_clusters'] += groups > 1
                file.write(json.dumps(dict(hash_kind=field, sha256=checksum, images=count,
                    datasets=datasets, groups=groups)) + '\n')
            results[field] = summary
    os.rename(temporary, output / 'exact-duplicates.jsonl')
    return results


def run(config, mode, attempt, preflight):
    root = Path(config['root'])
    source_path = root / 'inventory.json'
    source_ref = file_ref(source_path)
    source = json.loads(source_path.read_text())
    index_ref = dict(path=str(root / 'inventory.sqlite'), sha256=source['inventory_sha256'])
    if (source['stage'] != 'inventory_complete' or source['manifest_sha256'] != config['manifest_sha256']
            or source['dataset_counts'] != config['expected_counts']
            or source['images'] != sum(config['expected_counts'].values())):
        raise ValueError('Complete matching frozen inventory is required')
    checked_path(index_ref)
    policy = dict(schema_version=1, manifest_sha256=config['manifest_sha256'], inventory=index_ref,
        input='Original encoded bytes; no EXIF transform; only the frozen explicit quarter-turn rotation',
        canonical_hash='SHA256(width x height :RGB: prefix plus contiguous RGB uint8 bytes)',
        stored_hash_verification='NYU raw RGB bytes; IRS/Structured3D/TartanAir size-prefixed RGB bytes',
        descriptor='128 bits: horizontal9x8 + vertical8x9 comparisons; Pillow bilinear grayscale resize',
        retrieval_only=True, no_automatic_exclusions=True, full_census_perceptual_pair_scan=False,
        sample_selection='16 largest/smallest/hash scene groups (COCO128); first/middle/last/hash quartets',
        pillow=PIL.__version__, script_sha256=file_ref(Path(__file__))['sha256'],
        helper_sha256=file_ref(Path(__file__).parents[1] / 'src/radio_moge/rgb_audit.py')['sha256'])
    policy_hash = hashlib.sha256(canonical_json(policy)).hexdigest()
    if mode == 'census':
        previous = json.loads((root / preflight / 'report.json').read_text())
        if (previous['stage'] != 'rgb_preflight_complete_not_reviewed'
                or previous['failed_images'] != 0 or previous['policy_sha256'] != policy_hash):
            raise ValueError('Matching successful RGB identity preflight required')
    output = root / attempt
    output.mkdir(exist_ok=False)
    publish_json(output / 'policy.json', policy)
    temporary = output / 'rgb.sqlite.incoming'
    db = sqlite3.connect(temporary)
    db.execute('PRAGMA cache_size=-16384')
    db.execute('PRAGMA temp_store=FILE')
    db.execute('CREATE TABLE images(ordinal INTEGER PRIMARY KEY,dataset TEXT,scene_group TEXT,'
        'image_sha256 TEXT,raw_canonical_rgb_sha256 TEXT,canonical_rgb_sha256 TEXT,dhash128 TEXT,'
        'width INTEGER,height INTEGER,rotation INTEGER,stored_hash_verified INTEGER,error TEXT)')
    status = dict(stage='reading_rgb', mode=mode, started_at=time.time(), processed=0, failed_images=0,
        policy_sha256=policy_hash, training_admission=False, optimizer_updates=0)
    write_json(output / 'status.json', status)
    counts, examples = Counter(), []
    try:
        with readonly_db(index_ref['path']) as inventory:
            rows = probe_rows(inventory, source['source_groups']) if mode == 'preflight' else inventory_rows(inventory)
            expected = len(rows) if mode == 'preflight' else source['images']
            for row in rows:
                error, info = None, {}
                try:
                    image, info = inspect_rgb(row)
                    image.close()
                except Exception as caught:
                    error = type(caught).__name__ + ': ' + str(caught)[:900]
                    status['failed_images'] += 1
                db.execute('INSERT INTO images VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                    (row['corpus_ordinal'], row['dataset'], row['group'], row['sha256'],
                     info.get('raw_canonical_rgb_sha256'), info.get('canonical_rgb_sha256'),
                     info.get('dhash128'), info.get('width'), info.get('height'),
                     info.get('image_rotation_cw'), info.get('stored_decoded_hash_verified'), error))
                counts[row['dataset']] += 1
                status['processed'] += 1
                if mode == 'preflight':
                    examples.append(dict(row, **info, error=error))
                if status['processed'] % 128 == 0:
                    db.commit()
                    status.update(updated_at=time.time(), source_counts=dict(counts))
                    write_json(output / 'status.json', status)
                if status['processed'] % 2048 == 0:
                    print(json.dumps(status), flush=True)
            if status['processed'] != expected or (mode == 'census' and dict(counts) != source['dataset_counts']):
                raise ValueError('RGB census did not cover the exact selected scope')
        db.commit()
        pairs, visuals = None, []
        if mode == 'preflight':
            publish_json(output / 'samples.json', examples)
            if not status['failed_images']:
                pairs = pair_diagnostics(examples)
                visuals = sample_visuals(examples, pairs, output / 'visuals')
        duplicates = duplicate_census(db, output)
        if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('RGB audit database failed integrity check')
        db.close()
        checked_path(index_ref)
        checked_path(source_ref)
        os.rename(temporary, output / 'rgb.sqlite')
        status.update(stage=('rgb_preflight_complete_not_reviewed' if mode == 'preflight'
                             else 'rgb_census_complete_not_reviewed'), completed_at=time.time(),
            expected_scope=expected, source_counts=dict(counts), inventory=index_ref,
            source_report=source_ref, index=file_ref(output / 'rgb.sqlite'),
            exact_duplicate_summary=duplicates, sampled_pair_diagnostics=pairs, visuals=visuals,
            full_corpus_near_duplicates_checked=False, semantic_content_reviewed=False,
            automatic_exclusions=0, independent_scene_count_proven=False)
        publish_json(output / 'report.json', status)
        write_json(output / 'status.json', status)
        print(json.dumps({k: status[k] for k in ('stage', 'processed', 'failed_images', 'source_counts')}), flush=True)
        return status
    except BaseException as error:
        db.close()
        status.update(stage='failed', error=type(error).__name__, message=str(error)[:1000])
        write_json(output / 'status.json', status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'census'))
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--preflight', default='rgb-identity-preflight-01')
    args = parser.parse_args()
    if any(Path(n).name != n for n in (args.attempt, args.preflight)):
        parser.error('Use a new attempt basename')
    run(json.loads(args.config.read_text()), args.mode, args.attempt, args.preflight)
