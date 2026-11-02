"""Create a disk-backed, immutable full-corpus inventory. Never starts training.

Only manifest metadata is read here: file existence and pixel/depth correctness
remain separate gates. SQLite keeps a million-row inventory off workspace RAM.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

from indoor_download import write_json


def digest(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def image_path(manifest, row):
    return os.path.abspath(manifest.parent / row['image'])


def task_group(row):
    dataset = row['dataset']
    if dataset == 'arkitscenes':
        return str(row['video_id'])
    if dataset in ('hypersim', 'scannetpp_dslr'):
        return row['group']
    if dataset == 'tartanair':
        return row['source_url'].rsplit('/', 1)[0]
    if dataset in ('structured3d', 'irs'):
        return row['source_archive']
    return dataset


def heldout_manifests(data):
    original = data / 'attention-full-mix-v1-20261022'
    files = sorted(original.glob('val*.jsonl')) + sorted(original.glob('test*.jsonl'))
    if not files:
        raise FileNotFoundError('Original heldout manifests missing')
    files += [data / 'attention-scannetpp-snapshot-v5-20261101/val-scannetpp.jsonl',
              data / 'nyuv2-internal-val-s43/val.jsonl',
              data / 'scannetpp-laser-gt-pilot-v1-20261101/val/val.jsonl',
              data / 'scannetpp-iphone-lidar-val-v1-20261031/val-scannetpp-lidar.jsonl']
    if any(not path.is_file() for path in files):
        raise FileNotFoundError('A required heldout manifest is missing')
    return list(dict.fromkeys(files))


def create_index(config):
    data, root = Path(config['dataset_root']), Path(config['root'])
    manifest = data / config['manifest']
    root.mkdir(parents=True, exist_ok=True)
    final, temporary = root / 'inventory.sqlite', root / 'inventory.sqlite.incoming'
    if any(path.exists() for path in (final, temporary, root / 'inventory.json', root / 'plan.json')):
        raise FileExistsError('Inventory exists; inspect, do not implicitly restart/overwrite')
    write_json(root / 'authorization.json', config['authorization'])
    write_json(root / 'plan.json', config)
    status = dict(stage='indexing', started_at=time.time(), rows=0,
                  training_started=False, metadata_only=True, train_gate='CLOSED')
    write_json(root / 'inventory-status.json', status)
    connection = sqlite3.connect(temporary)
    connection.execute('PRAGMA cache_size=-32768')
    connection.execute('PRAGMA temp_store=FILE')
    connection.execute('''CREATE TABLE images (
        ordinal INTEGER PRIMARY KEY, dataset TEXT NOT NULL, scene_group TEXT NOT NULL,
        task_group TEXT NOT NULL, image TEXT NOT NULL UNIQUE, sha256 TEXT NOT NULL,
        cache_key TEXT NOT NULL UNIQUE, row_json TEXT NOT NULL)''')
    counts, groups, checksum = Counter(), {}, hashlib.sha256()
    try:
        with manifest.open('rb') as file:
            for line in file:
                checksum.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                dataset = row['dataset']
                if dataset not in config['expected_counts'] or row.get('split') != 'train':
                    raise ValueError('Unexpected dataset or non-training row')
                if not row.get('group') or len(row.get('sha256', '')) != 64:
                    raise ValueError('Missing scene identity or image checksum')
                if dataset == 'irs' and row.get('official_split') != 'train':
                    raise ValueError('IRS official test frame in optimizer corpus')
                if dataset == 'scannetpp_dslr':
                    scene = row['scene']
                    heldout = int(hashlib.sha256(('scannetpp-internal-val-v1:' + scene)
                                  .encode()).hexdigest()[:8], 16) % 20 == 0
                    if heldout or row.get('official_split') != 'nvs_sem_train':
                        raise ValueError('Ineligible ScanNet++ scene')
                path = image_path(manifest, row)
                key = hashlib.sha256(json.dumps([dataset, path, row['sha256'],
                    row.get('image_rotation_cw', 0)], separators=(',', ':')).encode()).hexdigest()
                ordinal = status['rows']
                connection.execute('INSERT INTO images VALUES (?,?,?,?,?,?,?,?)', (
                    ordinal, dataset, row['group'], task_group(row), path, row['sha256'], key,
                    json.dumps(row, separators=(',', ':'))))
                counts[dataset] += 1
                groups.setdefault(dataset, set()).add(row['group'])
                status['rows'] += 1
                if status['rows'] % 10000 == 0:
                    connection.commit()
                    write_json(root / 'inventory-status.json', status)
        if dict(counts) != config['expected_counts'] or checksum.hexdigest() != config['manifest_sha256']:
            raise ValueError('Authorized manifest identity or counts changed')
        for field in ('scene_group', 'sha256', 'task_group', 'dataset'):
            connection.execute(f'CREATE INDEX by_{field} ON images ({field})')
        connection.commit()
        proofs = []
        for path in heldout_manifests(data):
            frame_audit = path.name == 'test-irs-official.jsonl'
            proof = dict(path=str(path), sha256=digest(path), rows=0, overlap=0,
                         scene_disjoint_required=not frame_audit, shared_scene_rows=0)
            with path.open() as file:
                for line in file:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if frame_audit and (row.get('dataset') != 'irs'
                            or row.get('split') != 'official-test'
                            or row.get('official_split') != 'test'):
                        raise ValueError('Unexpected IRS frame-audit protocol')
                    shared = connection.execute('SELECT 1 FROM images WHERE scene_group=? LIMIT 1',
                                                (row.get('group'),)).fetchone() is not None
                    conflict = connection.execute('SELECT 1 FROM images WHERE image=? OR sha256=? LIMIT 1',
                            (image_path(path, row), row.get('sha256'))).fetchone() is not None
                    if conflict or (shared and not frame_audit):
                        raise ValueError(f'Train/heldout metadata overlap: {path.name}')
                    proof['rows'] += 1
                    proof['shared_scene_rows'] += int(shared)
            proofs.append(proof)
        duplicates = connection.execute('''SELECT SUM(n-1) FROM
            (SELECT COUNT(*) n FROM images GROUP BY sha256 HAVING n>1)''').fetchone()[0] or 0
        tasks = [dict(dataset=a, group=b, images=n) for a, b, n in connection.execute(
            'SELECT dataset, task_group, COUNT(*) FROM images GROUP BY dataset,task_group ORDER BY 1,2')]
        result = dict(stage='inventory_complete', metadata_only=True, training_started=False,
                      train_gate='CLOSED', manifest=str(manifest), manifest_sha256=checksum.hexdigest(),
                      images=status['rows'], dataset_counts=dict(counts),
                      independent_group_counts={k: len(v) for k, v in groups.items()},
                      duplicate_encoded_image_rows=duplicates, heldout_checks=proofs,
                      source_groups=tasks, pixel_and_depth_checks='NOT_YET_PERFORMED')
        connection.close()
        os.replace(temporary, final)
        result['inventory_sha256'] = digest(final)
        write_json(root / 'inventory.json', result)
        status.update(stage='complete', completed_at=time.time())
        write_json(root / 'inventory-status.json', status)
        print(json.dumps({k: result[k] for k in ('stage', 'images', 'dataset_counts',
                                                'independent_group_counts', 'train_gate')}), flush=True)
        return result
    except BaseException as error:
        connection.close()
        status.update(stage='failed', error=type(error).__name__, message=str(error))
        write_json(root / 'inventory-status.json', status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    create_index(json.loads(args.config.read_text()))
