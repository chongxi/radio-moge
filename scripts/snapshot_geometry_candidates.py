"""Freeze published geometry packs for a proposed first training stage.

Producers may continue writing new packs. This freezes an explicit receipt list,
checks all receipt RGB identities against the original inventory, and keeps
native flags, later quality flags and exclusions. It does NOT verify payloads,
accept a source, change the full-corpus gate, or authorize optimization.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

from indoor_download import write_json
from radio_moge.corpus_admission import SOURCE_CHECKS, checked_json, checked_path, file_ref, readonly_db
from radio_moge.geometry_cache import canonical_json, publish_json


def digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def capture_sources(root, sources, manifest, cutoff_ns):
    """Only filenames visible at the cutoff enter the immutable candidate list."""
    result = []
    for spec in sources:
        name = spec['attempt']
        if Path(name).name != name:
            raise ValueError('Use an attempt basename')
        directory = root / name / 'packs'
        policy_ref = file_ref(directory / 'policy.json')
        policy = checked_json(policy_ref)
        if policy['manifest_sha256'] != manifest or policy['dataset'] != spec['dataset']:
            raise ValueError('Candidate source policy differs from the frozen inventory')
        paths = sorted(p for p in directory.glob('pack-*.json') if p.stat().st_mtime_ns <= cutoff_ns)
        if not paths or any(p.name != f'pack-{i:06d}.json' for i, p in enumerate(paths)):
            raise ValueError('Published receipt sequence is empty or has a gap')
        status = json.loads((root / name / 'status.json').read_text())
        if status.get('optimizer_updates') != 0 or status.get('training_started') is not False:
            raise ValueError('Only offline label producers are supported')
        if status['policy_sha256'] != digest(policy):
            raise ValueError('Producer and published policy differ')
        result.append(dict(spec, policy=policy_ref, policy_sha256=digest(policy),
            producer_status_at_capture=status, paths=[str(p) for p in paths]))
    return result


def prepare_db(path):
    db = sqlite3.connect(path)
    db.execute('PRAGMA cache_size=-32768')
    db.execute('PRAGMA temp_store=FILE')
    db.executescript('''
        CREATE TABLE images(sha256 TEXT PRIMARY KEY,image TEXT UNIQUE NOT NULL,
            dataset TEXT NOT NULL,scene_group TEXT NOT NULL,row_sha256 TEXT NOT NULL,
            claimed_ordinal INTEGER,ordinal INTEGER UNIQUE);
        CREATE TABLE packs(id INTEGER PRIMARY KEY,source TEXT NOT NULL,
            receipt_json TEXT NOT NULL,tar_json TEXT NOT NULL,entries INTEGER NOT NULL);
        CREATE TABLE candidates(policy_sha256 TEXT,entry_key TEXT,image_sha256 TEXT,
            pack_id INTEGER,native_flag INTEGER,external_flag INTEGER,excluded INTEGER,
            reviewed_source INTEGER,label_kind TEXT,member_sha256 TEXT,
            PRIMARY KEY(policy_sha256,entry_key));
    ''')
    return db


def add_entry(db, entry, source, pack_id, external, exclusions):
    row, meta = dict(entry['image']), entry['geometry_metadata']
    ordinal = row.pop('corpus_ordinal', None)
    policy, dataset = source['policy_sha256'], source['dataset']
    if (row['split'] != 'train' or row['dataset'] != dataset or meta['dataset'] != dataset
            or meta['policy_sha256'] != policy or meta['image_sha256'] != row['sha256']
            or digest(meta) != entry['key'] or meta['label_kind'] not in source['label_kinds']
            or (ordinal is not None and (type(ordinal) is not int or ordinal < 0))):
        raise ValueError('Candidate RGB or geometry provenance changed')
    if meta['label_kind'] == 'teacher_pseudo' and meta['metric_calibrated'] is not False:
        raise ValueError('Pseudo labels cannot become calibrated GT')
    row_hash = digest(row)
    old = db.execute('SELECT image,dataset,scene_group,row_sha256,claimed_ordinal FROM images '
                     'WHERE sha256=?', (row['sha256'],)).fetchone()
    identity = (row['image'], dataset, row['group'], row_hash)
    if old:
        if old[:4] != identity or (old[4] is not None and ordinal is not None and old[4] != ordinal):
            raise ValueError('Alternative candidates changed their original RGB identity')
        if old[4] is None and ordinal is not None:
            db.execute('UPDATE images SET claimed_ordinal=? WHERE sha256=?', (ordinal, row['sha256']))
    else:
        db.execute('INSERT INTO images VALUES (?,?,?,?,?,?,NULL)',
                   (row['sha256'], *identity, ordinal))
    flag = external.pop(entry['key'], None)
    if flag is not None and (flag['member_sha256'] != entry['sha256']
            or any(flag[k] != row[k] for k in ('image', 'sha256', 'group'))
            or (ordinal is not None and flag['corpus_ordinal'] != ordinal)):
        raise ValueError('External quality flag does not identify this candidate')
    if flag is not None and ordinal is None:
        if old and old[4] is not None and old[4] != flag['corpus_ordinal']:
            raise ValueError('External quality flag changed the frozen ordinal')
        db.execute('UPDATE images SET claimed_ordinal=? WHERE sha256=?',
                   (flag['corpus_ordinal'], row['sha256']))
    decision = exclusions.get(row['sha256'])
    if decision is not None and (decision['image'] != row['image'] or decision['dataset'] != dataset):
        raise ValueError('Geometry exclusion changed its RGB identity')
    db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?)',
        (policy, entry['key'], row['sha256'], pack_id, bool(meta.get('source_needs_review', False)),
         flag is not None, decision is not None, bool(source.get('source_review')),
         meta['label_kind'], entry['sha256']))


def reconcile_inventory(db, inventory):
    matched = 0
    with readonly_db(inventory) as original:
        for ordinal, image, checksum, dataset, group, encoded in original.execute(
                'SELECT ordinal,image,sha256,dataset,scene_group,row_json FROM images ORDER BY ordinal'):
            found = db.execute('SELECT image,dataset,scene_group,row_sha256,claimed_ordinal '
                               'FROM images WHERE sha256=?', (checksum,)).fetchone()
            if found is None:
                continue
            row = dict(json.loads(encoded), image=image)
            if (found[:4] != (image, dataset, group, digest(row))
                    or (found[4] is not None and found[4] != ordinal)):
                raise ValueError('Candidate differs from the original frozen inventory')
            db.execute('UPDATE images SET ordinal=? WHERE sha256=?', (ordinal, checksum))
            matched += 1
    if db.execute('SELECT COUNT(*) FROM images WHERE ordinal IS NULL').fetchone()[0]:
        raise ValueError('Snapshot includes RGB outside the original inventory')
    db.commit()
    return matched


def run(config, plan, output):
    root = Path(config['root'])
    inventory_report_ref = file_ref(root / 'inventory.json')
    inventory_report = checked_json(inventory_report_ref)
    inventory_ref = dict(path=str(root / 'inventory.sqlite'), sha256=inventory_report['inventory_sha256'])
    if (inventory_report['stage'] != 'inventory_complete'
            or inventory_report['manifest_sha256'] != config['manifest_sha256']
            or inventory_report['dataset_counts'] != config['expected_counts']):
        raise ValueError('Original full inventory identity required')
    cutoff_ns = time.time_ns()
    sources = capture_sources(root, plan['sources'], config['manifest_sha256'], cutoff_ns)
    output.mkdir(exist_ok=False)
    publish_json(output / 'capture.json', dict(cutoff_ns=cutoff_ns, sources=sources,
        parent_inventory=inventory_ref, parent_inventory_report=inventory_report_ref,
        training_admission=False, optimizer_updates=0))
    publish_json(output / 'plan.json', plan)
    state = dict(stage='checking_receipt_metadata', processed_entries=0, processed_packs=0,
        started_at=time.time(), training_admission=False, optimizer_updates=0)
    write_json(output / 'status.json', state)
    exclusions_ref = plan['exclusions']
    exclusions = {}
    for line in checked_path(exclusions_ref).read_text().splitlines():
        value = json.loads(line)
        if value['decision'] != 'exclude' or value['sha256'] in exclusions or not value['evidence']:
            raise ValueError('Explicit unique geometry exclusions are required')
        exclusions[value['sha256']] = value
    with tempfile.TemporaryDirectory(prefix='geometry-stage1-') as temporary:
        db = prepare_db(Path(temporary) / 'candidates.sqlite')
        try:
            checked_path(inventory_ref)
            for source in sources:
                policy = checked_json(source['policy'])
                source['label_kinds'] = [policy['label_kind']] if 'label_kind' in policy else []
                external = {}
                if source.get('source_review'):
                    review = checked_json(source['source_review'])
                    if (review['stage'] != 'source_review_complete'
                            or review.get('decision') != 'eligible_with_flagged_rows_quarantined'
                            or any(review.get('checks', {}).get(k) is not True for k in SOURCE_CHECKS)
                            or review['dataset'] != source['dataset']
                            or review['policy_sha256'] != source['policy_sha256']
                            or review['manifest_sha256'] != config['manifest_sha256']):
                        raise ValueError('Previously completed source review has changed scope')
                    register = checked_json(review['candidate_quarantine'])
                    if (register['policy_sha256'] != source['policy_sha256']
                            or register['manifest_sha256'] != config['manifest_sha256']):
                        raise ValueError('Quality register changed scope')
                    external = {f['entry_key']: f for f in register['flags']}
                    if len(external) != len(register['flags']):
                        raise ValueError('Repeated external quality flag')
                for path in source['paths']:
                    if Path(path).stat().st_mtime_ns > cutoff_ns:
                        raise ValueError('A captured receipt changed after the cutoff')
                    receipt_ref = file_ref(path)
                    receipt = checked_json(receipt_ref)
                    tar = Path(path).parent / receipt['tar']
                    if (tar.parent != Path(path).parent or tar.stat().st_size != receipt['bytes']
                            or receipt['policy_sha256'] != source['policy_sha256']
                            or receipt['stage'] != 'cache_written_not_accepted'
                            or receipt['training_admission'] is not False):
                        raise ValueError('Published pack receipt identity differs')
                    tar_ref = dict(path=str(tar), sha256=receipt['sha256'], bytes=receipt['bytes'],
                                   payload_rehashed=False)
                    cursor = db.execute('INSERT INTO packs(source,receipt_json,tar_json,entries) '
                        'VALUES (?,?,?,?)', (source['attempt'], canonical_json(receipt_ref).decode(),
                            canonical_json(tar_ref).decode(), len(receipt['entries'])))
                    for entry in receipt['entries']:
                        add_entry(db, entry, source, cursor.lastrowid, external, exclusions)
                    state['processed_entries'] += len(receipt['entries'])
                    state['processed_packs'] += 1
                    if state['processed_packs'] % 32 == 0:
                        db.commit()
                        state['updated_at'] = time.time()
                        write_json(output / 'status.json', state)
                if external:
                    raise ValueError('A reviewed source snapshot omitted quarantined candidates')
                checked_path(source['policy'])
            state.update(stage='matching_original_inventory', updated_at=time.time())
            write_json(output / 'status.json', state)
            unique = reconcile_inventory(db, inventory_ref['path'])
            for checksum, decision in exclusions.items():
                found = db.execute('SELECT ordinal,scene_group FROM images WHERE sha256=?',
                                   (checksum,)).fetchone()
                if found and (found[0] != decision['corpus_ordinal']
                        or ('group' in decision and found[1] != decision['group'])):
                    raise ValueError('Geometry exclusion changed its frozen ordinal or group')
            known = db.execute('SELECT COUNT(DISTINCT image_sha256) FROM candidates WHERE excluded=1').fetchone()[0]
            reviewed = db.execute('SELECT COUNT(DISTINCT image_sha256) FROM candidates '
                'WHERE native_flag=0 AND external_flag=0 AND excluded=0 AND reviewed_source=1').fetchone()[0]
            counts = dict(db.execute('SELECT dataset,COUNT(*) FROM images GROUP BY dataset'))
            candidate_counts = Counter()
            for label, n in db.execute('SELECT label_kind,COUNT(*) FROM candidates GROUP BY label_kind'):
                candidate_counts[label] = n
            if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('Snapshot metadata index integrity failed')
            db.close()
            checked_path(inventory_ref)
            checked_path(inventory_report_ref)
            checked_path(exclusions_ref)
            incoming = output / 'candidates.sqlite.incoming'
            shutil.copyfile(Path(temporary) / 'candidates.sqlite', incoming)
            os.rename(incoming, output / 'candidates.sqlite')
            state.update(stage='candidate_snapshot_metadata_complete_not_accepted',
                completed_at=time.time(), unique_rgb=unique, dataset_counts=counts,
                candidate_label_counts=dict(candidate_counts), known_excluded_rgb_with_candidates=known,
                previously_source_reviewed_rgb=reviewed, parent_inventory=inventory_ref,
                parent_inventory_report=inventory_report_ref, exclusions=exclusions_ref,
                index=file_ref(output / 'candidates.sqlite'), capture=file_ref(output / 'capture.json'),
                code=file_ref(Path(__file__)), pack_payloads_verified=False,
                model_preflight_complete=False, gpu_used=False,
                remaining=['select exact eligible candidates and source reviews for this snapshot',
                    'verify complete selected pack payloads', 'canonical heldout isolation',
                    'source/scene/crop coverage and loss validation',
                    'frozen configuration and actual zero-update model probe'])
            publish_json(output / 'report.json', state)
            write_json(output / 'status.json', state)
            return state
        except BaseException as error:
            db.close()
            state.update(stage='failed', error=type(error).__name__, message=str(error))
            write_json(output / 'status.json', state)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = run(json.loads(args.config.read_text()), json.loads(args.plan.read_text()), args.output)
    print(json.dumps({k: result[k] for k in ('stage', 'unique_rgb', 'previously_source_reviewed_rgb')}))
