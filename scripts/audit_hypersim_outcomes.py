"""Account for every Hypersim native outcome after the producer's full readback.

This binds receipt metadata and the producer report, not an independent second
array readback or quality approval. Flagged and rejected rows remain unresolved.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

from radio_moge.corpus_admission import checked_json, checked_path, file_ref, readonly_db
from radio_moge.geometry_cache import canonical_json, publish_json


def outcome_kind(metadata):
    stats = metadata['source_quality']
    flagged = stats['inconsistent_supported_pixels'] > .01 * stats['supported_pixels']
    if metadata.get('source_needs_review') is not flagged:
        raise ValueError('Source-review flag differs from its frozen numeric policy')
    return 'flagged_cache' if flagged else 'unflagged_cache'


def bind_outcome(db, row, kind, reason, evidence, entry_key=None):
    found = db.execute('SELECT ordinal,row_json,outcome FROM images WHERE image=?', (row['image'],)).fetchone()
    if found is None:
        raise ValueError('Native outcome lies outside the frozen Hypersim scope')
    ordinal, original, previous = found
    candidate = dict(row)
    if (candidate.pop('corpus_ordinal', ordinal) != ordinal or candidate != json.loads(original)
            or previous is not None):
        raise ValueError('Changed or duplicate frozen native outcome')
    db.execute('UPDATE images SET outcome=?,reason=?,evidence_json=?,entry_key=? WHERE ordinal=?',
        (kind, reason, canonical_json(evidence).decode(), entry_key, ordinal))


def run(config, native_attempt, attempt):
    root = Path(config['root'])
    source_ref = file_ref(root / 'inventory.json')
    source = checked_json(source_ref)
    inventory_ref = dict(path=str(root / 'inventory.sqlite'), sha256=source['inventory_sha256'])
    checked_path(inventory_ref)
    native = root / native_attempt
    generation_ref, packs_ref, rejection_ref = [file_ref(native / p)
        for p in ('report.json', 'packs/packs.json', 'rejections.json')]
    generation, packs, rejections = [checked_json(r) for r in (generation_ref, packs_ref, rejection_ref)]
    policy = checked_json(file_ref(native / 'packs/policy.json'))
    expected = config['expected_counts']['hypersim']
    if (source['stage'] != 'inventory_complete' or source['manifest_sha256'] != config['manifest_sha256']
            or source['dataset_counts'] != config['expected_counts']
            or generation['stage'] != 'complete' or generation['mode'] != 'generate'
            or generation['training_admission'] is not False or generation['optimizer_updates'] != 0
            or generation['training_started'] is not False
            or generation['completed'] != generation['verified_cache_members']
            or generation['completed'] + generation['rejected'] != expected
            or generation['expected_scope'] != expected or packs['count'] != generation['completed']
            or generation['rejected'] != len(rejections)
            or generation['policy_sha256'] != packs['policy_sha256']
            or hashlib.sha256(canonical_json(policy)).hexdigest() != packs['policy_sha256']
            or policy['manifest_sha256'] != config['manifest_sha256'] or policy['dataset'] != 'hypersim'):
        raise ValueError('Complete matching native generation and producer readback required')
    output = root / attempt
    output.mkdir(exist_ok=False)
    temporary = output / 'outcomes.sqlite.incoming'
    db = sqlite3.connect(temporary)
    db.execute('PRAGMA cache_size=-16384')
    db.execute('CREATE TABLE images(ordinal INTEGER PRIMARY KEY,image TEXT UNIQUE,sha256 TEXT,'
        'scene_group TEXT,row_json TEXT,outcome TEXT,reason TEXT,evidence_json TEXT,entry_key TEXT)')
    try:
        with readonly_db(inventory_ref['path']) as inventory:
            for ordinal, image, checksum, group, encoded in inventory.execute(
                    "SELECT ordinal,image,sha256,scene_group,row_json FROM images WHERE dataset='hypersim' ORDER BY ordinal"):
                row = json.loads(encoded)
                row['image'] = image
                if row['dataset'] != 'hypersim' or row['split'] != 'train' or row['sha256'] != checksum:
                    raise ValueError('Frozen native scope identity differs')
                db.execute('INSERT INTO images VALUES (?,?,?,?,?,NULL,NULL,NULL,NULL)',
                           (ordinal, image, checksum, group, canonical_json(row).decode()))
        if db.execute('SELECT COUNT(*) FROM images').fetchone()[0] != expected:
            raise ValueError('Frozen native scope count differs')
        flags, rejected_flags, packed = 0, 0, 0
        for packet in packs['packs']:
            receipt_path = native / 'packs' / packet['receipt']
            if receipt_path.parent != native / 'packs':
                raise ValueError('Invalid native receipt path')
            receipt_ref = dict(path=str(receipt_path), sha256=packet['sha256'])
            receipt = checked_json(receipt_ref)
            if receipt['policy_sha256'] != packs['policy_sha256'] or len(receipt['entries']) != packet['count']:
                raise ValueError('Native receipt policy or count differs')
            for entry in receipt['entries']:
                meta, row = entry['geometry_metadata'], entry['image']
                if (meta['dataset'] != 'hypersim' or meta['image_sha256'] != row['sha256']
                        or meta['policy_sha256'] != packs['policy_sha256']):
                    raise ValueError('Native image/policy binding differs')
                kind = outcome_kind(meta)
                bind_outcome(db, row, kind, 'world_ray_mismatch_over_one_percent' if kind == 'flagged_cache'
                             else 'candidate_not_source_flagged', receipt_ref, entry['key'])
                flags += int(kind == 'flagged_cache')
                packed += 1
            db.commit()
        for rejection in rejections:
            reason = rejection['reason']
            if reason not in ('census_quarantined_native_pose', 'insufficient_world_verified_support'):
                raise ValueError('Unreviewed native rejection reason')
            stats = rejection.get('stats')
            if stats is not None:
                rejected_flags += int(stats['inconsistent_supported_pixels'] > .01 * stats['supported_pixels'])
            bind_outcome(db, rejection['image'], 'rejected', reason, rejection_ref)
        if (packed != packs['count'] or flags + rejected_flags != generation['source_review_images']
                or db.execute('SELECT COUNT(*) FROM images WHERE outcome IS NULL').fetchone()[0]):
            raise ValueError('Native outcomes or source-review counters do not reconcile')
        counts = dict(db.execute('SELECT outcome,COUNT(*) FROM images GROUP BY outcome'))
        reasons = dict(db.execute("SELECT reason,COUNT(*) FROM images WHERE outcome='rejected' GROUP BY reason"))
        scenes = {}
        for group, outcome, n in db.execute('SELECT scene_group,outcome,COUNT(*) FROM images GROUP BY 1,2'):
            scenes.setdefault(group, {})[outcome] = n
        review_counts = Counter()
        pending_path = output / 'unresolved-native.jsonl.incoming'
        with pending_path.open('x') as file:
            for ordinal, image, checksum, group, kind, reason, evidence, key in db.execute(
                    "SELECT ordinal,image,sha256,scene_group,outcome,reason,evidence_json,entry_key FROM images "
                    "WHERE outcome!='unflagged_cache' ORDER BY ordinal"):
                value = dict(corpus_ordinal=ordinal, image=image, sha256=checksum, dataset='hypersim', group=group,
                    native_outcome=kind, native_reason=reason, evidence=json.loads(evidence), entry_key=key)
                file.write(canonical_json(value).decode() + '\n')
                review_counts[group] += 1
        os.rename(pending_path, output / 'unresolved-native.jsonl')
        db.commit()
        if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('Outcome database integrity failure')
        db.close()
        for ref in (source_ref, generation_ref, packs_ref, rejection_ref):
            checked_path(ref)
        os.rename(temporary, output / 'outcomes.sqlite')
        publish_json(output / 'scenes.json', scenes)
        report = dict(stage='native_outcomes_accounted_not_accepted', dataset='hypersim', completed_at=time.time(),
            manifest_sha256=config['manifest_sha256'], expected_scope=expected, outcome_counts=counts,
            rejection_reasons=reasons, cached_source_flags=flags, rejected_source_flags=rejected_flags,
            source_review_counter=flags + rejected_flags, scenes=len(scenes), unresolved_scene_counts=dict(review_counts),
            unresolved_images=sum(review_counts.values()), source_report=source_ref, inventory=inventory_ref,
            generation_report=generation_ref, packs=packs_ref, rejections=rejection_ref,
            policy_sha256=packs['policy_sha256'], index=file_ref(output / 'outcomes.sqlite'),
            unresolved_manifest=file_ref(output / 'unresolved-native.jsonl'), scene_counts=file_ref(output / 'scenes.json'),
            producer_array_readback_complete=True, independent_array_readback_performed=False,
            receipt_metadata_only=True, source_quality_accepted=False, training_admission=False, optimizer_updates=0,
            audit_code=file_ref(Path(__file__)))
        publish_json(output / 'report.json', report)
        print(json.dumps({k: report[k] for k in ('stage', 'outcome_counts', 'unresolved_images', 'scenes')}), flush=True)
        return report
    except BaseException:
        db.close()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--native-attempt', default='hypersim-native-cache-02')
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if any(Path(n).name != n for n in (args.attempt, args.native_attempt)):
        parser.error('Use an attempt basename')
    run(json.loads(args.config.read_text()), args.native_attempt, args.attempt)
