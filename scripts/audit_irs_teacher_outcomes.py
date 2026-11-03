"""Full IRS teacher outcome identities and relative-depth receipt statistics.

Requires the producer's complete array readback. This audit reads receipt
metadata, not payload arrays, and never approves, edits or excludes a sample.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time

import numpy as np

from radio_moge.corpus_admission import checked_json, checked_path, file_ref, readonly_db
from radio_moge.geometry_cache import canonical_json, publish_json


def teacher_metrics(entry, policy_hash, revision, dataset='irs', image_shape=None):
    meta, row = entry['geometry_metadata'], entry['image']
    if (meta['label_kind'] != 'teacher_pseudo' or meta['camera_kind'] != 'teacher_estimated'
            or meta['metric_calibrated'] is not False or meta['dataset'] != dataset
            or meta['teacher_revision'] != revision or meta['policy_sha256'] != policy_hash
            or meta['image_sha256'] != row['sha256'] or meta['depth_convention'] != 'camera_z'
            or hashlib.sha256(canonical_json(meta)).hexdigest() != entry['key']):
        raise ValueError('Teacher provenance differs from the pinned candidate policy')
    quantiles = np.asarray(entry['depth_quantiles_m'], dtype=np.float64)
    h, w = meta['shape']
    # Legacy NYU rows have no dimension fields. Their caller must obtain the
    # shape from the hash-checked RGB, never from the geometry being audited.
    if image_shape is None:
        image_shape = (row['height'], row['width'])
    if (('height' in row and row['height'] != image_shape[0])
            or ('width' in row and row['width'] != image_shape[1])):
        raise ValueError('Teacher RGB dimensions differ from the frozen row')
    pixels = meta['valid_pixels']
    if (quantiles.shape != (5,) or not np.isfinite(quantiles).all()
            or not (quantiles > 0).all() or not (np.diff(quantiles) >= 0).all()
            or (h, w) != tuple(image_shape)
            or type(pixels) is not int or not 0 < pixels <= h * w
            or not math.isclose(entry['valid_fraction'], pixels / (h * w), rel_tol=0, abs_tol=1e-12)):
        raise ValueError('Teacher receipt support or quantiles are invalid')
    return (pixels / (h * w), float(quantiles[1] / quantiles[2]),
            float(quantiles[3] / quantiles[2]), float(quantiles[4] / quantiles[2]))


def bind_outcome(db, row, outcome, evidence, entry_key=None, metrics=None):
    found = db.execute('SELECT ordinal,row_json,outcome FROM images WHERE image=?', (row['image'],)).fetchone()
    if found is None:
        raise ValueError('Teacher outcome is outside the frozen IRS scope')
    ordinal, original, previous = found
    candidate = dict(row)
    if (candidate.pop('corpus_ordinal', ordinal) != ordinal or candidate != json.loads(original)
            or previous is not None):
        raise ValueError('Changed or duplicate frozen IRS outcome')
    values = metrics if metrics is not None else (None, None, None, None)
    db.execute('UPDATE images SET outcome=?,evidence_json=?,entry_key=?,support=?,p01_ratio=?,p99_ratio=?,'
               'max_ratio=? WHERE ordinal=?',
               (outcome, canonical_json(evidence).decode(), entry_key, *values, ordinal))


def distribution(db, group=None, eye=None):
    where, args = "outcome='teacher_candidate'", []
    if group is not None:
        where += ' AND scene_group=? AND eye=?'
        args = [group, eye]
    values = np.asarray(db.execute('SELECT support,p01_ratio,p99_ratio,max_ratio FROM images WHERE '
                                  + where + ' ORDER BY ordinal', args).fetchall(), dtype=np.float64)
    if not len(values):
        return dict(cached_images=0)
    probabilities = [0., .01, .5, .9, .99, 1.]
    return dict(cached_images=len(values), percentiles=probabilities,
        full_support_images=int((values[:, 0] == 1).sum()),
        support_percentiles=np.quantile(values[:, 0], probabilities).tolist(),
        p01_over_median_percentiles=np.quantile(values[:, 1], probabilities).tolist(),
        p99_over_median_percentiles=np.quantile(values[:, 2], probabilities).tolist(),
        max_over_median_percentiles=np.quantile(values[:, 3], probabilities).tolist(),
        p99_over_median_above={str(v): int((values[:, 2] > v).sum()) for v in (8, 16, 32, 64)},
        statistics_units='ratios to per-image valid median; no physical metric-scale claim')


def run(config, cache_attempt, attempt):
    root = Path(config['root'])
    source_ref = file_ref(root / 'inventory.json')
    source = checked_json(source_ref)
    inventory_ref = dict(path=str(root / 'inventory.sqlite'), sha256=source['inventory_sha256'])
    checked_path(inventory_ref)
    cache = root / cache_attempt
    generation_ref, packs_ref, rejection_ref, policy_ref = [file_ref(cache / p)
        for p in ('report.json', 'packs/packs.json', 'rejections.json', 'packs/policy.json')]
    generation, packs, rejections, policy = [checked_json(r)
        for r in (generation_ref, packs_ref, rejection_ref, policy_ref)]
    expected = config['expected_counts']['irs']
    policy_hash = hashlib.sha256(canonical_json(policy)).hexdigest()
    if (source['stage'] != 'inventory_complete' or source['manifest_sha256'] != config['manifest_sha256']
            or source['dataset_counts'] != config['expected_counts']
            or generation['stage'] != 'complete' or generation['mode'] != 'generate'
            or generation['training_admission'] is not False or generation['optimizer_updates'] != 0
            or generation['training_started'] is not False or generation['teacher_frozen'] is not True
            or generation['completed'] != generation['verified_cache_members']
            or generation['completed'] + generation['rejected'] != expected
            or generation['expected_scope'] != expected or generation['rejected'] != len(rejections)
            or packs['count'] != generation['completed'] or generation['policy_sha256'] != policy_hash
            or packs['policy_sha256'] != policy_hash or policy['dataset'] != 'irs'
            or policy['manifest_sha256'] != config['manifest_sha256']
            or policy['label_kind'] != 'teacher_pseudo' or policy['native_metric_gt'] is not False
            or policy['teacher'] != config['teacher']):
        raise ValueError('Complete matching teacher producer readback and frozen policy required')
    output = root / attempt
    output.mkdir(exist_ok=False)
    temporary = output / 'outcomes.sqlite.incoming'
    db = sqlite3.connect(temporary)
    db.execute('PRAGMA cache_size=-16384')
    db.execute('PRAGMA temp_store=FILE')
    db.execute('CREATE TABLE images(ordinal INTEGER PRIMARY KEY,image TEXT UNIQUE,sha256 TEXT,'
        'scene_group TEXT,eye TEXT,row_json TEXT,outcome TEXT,evidence_json TEXT,entry_key TEXT UNIQUE,'
        'support REAL,p01_ratio REAL,p99_ratio REAL,max_ratio REAL)')
    try:
        with readonly_db(inventory_ref['path']) as inventory:
            for ordinal, image, checksum, group, encoded in inventory.execute(
                    "SELECT ordinal,image,sha256,scene_group,row_json FROM images WHERE dataset='irs' ORDER BY ordinal"):
                row = json.loads(encoded)
                row['image'] = image
                if (row['dataset'] != 'irs' or row['split'] != 'train' or row['sha256'] != checksum
                        or row['group'] != group or row['eye'] not in ('left', 'right')
                        or row.get('image_rotation_cw', 0) != 0):
                    raise ValueError('Frozen IRS scope identity differs')
                db.execute('INSERT INTO images(ordinal,image,sha256,scene_group,eye,row_json) VALUES (?,?,?,?,?,?)',
                           (ordinal, image, checksum, group, row['eye'], canonical_json(row).decode()))
        if db.execute('SELECT COUNT(*) FROM images').fetchone()[0] != expected:
            raise ValueError('Frozen IRS scope count differs')
        db.execute('CREATE INDEX image_strata ON images(scene_group,eye)')
        packed = 0
        for packet in packs['packs']:
            receipt_path = cache / 'packs' / packet['receipt']
            if receipt_path.parent != cache / 'packs':
                raise ValueError('Invalid teacher receipt path')
            receipt_ref = dict(path=str(receipt_path), sha256=packet['sha256'])
            receipt = checked_json(receipt_ref)
            pack_path = receipt_path.parent / receipt['tar']
            if (pack_path.parent != receipt_path.parent or not pack_path.is_file()
                    or pack_path.stat().st_size != receipt['bytes']
                    or receipt['policy_sha256'] != policy_hash or len(receipt['entries']) != packet['count']):
                raise ValueError('Teacher receipt policy, pack size or count differs')
            for entry in receipt['entries']:
                metrics = teacher_metrics(entry, policy_hash, config['teacher']['revision'])
                bind_outcome(db, entry['image'], 'teacher_candidate', receipt_ref, entry['key'], metrics)
                packed += 1
            db.commit()
        rejected_reasons = Counter()
        for rejection in rejections:
            if rejection['reason'] != 'teacher_has_no_valid_depth':
                raise ValueError('Unreviewed teacher rejection reason')
            bind_outcome(db, rejection['image'], 'rejected', rejection_ref)
            rejected_reasons[rejection['reason']] += 1
        strata = db.execute('SELECT scene_group,eye,COUNT(*) FROM images GROUP BY 1,2 ORDER BY 1,2').fetchall()
        if (packed != generation['completed'] or len(strata) != generation['family_eye_pairs_seen']
                or db.execute('SELECT COUNT(*) FROM images WHERE outcome IS NULL').fetchone()[0]):
            raise ValueError('Teacher outcomes do not reconcile with the entire frozen scope')
        summary = distribution(db)
        groups, selected = [], {}
        for group, eye, count in strata:
            groups.append(dict(group=group, eye=eye, scope_images=count, **distribution(db, group, eye)))
            for order, reason in (('p99_ratio DESC', 'largest_relative_p99'), ('support ASC', 'least_support')):
                for ordinal, in db.execute("SELECT ordinal FROM images WHERE outcome='teacher_candidate' "
                        'AND scene_group=? AND eye=? ORDER BY ' + order + ',ordinal LIMIT 2', (group, eye)):
                    selected.setdefault(ordinal, []).append(reason)
        review_rows = []
        for ordinal, reasons in sorted(selected.items()):
            group, eye, image, checksum, key, evidence, support, low, high, maximum = db.execute(
                'SELECT scene_group,eye,image,sha256,entry_key,evidence_json,support,p01_ratio,p99_ratio,max_ratio '
                'FROM images WHERE ordinal=?', (ordinal,)).fetchone()
            review_rows.append(dict(corpus_ordinal=ordinal, group=group, eye=eye, image=image, sha256=checksum,
                key=key, receipt=json.loads(evidence), selection_reasons=reasons, valid_fraction=support,
                p01_over_median=low, p99_over_median=high, max_over_median=maximum))
        db.commit()
        if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('Teacher outcome index integrity failure')
        db.close()
        for ref in (source_ref, generation_ref, packs_ref, rejection_ref, policy_ref):
            checked_path(ref)
        os.rename(temporary, output / 'outcomes.sqlite')
        publish_json(output / 'strata.json', groups)
        publish_json(output / 'review-candidates.json', review_rows)
        report = dict(stage='teacher_outcomes_accounted_not_accepted', completed_at=time.time(), dataset='irs',
            expected_scope=expected, candidate_images=packed, rejected_images=len(rejections),
            rejection_reasons=dict(rejected_reasons), family_eye_pairs=len(strata),
            families=len({g for g, _, _ in strata}), relative_statistics=summary,
            manifest_sha256=config['manifest_sha256'], policy_sha256=policy_hash,
            source_report=source_ref, inventory=inventory_ref, generation_report=generation_ref,
            packs=packs_ref, rejections=rejection_ref, index=file_ref(output / 'outcomes.sqlite'),
            strata=file_ref(output / 'strata.json'), review_candidates=file_ref(output / 'review-candidates.json'),
            review_candidates_count=len(review_rows), producer_array_readback_complete=True,
            independent_array_readback_performed=False, pack_payload_hashes_rechecked=False,
            receipt_metadata_only=True, source_quality_accepted=False, training_admission=False,
            optimizer_updates=0, automatic_exclusions=0, audit_code=file_ref(Path(__file__)))
        publish_json(output / 'report.json', report)
        print(json.dumps({k: report[k] for k in ('stage', 'candidate_images', 'rejected_images',
                                               'relative_statistics', 'review_candidates_count')}), flush=True)
        return report
    except BaseException:
        db.close()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--cache-attempt', default='irs-teacher-cache-01')
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if any(Path(n).name != n for n in (args.cache_attempt, args.attempt)):
        parser.error('Use attempt basenames')
    run(json.loads(args.config.read_text()), args.cache_attempt, args.attempt)
