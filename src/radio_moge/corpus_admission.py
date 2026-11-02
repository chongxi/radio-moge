"""Full-corpus accounting and a fail-closed, content-bound training gate.

Cache generation, source review, whole-corpus verification, and permission to
optimize are different artifacts. None of the first three opens the last gate.
SQLite keeps million-image metadata out of the trainer's Python heap.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sqlite3

import numpy as np

from .geometry_cache import (HEX64, canonical_json, publish_json,
                             read_geometry_member, sha_file)
from .geometry_sampling import camera_matrix, source_supervision


SOURCE_CHECKS = ('documentation_and_paper', 'units_and_camera', 'valid_support',
                 'geometric_quality', 'visual_quality')
FINAL_CHECKS = ('source_quality', 'coverage', 'heldout_isolation', 'loss_tests',
                'model_zero_update')


def file_ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha_file(path))


def checked_path(ref):
    if (not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}
            or not isinstance(ref['path'], str) or not Path(ref['path']).is_absolute()
            or not isinstance(ref['sha256'], str) or not HEX64.fullmatch(ref['sha256'])):
        raise ValueError('Evidence must be an absolute path and exact SHA256')
    path = Path(ref['path'])
    if sha_file(path) != ref['sha256']:
        raise ValueError(f'Evidence changed: {path.name}')
    return path


def checked_json(ref):
    return json.loads(checked_path(ref).read_text())


def readonly_db(path):
    # Do NOT use immutable=1: that would hide an accidental in-place replacement.
    db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    db.execute('PRAGMA query_only=ON')
    db.execute('PRAGMA cache_size=-16384')
    return db


def training_contract_hash(config):
    """Exclude only the seal's own reference to avoid a cryptographic cycle."""
    value = deepcopy(config)
    value['data'].pop('geometry_corpus', None)
    return hashlib.sha256(canonical_json(value)).hexdigest()


def implementation_hashes():
    directory = Path(__file__).parent
    names = ('corpus_admission.py', 'corpus_data.py', 'geometry_sampling.py',
             'corpus_losses.py', 'geometry_cache.py', 'official_gt.py', 'train.py',
             'data.py', 'feature_replay.py', 'decoder.py', 'system.py', 'losses.py',
             'adapters.py', 'architecture_screen.py', 'spatial_heads.py',
             'feature_refinement.py', 'upstream.py', 'config.py', 'tracking.py')
    return {name: sha_file(directory / name) for name in names}


def normalized_inventory_row(encoded, image):
    row = json.loads(encoded)
    row['image'] = image
    return row


def validate_source_review(review, dataset, policy, manifest):
    if (review.get('stage') != 'source_review_complete'
            or review.get('decision') != 'eligible_with_flagged_rows_quarantined'
            or review.get('dataset') != dataset or review.get('policy_sha256') != policy
            or review.get('manifest_sha256') != manifest
            or any(review.get('checks', {}).get(name) is not True for name in SOURCE_CHECKS)
            or not review.get('evidence') or not review.get('label_kinds')):
        raise ValueError('A complete, policy-bound source quality review is required')
    for ref in review['evidence']:
        checked_path(ref)


def build_accounting(plan, output):
    """Verify every selected pack member and every frozen row; never open a gate.

    All sources need explicit reviews. Flagged candidates are counted but never
    selected; their images still need a separately reviewed pseudo candidate or
    an explicit, evidenced exclusion. Duplicate selections are errors, not an
    implicit preference for whichever cache was read last.
    """
    scope = checked_json(plan['scope'])
    inventory_report = checked_json(plan['inventory_report'])
    inventory_path = checked_path(plan['inventory'])
    expected = scope['expected_counts']
    manifest = scope['manifest_sha256']
    if (inventory_report.get('stage') != 'inventory_complete'
            or inventory_report['manifest_sha256'] != manifest
            or inventory_report['inventory_sha256'] != plan['inventory']['sha256']
            or inventory_report['dataset_counts'] != expected
            or inventory_report['images'] != sum(expected.values())
            or inventory_report['duplicate_encoded_image_rows'] != 0):
        raise ValueError('Full frozen inventory identity/counts are required')
    if sha_file(inventory_report['manifest']) != manifest:
        raise ValueError('Original training manifest changed')
    for check in inventory_report['heldout_checks']:
        checked_path(dict(path=check['path'], sha256=check['sha256']))
        if check['overlap'] or (check['scene_disjoint_required'] and check['shared_scene_rows']):
            raise ValueError('Inventory reports train/heldout leakage')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    publish_json(output / 'plan.json', plan)
    temporary = output / 'accounting.sqlite.incoming'
    db = sqlite3.connect(temporary)
    db.execute('PRAGMA cache_size=-32768')
    db.execute('PRAGMA temp_store=FILE')
    db.executescript('''
        CREATE TABLE images (
            ordinal INTEGER PRIMARY KEY, image TEXT NOT NULL UNIQUE,
            sha256 TEXT NOT NULL UNIQUE, dataset TEXT NOT NULL, scene_group TEXT NOT NULL,
            row_json TEXT NOT NULL, disposition TEXT NOT NULL DEFAULT 'unresolved',
            pack TEXT, entry_json TEXT, decision_json TEXT, label_kind TEXT,
            valid_fraction REAL, median_depth REAL);
        CREATE INDEX by_group ON images(scene_group);
        CREATE INDEX by_dataset ON images(dataset);
        CREATE TABLE packs (path TEXT PRIMARY KEY, sha256 TEXT NOT NULL,
            bytes INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, receipt_json TEXT NOT NULL);
        CREATE TABLE samples (sample INTEGER PRIMARY KEY, ordinal INTEGER NOT NULL UNIQUE);
    ''')
    selected, quarantine, reports = Counter(), Counter(), []
    try:
        with readonly_db(inventory_path) as source:
            for ordinal, image, checksum, dataset, group, encoded in source.execute(
                    'SELECT ordinal,image,sha256,dataset,scene_group,row_json FROM images ORDER BY ordinal'):
                row = normalized_inventory_row(encoded, image)
                if row.get('split') != 'train' or row.get('group') != group or row['sha256'] != checksum:
                    raise ValueError('Malformed frozen training row')
                db.execute('INSERT INTO images(ordinal,image,sha256,dataset,scene_group,row_json) '
                           'VALUES (?,?,?,?,?,?)',
                           (ordinal, image, checksum, dataset, group, canonical_json(row).decode()))
        actual = dict(db.execute('SELECT dataset,COUNT(*) FROM images GROUP BY dataset'))
        if actual != expected:
            raise ValueError('Inventory database does not cover the authorized scope')
        db.commit()
        for candidate in plan['sources']:
            generation = checked_json(candidate['generation_report'])
            review = checked_json(candidate['review'])
            index_path = checked_path(candidate['packs_index'])
            index = json.loads(index_path.read_text())
            policy = index['policy_sha256']
            dataset = candidate['dataset']
            if (dataset not in expected or generation.get('stage') != 'complete'
                    or generation.get('mode') != 'generate'
                    or generation.get('optimizer_updates') != 0
                    or generation.get('training_started') is not False
                    or generation['policy_sha256'] != policy
                    or generation['verified_cache_members'] != generation['completed']
                    or index['count'] != generation['completed']
                    or generation['completed'] + generation['rejected'] != candidate['scope_count']):
                raise ValueError('A completed, fully reread candidate generation is required')
            validate_source_review(review, dataset, policy, manifest)
            policy_value = json.loads((index_path.parent / 'policy.json').read_text())
            if (hashlib.sha256(canonical_json(policy_value)).hexdigest() != policy
                    or policy_value['manifest_sha256'] != manifest):
                raise ValueError('Cache policy is not bound to the frozen manifest')
            verified = 0
            for pack_item in index['packs']:
                receipt_path = index_path.parent / pack_item['receipt']
                if receipt_path.parent != index_path.parent:
                    raise ValueError('Unexpected pack receipt path')
                receipt_ref = dict(path=str(receipt_path), sha256=pack_item['sha256'])
                receipt = checked_json(receipt_ref)
                tar = receipt_path.parent / receipt['tar']
                if (tar.parent != receipt_path.parent or receipt['policy_sha256'] != policy
                        or len(receipt['entries']) != pack_item['count']
                        or sha_file(tar) != receipt['sha256']):
                    raise ValueError('Completed pack identity changed')
                stat = tar.stat()
                if stat.st_size != receipt['bytes']:
                    raise ValueError('Pack byte count changed')
                db.execute('INSERT INTO packs VALUES (?,?,?,?,?)', (str(tar), receipt['sha256'],
                    stat.st_size, stat.st_mtime_ns, canonical_json(receipt_ref).decode()))
                for entry in receipt['entries']:
                    depth, valid, k, meta = read_geometry_member(tar, entry)
                    camera_matrix(k)
                    row = dict(entry['image'])
                    identity = db.execute('SELECT ordinal,row_json,disposition FROM images WHERE image=?',
                                          (row['image'],)).fetchone()
                    if identity is None:
                        raise ValueError('Candidate is outside the frozen corpus')
                    ordinal, encoded, disposition = identity
                    if row.pop('corpus_ordinal', ordinal) != ordinal or row != json.loads(encoded):
                        raise ValueError('Candidate changed a frozen RGB record')
                    if (meta['policy_sha256'] != policy or meta['dataset'] != dataset
                            or row['dataset'] != dataset or meta['image_sha256'] != row['sha256']):
                        raise ValueError('Candidate metadata does not match source/image policy')
                    verified += 1
                    if meta.get('source_needs_review', False):
                        quarantine[dataset] += 1
                        continue
                    source_supervision(meta)
                    if meta['label_kind'] not in review['label_kinds'] or valid.sum() < 32:
                        raise ValueError('Unreviewed label kind or insufficient observations')
                    if disposition != 'unresolved':
                        raise ValueError('Duplicate image selection; explicit routing is required')
                    if not np.isclose(valid.mean(), entry['valid_fraction'], rtol=0, atol=1e-12):
                        raise ValueError('Cached valid-support statistic differs')
                    db.execute('UPDATE images SET disposition=?,pack=?,entry_json=?,decision_json=?, '
                               'label_kind=?,valid_fraction=?,median_depth=? WHERE ordinal=?',
                               ('selected', str(tar), canonical_json(entry).decode(),
                                canonical_json(candidate['review']).decode(), meta['label_kind'],
                                float(valid.mean()), float(np.median(depth[valid])), ordinal))
                    selected[dataset] += 1
                db.commit()
            if verified != index['count']:
                raise ValueError('Not every published cache entry was reread')
            reports.append(dict(dataset=dataset, policy_sha256=policy, verified=verified,
                                generation_report=candidate['generation_report'], review=candidate['review']))
        excluded = Counter()
        with checked_path(plan['exclusions']).open() as file:
            for line in file:
                if not line.strip():
                    continue
                decision = json.loads(line)
                identity = db.execute('SELECT ordinal,sha256,dataset,disposition FROM images WHERE image=?',
                                      (decision['image'],)).fetchone()
                if (identity is None or identity[1] != decision['sha256']
                        or identity[3] != 'unresolved' or not decision.get('reason')
                        or not decision.get('evidence') or decision.get('decision') != 'exclude'):
                    raise ValueError('Exclusion must be explicit, unique and evidenced')
                for ref in decision['evidence']:
                    checked_path(ref)
                db.execute('UPDATE images SET disposition=?,decision_json=? WHERE ordinal=?',
                           ('excluded', canonical_json(decision).decode(), identity[0]))
                excluded[identity[2]] += 1
        unresolved = db.execute("SELECT COUNT(*) FROM images WHERE disposition='unresolved'").fetchone()[0]
        if unresolved:
            raise ValueError(f'{unresolved} frozen images have neither verified labels nor reviewed exclusion')
        if set(selected) != set(expected):
            raise ValueError('Every source needs accepted geometry candidates, not only exclusions')
        for sample, (ordinal,) in enumerate(db.execute(
                "SELECT ordinal FROM images WHERE disposition='selected' ORDER BY ordinal")):
            db.execute('INSERT INTO samples VALUES (?,?)', (sample, ordinal))
        coverage = [dict(dataset=d, label_kind=k, images=n, groups=g,
                         min_valid_fraction=lo, mean_valid_fraction=mean,
                         min_image_median_depth=zlo, max_image_median_depth=zhi)
                    for d, k, n, g, lo, mean, zlo, zhi in db.execute('''
                        SELECT dataset,label_kind,COUNT(*),COUNT(DISTINCT scene_group),
                            MIN(valid_fraction),AVG(valid_fraction),MIN(median_depth),MAX(median_depth)
                        FROM images WHERE disposition='selected' GROUP BY dataset,label_kind''')]
        db.commit()
        if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('Accounting database integrity check failed')
        db.close()
        final = output / 'accounting.sqlite'
        os.replace(temporary, final)
        report = dict(schema_version=1, stage='accounting_verified_not_accepted', training_admission=False,
            optimizer_updates=0, manifest_sha256=manifest, expected_counts=expected,
            total_images=sum(expected.values()), selected_counts=dict(selected),
            excluded_counts=dict(excluded), quarantined_candidate_counts=dict(quarantine),
            unresolved=0, selected_images=sum(selected.values()), sources=reports, coverage=coverage,
            inventory=plan['inventory'], inventory_report=plan['inventory_report'],
            scope=plan['scope'], plan=file_ref(output / 'plan.json'), index=file_ref(final))
        publish_json(output / 'report.json', report)
        return report
    except BaseException as error:
        db.close()
        publish_json(output / 'failure.json', dict(stage='failed', training_admission=False,
            error=type(error).__name__, message=str(error)[:1000]))
        raise


def checked_gate_evidence(name, ref, accounting_ref, accounting):
    value = checked_json(ref)
    if (value.get('gate') != name or value.get('stage') != 'complete'
            or value.get('passed') is not True
            or value.get('accounting_report_sha256') != accounting_ref['sha256']
            or value.get('index_sha256') != accounting['index']['sha256']
            or value.get('manifest_sha256') != accounting['manifest_sha256']
            or not value.get('evidence')):
        raise ValueError(f'Final {name} gate is incomplete or bound to other data')
    for item in value['evidence']:
        checked_path(item)
    return value


def validate_pre_model_evidence(accounting_ref, checks):
    """Verify all nonmodel gates without issuing or imitating a training seal."""
    accounting = checked_json(accounting_ref)
    if (accounting.get('stage') != 'accounting_verified_not_accepted'
            or accounting.get('training_admission') is not False or accounting.get('unresolved') != 0
            or accounting['total_images'] != sum(accounting['expected_counts'].values())
            or accounting['selected_images'] < 1
            or sum(accounting['selected_counts'].values()) != accounting['selected_images']
            or accounting['selected_images'] + sum(accounting['excluded_counts'].values())
                != accounting['total_images']):
        raise ValueError('Full verified accounting is required before final acceptance')
    for dataset, count in accounting['expected_counts'].items():
        if (accounting['selected_counts'].get(dataset, 0)
                + accounting['excluded_counts'].get(dataset, 0) != count):
            raise ValueError('Per-source accounting does not cover the full scope')
    checked_path(accounting['plan'])
    scope = checked_json(accounting['scope'])
    if (scope['manifest_sha256'] != accounting['manifest_sha256']
            or scope['expected_counts'] != accounting['expected_counts']):
        raise ValueError('Accounting differs from the authorized scope')
    for source in accounting['sources']:
        checked_json(source['generation_report'])
        validate_source_review(checked_json(source['review']), source['dataset'],
                               source['policy_sha256'], accounting['manifest_sha256'])
    names = set(FINAL_CHECKS) - {'model_zero_update'}
    if set(checks) != names:
        raise ValueError('All source, coverage, leakage and loss gates are required before model preflight')
    evidence = {name: checked_gate_evidence(name, checks[name], accounting_ref, accounting)
                for name in names}
    if (evidence['source_quality'].get('unresolved') != 0
            or evidence['source_quality'].get('reviewed_images') != accounting['total_images']
            or evidence['coverage'].get('reviewed_counts') != accounting['expected_counts']
            or evidence['heldout_isolation'].get('overlaps') != 0):
        raise ValueError('Quality, coverage and heldout review must cover the entire frozen scope')
    hashes = implementation_hashes()
    if evidence['loss_tests'].get('implementation_sha256') != hashes:
        raise ValueError('Loss/model tests refer to a different implementation')
    checked_path(accounting['index'])
    return accounting, hashes


def validate_final_evidence(accounting_ref, checks, config):
    if set(checks) != set(FINAL_CHECKS):
        raise ValueError('All final source, coverage, leakage, loss and model gates are required')
    accounting, hashes = validate_pre_model_evidence(
        accounting_ref, {name: ref for name, ref in checks.items() if name != 'model_zero_update'})
    model = checked_gate_evidence('model_zero_update', checks['model_zero_update'], accounting_ref, accounting)
    if model.get('implementation_sha256') != hashes:
        raise ValueError('Loss/model tests refer to a different implementation')
    if (model.get('optimizer_updates') != 0 or model.get('actual_model_forward_backward') is not True
            or model.get('frozen_backbones_unchanged') is not True
            or model.get('adapter_unchanged') is not True
            or model.get('finite_nonzero_adapter_gradient') is not True
            or model.get('training_contract_sha256') != training_contract_hash(config)):
        raise ValueError('Actual unchanged-model zero-update preflight is required')
    checked_path(model['initial_checkpoint'])
    return accounting, model, hashes


def seal_acceptance(accounting_ref, checks, config, output):
    """No automatic review: all hash-bound evidence must already exist and pass."""
    accounting, model, hashes = validate_final_evidence(accounting_ref, checks, config)
    result = dict(schema_version=1, stage='accepted_for_training', training_admission=True,
        accounting_report=accounting_ref, checks=checks, index=accounting['index'],
        manifest_sha256=accounting['manifest_sha256'], selected_images=accounting['selected_images'],
        initial_checkpoint=model['initial_checkpoint'], implementation_sha256=hashes,
        training_contract_sha256=training_contract_hash(config))
    publish_json(output, result)
    return result


def open_training_gate(config, init=None, resume=False):
    """Called BEFORE model/optimizer construction, also on every exact resume."""
    gate = checked_json(config['data']['geometry_corpus'])
    if (gate.get('schema_version') != 1 or gate.get('stage') != 'accepted_for_training'
            or gate.get('training_admission') is not True
            or gate.get('training_contract_sha256') != training_contract_hash(config)
            or gate.get('implementation_sha256') != implementation_hashes()):
        raise ValueError('Corpus training gate is closed, stale, or for a different configuration')
    accounting, model, _ = validate_final_evidence(gate['accounting_report'], gate['checks'], config)
    if (gate['index'] != accounting['index'] or gate['manifest_sha256'] != accounting['manifest_sha256']
            or gate['selected_images'] != accounting['selected_images']
            or gate['initial_checkpoint'] != model['initial_checkpoint']):
        raise ValueError('Acceptance seal does not match its verified accounting/preflight')
    if not resume and (init is None or sha_file(init) != gate['initial_checkpoint']['sha256']):
        raise ValueError('Corpus stage requires the exact preflight initialization checkpoint')
    if sha_file(config['data']['train_manifest']) != gate['manifest_sha256']:
        raise ValueError('Configured training manifest differs from the accepted corpus')
    with readonly_db(gate['index']['path']) as db:
        if db.execute('SELECT COUNT(*) FROM samples').fetchone()[0] != gate['selected_images']:
            raise ValueError('Accepted sample count differs')
        for path, size, mtime in db.execute('SELECT path,bytes,mtime_ns FROM packs'):
            info = Path(path).stat()
            if (info.st_size, info.st_mtime_ns) != (size, mtime):
                raise ValueError('Accepted geometry pack changed; new full verification required')
    # Each read still verifies the actual RGB and geometry payload SHA256.
    return gate
