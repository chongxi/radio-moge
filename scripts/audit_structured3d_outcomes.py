"""Reconcile recovered and new Structured3D outputs without accepting geometry.

The producer/recovery reports bind earlier full payload readback. This audit
checks every frozen identity, receipt and rejection; it does not repeat arrays
or claim source quality. Use a local temporary SQLite file for bounded reads.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

from radio_moge.corpus_admission import checked_json, checked_path, file_ref, readonly_db
from radio_moge.geometry_cache import canonical_json, publish_json
from radio_moge.structured3d_depth import is_erratum, parse_errata


def candidate_metrics(entry, policy_hash, errata):
    row, meta = entry['image'], entry['geometry_metadata']
    stats = meta['source_quality']
    h, w = meta['shape']
    fraction = meta['valid_pixels'] / (h * w)
    quantiles = entry['depth_quantiles_m']
    if (meta['dataset'] != 'structured3d' or row['dataset'] != 'structured3d'
            or row['split'] != 'train' or meta['image_sha256'] != row['sha256']
            or meta['policy_sha256'] != policy_hash
            or hashlib.sha256(canonical_json(meta)).hexdigest() != entry['key']
            or meta['label_kind'] != 'synthetic_renderer' or meta['camera_kind'] != 'synthetic_camera'
            or meta['metric_calibrated'] is not True or meta['depth_convention'] != 'camera_z'
            or meta['units'] != 'meters' or [h, w] != [row['height'], row['width']]
            or meta['source_needs_review'] is not False or stats['source_needs_review'] is not False
            or stats['layout_verified_pixels'] < 100 or not .05 <= fraction <= 1
            or not math.isclose(fraction, entry['valid_fraction'], rel_tol=0, abs_tol=1e-12)
            or not math.isclose(fraction, stats['valid_fraction'], rel_tol=0, abs_tol=1e-12)
            or len(quantiles) != 5 or any(not math.isfinite(x) or x <= 0 for x in quantiles)
            or sorted(quantiles) != quantiles or is_erratum(row, errata)):
        raise ValueError('Structured3D candidate identity, support, units or source flag differs')
    return dict(valid_fraction=fraction, median_depth_m=quantiles[2],
                layout_verified_pixels=stats['layout_verified_pixels'],
                layout_conflicting_pixels=stats['layout_conflicting_pixels'])


def rejection_kind(item, errata, errata_proof):
    reason, row = item['reason'], item['image']
    if errata_proof not in item['source_proofs']:
        raise ValueError('Rejection has no bound upstream errata reference')
    if reason == 'upstream_errata_scene_or_room':
        if not is_erratum(row, errata):
            raise ValueError('Errata rejection does not match the published list')
        return 'upstream_errata', False
    if reason != 'insufficient_verified_support' or is_erratum(row, errata):
        raise ValueError('Unexpected native rejection reason')
    stats = item['stats']
    flagged = stats['layout_verified_pixels'] < 100
    valid = stats['valid_fraction']
    if (stats['source_needs_review'] is not flagged or not math.isfinite(valid)
            or not 0 <= valid < .05 or (flagged and valid != 0)):
        raise ValueError('Rejection violates the unchanged layout/support policy')
    return ('insufficient_layout_reference' if flagged else 'insufficient_valid_fraction'), flagged


def bind(db, row, outcome, reason, policy, evidence, key=None, metrics=None):
    found = db.execute('SELECT ordinal,row_json,outcome FROM images WHERE image=?',
                       (row['image'],)).fetchone()
    if found is None:
        raise ValueError('Outcome outside frozen Structured3D scope')
    ordinal, encoded, previous = found
    value = dict(row)
    if value.pop('corpus_ordinal', ordinal) != ordinal or value != json.loads(encoded) or previous:
        raise ValueError('Changed or duplicate Structured3D outcome')
    db.execute('UPDATE images SET outcome=?,reason=?,policy=?,evidence_json=?,entry_key=?,metrics_json=? '
               'WHERE ordinal=?', (outcome, reason, policy, canonical_json(evidence).decode(), key,
                                   canonical_json(metrics or {}).decode(), ordinal))
    return ordinal


def run(config, recovered_name, native_name, attempt):
    root = Path(config['root'])
    source_ref = file_ref(root / 'inventory.json')
    source = checked_json(source_ref)
    inventory_ref = dict(path=str(root / 'inventory.sqlite'), sha256=source['inventory_sha256'])
    checked_path(inventory_ref)
    expected = config['expected_counts']['structured3d']
    if (source['stage'] != 'inventory_complete' or source['manifest_sha256'] != config['manifest_sha256']
            or source['dataset_counts'] != config['expected_counts']):
        raise ValueError('Matching frozen inventory required')
    errata_path = root / 'native-source/structured3d-metadata/errata.txt'
    errata_ref = file_ref(errata_path)
    errata_proof = checked_json(file_ref(errata_path.with_name('errata.txt.receipt.json')))
    if errata_proof['sha256'] != errata_ref['sha256']:
        raise ValueError('Stored errata changed')
    errata = parse_errata(errata_path.read_text())
    sources = []
    for name in (recovered_name, native_name):
        base = root / name
        refs = {key: file_ref(base / path) for key, path in
                [('generation', 'report.json'), ('packs', 'packs/packs.json'), ('policy', 'packs/policy.json')]}
        gen, packs, policy = [checked_json(refs[key]) for key in ('generation', 'packs', 'policy')]
        policy_hash = hashlib.sha256(canonical_json(policy)).hexdigest()
        if (gen['stage'] != 'complete' or gen['mode'] != 'generate'
                or gen['training_started'] is not False or gen['training_admission'] is not False
                or gen['optimizer_updates'] != 0 or gen['completed'] != gen['verified_cache_members']
                or gen['completed'] != packs['count'] or gen['policy_sha256'] != policy_hash
                or packs['policy_sha256'] != policy_hash or policy['dataset'] != 'structured3d'
                or policy['manifest_sha256'] != config['manifest_sha256']
                or policy['upstream_errata'] != errata_proof or policy['minimum_image_valid_fraction'] != .05
                or policy['native_holes_filled'] is not False or policy['label_kind'] != 'synthetic_renderer'):
            raise ValueError('Complete matching producer/recovery payload readback required')
        sources.append(dict(base=base, refs=refs, gen=gen, packs=packs, policy=policy, hash=policy_hash))
    recovered, native = sources
    scope_ref = file_ref(recovered['base'] / 'scope.jsonl')
    if (recovered['gen'].get('recovered_verified_subset') is not True
            or recovered['gen']['scope_sha256'] != scope_ref['sha256']
            or recovered['gen']['rejected'] != 0
            or recovered['gen']['expected_scope'] != recovered['gen']['completed']
            or native['gen']['recovered_subset'] != recovered['refs']['generation']
            or native['gen']['recovered_skipped'] != recovered['gen']['completed']
            or native['gen']['completed'] + native['gen']['rejected'] != native['gen']['expected_scope']
            or native['gen']['expected_scope'] + recovered['gen']['completed'] != expected):
        raise ValueError('Recovery and remainder scopes do not reconcile')
    scope_rows = [json.loads(line) for line in checked_path(scope_ref).read_text().splitlines()]
    scope = {(x['ordinal'], x['sha256']) for x in scope_rows}
    if len(scope) != len(scope_rows) or len(scope) != recovered['gen']['completed']:
        raise ValueError('Recovered scope has duplicate or missing identities')
    rejection_ref = file_ref(native['base'] / 'rejections.json')
    rejections = checked_json(rejection_ref)
    if len(rejections) != native['gen']['rejected']:
        raise ValueError('Native rejection count differs')
    output = root / attempt
    output.mkdir(exist_ok=False)
    counts, reasons, source_counts = Counter(), Counter(), Counter()
    with tempfile.TemporaryDirectory(prefix='structured3d-outcomes-') as scratch:
        temporary = Path(scratch) / 'outcomes.sqlite'
        with sqlite3.connect(temporary) as db:
            db.execute('PRAGMA cache_size=-32768')
            db.execute('CREATE TABLE images(ordinal INTEGER PRIMARY KEY,image TEXT UNIQUE,sha256 TEXT,'
                'scene_group TEXT,row_json TEXT,outcome TEXT,reason TEXT,policy TEXT,'
                'evidence_json TEXT,entry_key TEXT UNIQUE,metrics_json TEXT)')
            with readonly_db(inventory_ref['path']) as original:
                for ordinal, image, checksum, group, encoded in original.execute(
                        "SELECT ordinal,image,sha256,scene_group,row_json FROM images WHERE dataset='structured3d' ORDER BY ordinal"):
                    row = dict(json.loads(encoded), image=image)
                    if row['sha256'] != checksum or row['group'] != group or row['split'] != 'train':
                        raise ValueError('Frozen image identity differs')
                    db.execute('INSERT INTO images VALUES (?,?,?,?,?,NULL,NULL,NULL,NULL,NULL,NULL)',
                               (ordinal, image, checksum, group, canonical_json(row).decode()))
            if db.execute('SELECT COUNT(*) FROM images').fetchone()[0] != expected:
                raise ValueError('Frozen source count differs')
            for src in sources:
                packed = 0
                seen_receipts = set()
                for item in src['packs']['packs']:
                    name = item['receipt']
                    if Path(name).name != name or name in seen_receipts:
                        raise ValueError('Duplicate or invalid receipt path')
                    seen_receipts.add(name)
                    path = src['base'] / 'packs' / name
                    receipt_ref = dict(path=str(path), sha256=item['sha256'])
                    receipt = checked_json(receipt_ref)
                    if (Path(receipt['tar']).name != receipt['tar']
                            or (path.parent / receipt['tar']).stat().st_size != receipt['bytes']
                            or receipt['policy_sha256'] != src['hash']
                            or len(receipt['entries']) != item['count']):
                        raise ValueError('Receipt policy, count or payload size changed')
                    for entry in receipt['entries']:
                        metrics = candidate_metrics(entry, src['hash'], errata)
                        ordinal = bind(db, entry['image'], 'native_candidate', 'quality_review_pending',
                                       src['hash'], receipt_ref, entry['key'], metrics)
                        if src is recovered:
                            identity = (ordinal, entry['image']['sha256'])
                            if identity not in scope:
                                raise ValueError('Recovered entry differs from recovery scope')
                            scope.remove(identity)
                        packed += 1
                    db.commit()
                if packed != src['gen']['completed']:
                    raise ValueError('Not every producer member was accounted')
                source_counts[src['hash']] += packed
            if scope:
                raise ValueError('Recovered scope is incomplete')
            flags = 0
            for item in rejections:
                reason, flagged = rejection_kind(item, errata, errata_proof)
                bind(db, item['image'], 'rejected_needs_disposition', reason, native['hash'], rejection_ref,
                     metrics=item.get('stats'))
                reasons[reason] += 1
                flags += flagged
            if flags != native['gen']['source_review_images']:
                raise ValueError('Native source flag counter differs')
            if db.execute('SELECT COUNT(*) FROM images WHERE outcome IS NULL').fetchone()[0]:
                raise ValueError('Frozen images are missing outcomes')
            counts.update(dict(db.execute('SELECT outcome,COUNT(*) FROM images GROUP BY outcome')))
            with (output / 'unresolved-native.jsonl.incoming').open('x') as file:
                for ordinal, encoded, reason, proof in db.execute(
                        "SELECT ordinal,row_json,reason,evidence_json FROM images WHERE outcome!='native_candidate' ORDER BY ordinal"):
                    file.write(canonical_json(dict(image=json.loads(encoded), corpus_ordinal=ordinal,
                        native_reason=reason, evidence=json.loads(proof))).decode() + '\n')
            os.replace(output / 'unresolved-native.jsonl.incoming', output / 'unresolved-native.jsonl')
            choices = {}
            for policy in source_counts:
                for variant in ('empty', 'full'):
                    for metric, direction in [('valid_fraction', 'ASC'), ('median_depth_m', 'ASC'),
                                              ('median_depth_m', 'DESC'), ('layout_verified_pixels', 'ASC')]:
                        query = ("SELECT ordinal,row_json,entry_key,evidence_json,metrics_json FROM images "
                                 "WHERE outcome='native_candidate' AND policy=? AND json_extract(row_json,'$.configuration')=? "
                                 f"ORDER BY json_extract(metrics_json,'$.{metric}') {direction},ordinal LIMIT 2")
                        for ordinal, encoded, key, proof, stats in db.execute(query, (policy, variant)):
                            value = choices.setdefault(ordinal, dict(corpus_ordinal=ordinal, image=json.loads(encoded),
                                entry_key=key, receipt=json.loads(proof), policy_sha256=policy,
                                metrics=json.loads(stats), selection_reasons=[]))
                            value['selection_reasons'].append(f'{variant}:{metric}:{direction}')
            publish_json(output / 'review-candidates.json', [choices[k] for k in sorted(choices)])
            db.commit()
            if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('Outcome database integrity failure')
        shutil.copyfile(temporary, output / 'outcomes.sqlite.incoming')
        os.replace(output / 'outcomes.sqlite.incoming', output / 'outcomes.sqlite')
    all_refs = [source_ref, inventory_ref, errata_ref, scope_ref, rejection_ref,
                *(ref for src in sources for ref in src['refs'].values())]
    for ref in all_refs:
        checked_path(ref)
    report = dict(stage='native_outcomes_accounted_not_accepted', dataset='structured3d',
        completed_at=time.time(), expected_scope=expected, manifest_sha256=config['manifest_sha256'],
        outcome_counts=dict(counts), rejection_categories=dict(reasons), rejected_source_flags=flags,
        cached_source_flags=0, candidate_policy_counts=dict(source_counts), evidence=all_refs,
        index=file_ref(output / 'outcomes.sqlite'), unresolved_manifest=file_ref(output / 'unresolved-native.jsonl'),
        review_candidates=file_ref(output / 'review-candidates.json'), review_candidate_count=len(choices),
        producer_array_readback_complete=True, receipt_metadata_only=True,
        independent_array_readback_performed=False, pack_payload_hashes_rechecked=False,
        source_quality_accepted=False, training_admission=False, optimizer_updates=0,
        audit_code=file_ref(Path(__file__)))
    publish_json(output / 'report.json', report)
    print(json.dumps({key: report[key] for key in ('stage', 'outcome_counts', 'rejection_categories')}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--recovered', default='structured3d-recovered-01')
    parser.add_argument('--native', default='structured3d-native-cache-02')
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if any(Path(n).name != n for n in (args.recovered, args.native, args.attempt)):
        parser.error('Use attempt basenames')
    run(json.loads(args.config.read_text()), args.recovered, args.native, args.attempt)
