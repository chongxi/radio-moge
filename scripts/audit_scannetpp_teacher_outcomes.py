"""Account recovered and newly generated DSLR teacher candidates exactly once.

Producer payload readback must already be complete. This audit checks receipt
provenance, scope and relative statistics; it neither clears native flags nor
approves labels, repeats inference or rereads all payload arrays.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

import numpy as np

from audit_irs_teacher_outcomes import teacher_metrics
from cache_scannetpp_review_teacher import review_scope
from radio_moge.cache_recovery import validate_teacher_recovery_policy
from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import HEX64, canonical_json, publish_json


def validate_generation(recovery, generation, recovery_ref, total):
    for value in (recovery, generation):
        if (value.get('stage') != 'complete' or value.get('mode') != 'generate'
                or value.get('training_started') is not False
                or value.get('training_admission') is not False or value.get('optimizer_updates') != 0
                or value.get('completed') != value.get('verified_cache_members')
                or value.get('completed', -1) + value.get('rejected', -1) != value.get('expected_scope')):
            raise ValueError('Complete producer payload readback is required')
    if (recovery.get('recovered_verified_subset') is not True or recovery['rejected'] != 0
            or generation.get('teacher_frozen') is not True
            or generation.get('recovered_subset') != recovery_ref
            or generation.get('recovered_skipped') != recovery['completed']
            or generation.get('original_scope') != total
            or recovery['expected_scope'] + generation['expected_scope'] != total):
        raise ValueError('Recovery and remainder generation do not form the exact routed scope')


def checked_metrics(entry, policy, policy_hash, route_ref):
    metrics = teacher_metrics(entry, policy_hash, policy['teacher']['revision'], 'scannetpp_dslr')
    row, meta = entry['image'], entry['geometry_metadata']
    stats = meta['source_quality']
    mask_hash = stats.get('anonymous_mask_sha256', '')
    expected = [dict(source=policy['teacher']['repo'] + '/model.pt', sha256=policy['checkpoint_sha256']),
        dict(source=row['image'], sha256=row['sha256']), dict(source=row['mask'], sha256=mask_hash),
        dict(source=route_ref['path'], sha256=route_ref['sha256'])]
    if (row['dataset'] != 'scannetpp_dslr' or row['split'] != 'train'
            or row['official_split'] != 'nvs_sem_train' or row['group'] != 'scannetpp:' + row['scene']
            or row.get('image_rotation_cw', 0) != 0
            or meta.get('native_source_review_unresolved') is not True
            or meta.get('conversion') != 'pinned_moge2_infer_forced_projection_with_original_anonymity_mask'
            or meta.get('source_id') != row['scene'] + '/' + row['view']
            or meta.get('source_proofs') != expected or not HEX64.fullmatch(mask_hash)
            or stats.get('valid_pixels') != meta['valid_pixels']
            or stats.get('total_pixels') != row['height'] * row['width']
            or not math.isclose(stats['valid_fraction'], metrics[0], rel_tol=0, abs_tol=1e-12)
            or not policy['minimum_valid_fraction'] <= metrics[0] <= stats['teacher_valid_fraction'] <= 1
            or not 0 <= stats['projection_max_scaled_error'] <= 2e-5
            or not math.isfinite(stats['projection_max_abs_error_m'])
            or stats['projection_max_abs_error_m'] < 0):
        raise ValueError('Teacher masks, projection, provenance or retained native flags differ')
    return metrics


def bind_outcome(db, row, subset, recovered, policy, proof, key=None, metrics=None, reason=None):
    ordinal = row.get('corpus_ordinal')
    found = db.execute('SELECT row_json,outcome FROM images WHERE ordinal=?', (ordinal,)).fetchone()
    if (found is None or row != json.loads(found[0]) or found[1] is not None
            or (ordinal in recovered) != (subset == 'recovered')):
        raise ValueError('Changed, duplicate, outside-scope or wrongly recovered RGB outcome')
    values = metrics if metrics is not None else (None, None, None, None)
    db.execute('UPDATE images SET outcome=?,subset=?,policy=?,evidence_json=?,entry_key=?,support=?,'
        'p01_ratio=?,p99_ratio=?,max_ratio=?,reason=? WHERE ordinal=?',
        ('teacher_candidate' if key is not None else 'teacher_rejection_needs_review', subset, policy,
         canonical_json(proof).decode(), key, *values, reason, ordinal))


def statistics(values):
    probabilities = [0., .01, .5, .9, .99, 1.]
    values = np.asarray(values, np.float64)
    return dict(images=len(values), percentiles=probabilities,
        **{name: np.quantile(values[:, i], probabilities).tolist() if len(values) else []
           for i, name in enumerate(('support', 'p01_over_median', 'p99_over_median', 'max_over_median'))},
        units='ratios to per-image valid depth median; no calibrated metric-scale claim')


def run(config, attempt, recovery_name='scannetpp-review-teacher-recovered-01',
        cache_name='scannetpp-review-teacher-cache-02', preflight='scannetpp-review-teacher-preflight-02'):
    root = Path(config['root'])
    names = (recovery_name, cache_name)
    sources = []
    for name in names:
        refs = {k: file_ref(root / name / v) for k, v in
            dict(generation='report.json', packs='packs/packs.json', policy='packs/policy.json').items()}
        sources.append(dict(name=name, refs=refs, **{k: checked_json(v) for k, v in refs.items()}))
    recovery, fresh = sources
    route_ref = file_ref(root / 'scannetpp-review-teacher-routing-20261102.json')
    route = checked_json(route_ref)
    validate_generation(recovery['generation'], fresh['generation'], recovery['refs']['generation'],
                        route['scope_images'])
    validate_teacher_recovery_policy(recovery['policy'], fresh['policy'])
    if (route['dataset'] != 'scannetpp_dslr' or route['label_kind'] != 'teacher_pseudo'
            or route['manifest_sha256'] != config['manifest_sha256'] or route['metric_calibrated'] is not False
            or route['fill_native_holes'] is not False or route['training_admission'] is not False):
        raise ValueError('Explicit unchanged nonmetric scene routing required')
    evidence = [route_ref, route['source_audit']]
    for src in sources:
        gen, packs, policy = (src[k] for k in ('generation', 'packs', 'policy'))
        policy_hash = hashlib.sha256(canonical_json(policy)).hexdigest()
        if (policy_hash != gen['policy_sha256'] or policy_hash != packs['policy_sha256']
                or packs['count'] != gen['completed'] or policy['dataset'] != 'scannetpp_dslr'
                or policy['manifest_sha256'] != config['manifest_sha256'] or policy['teacher'] != config['teacher']
                or policy['candidate_routing_sha256'] != route_ref['sha256']
                or policy['native_geometry_read'] is not False or policy['native_metric_gt'] is not False
                or policy['force_projection'] is not True or policy['minimum_valid_fraction'] != .05
                or policy['anonymity'] != 'original mask==255 AND 5x5 erosion AND teacher support; no resizing/filling'):
            raise ValueError('Teacher policy or complete pack count differs')
        src['policy_hash'] = policy_hash
        evidence.extend(src['refs'].values())
    previous_ref = file_ref(root / preflight / 'report.json')
    visual_ref = file_ref(root / preflight / 'visual-review.json')
    previous, visual = checked_json(previous_ref), checked_json(visual_ref)
    if (previous['stage'] != 'complete' or previous['passed'] is not True or previous['mode'] != 'preflight'
            or previous['verified_cache_members'] != 2 * len(route['scene_counts'])
            or previous['rejected'] != 0 or previous['policy_sha256'] != fresh['policy_hash']
            or visual['stage'] != 'visual_review_complete' or visual['passed'] is not True
            or visual['policy_sha256'] != fresh['policy_hash']
            or visual['preflight_report_sha256'] != previous_ref['sha256']
            or set(visual['reviewed_scenes']) != set(route['scene_counts']) or not visual['evidence']):
        raise ValueError('Matching numeric and all-scene visual preflight required')
    evidence.extend([previous_ref, visual_ref, *visual['evidence']])
    inventory_ref = file_ref(root / 'inventory.json')
    inventory = checked_json(inventory_ref)
    index_ref = dict(path=str(root / 'inventory.sqlite'), sha256=inventory['inventory_sha256'])
    if (inventory['stage'] != 'inventory_complete' or inventory['manifest_sha256'] != config['manifest_sha256']
            or inventory['dataset_counts'] != config['expected_counts']):
        raise ValueError('Complete matching frozen inventory required')
    checked_path(index_ref)
    evidence.extend([inventory_ref, index_ref])
    rows, _ = review_scope(root, route, config['manifest_sha256'])
    scope_ref = dict(path=str(root / recovery_name / 'scope.jsonl'),
                     sha256=recovery['generation']['scope_sha256'])
    recovered_rows = [json.loads(line) for line in checked_path(scope_ref).read_text().splitlines()]
    recovered = {r['ordinal']: r['sha256'] for r in recovered_rows}
    exact = {r['corpus_ordinal']: r for r in rows}
    if (len(exact) != len(rows) or len(recovered) != len(recovered_rows)
            or len(recovered) != recovery['generation']['completed']
            or any(i not in exact or exact[i]['sha256'] != h for i, h in recovered.items())):
        raise ValueError('Recovered scope is not an exact unique subset of the original RGB identities')
    evidence.append(scope_ref)
    rejection_ref = file_ref(root / cache_name / 'rejections.json')
    rejections = checked_json(rejection_ref)
    if len(rejections) != fresh['generation']['rejected']:
        raise ValueError('Remainder rejection count differs')
    evidence.append(rejection_ref)
    output = root / attempt
    output.mkdir(exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='scannet-teacher-outcomes-') as temporary:
        local = Path(temporary) / 'outcomes.sqlite'
        with sqlite3.connect(local) as db:
            db.execute('CREATE TABLE images(ordinal INTEGER PRIMARY KEY,image TEXT UNIQUE,sha256 TEXT UNIQUE,'
                'scene_group TEXT,row_json TEXT,outcome TEXT,subset TEXT,policy TEXT,evidence_json TEXT,'
                'entry_key TEXT UNIQUE,support REAL,p01_ratio REAL,p99_ratio REAL,max_ratio REAL,reason TEXT)')
            db.executemany('INSERT INTO images(ordinal,image,sha256,scene_group,row_json) VALUES (?,?,?,?,?)',
                [(r['corpus_ordinal'],r['image'],r['sha256'],r['group'],canonical_json(r).decode()) for r in rows])
            for src, subset in zip(sources, ('recovered', 'remainder'), strict=True):
                count = 0
                for item in src['packs']['packs']:
                    receipt_path = root / src['name'] / 'packs' / item['receipt']
                    if receipt_path.parent != root / src['name'] / 'packs':
                        raise ValueError('Invalid pack receipt path')
                    receipt_ref = dict(path=str(receipt_path), sha256=item['sha256'])
                    receipt = checked_json(receipt_ref)
                    tar = receipt_path.parent / receipt['tar']
                    if (tar.parent != receipt_path.parent or tar.stat().st_size != receipt['bytes']
                            or receipt['policy_sha256'] != src['policy_hash']
                            or len(receipt['entries']) != item['count']):
                        raise ValueError('Teacher pack size, policy or entry count differs')
                    for entry in receipt['entries']:
                        metrics = checked_metrics(entry, src['policy'], src['policy_hash'], route_ref)
                        bind_outcome(db, entry['image'], subset, recovered, src['policy_hash'],
                                     receipt_ref, entry['key'], metrics)
                        count += 1
                    db.commit()
                if count != src['generation']['completed']:
                    raise ValueError('Published receipt count differs from producer readback')
            for item in rejections:
                if item['reason'] not in ('teacher_has_no_valid_depth', 'insufficient_teacher_and_anonymous_support'):
                    raise ValueError('Unknown teacher rejection reason')
                bind_outcome(db, item['image'], 'remainder', recovered, fresh['policy_hash'],
                             rejection_ref, reason=item['reason'])
            if db.execute('SELECT COUNT(*) FROM images WHERE outcome IS NULL').fetchone()[0]:
                raise ValueError('Not every routed original RGB has exactly one outcome')
            observed = dict(db.execute('SELECT substr(scene_group,11),COUNT(*) FROM images GROUP BY scene_group'))
            if observed != route['scene_counts']:
                raise ValueError('Per-scene outcomes differ from the routed native review scope')
            groups, selected = [], {}
            for scene in sorted(route['scene_counts']):
                group = 'scannetpp:' + scene
                values = db.execute("SELECT support,p01_ratio,p99_ratio,max_ratio FROM images "
                    "WHERE scene_group=? AND outcome='teacher_candidate' ORDER BY ordinal", (group,)).fetchall()
                groups.append(dict(group=group, original_images=route['scene_counts'][scene], **statistics(values)))
                for order, reason in (('support ASC','least_support'),('p99_ratio DESC','largest_relative_p99')):
                    for ordinal, in db.execute("SELECT ordinal FROM images WHERE scene_group=? "
                            "AND outcome='teacher_candidate' ORDER BY " + order + ',ordinal LIMIT 2', (group,)):
                        selected.setdefault(ordinal, []).append(reason)
            candidates = []
            for ordinal, reasons in sorted(selected.items()):
                image, checksum, group, policy, key, proof, support, low, high, maximum = db.execute(
                    'SELECT image,sha256,scene_group,policy,entry_key,evidence_json,support,p01_ratio,p99_ratio,max_ratio '
                    'FROM images WHERE ordinal=?', (ordinal,)).fetchone()
                candidates.append(dict(corpus_ordinal=ordinal, image=image, sha256=checksum, group=group,
                    policy_sha256=policy,key=key,receipt=json.loads(proof),selection_reasons=reasons,
                    valid_fraction=support,p01_over_median=low,p99_over_median=high,max_over_median=maximum))
            counts = dict(db.execute('SELECT outcome,COUNT(*) FROM images GROUP BY outcome'))
            policy_counts = dict(db.execute("SELECT policy,COUNT(*) FROM images WHERE outcome='teacher_candidate' GROUP BY policy"))
            summary = statistics(db.execute("SELECT support,p01_ratio,p99_ratio,max_ratio FROM images "
                                             "WHERE outcome='teacher_candidate' ORDER BY ordinal").fetchall())
            db.commit()
            if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('Outcome index integrity check failed')
        for ref in evidence:
            checked_path(ref)
        shutil.copyfile(local, output / 'outcomes.sqlite.incoming')
        (output / 'outcomes.sqlite.incoming').rename(output / 'outcomes.sqlite')
    publish_json(output / 'strata.json', groups)
    publish_json(output / 'review-candidates.json', candidates)
    report = dict(stage='teacher_outcomes_accounted_not_accepted', dataset='scannetpp_dslr',completed_at=time.time(),
        expected_scope=len(rows),outcome_counts=counts,scenes=len(groups),candidate_policy_counts=policy_counts,
        candidate_images=counts.get('teacher_candidate',0),rejected_images=len(rejections),
        rejection_reasons=dict(Counter(x['reason'] for x in rejections)),relative_statistics=summary,
        index=file_ref(output / 'outcomes.sqlite'),strata=file_ref(output / 'strata.json'),
        review_candidates=file_ref(output / 'review-candidates.json'),review_candidates_count=len(candidates),
        manifest_sha256=config['manifest_sha256'],routing=route_ref,evidence=evidence,
        sources=[dict(name=s['name'],policy_sha256=s['policy_hash'],**s['refs']) for s in sources],
        producer_array_readback_complete=True,receipt_metadata_only=True,independent_array_readback_performed=False,
        pack_payload_hashes_rechecked=False,native_source_flags_cleared=False,source_quality_accepted=False,
        training_admission=False,optimizer_updates=0,automatic_exclusions=0,model_inference_performed=False,
        audit_code=file_ref(Path(__file__)),metrics_code=file_ref(Path(__file__).with_name('audit_irs_teacher_outcomes.py')))
    publish_json(output / 'report.json',report)
    print(json.dumps({k:report[k] for k in ('stage','outcome_counts','scenes','review_candidates_count')}),flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--attempt',required=True)
    args = parser.parse_args()
    if Path(args.attempt).name != args.attempt:
        parser.error('Use a new attempt basename')
    run(json.loads(args.config.read_text()), args.attempt)
