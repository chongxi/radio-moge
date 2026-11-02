"""Reconcile native Hypersim and review-teacher outcomes without accepting labels.

All frozen inputs receive one proposed route. Native flags remain in the record;
teacher replacements never create additional RGB identities. This reads bound
receipt metadata after producer payload readback, not a second array audit.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

from audit_irs_teacher_outcomes import teacher_metrics
from radio_moge.corpus_admission import checked_json, checked_path, file_ref, readonly_db
from radio_moge.geometry_cache import canonical_json, publish_json
from radio_moge.native_teacher_scope import review_scope, require_visual_preflight
from radio_moge.teacher_rgb_review import REASON, reviewed_rgb_exclusions


def bind_review_outcome(pending, bindings, row, outcomes, value):
    ordinal = row.get('corpus_ordinal')
    if ordinal not in pending or row != pending[ordinal] or ordinal in outcomes:
        raise ValueError('Changed, outside-scope or duplicate review-teacher RGB outcome')
    native = bindings[ordinal]
    outcomes[ordinal] = dict(corpus_ordinal=ordinal, image=row['image'], sha256=row['sha256'],
        group=row['group'], native_outcome=native['native_outcome'],
        native_reason=native['native_reason'], **value)
    del pending[ordinal]


def reconcile_review(rows, bindings, entries, rejections, excluded, policy_hash, revision,
                     minimum_support, rgb_review_ref, rejection_ref):
    pending = {r['corpus_ordinal']: r for r in rows}
    if len(pending) != len(rows) or set(pending) != set(bindings) or not set(excluded) <= set(pending):
        raise ValueError('Review scope identities differ')
    outcomes, keys = {}, set()
    for entry, receipt_ref in entries:
        row, meta = entry['image'], entry['geometry_metadata']
        ordinal = row.get('corpus_ordinal')
        if ordinal not in bindings or ordinal in excluded or entry['key'] in keys:
            raise ValueError('Unreviewed or excluded teacher candidate identity')
        native = bindings[ordinal]
        support, low, high, maximum = teacher_metrics(entry, policy_hash, revision, 'hypersim')
        if (meta.get('native_source_review_unresolved') is not True
                or meta.get('native_outcome') != native['native_outcome']
                or meta.get('native_reason') != native['native_reason']
                or support < minimum_support):
            raise ValueError('Teacher candidate changed native flags or lacks required support')
        keys.add(entry['key'])
        bind_review_outcome(pending, bindings, row, outcomes, dict(route='teacher_candidate',
            entry_key=entry['key'], evidence=receipt_ref, valid_fraction=support,
            p01_over_median=low, p99_over_median=high, max_over_median=maximum))
    for item in rejections:
        row, reason = item['image'], item['reason']
        ordinal = row.get('corpus_ordinal')
        if ordinal in excluded:
            if (reason != REASON or item.get('teacher_inference_performed') is not False
                    or item.get('constant_rgb') != excluded[ordinal]['channel_min']
                    or item.get('rgb_input_review') != rgb_review_ref):
                raise ValueError('Reviewed blank RGB rejection differs from its evidence')
            kind = 'reviewed_input_exclusion'
        elif reason in ('teacher_has_no_valid_depth', 'insufficient_teacher_support'):
            kind = 'teacher_rejection_needs_review'
        else:
            raise ValueError('Unreviewed teacher rejection reason')
        bind_review_outcome(pending, bindings, row, outcomes,
                            dict(route=kind, reason=reason, evidence=rejection_ref))
    if pending:
        raise ValueError('Not every unresolved native image has a teacher outcome')
    return outcomes


def run(config, cache_attempt, preflight, attempt):
    root = Path(config['root'])
    source_ref = file_ref(root / 'inventory.json')
    source = checked_json(source_ref)
    inventory_ref = dict(path=str(root / 'inventory.sqlite'), sha256=source['inventory_sha256'])
    checked_path(inventory_ref)
    route_ref = file_ref(root / 'hypersim-review-teacher-routing-20261102.json')
    route = checked_json(route_ref)
    if (source['stage'] != 'inventory_complete' or source['manifest_sha256'] != config['manifest_sha256']
            or source['dataset_counts'] != config['expected_counts']):
        raise ValueError('Complete matching frozen inventory required')
    rows, bindings = review_scope(root, route, config)
    native = checked_json(route['native_outcomes'])
    rgb_ref = file_ref(root / 'hypersim-review-rgb-degeneracy-01/review.json')
    excluded = reviewed_rgb_exclusions(rgb_ref, rows, route['native_outcomes'])
    cache = root / cache_attempt
    gen_ref, packs_ref, rejection_ref, policy_ref = [file_ref(cache / p) for p in
        ('report.json', 'packs/packs.json', 'rejections.json', 'packs/policy.json')]
    gen, packs, rejections, policy = [checked_json(r) for r in
        (gen_ref, packs_ref, rejection_ref, policy_ref)]
    policy_hash = hashlib.sha256(canonical_json(policy)).hexdigest()
    if (gen['stage'] != 'complete' or gen['mode'] != 'generate'
            or gen['training_admission'] is not False or gen['training_started'] is not False
            or gen['optimizer_updates'] != 0 or gen['teacher_frozen'] is not True
            or gen['completed'] != gen['verified_cache_members'] or packs['count'] != gen['completed']
            or gen['completed'] + gen['rejected'] != len(rows) or gen['expected_scope'] != len(rows)
            or gen['rejected'] != len(rejections) or gen['policy_sha256'] != policy_hash
            or packs['policy_sha256'] != policy_hash or policy['dataset'] != 'hypersim'
            or policy['label_kind'] != 'teacher_pseudo' or policy['native_metric_gt'] is not False
            or policy['teacher'] != config['teacher'] or policy['manifest_sha256'] != config['manifest_sha256']
            or policy['candidate_routing_sha256'] != route_ref['sha256']
            or policy['rgb_input_review'] != rgb_ref or policy['input_exclusion_reason'] != REASON):
        raise ValueError('Complete matching teacher producer readback and policy required')
    require_visual_preflight(root / preflight / 'report.json', policy_hash, route['scene_counts'])

    def entries():
        for item in packs['packs']:
            receipt_path = cache / 'packs' / item['receipt']
            if receipt_path.parent != cache / 'packs':
                raise ValueError('Invalid receipt path')
            receipt_ref = dict(path=str(receipt_path), sha256=item['sha256'])
            receipt = checked_json(receipt_ref)
            tar = receipt_path.parent / receipt['tar']
            if (tar.parent != receipt_path.parent or not tar.is_file()
                    or tar.stat().st_size != receipt['bytes'] or receipt['policy_sha256'] != policy_hash
                    or len(receipt['entries']) != item['count']):
                raise ValueError('Receipt policy, size or count differs')
            for entry in receipt['entries']:
                yield entry, receipt_ref

    outcomes = reconcile_review(rows, bindings, entries(), rejections, excluded, policy_hash,
        config['teacher']['revision'], policy['minimum_valid_fraction'], rgb_ref, rejection_ref)
    teacher_counts = Counter(v['route'] for v in outcomes.values())
    if (teacher_counts['teacher_candidate'] != gen['completed']
            or teacher_counts['reviewed_input_exclusion'] != gen['reviewed_input_exclusions']
            or teacher_counts['reviewed_input_exclusion'] != len(excluded)):
        raise ValueError('Teacher outcomes do not reconcile with producer counters')
    output = root / attempt
    output.mkdir(exist_ok=False)
    incoming = output / 'routes.jsonl.incoming'
    counts, native_counts, transitions, scene_counts = Counter(), Counter(), Counter(), Counter()
    with incoming.open('x') as file, readonly_db(native['index']['path']) as db:
        for ordinal, image, checksum, group, original, kind, reason, evidence, key in db.execute(
                'SELECT ordinal,image,sha256,scene_group,row_json,outcome,reason,evidence_json,entry_key '
                'FROM images ORDER BY ordinal'):
            native_counts[kind] += 1
            if kind == 'unflagged_cache':
                if ordinal in outcomes:
                    raise ValueError('Teacher attempted to replace an unflagged native image')
                value = dict(corpus_ordinal=ordinal, image=image, sha256=checksum, group=group,
                    route='native_candidate', native_outcome=kind, native_reason=reason,
                    entry_key=key, evidence=json.loads(evidence))
            else:
                value = outcomes.pop(ordinal)
                row = json.loads(original)
                if (value['image'] != row['image'] or value['sha256'] != checksum
                        or value['group'] != group or value['native_outcome'] != kind
                        or value['native_reason'] != reason):
                    raise ValueError('Teacher route differs from the original native outcome')
            counts[value['route']] += 1
            transitions[kind + ' -> ' + value['route']] += 1
            scene_counts[group] += 1
            file.write(canonical_json(value).decode() + '\n')
    if (outcomes or sum(counts.values()) != config['expected_counts']['hypersim']
            or dict(native_counts) != native['outcome_counts']):
        raise ValueError('Combined routes do not cover every frozen Hypersim input once')
    for ref in (source_ref, route_ref, rgb_ref, gen_ref, packs_ref, rejection_ref, policy_ref):
        checked_path(ref)
    checked_path(native['index'])
    os.rename(incoming, output / 'routes.jsonl')
    report = dict(stage='source_routes_accounted_not_accepted', dataset='hypersim',
        completed_at=time.time(), expected_scope=sum(counts.values()), route_counts=dict(counts),
        native_outcome_counts=dict(native_counts), transitions=dict(transitions), scenes=len(scene_counts),
        teacher_scope=len(rows), teacher_rejections_needing_review=teacher_counts['teacher_rejection_needs_review'],
        manifest_sha256=config['manifest_sha256'], policy_sha256=policy_hash,
        routes=file_ref(output / 'routes.jsonl'), native_outcomes=route['native_outcomes'],
        generation_report=gen_ref, packs=packs_ref, rejections=rejection_ref, policy=policy_ref,
        inventory=inventory_ref, source_report=source_ref, candidate_routing=route_ref, rgb_input_review=rgb_ref,
        preflight_visual_review=file_ref(root / preflight / 'visual-review.json'),
        producer_array_readback_complete=True, independent_array_readback_performed=False,
        pack_payload_hashes_rechecked=False, receipt_metadata_only=True,
        source_quality_accepted=False, training_admission=False, optimizer_updates=0,
        native_source_flags_cleared=False, audit_code=file_ref(Path(__file__)),
        receipt_helper=file_ref(Path(__file__).with_name('audit_irs_teacher_outcomes.py')))
    publish_json(output / 'report.json', report)
    print(json.dumps({k: report[k] for k in ('stage', 'route_counts', 'transitions')}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--cache-attempt', default='hypersim-review-teacher-cache-01')
    parser.add_argument('--preflight', default='hypersim-review-teacher-preflight-02')
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if any(Path(n).name != n for n in (args.cache_attempt, args.preflight, args.attempt)):
        parser.error('Use attempt basenames')
    run(json.loads(args.config.read_text()), args.cache_attempt, args.preflight, args.attempt)
