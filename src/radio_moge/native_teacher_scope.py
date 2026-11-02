"""Bind separate teacher candidates to exactly the unresolved native RGB rows."""
from collections import Counter, defaultdict
import json

from .corpus_admission import checked_json, checked_path, readonly_db


def review_scope(root, route, config):
    if (route['dataset'] != 'hypersim' or route['manifest_sha256'] != config['manifest_sha256']
            or route['label_kind'] != 'teacher_pseudo' or route['metric_calibrated'] is not False
            or route['fill_native_holes'] is not False or route['training_admission'] is not False
            or not route.get('evidence') or route['scope_images'] <= 0):
        raise ValueError('Explicit nonmetric Hypersim candidate routing required')
    audit = checked_json(route['native_outcomes'])
    generation = checked_json(audit['generation_report'])
    if (audit['stage'] != 'native_outcomes_accounted_not_accepted'
            or audit['dataset'] != route['dataset'] or audit['manifest_sha256'] != config['manifest_sha256']
            or audit['expected_scope'] != config['expected_counts']['hypersim']
            or audit['producer_array_readback_complete'] is not True
            or audit['unresolved_images'] != route['scope_images']
            or generation['stage'] != 'complete' or generation['completed'] != generation['verified_cache_members']
            or generation['policy_sha256'] != audit['policy_sha256']):
        raise ValueError('Complete matching native outcomes and producer readback required')
    for ref in route['evidence']:
        checked_path(ref)
    checked_path(audit['index'])
    checked_path(audit['unresolved_manifest'])
    reasons, groups, pending = Counter(), Counter(), {}
    with checked_path(audit['unresolved_manifest']).open() as file:
        for line in file:
            item = json.loads(line)
            ordinal = item['corpus_ordinal']
            if ordinal in pending:
                raise ValueError('Repeated unresolved native identity')
            pending[ordinal] = item
    rows, bindings = [], {}
    with readonly_db(audit['index']['path']) as outcomes, readonly_db(root / 'inventory.sqlite') as inventory:
        for ordinal, image, checksum, group, encoded, outcome, reason in outcomes.execute(
                "SELECT ordinal,image,sha256,scene_group,row_json,outcome,reason FROM images "
                "WHERE outcome!='unflagged_cache' ORDER BY ordinal"):
            if outcome not in ('flagged_cache', 'rejected'):
                raise ValueError('Only unresolved native rows belong in teacher fallback')
            row = json.loads(encoded)
            frozen = inventory.execute('SELECT image,row_json FROM images WHERE ordinal=?', (ordinal,)).fetchone()
            if frozen is None:
                raise ValueError('Native outcome is outside frozen inventory')
            original = json.loads(frozen[1])
            original['image'] = frozen[0]
            item = pending.pop(ordinal, None)
            if (row != original or row['image'] != image or row['sha256'] != checksum or row['group'] != group
                    or row['dataset'] != 'hypersim' or row['split'] != 'train'
                    or row.get('image_rotation_cw', 0) != 0
                    or item is None or item['image'] != image or item['sha256'] != checksum
                    or item['native_outcome'] != outcome or item['native_reason'] != reason):
                raise ValueError('Native review manifest and frozen RGB identity differ')
            row['corpus_ordinal'] = ordinal
            rows.append(row)
            bindings[ordinal] = item
            groups[group] += 1
            reasons[reason] += 1
    if (pending or len(rows) != route['scope_images'] or dict(groups) != audit['unresolved_scene_counts']
            or dict(groups) != route['scene_counts'] or dict(reasons) != route['reason_counts']):
        raise ValueError('Explicit fallback scope does not cover every unresolved native row exactly once')
    return rows, bindings


def select_preflight(rows, bindings):
    """First/last RGB per scene and native failure class; no geometry-fitted selection."""
    bins = defaultdict(list)
    for row in rows:
        bins[(row['group'], bindings[row['corpus_ordinal']]['native_reason'])].append(row)
    selected = {}
    for values in bins.values():
        ordered = sorted(values, key=lambda r: r['corpus_ordinal'])
        for row in (ordered[0], ordered[-1]):
            selected[row['corpus_ordinal']] = row
    return [selected[k] for k in sorted(selected)]


def require_visual_preflight(previous_path, policy_hash, groups):
    """A numeric preflight alone cannot authorize this separate bulk producer."""
    from .geometry_cache import sha_file
    previous = json.loads(previous_path.read_text())
    review = json.loads((previous_path.parent / 'visual-review.json').read_text())
    if (previous['stage'] != 'complete' or previous.get('passed') is not True
            or previous['policy_sha256'] != policy_hash or review.get('passed') is not True
            or review.get('stage') != 'visual_review_complete'
            or review.get('preflight_report_sha256') != sha_file(previous_path)
            or review.get('policy_sha256') != policy_hash
            or set(review.get('reviewed_scenes', [])) != set(groups) or not review.get('evidence')):
        raise ValueError('Matching numeric and all-scene visual preflight required')
    for ref in review['evidence']:
        checked_path(ref)
