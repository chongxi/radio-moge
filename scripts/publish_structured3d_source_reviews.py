"""Bind completed Structured3D reviews to both immutable native policies.

Reuse the full outcome audit and actual source/visual checks. This records
source eligibility, not crop coverage, corpus admission or permission to train.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time

from publish_hypersim_native_source_review import validate_exclusion_identity
from radio_moge.corpus_admission import (SOURCE_CHECKS, checked_candidate_quarantine,
    checked_json, checked_path, file_ref, readonly_db, validate_source_review)
from radio_moge.geometry_cache import canonical_json, publish_json


def source_disposition(ordinal, row, outcome, reason, key, metrics, exclusions, flags):
    decision, flag = exclusions.get(ordinal), flags.get(key)
    if decision is not None:
        validate_exclusion_identity(decision, row, ordinal)
        if decision.get('scope') != 'geometry_only' or decision.get('rgb_replay_excluded') is not False:
            raise ValueError('Structured3D disposition must retain its original RGB')
    if outcome == 'rejected_needs_disposition':
        if decision is None or decision.get('native_reason') != reason or key is not None:
            raise ValueError('Native rejection lacks its exact reviewed disposition')
        return 'excluded_geometry'
    if outcome != 'native_candidate':
        raise ValueError('Unknown native outcome')
    if flag is not None:
        if (flag['corpus_ordinal'] != ordinal or decision is None
                or any(flag[k] != row[k] for k in ('image', 'sha256', 'group'))):
            raise ValueError('Quarantined candidate lacks a matching RGB exclusion')
        return 'excluded_geometry'
    if decision is not None:
        raise ValueError('Excluded native candidate requires an exact policy quarantine flag')
    if (not key or not all(math.isfinite(metrics[k]) for k in
                          ('valid_fraction', 'layout_verified_pixels', 'median_depth_m'))
            or not .05 <= metrics['valid_fraction'] <= 1
            or metrics['layout_verified_pixels'] < 100 or metrics['median_depth_m'] <= 0):
        raise ValueError('Audited native support or geometry differs')
    return 'native_source_eligible'


def run(root, output):
    names = ('structured3d-outcomes-01/report.json',
        'structured3d-rejection-decisions-01/report.json',
        'structured3d-scene00103-decisions-01/report.json',
        'structured3d-camera-census-01/report.json',
        'structured3d-cache-preflight-02/report.json',
        'structured3d-cache-preflight-02/visual-review-01.json',
        'structured3d-cache-preflight-02/array-parity-review-01.json',
        'structured3d-failure-regression-01/report.json')
    refs = {name: file_ref(root / name) for name in names}
    audit, decisions, external, cameras, preflight, visual, parity, regression = (
        checked_json(refs[name]) for name in names)
    if (refs[names[0]]['sha256'] != '5fa30bc41e30aa60c84a322f29cfb04ef6deb3c6a5bfe4db79ed5e962c5b4e71'
            or refs[names[1]]['sha256'] != '8c97135b4c6bc8dbba1b1d78fb95b04d1a1f89aa78e2f8c463df429248e13248'
            or audit['expected_scope'] != 134749 or audit['producer_array_readback_complete'] is not True
            or audit['cached_source_flags'] != 0 or audit['rejected_source_flags'] != 686
            or decisions['source_native_candidates_after_flags'] != 133563
            or decisions['source_geometry_exclusions'] != 1186
            or decisions['rejections_without_disposition'] != 0
            or cameras['stage'] != 'complete' or cameras['completed'] != 134749 or cameras['rejected'] != 0
            or preflight['stage'] != 'complete' or preflight['passed'] is not True
            or preflight['verified_cache_members'] != 58 or preflight['rejected'] != 2
            or visual['stage'] != 'preflight_visual_review_complete'
            or visual['numerical_preflight'] != refs[names[4]]
            or parity['all_depth_mask_intrinsics_bitwise_equal'] is not True
            or parity['matching_original_entries'] != 56
            or regression['stage'] != 'diagnostic_complete'
            or regression['parser_sha256'] != 'fe8dfe9b6ee489c31a2ce216bb94760bbf8034699f000f11202080f37bb6bea3'):
        raise ValueError('Completed Structured3D source evidence differs')
    # These references bind the previously inspected 12 candidate and 24 rejection pages.
    more_refs = [external['visual_review'], decisions['visual_review'], decisions['reproduction'],
                 decisions['known_geometry_exclusions'], audit['index']]
    for ref in more_refs:
        checked_path(ref)
    if (checked_json(external['visual_review'])['all_rendered_pages_inspected'] is not True
            or checked_json(decisions['visual_review'])['stage'] != 'rejection_visual_review_complete'):
        raise ValueError('Completed visual reviews required')
    exclusions = {}
    for line in checked_path(decisions['known_geometry_exclusions']).read_text().splitlines():
        row = json.loads(line)
        if row['dataset'] == 'structured3d':
            if row['corpus_ordinal'] in exclusions:
                raise ValueError('Duplicate source exclusion')
            exclusions[row['corpus_ordinal']] = row
    if len(exclusions) != 1186:
        raise ValueError('Incomplete source exclusions')
    sources, all_flags = {}, {}
    audited_refs = {canonical_json(ref) for ref in audit['evidence']}
    for name in ('structured3d-recovered-01', 'structured3d-native-cache-02'):
        generation_ref = file_ref(root / name / 'report.json')
        packs_ref = file_ref(root / name / 'packs/packs.json')
        policy_ref = file_ref(root / name / 'packs/policy.json')
        if any(canonical_json(ref) not in audited_refs for ref in (generation_ref, packs_ref, policy_ref)):
            raise ValueError('Source is outside the completed outcome audit')
        gen, packs, policy = (checked_json(ref) for ref in (generation_ref, packs_ref, policy_ref))
        policy_hash = hashlib.sha256(canonical_json(policy)).hexdigest()
        if (gen['stage'] != 'complete' or gen['mode'] != 'generate'
                or gen['completed'] != gen['verified_cache_members'] or gen['completed'] != packs['count']
                or gen['policy_sha256'] != policy_hash or packs['policy_sha256'] != policy_hash
                or gen['completed'] != audit['candidate_policy_counts'][policy_hash]
                or policy['manifest_sha256'] != audit['manifest_sha256']
                or policy['native_holes_filled'] is not False or gen['optimizer_updates'] != 0):
            raise ValueError('Source full readback or policy changed')
        quarantine = external['registers'][name]['register']
        flags = checked_candidate_quarantine(quarantine, 'structured3d', policy_hash,
                                              audit['manifest_sha256'], packs_ref)
        if len(flags) != external['registers'][name]['flags'] or set(flags) & set(all_flags):
            raise ValueError('Quarantine policies overlap or changed')
        all_flags.update(flags)
        sources[policy_hash] = dict(name=name, generation=generation_ref, packs=packs_ref,
            policy=policy_ref, quarantine=quarantine, flags=flags, count=packs['count'])
    if len(all_flags) != 9:
        raise ValueError('Exactly nine external candidates must remain quarantined')
    output.mkdir(exist_ok=False)
    counts, by_policy, groups, seen_exclusions, seen_flags = Counter(), Counter(), {}, set(), set()
    incoming = output / 'source-routes.jsonl.incoming'
    with readonly_db(audit['index']['path']) as db, incoming.open('x') as file:
        for ordinal, image, checksum, group, encoded, outcome, reason, policy, proof, key, stats in db.execute(
                'SELECT ordinal,image,sha256,scene_group,row_json,outcome,reason,policy,evidence_json,'
                'entry_key,metrics_json FROM images ORDER BY ordinal'):
            row = json.loads(encoded)
            if (policy not in sources or row['image'] != image or row['sha256'] != checksum
                    or row['group'] != group or row['dataset'] != 'structured3d' or row['split'] != 'train'):
                raise ValueError('Audited frozen RGB identity or policy differs')
            disposition = source_disposition(ordinal, row, outcome, reason, key, json.loads(stats),
                                              exclusions, sources[policy]['flags'])
            counts[disposition] += 1
            by_policy[policy, disposition] += 1
            if disposition == 'native_source_eligible':
                groups.setdefault(policy, set()).add(group)
            if ordinal in exclusions:
                seen_exclusions.add(ordinal)
            if key in all_flags:
                seen_flags.add(key)
            file.write(canonical_json(dict(corpus_ordinal=ordinal, image=image, sha256=checksum,
                group=group, policy_sha256=policy, entry_key=key, native_outcome=outcome,
                native_reason=reason, source_disposition=disposition, source_flags_cleared=False,
                rgb_replay_excluded=False, candidate_evidence=json.loads(proof),
                decision_evidence=decisions['known_geometry_exclusions'] if ordinal in exclusions
                    else refs[names[0]])).decode() + '\n')
        file.flush()
        os.fsync(file.fileno())
    if (counts != Counter(native_source_eligible=133563, excluded_geometry=1186)
            or seen_exclusions != set(exclusions) or seen_flags != set(all_flags)):
        raise ValueError('Source routes did not reconcile exactly once')
    os.rename(incoming, output / 'source-routes.jsonl')
    documentation = dict(stage='primary_source_review_complete', references=[
        dict(url='https://arxiv.org/abs/1908.00222', finding='Synthetic renderings and structural annotations from professional interior designs; not real sensor measurements.'),
        dict(url='https://github.com/bertjiazheng/Structured3D/blob/master/data_organization.md', finding='Perspective depth is uint16 millimeters with zero missing; distinct panorama format, encoded normals and half-FOV camera angles.'),
        dict(url='https://github.com/bertjiazheng/Structured3D/blob/master/misc/utils.py', finding='Publisher normalizes look/up and constructs cross-product axes. Floating-point intrinsics retain the declared projection origin.')],
        source_consistency='Reproduced camera-Z plane checks discriminate the radial interpretation; renderer geometry and layout are not independent physical calibration.',
        training_admission=False, optimizer_updates=0)
    publish_json(output / 'documentation-review.json', documentation)
    published = []
    for policy, src in sources.items():
        eligible = by_policy[policy, 'native_source_eligible']
        if eligible + len(src['flags']) != src['count']:
            raise ValueError('Policy native candidates do not reconcile')
        review = dict(stage='source_review_complete', decision='eligible_with_flagged_rows_quarantined',
            dataset='structured3d', policy_sha256=policy, manifest_sha256=audit['manifest_sha256'],
            checks=dict.fromkeys(SOURCE_CHECKS, True), candidate_quarantine=src['quarantine'],
            evidence=[*refs.values(), *more_refs, src['generation'], src['packs'], src['policy'],
                file_ref(output / 'documentation-review.json'), file_ref(output / 'source-routes.jsonl')],
            label_kinds=['synthetic_renderer'], supervision='metric_native',
            eligible_candidate_images=eligible, quarantined_candidate_images=len(src['flags']),
            selected_source_groups=len(groups[policy]), original_source_flags_retained=True,
            check_details=dict(documentation_and_paper='Official paper, formats and camera parser reviewed.',
                units_and_camera='134749 camera records checked; camera-Z/mm source plane regressions and 56 bitwise old/new arrays bind the two preserved conversion policies.',
                valid_support='Full producer/recovery payload readback complete; audited original masks meet unchanged support, semantic, normal and layout rules. No filling.',
                geometric_quality='All frozen outcomes audited; 467 errata and 710 exactly reproduced source rejections explicitly excluded. Nine reviewed candidates quarantined by exact policy/member/RGB identity.',
                visual_quality='Reuse eight preflight sheets, three source regressions, all 12 candidate-extreme pages, nine scene RGBs and all 24 rejection pages already inspected.'),
            physical_scale_independently_measured=False, source_flags_cleared=False,
            full_array_readback_repeated=False, completed_outcome_audit_reused=True,
            crop_coverage_accepted=False, canonical_heldout_isolation_accepted=False,
            actual_archived_model_probe_performed=False, training_admission=False, optimizer_updates=0)
        validate_source_review(review, 'structured3d', policy, audit['manifest_sha256'], src['packs'])
        path = output / (src['name'] + '-source-review.json')
        publish_json(path, review)
        published.append(file_ref(path))
    for ref in [*refs.values(), *more_refs]:
        checked_path(ref)
    report = dict(stage='structured3d_source_reviews_and_routes_complete_not_corpus_admission',
        source_reviews=published, source_routes=file_ref(output / 'source-routes.jsonl'),
        source_disposition_counts=dict(counts), original_images=sum(counts.values()),
        native_source_groups=len(set.union(*groups.values())),
        source_policy_counts={src['name']: dict(eligible=by_policy[p, 'native_source_eligible'],
            quarantined=len(src['flags']), excluded=by_policy[p, 'excluded_geometry']) for p, src in sources.items()},
        known_geometry_exclusions=decisions['known_geometry_exclusions'],
        full_array_readback_repeated=False, completed_outcome_audit_reused=True,
        source_flags_cleared=False, crop_coverage_accepted=False, training_admission=False,
        optimizer_updates=0, active_training_changed=False, completed_at=time.time(), code=file_ref(Path(__file__)))
    publish_json(output / 'report.json', report)
    print(json.dumps(dict(source_disposition_counts=dict(counts), source_reviews=published)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
