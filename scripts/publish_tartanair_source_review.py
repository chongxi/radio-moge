"""Bind completed TartanAir source/crop evidence without admitting a new run.

Reuse producer readback, full outcome accounting and the actually inspected
candidate/rejection pages. A synthetic-source eligibility record does not prove
independent physical accuracy or change any accepted training inputs.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import time

from download_tartanair_variety import family
from publish_coco_nyu_source_reviews import write_lines
from publish_hypersim_native_source_review import validate_exclusion_identity
from radio_moge.corpus_admission import (SOURCE_CHECKS, checked_json, checked_path,
    file_ref, readonly_db, validate_source_review)
from radio_moge.geometry_cache import HEX64, canonical_json, publish_json


PINNED = {
    'tartanair-outcomes-01/report.json':
        '630e082e1e02cd9e58e5f7edc14e355c2744ebfa78b11a0e4ea261b488fcac20',
    'tartanair-outcomes-01/rejection-closure-01.json':
        'c28eaec3bdfd027b71a9d553cc21229a184d2d8193b7ba0c9aa1f26a33cf7b0a',
    'tartanair-candidate-visuals-01/index.json':
        '4c4f14c3f126dfa701cc832a2d69e8c030e05b2501ea340df1b7d5cd422a1eeb',
    'tartanair-candidate-visuals-01/visual-review-01.json':
        '23045e99d4f8405adb60c769ebf0c6ebc430850024d10701248dbde7bc9c9215',
    'tartanair-candidate-crops-01/report.json':
        '2746a2ddf41dcac0c8ff98eb744e40505397c0d0d3796e0a26e367e5c1718c7d',
    'tartanair-sparse-crops-01/visual-review-01.json':
        '88bbd496acba2f354e6517a692bdce537a795772317b74287093bef374f123ec',
    'tartanair-cache-preflight-01/report.json':
        'd248321c8a8a28919f497902d3f564c4998052a7b5ffba0c5461ac81606c7d58',
    'tartanair-cache-preflight-01/visual-review-20261102T1045Z.json':
        '95bbae77e7381d5c43903b5eedbfe26830836e36c8c7cdd18577d3f3e7624b9b',
}


def source_disposition(ordinal, row, outcome, reason, key, metrics, proof, exclusions, closed):
    if row['dataset'] != 'tartanair' or row['split'] != 'train':
        raise ValueError('Frozen source or split changed')
    support = metrics.get('support')
    if (not isinstance(support, (int, float)) or isinstance(support, bool)
            or not math.isfinite(support) or not 0 <= support <= 1):
        raise ValueError('Audited native support differs')
    decision = exclusions.get(ordinal)
    if outcome == 'rejected':
        if (ordinal not in closed or decision is None or key is not None
                or reason != 'insufficient_stereo_verified_support' or support >= .05):
            raise ValueError('Rejection lacks its exact completed disposition')
        validate_exclusion_identity(decision, row, ordinal)
        if (decision.get('scope') not in ('geometry_only', 'geometry_supervision')
                or decision.get('rgb_replay_excluded') is not False
                or proof not in decision.get('evidence', [])):
            raise ValueError('Reviewed geometry-only rejection evidence changed')
        # Legacy decisions omit redundant statistics; the exact per-image
        # rejection ref and completed closure bind them, rather than guessing.
        if ('valid_fraction' in decision and decision['valid_fraction'] != support
                or 'minimum_image_valid_fraction' in decision
                and decision['minimum_image_valid_fraction'] != .05):
            raise ValueError('Repeated rejection statistics differ')
        return 'excluded_geometry'
    if (outcome != 'native_candidate' or reason != 'candidate_pending_source_quality_review'
            or decision is not None or ordinal in closed or not isinstance(key, str)
            or not HEX64.fullmatch(key) or support < .05):
        raise ValueError('Unknown, excluded or flagged native candidate')
    for name, lo, hi in (('task_support', support, 1), ('median_depth', .1 - 1e-7, 200),
                         ('p99_depth', .1 - 1e-7, 200), ('rgb_mae', 0, 1),
                         ('max_stereo_error', 0, 1.010001)):
        value = metrics.get(name)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or not lo <= value <= hi):
            raise ValueError('Audited native range or stereo statistics differ')
    if metrics['p99_depth'] < metrics['median_depth']:
        raise ValueError('Audited depth quantiles differ')
    return 'native_source_eligible'


def check_candidate_scope(selection, rendered, visual, crops, views, sparse):
    chosen = {r['corpus_ordinal']: r for r in selection}
    actual = {r['corpus_ordinal']: r for r in rendered['records']}
    if (len(chosen) != len(selection) or len(actual) != len(rendered['records'])
            or set(chosen) != set(actual) or visual['reviewed_images'] != len(chosen)
            or visual['all_rendered_images_viewed'] is not True
            or visual['pages_viewed'] != rendered['pages']
            or rendered['arrays_camera_mask_and_statistics_reproduced_exactly'] is not True
            or rendered['source_member_sha_crc_verified'] is not True
            or rendered['all_selected_views_included'] is not True
            or crops['actual_production_loader_used'] is not True
            or crops['original_mask_unchanged'] is not True
            or crops['all_observations_unique_original_samples'] is not True
            or crops['images'] != len(chosen) or crops['views'] != len(views)):
        raise ValueError('Completed actual source/visual/crop scope differs')
    for ordinal, row in actual.items():
        if (row['entry_key'] != chosen[ordinal]['entry_key'] or row['receipt'] != chosen[ordinal]['receipt']
                or row['arrays_camera_mask_and_statistics_reproduced_exactly'] is not True):
            raise ValueError('Selected native candidate identity differs')
    counts, combinations, sparse_rows = Counter(), {}, []
    for row in views:
        ordinal = row['corpus_ordinal']
        if (ordinal not in actual or row['entry_key'] != actual[ordinal]['entry_key']
                or row['receipt'] != actual[ordinal]['receipt']
                or any(row[k] != actual[ordinal]['image'][k] for k in ('image', 'sha256'))
                or type(row['selected_observations']) is not int or row['selected_observations'] < 0
                or row['sparse_geometry_skipped'] is not (row['selected_observations'] < 32)
                or not math.isfinite(row['native_to_target_max_relative_error'])
                or not 0 <= row['native_to_target_max_relative_error'] <= 1e-6):
            raise ValueError('Actual crop identity, observation mapping or skip rule differs')
        counts[ordinal] += 1
        combinations.setdefault(ordinal, set()).add((row['augmented'], row['seed'], row['flipped']))
        if row['sparse_geometry_skipped']:
            sparse_rows.append(row)
    expected = {(False, 0, False), (True, 43, False), (True, 0, True)}
    if (set(counts) != set(chosen) or any(n != 3 for n in counts.values())
            or any(v != expected for v in combinations.values())
            or sparse['all_rendered_pages_inspected'] is not True
            or sparse['minimum_observations_unchanged'] != 32
            or sparse['new_geometry_exclusions'] != 0 or sparse['rgb_excluded'] is not False
            or len(sparse_rows) != crops['sparse_views']
            or {r['corpus_ordinal'] for r in sparse_rows} != {sparse['corpus_ordinal']}
            or [r['selected_observations'] for r in views
                if r['corpus_ordinal'] == sparse['corpus_ordinal']] != sparse['selected_counts']):
        raise ValueError('Sparse crop disposition or exercised view cases differ')


def run(root, output):
    refs = {name: dict(path=str(root / name), sha256=sha) for name, sha in PINNED.items()}
    audit, closure, rendered, visual, crops, sparse, preflight, old_visual = (
        checked_json(ref) for ref in refs.values())
    names = list(refs)
    if (audit['stage'] != 'native_outcomes_accounted_not_accepted'
            or audit['dataset'] != 'tartanair' or audit['expected_scope'] != 229924
            or audit['outcome_counts'] != dict(native_candidate=229862, rejected=62)
            or audit['producer_array_readback_complete'] is not True
            or closure['outcomes'] != refs[names[0]] or closure['generation_report'] != audit['generation_report']
            or closure['all_final_rejections_exactly_equal_bound_retained_per_image_rejection'] is not True
            or closure['all_rgb_retained'] is not True or closure['new_exclusions'] != 0
            or closure['final_rejected_images'] != 62
            or rendered['audit'] != refs[names[0]] or rendered['selection'] != audit['review_candidates']
            or visual['evidence'] != refs[names[2]] or visual['new_geometry_exclusions'] != 0
            or crops['visual_review'] != refs[names[3]] or crops['selection'] != audit['review_candidates']
            or sparse['original_probe'] != refs[names[4]]
            or preflight['stage'] != 'complete' or preflight['passed'] is not True
            or preflight['verified_cache_members'] != 47 or preflight['rejected'] != 1
            or preflight['policy_sha256'] != audit['policy_sha256']
            or old_visual['policy_sha256'] != audit['policy_sha256']
            or len(old_visual['contact_sheets_viewed']) != 6):
        raise ValueError('Pinned completed source evidence differs')
    selection, views = checked_json(audit['review_candidates']), checked_json(crops['views_file'])
    if len(selection) != 113 or len(views) != 339 or crops['source_bundles'] != 24 or crops['groups'] != 11:
        raise ValueError('Bounded source/crop scope differs')
    check_candidate_scope(selection, rendered, visual, crops, views, sparse)
    for ref in [*rendered['pages'], sparse['evidence'], sparse['page']]:
        checked_path(ref)
    gen, packs, policy = (checked_json(audit[k]) for k in ('generation_report', 'packs', 'policy'))
    if (gen['stage'] != 'complete' or gen['mode'] != 'generate' or gen['completed'] != 229862
            or gen['verified_cache_members'] != gen['completed'] or gen['rejected'] != 62
            or gen['training_started'] is not False or gen['training_admission'] is not False
            or gen['optimizer_updates'] != 0 or packs['count'] != gen['completed']
            or gen['policy_sha256'] != audit['policy_sha256'] or packs['policy_sha256'] != audit['policy_sha256']
            or hashlib.sha256(canonical_json(policy)).hexdigest() != audit['policy_sha256']
            or policy['manifest_sha256'] != audit['manifest_sha256']
            or policy['label_kind'] != 'synthetic_renderer' or policy['native_holes_filled'] is not False
            or policy['independent_physical_depth_accuracy_checked'] is not False):
        raise ValueError('Complete matching native producer/policy required')
    decisions_ref = closure['known_exclusions']
    decisions = [json.loads(s) for s in checked_path(decisions_ref).read_text().splitlines()]
    exclusions = {r['corpus_ordinal']: r for r in decisions if r['dataset'] == 'tartanair'}
    closed = set(closure['unique_ordinals'])
    if (len(exclusions) != 62 or len(closed) != len(closure['unique_ordinals']) or set(exclusions) != closed
            or len(decisions) != 4101 or [r['corpus_ordinal'] for r in decisions if r.get('rgb_replay_excluded')] != [210812]):
        raise ValueError('Frozen source/global exclusions differ')
    documentation_ref = dict(path=str(root / 'code-tartanair-v1/docs/geometry-corpus-acceptance.md'),
        sha256='6bf44feea0e5dee7e807db607d80acf6d1145cb8b350a4d624cfc6039432454c')
    checked_path(documentation_ref)
    receipts = {str(Path(audit['packs']['path']).parent / p['receipt']): p['sha256'] for p in packs['packs']}
    output.mkdir(exist_ok=False)
    counts, groups, bundles, environments, trajectories, seen_exclusions = Counter(), set(), set(), set(), set(), set()
    with tempfile.TemporaryDirectory(prefix='tartanair-source-review-') as temporary:
        local_index = Path(temporary) / 'outcomes.sqlite'
        shutil.copyfile(audit['index']['path'], local_index)
        checked_path(dict(path=str(local_index), sha256=audit['index']['sha256']))

        def routes():
            with readonly_db(local_index) as db:
                for ordinal, image, checksum, group, encoded, kind, reason, key, stats, evidence in db.execute(
                        'SELECT ordinal,image,sha256,scene_group,row_json,outcome,reason,entry_key,'
                        'details_json,evidence_json FROM images ORDER BY ordinal'):
                    row, metrics, proof = json.loads(encoded), json.loads(stats), json.loads(evidence)
                    if (row['image'] != image or row['sha256'] != checksum or row['group'] != group
                            or group != 'tartanair:environment:' + family(row['environment'])
                            or kind == 'native_candidate' and receipts.get(proof['path']) != proof['sha256']):
                        raise ValueError('Frozen RGB, family or candidate receipt identity differs')
                    disposition = source_disposition(ordinal, row, kind, reason, key, metrics,
                                                       proof, exclusions, closed)
                    counts[disposition] += 1
                    groups.add(group)
                    bundles.add(row['source_url'])
                    environments.add(row['environment'])
                    trajectories.add(row['trajectory'])
                    if ordinal in exclusions:
                        seen_exclusions.add(ordinal)
                    yield dict(corpus_ordinal=ordinal, image=image, sha256=checksum, group=group,
                        entry_key=key, policy_sha256=audit['policy_sha256'], native_outcome=kind,
                        native_reason=reason, source_disposition=disposition, candidate_evidence=proof,
                        disposition_evidence=decisions_ref if ordinal in exclusions else refs[names[3]],
                        source_flags_cleared=False, rgb_replay_excluded=False)

        write_lines(output / 'source-routes.jsonl', routes())
    if (counts != Counter(native_source_eligible=229862, excluded_geometry=62)
            or seen_exclusions != closed or (len(groups), len(bundles), len(environments), len(trajectories)) != (11, 24, 12, 272)):
        raise ValueError('Every frozen native outcome must reconcile once')
    publish_json(output / 'candidate-quarantine.json', dict(schema_version=1,
        stage='candidate_quality_quarantine_recorded', dataset='tartanair',
        policy_sha256=audit['policy_sha256'], manifest_sha256=audit['manifest_sha256'],
        packs_index=audit['packs'], flags=[], training_admission=False, optimizer_updates=0))
    review = dict(stage='source_review_complete', decision='eligible_with_flagged_rows_quarantined',
        dataset='tartanair', policy_sha256=audit['policy_sha256'], manifest_sha256=audit['manifest_sha256'],
        checks=dict.fromkeys(SOURCE_CHECKS, True), candidate_quarantine=file_ref(output / 'candidate-quarantine.json'),
        evidence=[*refs.values(), audit['generation_report'], audit['packs'], audit['policy'], audit['index'],
            audit['review_candidates'], crops['views_file'], decisions_ref, documentation_ref,
            file_ref(output / 'source-routes.jsonl')],
        label_kinds=['synthetic_renderer'], supervision='metric_native', eligible_candidate_images=229862,
        quarantined_candidate_images=0, geometry_excluded_images=62, selected_source_groups=11,
        original_source_flags_retained=True, check_details=dict(
            documentation_and_paper='Reuse the November2 primary paper/V1 format review in the immutable source execution snapshot.',
            units_and_camera='Released native camera-Z meters and 640x480 synthetic calibration;113 retained stereo source conversions exactly reproduce cached depth/mask/K/statistics.',
            valid_support='All229862 producer members read back; full outcome audit reused. Unchanged task range0.1-200m, stereo/stability/photometric masks and5percent original support; no filling.',
            geometric_quality='Original synthetic camera geometry and same-renderer stereo checks; not independent physical calibration. All62 final rejections bind exact reviewed geometry-only decisions, RGB retained.',
            visual_quality='Reuse six preflight sheets and all62 prior rejection reviews; all29 candidate pages/113 extremes and the sparse crop page were actually inspected.'),
        limitations=['Synthetic rendering does not establish measured physical accuracy, including water, reflections and translucency.',
            '113 selected extremes are not random prevalence; display-resolution review is not exhaustive per-pixel review.',
            '272 trajectories and12 environment names reduce to11 leakage families; day/night factory is one family.',
            'Two of339 checked crop views have no supported points and take the unchanged<32 geometry-skip path; feature supervision remains.'],
        selected_crop_review_complete=True, selected_crop_images=113, selected_crop_views=339,
        independent_physical_depth_accuracy_checked=False, full_corpus_coverage_accepted=False,
        full_array_readback_repeated=False, completed_outcome_audit_reused=True,
        actual_archived_model_probe_performed=False, source_flags_cleared=False,
        training_admission=False, optimizer_updates=0)
    validate_source_review(review, 'tartanair', audit['policy_sha256'], audit['manifest_sha256'], audit['packs'])
    publish_json(output / 'source-review.json', review)
    report = dict(stage='tartanair_source_review_and_routes_complete_not_corpus_admission',
        source_review=file_ref(output / 'source-review.json'), source_routes=file_ref(output / 'source-routes.jsonl'),
        candidate_quarantine=review['candidate_quarantine'], source_disposition_counts=dict(counts),
        original_images=229924, source_bundles=24, environments=12, source_groups=11, trajectories=272,
        known_geometry_exclusions=decisions_ref, new_exclusions=0, rgb_exclusions_added=0,
        full_array_readback_repeated=False, completed_outcome_audit_reused=True,
        selected_crop_review_complete=True, actual_archived_model_probe_performed=False,
        training_admission=False, optimizer_updates=0, active_training_changed=False,
        completed_at=time.time(), code=file_ref(Path(__file__)))
    publish_json(output / 'report.json', report)
    print(json.dumps(dict(source_disposition_counts=dict(counts), report=file_ref(output / 'report.json'))), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
