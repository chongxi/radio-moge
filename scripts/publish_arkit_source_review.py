"""Publish qualified native-sensor eligibility from completed ARKit evidence.

Reuse the full outcome/readback and actual visual reviews. Explicit frame
decisions retain cross-sensor limitations; they do not calibrate either sensor
or establish population accuracy. Active training artifacts are never changed.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

from publish_coco_nyu_source_reviews import write_lines
from publish_hypersim_native_source_review import validate_exclusion_identity
from radio_moge.corpus_admission import (SOURCE_CHECKS, checked_candidate_quarantine,
    checked_json, checked_path, file_ref, readonly_db, validate_source_review)
from radio_moge.geometry_cache import canonical_json, publish_json


# These are individual decisions from the already inspected exact-camera pages,
# not a numerical error cutoff or a rule to propagate to neighboring frames.
FARO_EXCLUSIONS = {
    27177: 'Broad near-wall residual persists with exact publisher K; conservatively omit geometry.',
    27185: 'Wall/baseboard residual bands and visibility ambiguity persist; conservatively omit geometry.',
    41530: 'Mirror/ceiling surface identity remains ambiguous; agreement does not establish physical mirror depth.',
    41531: 'Mirror/sink visibility and basin boundaries remain ambiguous; conservatively omit geometry.',
    41546: 'Panel residual and bowl/boundary disagreement persist with exact publisher K; omit geometry.',
}
FARO_RETAINED = {4591, 4600, 4606, 27155}


def checked_faro_records(review, diagnostic):
    if (review['stage'] != 'exact_publisher_camera_visual_review_complete'
            or review['images_reviewed'] != 9 or review['pages_reviewed'] != 9
            or review['exact_highres_intrinsics_ambiguity_resolved_for_these_nine'] is not True
            or diagnostic['exact_highres_intrinsics_now_obtained'] is not True
            or diagnostic['original_native_arrays_and_masks_unchanged'] is not True
            or diagnostic['calibration_fit_performed'] is not False):
        raise ValueError('Completed exact-camera diagnostic and actual review required')
    records = {x['corpus_ordinal']: x for x in diagnostic['records']}
    reviewed = {x['corpus_ordinal']: x for x in review['records']}
    if (len(records) != len(diagnostic['records']) or len(reviewed) != len(review['records'])
            or set(records) != set(FARO_EXCLUSIONS) | FARO_RETAINED or set(reviewed) != set(records)):
        raise ValueError('Exact nine-frame reviewed scope differs')
    for ordinal, record in records.items():
        view = reviewed[ordinal]
        if (view['page_actually_viewed'] is not True or view['page'] != record['page']
                or view['image'] != record['image'] or not view['observation']):
            raise ValueError('Reviewed exact image or page differs')
    return records


def source_disposition(ordinal, row, outcome, reason, key, metrics, exclusions, flags):
    decision, flag = exclusions.get(ordinal), flags.get(key)
    if decision is not None:
        validate_exclusion_identity(decision, row, ordinal)
        if (decision.get('scope') not in ('geometry_only', 'geometry_supervision')
                or decision.get('rgb_replay_excluded') is not False):
            raise ValueError('ARKit geometry decisions must retain original RGB')
    if outcome == 'rejected':
        if decision is None or decision.get('native_reason') != reason or key is not None:
            raise ValueError('Native rejection lacks its exact reviewed disposition')
        return 'excluded_geometry'
    if outcome != 'native_candidate':
        raise ValueError('Unknown or unresolved native outcome')
    if flag is not None:
        if (flag['corpus_ordinal'] != ordinal or decision is None
                or any(flag[k] != row[k] for k in ('image', 'sha256', 'group'))):
            raise ValueError('Quarantined candidate lacks an exact RGB disposition')
        return 'excluded_geometry'
    if decision is not None:
        raise ValueError('Excluded candidate lacks its mandatory policy flag')
    if (not key or not all(math.isfinite(metrics[k]) for k in
                          ('support', 'median_depth', 'rgb_mae', 'rgb_p90'))
            or not .05 <= metrics['support'] <= 1 or metrics['median_depth'] <= 0
            or not 0 <= metrics['rgb_mae'] <= .04 or not 0 <= metrics['rgb_p90'] <= .08):
        raise ValueError('Audited original support or registration differs')
    return 'native_source_eligible'


def run(root, output):
    names = ('arkit-outcomes-01/report.json', 'arkit-rejection-decisions-01/report.json',
        'arkit-candidate-decisions-01/report.json', 'arkit-exact-faro-camera-01/visual-review-01.json',
        'arkit-exact-faro-camera-01/report.json', 'arkit-outcome-visuals-01/visual-review-01.json',
        'arkit-outcome-visuals-01/report.json', 'arkit-training-crops-01/report.json',
        'arkit-sparse-crop-visuals-01/visual-review.json', 'arkit-cache-preflight-03/report.json')
    refs = {n: file_ref(root / n) for n in names}
    audit, rejected, prior, visual, diagnostic, candidate_visual, rendered, crops, sparse, preflight = (
        checked_json(refs[n]) for n in names)
    if (refs[names[0]]['sha256'] != 'c815c418169c87445cdd76ac07bbb8b5a51af5eca6474d168997cc3ade10f27a'
            or refs[names[3]]['sha256'] != '0002e9d0a069aa74435bf885ef99fe4cfff66d7eac9fac2b61234f770f3297c2'
            or audit['outcome_counts'] != dict(native_candidate=194924, rejected=2618)
            or audit['expected_scope'] != 197542 or audit['producer_array_readback_complete'] is not True
            or rejected['excluded_images'] != 2618 or rejected['source_geometry_exclusions'] != 2621
            or prior['ordinals'] != [153819, 167294, 182561]
            or visual['report'] != refs[names[4]]
            or candidate_visual['all_36_selected_views_inspected'] is not True
            or candidate_visual['evidence'] != refs[names[6]]
            or rendered['source_array_and_camera_parity'] is not True or rendered['rendered_images'] != 36
            or crops['views'] != 381 or crops['images'] != 127 or crops['original_support_unchanged'] is not True
            or sparse['all_rendered_pages_inspected'] is not True
            or preflight['passed'] is not True or preflight['verified_cache_members'] != 24
            or preflight['policy_sha256'] != audit['policy_sha256']):
        raise ValueError('Completed source evidence differs')
    faro = checked_faro_records(visual, diagnostic)
    gen, packs, policy = (checked_json(audit[n]) for n in ('generation_report', 'packs', 'policy'))
    if (gen['stage'] != 'complete' or gen['mode'] != 'generate' or gen['completed'] != 194924
            or gen['verified_cache_members'] != 194924 or packs['count'] != 194924
            or gen['optimizer_updates'] != 0 or gen['training_started'] is not False
            or gen['policy_sha256'] != audit['policy_sha256'] or packs['policy_sha256'] != audit['policy_sha256']
            or hashlib.sha256(canonical_json(policy)).hexdigest() != audit['policy_sha256']
            or policy['manifest_sha256'] != audit['manifest_sha256']):
        raise ValueError('Completed native policy or readback differs')
    old_flags = checked_candidate_quarantine(prior['quarantine'], 'arkitscenes',
        audit['policy_sha256'], audit['manifest_sha256'], audit['packs'])
    if len(old_flags) != 3:
        raise ValueError('Previous flags changed')
    previous_ref = rejected['known_geometry_exclusions']
    if previous_ref['sha256'] != 'a80a5da641bf914e0f590a9e9314e269bdae01e1ca1c54b1e68e7fce431c99c5':
        raise ValueError('Previous global decisions changed')
    previous = [json.loads(line) for line in checked_path(previous_ref).read_text().splitlines()]
    if len(previous) != 4078 or set(FARO_EXCLUSIONS) & {r['corpus_ordinal'] for r in previous}:
        raise ValueError('Repeated or incomplete prior decisions')
    index = checked_path(audit['index'])
    selected = {}
    with readonly_db(index) as db:
        # One bounded sequential pass, without NFS point lookups or array rereads.
        for ordinal, encoded, outcome, proof, key in db.execute(
                'SELECT ordinal,row_json,outcome,evidence_json,entry_key FROM images ORDER BY ordinal'):
            if ordinal in faro:
                row = json.loads(encoded)
                if (outcome != 'native_candidate' or faro[ordinal]['image'] !=
                        dict(path=row['image'], sha256=row['sha256'])):
                    raise ValueError('Reviewed FARO image differs from frozen native candidate')
                selected[ordinal] = row, json.loads(proof), key
    if set(selected) != set(faro):
        raise ValueError('Missing reviewed native candidate')
    new_flags, decisions = [], []
    for ordinal, reason in FARO_EXCLUSIONS.items():
        row, receipt_ref, key = selected[ordinal]
        receipt = checked_json(receipt_ref)
        entries = [e for e in receipt['entries'] if e['key'] == key]
        if len(entries) != 1 or receipt['policy_sha256'] != audit['policy_sha256']:
            raise ValueError('Reviewed native receipt changed')
        entry = entries[0]
        original = dict(entry['image'])
        original.pop('corpus_ordinal', None)
        frozen = dict(row)
        frozen.pop('corpus_ordinal', None)
        if original != frozen:
            raise ValueError('Reviewed native receipt RGB differs')
        evidence = [refs[names[3]], refs[names[4]], faro[ordinal]['page'], receipt_ref]
        for ref in evidence:
            checked_path(ref)
        common = dict(corpus_ordinal=ordinal, image=row['image'], sha256=row['sha256'],
            group=row['group'], reason=reason, evidence=evidence)
        new_flags.append(dict(common, decision='quarantine_candidate', entry_key=key,
                              member_sha256=entry['sha256'], receipt=receipt_ref))
        decisions.append(dict(common, dataset='arkitscenes', decision='exclude', scope='geometry_only',
            rgb_replay_excluded=False, candidate_policy_sha256=audit['policy_sha256'],
            physical_error_proven=False, automatic_numeric_threshold=False, neighbor_scope_expanded=False))
    replay = root / 'reviewed-stage1-replay-02/rgb.jsonl'
    wanted = {selected[i][0]['image']: selected[i][0] for i in FARO_EXCLUSIONS}
    found, count, digest = set(), 0, hashlib.sha256()
    with replay.open('rb') as file:
        for line in file:
            digest.update(line)
            count += 1
            row = json.loads(line)
            if row['image'] in wanted:
                if row['image'] in found or any(row[k] != wanted[row['image']][k] for k in ('sha256', 'group')):
                    raise ValueError('Retained replay RGB identity changed')
                found.add(row['image'])
    if (found != set(wanted) or count != 796477
            or digest.hexdigest() != '1ff48734105a0ca14532f2a511916d7db9cc057ece37d2d74210bc860ea81aaa'):
        raise ValueError('Original RGB must remain in the unchanged accepted replay')
    combined = sorted(previous + decisions, key=lambda x: x['corpus_ordinal'])
    if (len({x['corpus_ordinal'] for x in combined}) != 4083 or len({x['image'] for x in combined}) != 4083
            or [x['corpus_ordinal'] for x in combined if x.get('rgb_replay_excluded')] != [210812]):
        raise ValueError('Combined global exclusion scope differs')
    output.mkdir(exist_ok=False)
    write_lines(output / 'geometry-exclusions.jsonl', decisions)
    write_lines(output / 'known-geometry-exclusions.jsonl', combined)
    quarantine = deepcopy(checked_json(prior['quarantine']))
    quarantine['flags'] += new_flags
    publish_json(output / 'candidate-quarantine.json', quarantine)
    quarantine_ref = file_ref(output / 'candidate-quarantine.json')
    flags = checked_candidate_quarantine(quarantine_ref, 'arkitscenes', audit['policy_sha256'],
                                        audit['manifest_sha256'], audit['packs'])
    exclusions = {x['corpus_ordinal']: x for x in combined if x['dataset'] == 'arkitscenes'}
    counts, groups, seen_exclusions, seen_flags = Counter(), set(), set(), set()
    def routes():
        with readonly_db(index) as db:
            for ordinal, encoded, outcome, reason, key, stats, proof in db.execute(
                    'SELECT ordinal,row_json,outcome,reason,entry_key,details_json,evidence_json '
                    'FROM images ORDER BY ordinal'):
                row = json.loads(encoded)
                if row['dataset'] != 'arkitscenes' or row['split'] != 'train':
                    raise ValueError('Source or split changed')
                disposition = source_disposition(ordinal, row, outcome, reason, key,
                                                 json.loads(stats), exclusions, flags)
                counts[disposition] += 1
                if disposition == 'native_source_eligible':
                    groups.add(row['group'])
                if ordinal in exclusions:
                    seen_exclusions.add(ordinal)
                if key in flags:
                    seen_flags.add(key)
                yield dict(corpus_ordinal=ordinal, image=row['image'], sha256=row['sha256'],
                    group=row['group'], native_outcome=outcome, native_reason=reason,
                    entry_key=key, source_disposition=disposition, candidate_evidence=json.loads(proof),
                    source_flags_cleared=False, rgb_replay_excluded=False)
    write_lines(output / 'source-routes.jsonl', routes())
    if (counts != Counter(native_source_eligible=194916, excluded_geometry=2626)
            or seen_exclusions != set(exclusions) or seen_flags != set(flags) or len(flags) != 8):
        raise ValueError('Every native source outcome must reconcile exactly once')
    publish_json(output / 'faro-dispositions.json', dict(stage='reviewed_faro_cases_disposed',
        review=refs[names[3]], diagnostic=refs[names[4]], excluded_ordinals=sorted(FARO_EXCLUSIONS),
        retained_ordinals=sorted(FARO_RETAINED), retained_basis='Broad observed structure is usable under the unchanged native confidence/stability/registration masks. Small residuals and reference holes remain; not error-free or population accuracy evidence.',
        excluded_basis=FARO_EXCLUSIONS, calibration_fit_performed=False, depth_or_masks_modified=False,
        population_accuracy_estimated=False, physical_reference_resolved=False, rgb_replay_modified=False,
        replay=dict(path=str(replay), sha256=digest.hexdigest()), actual_rgb_memberships_checked=len(found)))
    documentation = dict(stage='primary_source_review_complete', references=[
        dict(url='https://arxiv.org/abs/2111.08897', finding='Mobile RGB-D sensor data and stationary laser-scanner reference are distinct measurement sources.'),
        dict(url='https://github.com/apple/ARKitScenes/blob/main/DATA.md', finding='Depth uint16 millimeters, confidence 0 to 2, and timestamp-specific pincam focal/principal values.'),
        dict(url='https://github.com/apple/ARKitScenes/blob/main/threedod/benchmark_scripts/utils/tenFpsDataLoader.py', finding='Native integer-pixel camera-Z backprojection; preserve native grid and exact RGB camera mapping.')],
        retained_documentation=visual['documentation'], exact_camera_correction=visual['correction'],
        mobile_depth_is_faro_ground_truth=False, population_physical_accuracy_guarantee=False)
    publish_json(output / 'documentation-review.json', documentation)
    more = [audit[n] for n in ('generation_report', 'packs', 'policy', 'index')]
    more += [rejected['visual_review'], rejected['source_reproduction'], previous_ref,
        file_ref(output / 'known-geometry-exclusions.jsonl'), file_ref(output / 'source-routes.jsonl'),
        file_ref(output / 'faro-dispositions.json'), file_ref(output / 'documentation-review.json')]
    review = dict(stage='source_review_complete', decision='eligible_with_flagged_rows_quarantined',
        dataset='arkitscenes', policy_sha256=audit['policy_sha256'], manifest_sha256=audit['manifest_sha256'],
        checks=dict.fromkeys(SOURCE_CHECKS, True), candidate_quarantine=quarantine_ref,
        evidence=[*refs.values(), *more], label_kinds=['measured_sensor'], supervision='metric_native',
        eligible_candidate_images=194916, quarantined_candidate_images=8, selected_source_groups=len(groups),
        check_details=dict(documentation_and_paper='Official paper/formats/camera implementation and exact-wide-K correction reviewed.',
            units_and_camera='Meters camera-Z, original native grid, timestamp-specific native and RGB K; all four orientations checked without blind depth resizing.',
            valid_support='Completed full producer readback and outcome audit; unchanged confidence 2, stable-depth and exact-camera RGB registration support. No hole filling.',
            geometric_quality='Qualified noisy native-sensor supervision. All 2618 producer rejections explicitly excluded; eight exact candidates quarantined. Nine exact-camera FARO cases have explicit dispositions, with residual/visibility limitations retained.',
            visual_quality='Reuse 24 preflight, 36 source-reproduced candidate views, all 18 rejection pages, nine exact-camera FARO pages and 381 production-loader crop views; no random prevalence claim.'),
        limitations=['FARO comparisons cover only three videos and publisher-filtered frames; no source-wide accuracy estimate.',
            'Known metric units and camera calibration do not imply error-free sensor depth. Photometric agreement is not physical depth validation.',
            'Cross-sensor residual, mirror, timing and occlusion uncertainty remain. Exact five exclusions do not establish correctness of nearby frames.',
            'Sparse training crops retain RGB/feature supervision and follow the existing fewer-than-32-observations zero-geometry path.'],
        original_source_flags_retained=True, source_flags_cleared=False, faro_physical_reference_resolved=False,
        calibration_fit_performed=False, full_array_readback_repeated=False, completed_outcome_audit_reused=True,
        crop_coverage_accepted=False, actual_archived_model_probe_performed=False,
        training_admission=False, optimizer_updates=0, active_training_changed=False)
    validate_source_review(review, 'arkitscenes', audit['policy_sha256'], audit['manifest_sha256'], audit['packs'])
    publish_json(output / 'source-review.json', review)
    report = dict(stage='arkit_source_review_and_routes_complete_not_corpus_admission', completed_at=time.time(),
        source_review=file_ref(output / 'source-review.json'), source_routes=file_ref(output / 'source-routes.jsonl'),
        source_disposition_counts=dict(counts), original_images=sum(counts.values()), source_groups=len(groups),
        new_geometry_exclusions=file_ref(output / 'geometry-exclusions.jsonl'), candidate_quarantine=quarantine_ref,
        known_geometry_exclusions=file_ref(output / 'known-geometry-exclusions.jsonl'), known_geometry_exclusion_count=4083,
        original_source_flags_retained=True, full_array_readback_repeated=False, completed_outcome_audit_reused=True,
        active_training_changed=False, training_admission=False, optimizer_updates=0, code=file_ref(Path(__file__)))
    publish_json(output / 'report.json', report)
    print(json.dumps(dict(counts=dict(counts), report=file_ref(output / 'report.json'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
