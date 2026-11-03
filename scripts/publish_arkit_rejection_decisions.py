"""Publish exact geometry-only ARKit rejection decisions after the source review.

This preserves prior decisions and RGB replay, and never accepts a source or
modifies training inputs. Only the already reproduced native rejection scope is
eligible for these decisions; candidate quarantine remains a separate artifact.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from review_arkit_rejections import directory_absences, missing_request
from radio_moge.corpus_admission import checked_json, checked_path, file_ref, readonly_db
from radio_moge.geometry_cache import canonical_json, publish_json


def run(root, attempt):
    review_ref = file_ref(root / 'arkit-rejection-census-01/visual-review-01.json')
    review = checked_json(review_ref)
    census = checked_json(review['evidence'])
    outcomes = checked_json(census['outcomes'])
    if (review['stage'] != 'rejection_census_visual_review_complete'
            or review['all_selected_pages_inspected'] is not True
            or review['source_quality_accepted'] is not False
            or review['records_evidence'] != census['records']
            or census['stage'] != 'rejection_reproduction_complete_not_reviewed'
            or census['mode'] != 'census' or census['processed'] != 2618
            or census['counts'] != outcomes['rejection_reasons']
            or outcomes['expected_scope'] != 197542
            or outcomes['outcome_counts'] != dict(native_candidate=194924, rejected=2618)
            or census['rejections'] != outcomes['rejections']
            or census['policy_sha256'] != outcomes['policy_sha256']
            or census['no_threshold_changes'] is not True
            or census['native_payload_network_downloads'] != 0):
        raise ValueError('Complete matching rejection census and actual visual review required')
    selected = {p['corpus_ordinal']: p['page'] for p in census['pages']}
    if ({p['corpus_ordinal']: p['page'] for p in review['records']} != selected
            or len(selected) != len(census['pages'])
            or len(selected) != len(review['records'])
            or len(selected) != review['selected_images'] or not selected):
        raise ValueError('Reviewed visual scope differs')
    for ref in selected.values():
        checked_path(ref)
    for name in ('contacts', 'preflight_visual_review', 'tooling_tests'):
        checked_path(review[name])
    for proof in census['directories']:
        data = checked_path(proof['retained_directory']).read_bytes()
        stats = directory_absences(data, proof['directory_start'], proof['archive_bytes'],
                                   proof['absent_members'])
        if any(stats[k] != proof[k] for k in stats) or proof['payload_members_downloaded'] != 0:
            raise ValueError('Retained publisher directory proof changed')
    directory_by_url = {p['source']: p for p in census['directories']}
    with checked_path(census['records']).open() as file:
        records = [json.loads(line) for line in file]
    with readonly_db(checked_path(outcomes['index'])) as db:
        rejected = {ordinal: (json.loads(row), reason) for ordinal, row, reason in db.execute(
            "SELECT ordinal,row_json,reason FROM images WHERE outcome='rejected'")}
    if (len(records) != len(rejected) or len(records) != 2618
            or {x['corpus_ordinal'] for x in records} != set(rejected)
            or len({x['corpus_ordinal'] for x in records}) != len(records)):
        raise ValueError('Each frozen rejected image must appear exactly once')
    source_rejections = {x['image']['image']: x for x in checked_json(census['rejections'])}
    for item in records:
        original, reason = rejected[item['corpus_ordinal']]
        if item['image'] != original or item['kind'] != reason:
            raise ValueError('Decision changed a frozen rejection identity')
        producer = source_rejections[original['image']]
        if reason == 'missing_exact_timestamp_native_member':
            proof = item['publisher_directory']
            url, member = missing_request(producer)
            if (proof != directory_by_url[url] or proof['source'] != url
                    or member != item['producer_detail']
                    or item['producer_detail'] != producer['stats']['detail']
                    or item['producer_detail'] not in proof['absent_members']
                    or item['exact_member_absence_verified'] is not True):
                raise ValueError('Exact publisher member absence is not verified')
        elif (item['statistics_exactly_reproduced'] is not True
                or canonical_json(item['stats']) != canonical_json(producer['stats'])
                or len(item['source_proofs']) != 5 or item['stats']['valid_fraction'] >= .05):
            raise ValueError('Numeric failure differs from the unchanged producer')
    previous_ref = file_ref(root / 'arkit-candidate-decisions-01/known-geometry-exclusions.jsonl')
    if previous_ref['sha256'] != '62e76a8189c8cfd53ad05c29645b1849796cd1114d6a57557f6aeeb299871aac':
        raise ValueError('Previous explicit exclusion set changed')
    with checked_path(previous_ref).open() as file:
        previous = [json.loads(line) for line in file]
    if len(previous) != 1460 or {x['corpus_ordinal'] for x in previous} & set(rejected):
        raise ValueError('New rejection decisions overlap previous exclusions')
    replay_path = root / 'reviewed-stage1-replay-02/rgb.jsonl'
    expected_replay_hash = '1ff48734105a0ca14532f2a511916d7db9cc057ece37d2d74210bc860ea81aaa'
    wanted = {x['image']['image']: x['image'] for x in records}
    found, count, digest = set(), 0, hashlib.sha256()
    before_stat = replay_path.stat()
    with replay_path.open('rb') as file:
        for line in file:
            digest.update(line)
            row = json.loads(line)
            count += 1
            if row['image'] in wanted:
                original = wanted[row['image']]
                if row['sha256'] != original['sha256'] or row['group'] != original['group']:
                    raise ValueError('Retained RGB identity changed')
                if row['image'] in found:
                    raise ValueError('Duplicate retained RGB')
                found.add(row['image'])
    after_stat = replay_path.stat()
    if (len(found) != len(records) or count != 796477 or digest.hexdigest() != expected_replay_hash
            or before_stat.st_size != after_stat.st_size or before_stat.st_mtime_ns != after_stat.st_mtime_ns):
        raise ValueError('All actual RGB must remain in the unchanged accepted replay')
    decisions = []
    for item in sorted(records, key=lambda x: x['corpus_ordinal']):
        row = item['image']
        evidence = [review_ref, review['evidence'], census['records'], census['rejections']]
        if item['kind'] == 'missing_exact_timestamp_native_member':
            evidence.append(item['publisher_directory']['retained_directory'])
        decisions.append(dict(corpus_ordinal=item['corpus_ordinal'], image=row['image'],
            sha256=row['sha256'], dataset=row['dataset'], group=row['group'], decision='exclude',
            scope='geometry_supervision', rgb_replay_excluded=False,
            reason='reviewed_' + item['kind'], native_reason=item['kind'],
            candidate_policy_sha256=census['policy_sha256'], evidence=evidence,
            interpretation='Exact native rejection under unchanged source policy; retain RGB. '
                'No physical-error, fitted-calibration, neighbor or scene-wide exclusion claim.'))
    combined = sorted(previous + decisions, key=lambda x: x['corpus_ordinal'])
    if (len({x['corpus_ordinal'] for x in combined}) != 4078
            or len({x['image'] for x in combined}) != 4078
            or [x['corpus_ordinal'] for x in combined if x.get('rgb_replay_excluded')] != [210812]):
        raise ValueError('Merged exclusion identities or RGB scope differ')
    for ref in (review_ref, review['evidence'], census['records'], census['outcomes'], previous_ref):
        checked_path(ref)
    output = root / attempt
    output.mkdir(exist_ok=False)
    for name, rows in (('geometry-exclusions.jsonl', decisions), ('known-geometry-exclusions.jsonl', combined)):
        temp = output / (name + '.incoming')
        with temp.open('x') as file:
            for row in rows:
                file.write(canonical_json(row).decode() + '\n')
        temp.rename(output / name)
    publish_json(output / 'rgb-preservation.json', dict(stage='actual_rgb_membership_verified',
        verified_images=len(found), total_replay_images=count,
        ordinals=sorted(rejected), rgb_replay=dict(path=str(replay_path), sha256=digest.hexdigest()),
        geometry_only=True, rgb_replay_modified=False, training_configuration_modified=False))
    report = dict(stage='explicit_geometry_exclusions_recorded', dataset='arkitscenes',
        completed_at=time.time(), excluded_images=len(decisions), counts=dict(Counter(x['native_reason'] for x in decisions)),
        source_scope=197542, prior_candidate_exclusions=3, source_geometry_exclusions=2621,
        remaining_native_candidates=194921, remaining_candidates_accepted=False,
        geometry_exclusions=file_ref(output / 'geometry-exclusions.jsonl'), previous_exclusions=previous_ref,
        known_geometry_exclusions=file_ref(output / 'known-geometry-exclusions.jsonl'),
        known_geometry_exclusion_count=len(combined), visual_review=review_ref,
        rgb_preservation=file_ref(output / 'rgb-preservation.json'),
        publisher_directory_proofs_rechecked=len(directory_by_url), source_reproduction=review['evidence'],
        publication_code=file_ref(Path(__file__)), source_quality_accepted=False,
        faro_visibility_and_calibration_resolved=False, native_flags_preserved=True,
        training_admission=False, optimizer_updates=0)
    publish_json(output / 'report.json', report)
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if Path(args.attempt).name != args.attempt:
        parser.error('Use a new attempt basename')
    run(args.root.resolve(), args.attempt)
