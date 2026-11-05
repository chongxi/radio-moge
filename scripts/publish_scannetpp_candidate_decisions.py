"""Dispose the nineteen reviewed native candidates; preserve RGB and all flags.

This is a conservative geometry-only decision, not source acceptance, a claim
of independently calibrated physical error, or permission to replace labels.
"""
import argparse
import hashlib
import json
from pathlib import Path

from radio_moge.corpus_admission import (
    checked_candidate_quarantine, checked_json, checked_path, file_ref,
)
from radio_moge.geometry_cache import canonical_json, publish_json

ORDINALS = {853213, 853223, 853224, 853227, 1002327, 1002328, 1002329,
            1284720, 1284721, 1284775, 1284776, 1284777, 1285067, 1285068,
            1461782, 1461783, 1461784, 1470794, 1470795}
PREVIOUS_SHA = 'efe9fdc2589735ccc9caa0454bb351b01e4e4ffd1d3eaa4f28b6cda4a951c669'


def reviewed_candidates(review, flags):
    """Require the exact previously reviewed register, including every identity."""
    if (review['stage'] != 'visual_review_complete_source_quality_unresolved'
            or review['source_quality_accepted'] is not False
            or set(review['quarantined_candidate_ordinals']) != ORDINALS
            or len(flags) != 19 or {f['corpus_ordinal'] for f in flags} != ORDINALS):
        raise ValueError('Exactly nineteen previously quarantined candidates required')
    records = [r for r in review['records'] if r['decision'] == 'quarantine_candidate']
    if len(records) != 19 or {r['corpus_ordinal'] for r in records} != ORDINALS:
        raise ValueError('Missing or duplicated visual review')
    by_ordinal = {r['corpus_ordinal']: r for r in records}
    for flag in flags:
        row = by_ordinal[flag['corpus_ordinal']]
        if (row['actual_page_viewed'] is not True
                or row['image'] != dict(path=flag['image'], sha256=flag['sha256'])
                or any(row[k] != flag[k] for k in
                       ('group', 'entry_key', 'member_sha256', 'receipt'))
                or row['policy_sha256'] != flag['policy_sha256']):
            raise ValueError('Candidate identity or actual visual review differs')
    return by_ordinal


def verify_rgb_membership(manifest, flags):
    path = Path(manifest['path'])
    before = path.stat()
    wanted = {f['image']: f for f in flags}
    if len(wanted) != len(flags):
        raise ValueError('Duplicate RGB identity')
    digest, seen, count = hashlib.sha256(), set(), 0
    with path.open('rb') as stream:
        for line in stream:
            digest.update(line)
            count += 1
            row = json.loads(line)
            if row['image'] in wanted:
                expected = wanted[row['image']]
                if (row['image'] in seen or row['dataset'] != 'scannetpp_dslr'
                        or any(row[k] != expected[k] for k in ('sha256', 'group'))):
                    raise ValueError('Duplicate or changed retained RGB')
                seen.add(row['image'])
    after = path.stat()
    if (digest.hexdigest() != manifest['sha256'] or count != 1637908
            or len(seen) != len(flags) or any(getattr(before, k) != getattr(after, k)
                for k in ('st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))):
        raise ValueError('All nineteen RGB must remain in the unchanged full pool')
    return dict(manifest=manifest, count=count, verified_retained_images=len(seen))


def run(root, output):
    validation_ref = file_ref(root/'scannetpp-native-visual-review-02/quarantine-validation.json')
    validation = checked_json(validation_ref)
    review_ref = validation['visual_review']
    review = checked_json(review_ref)
    flags, registers = [], []
    for item in validation['registers']:
        register_ref = item['register']
        register = checked_json(register_ref)
        values = checked_candidate_quarantine(register_ref, 'scannetpp_dslr',
            item['policy_sha256'], register['manifest_sha256'], item['packs_index'])
        if (len(values) != item['count'] or sorted(f['corpus_ordinal'] for f in values.values())
                != item['ordinals']):
            raise ValueError('Existing validated quarantine scope differs')
        flags.extend(dict(f, policy_sha256=item['policy_sha256'], register=register_ref)
                     for f in values.values())
        registers.append(register_ref)
    records = reviewed_candidates(review, flags)
    for row in records.values():
        checked_path(row['page'])
    prior_ref = file_ref(root/'scannetpp-support-rejection-decisions-01/known-geometry-exclusions.jsonl')
    if prior_ref['sha256'] != PREVIOUS_SHA:
        raise ValueError('Prior 4171 exclusions differ')
    previous = [json.loads(s) for s in checked_path(prior_ref).read_text().splitlines()]
    if len(previous) != 4171 or {r['corpus_ordinal'] for r in previous} & ORDINALS:
        raise ValueError('Prior exclusion scope overlaps or differs')
    replay_ref = file_ref(root/'rgb-replay-candidates-01/report.json')
    membership = verify_rgb_membership(checked_json(replay_ref)['manifest'], flags)
    decisions = []
    for flag in sorted(flags, key=lambda f: f['corpus_ordinal']):
        row = records[flag['corpus_ordinal']]
        decisions.append(dict(corpus_ordinal=flag['corpus_ordinal'], image=flag['image'],
            sha256=flag['sha256'], dataset='scannetpp_dslr', group=flag['group'],
            decision='exclude', scope='geometry_supervision', rgb_replay_excluded=False,
            reason='reviewed_native_near_surface_quality_uncertainty',
            candidate_policy_sha256=flag['policy_sha256'], candidate_entry_key=flag['entry_key'],
            member_sha256=flag['member_sha256'], native_candidate_flags_retained=True,
            evidence=[review_ref, validation_ref, flag['register'], flag['receipt'], row['page']],
            interpretation='Conservative exclusion of this exact image from geometry supervision based '
            'on its reviewed supported near-surface anomaly. Source physical cause remains unresolved. '
            'No per-image scanner proof is claimed where none exists; no neighbor/scene expansion.',
            reviewed_finding=row['finding']))
    combined = sorted(previous + decisions, key=lambda x: x['corpus_ordinal'])
    if (len({r['corpus_ordinal'] for r in combined}) != 4190
            or len({r['image'] for r in combined}) != 4190
            or [r['corpus_ordinal'] for r in combined if r.get('rgb_replay_excluded')] != [210812]):
        raise ValueError('Combined geometry or RGB exclusion scope differs')
    output.mkdir(exist_ok=False)
    for name, rows in [('geometry-exclusions.jsonl', decisions),
                       ('known-geometry-exclusions.jsonl', combined)]:
        incoming = output/(name+'.incoming')
        with incoming.open('xb') as stream:
            for row in rows:
                stream.write(canonical_json(row)+b'\n')
        incoming.rename(output/name)
    publish_json(output/'report.json', dict(stage='explicit_geometry_exclusions_recorded',
        review=review_ref, quarantine_validation=validation_ref, mandatory_registers=registers,
        prior_exclusions=prior_ref, new_exclusions=file_ref(output/'geometry-exclusions.jsonl'),
        all_known_exclusions=file_ref(output/'known-geometry-exclusions.jsonl'),
        new_geometry_exclusions=19, total_geometry_exclusions=4190,
        actual_rgb_membership=membership, replay_report=replay_ref,
        candidate_flags_retained=19, pending_candidate_dispositions=0,
        whole_scene_exclusions=0, rgb_excluded=False, new_teacher_replacements=0,
        source_quality_accepted=False, v2_labels_accepted=False,
        historical_training_failure_cause_established=False,
        generator=file_ref(Path(__file__).resolve()),
        training_configuration_modified=False, training_admission=False, optimizer_updates=0))
    print(json.dumps(file_ref(output/'report.json')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
