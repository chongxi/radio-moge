"""Carry an actual previous visual review only across exact teacher array parity.

No inference or visual inspection is claimed here. Changed RGB, masks, cameras,
depth or inference policies fail and require a new review.
"""
import argparse
import json
from pathlib import Path
import time

from radio_moge.cache_recovery import validate_teacher_recovery_policy
from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import publish_json, read_geometry_member
from visualize_teacher_preflight import entries_for_preflight


def check_pair(old_row, new_row, old_arrays, new_arrays):
    if old_row != new_row:
        raise ValueError('Preflight RGB identity changed')
    old_meta, new_meta = old_arrays[3], new_arrays[3]
    if ({k: v for k, v in old_meta.items() if k != 'policy_sha256'}
            != {k: v for k, v in new_meta.items() if k != 'policy_sha256'}):
        raise ValueError('Teacher provenance or original support changed')
    for name, old, new in zip(('depth', 'valid', 'intrinsics'), old_arrays[:3], new_arrays[:3]):
        if old.dtype != new.dtype or old.shape != new.shape or old.tobytes() != new.tobytes():
            raise ValueError(f'Preflight {name} arrays differ')


def run(previous, current):
    visual_ref = file_ref(previous / 'visual-review.json')
    visual = checked_json(visual_ref)
    old_entries, old_report, old_ref = entries_for_preflight(previous)
    new_entries, new_report, new_ref = entries_for_preflight(current)
    old_policy_ref, new_policy_ref = (file_ref(p / 'packs/policy.json') for p in (previous, current))
    validate_teacher_recovery_policy(checked_json(old_policy_ref), checked_json(new_policy_ref))
    if (visual.get('stage') != 'visual_review_complete' or visual.get('passed') is not True
            or visual.get('preflight_report_sha256') != old_ref['sha256']
            or visual.get('policy_sha256') != old_report['policy_sha256']
            or not visual.get('evidence') or not visual.get('reviewed_scenes')
            or old_report['passed'] is not True or new_report['passed'] is not True
            or old_report['rejected'] != 0 or new_report['rejected'] != 0
            or len(old_entries) != len(new_entries)):
        raise ValueError('Both complete numeric preflights and actual previous visual review required')
    for ref in visual['evidence']:
        checked_path(ref)
    old = {entry['image']['sha256']: (pack, entry) for pack, entry, _, _ in old_entries}
    if len(old) != len(old_entries):
        raise ValueError('Duplicate previous RGB')
    records, scenes = [], set()
    for pack, entry, receipt, pack_ref in new_entries:
        row = entry['image']
        found = old.pop(row['sha256'], None)
        if found is None:
            raise ValueError('Duplicate or outside-scope new RGB')
        old_pack, old_entry = found
        check_pair(old_entry['image'], row, read_geometry_member(old_pack, old_entry),
                   read_geometry_member(pack, entry))
        scenes.add(row['scene'])
        records.append(dict(image=row, old_key=old_entry['key'], new_key=entry['key'],
                            current_receipt=receipt, current_pack=pack_ref, arrays_bitwise_equal=True))
    if old or scenes != set(visual['reviewed_scenes']):
        raise ValueError('Incomplete or changed reviewed scene selection')
    proofs = [visual_ref, old_ref, new_ref, old_policy_ref, new_policy_ref]
    for ref in proofs:
        checked_path(ref)
    parity_path = current / 'array-parity-to-reviewed-01.json'
    publish_json(parity_path, dict(stage='exact_preflight_array_parity_complete', passed=True,
        completed_at=time.time(), entries=len(records), records=records, evidence=proofs,
        depth_mask_intrinsics_bitwise_equal=True, frozen_rgb_and_provenance_equal=True,
        inference_performed=False, new_visual_inspection_performed=False,
        code=file_ref(Path(__file__)), training_admission=False, optimizer_updates=0))
    review = dict(stage='visual_review_complete', passed=True,
        preflight_report_sha256=new_ref['sha256'], policy_sha256=new_report['policy_sha256'],
        reviewed_scenes=sorted(scenes), evidence=[*proofs, file_ref(parity_path)],
        method='Reuse previous actual visual inspection only after exact RGB, depth, mask, K and provenance parity.',
        prior_actual_visual_review=visual_ref, new_visual_inspection_performed=False,
        candidate_generation_only=True, source_quality_accepted=False,
        training_admission=False, optimizer_updates=0)
    publish_json(current / 'visual-review.json', review)
    print(json.dumps(dict(entries=len(records), scenes=len(scenes), visual_review=file_ref(current / 'visual-review.json'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--current', type=Path, required=True)
    args = parser.parse_args()
    run(args.previous.resolve(), args.current.resolve())
