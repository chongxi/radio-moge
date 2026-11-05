"""Read back a closed mixed-feature endpoint and preserve its fixed checkpoint."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import torch

from radio_moge.corpus_admission import checked_json, file_ref
from radio_moge.geometry_cache import publish_json
from verify_corpus_stage_startup import equal_finite


def verify(root):
    completion_ref = file_ref(root / 'completed.json')
    completion = checked_json(completion_ref)
    plan = checked_json(file_ref(root / 'plan.json'))
    config = checked_json(plan['config'])
    review_ref = file_ref(root / 'validation-234538-review-01/report.json')
    review = checked_json(review_ref)
    provenance_ref = file_ref(root / 'training/provenance.json')
    provenance = checked_json(provenance_ref)
    audit = checked_json(completion['actual_inputs'])
    initial_proof = checked_json(file_ref(root / 'initial-verification.json'))
    endpoint = config['train']['steps']
    if (endpoint != 234538 or initial_proof['step'] != 233538
            or completion['new_updates'] != 1000 or not completion['fixed1000_complete']
            or completion['stop_report']['reason'] != 'step_limit'
            or completion['stop_report']['step'] != endpoint
            or completion['stop_report']['final_evaluation'] != 'complete'
            or review['step'] != endpoint or not review['fixed1000_complete']
            or review['actual_rgb_presentations'] != 32000
            or review['actual_synthetic_presentations'] != 4000
            or not review['actual_planned_keys_matched']
            or audit['optimizer_updates'] != 1000 or audit['cursor'] != 8000
            or audit['presentations'] != 32000
            or audit['source_counts']['procthor_empty'] != 4000
            or sum(audit['source_counts'].values()) != 32000
            or sum(audit['resolution_counts'].values()) != 32000
            or not audit['planned_keys_matched']
            or config['loss'] != dict(spatial=1., summary=1., cosine=1., summary_cosine=1.)
            or any(k in config for k in ('ordinary_geometry', 'geometry_supervision', 'feature_replay'))):
        raise ValueError('The closed endpoint differs from the declared feature-only run')
    records = [json.loads(line) for line in (root / 'training/train.jsonl').read_text().splitlines()]
    if [r['step'] for r in records] != list(range(233539, endpoint + 1)):
        raise ValueError('Missing or repeated optimizer updates')
    max_total_difference = 0.
    for number, row in enumerate(records, 1):
        if (any(not math.isfinite(v) for v in row.values() if isinstance(v, (int, float)))
                or row['effective_batch'] != 32 or row['logical_data_cursor'] != number * 8
                or row['lr_used'] != 1e-5 or any(k.startswith('gt_') for k in row)):
            raise ValueError('Invalid feature update, cursor or unexpected geometry metric')
        total = sum(row[k] for k in ('spatial', 'summary', 'cosine', 'summary_cosine'))
        difference = abs(total - row['total'])
        max_total_difference = max(max_total_difference, difference)
        if not math.isclose(total, row['total'], rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError('Feature total differs beyond float32 summation rounding')
    source = Path(completion['final_checkpoint']['path'])
    before = source.stat()
    state = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    initial = torch.load(root / 'training/initial.pt', map_location='cpu', weights_only=True, mmap=True)
    if (state['step'] != endpoint or state['cursor'] != 8000 or state['config'] != config
            or state['data_digests'] != provenance['data_digests']
            or state['revisions'] != provenance['revisions']
            or state['execution_revision'] != provenance['execution_revision']
            or state['adapter_config'] != initial['adapter_config']
            or state['model_config'] != initial['model_config']
            or state['performance'] != provenance['performance']
            or not equal_finite(state['statistics'], initial['statistics'])):
        raise ValueError('Final state, identity or fixed statistics mismatch')
    for section in ('adapter', 'optimizer', 'rng'):
        if not equal_finite(state[section], state[section]):
            raise ValueError('Nonfinite final state: ' + section)
    steps = sorted({int(v['step']) for v in state['optimizer']['state'].values()})
    if len(state['optimizer']['state']) != 100 or steps != [endpoint]:
        raise ValueError('Final Adam states or update counters differ')
    after = source.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('Final checkpoint changed during readback')
    output = root / 'finish-verification-01'
    output.mkdir(exist_ok=False)
    fixed = output / f'checkpoint-step{endpoint}.pt'
    os.link(source, fixed)
    if not os.path.samefile(source, fixed):
        raise ValueError('Fixed checkpoint is not the closed final inode')
    report = dict(stage='bounded1000_empty_room_feature_final_verified', step=endpoint,
        new_updates=1000, cursor=8000, optimizer_states=100, optimizer_step_values=steps,
        all_adapter_and_adam_finite=True, fixed_statistics_unchanged=True,
        model_and_data_identities_unchanged=True, all1000_records_finite=True,
        feature_total_arithmetic_checked=True, maximum_logged_total_roundoff=max_total_difference,
        geometry_loss=False, actual_rgb_presentations=32000, actual_synthetic_presentations=4000,
        actual_input_audit=completion['actual_inputs'], completion=completion_ref,
        result_review=review_ref, provenance=provenance_ref, checkpoint=dict(path=str(fixed),
            sha256=completion['final_checkpoint']['sha256'], hardlink_to_closed_final=True,
            hash_reused_from_controller_after_final_close=True),
        model_inference=False, optimizer_updates_by_verifier=0, created_at=time.time(),
        verifier=file_ref(Path(__file__)))
    publish_json(output / 'report.json', report)
    print(json.dumps(dict(report=file_ref(output / 'report.json'), step=endpoint, cursor=8000)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    verify(args.root.resolve())
