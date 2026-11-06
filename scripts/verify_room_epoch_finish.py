"""Verify the closed full-image epoch once and preserve its final checkpoint."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import torch

from radio_moge.corpus_admission import checked_json, file_ref
from radio_moge.geometry_cache import publish_json
from run_room_feature_epoch import IMAGES, SOURCES
from review_room_feature_epoch import epoch_shape
from verify_corpus_stage_startup import equal_finite


def check_closure(plan, config, completion, review, audit, coverage, counts):
    batch, updates, _ = epoch_shape(plan)
    accumulation = batch//4
    endpoint = plan['source_step']+updates
    actual = review['actual_inputs']
    if (plan['additional_updates'] != updates or plan['synthetic_images'] != IMAGES
            or config['train']['steps'] != endpoint or completion['new_updates'] != updates
            or not completion['full_epoch_complete'] or not review['full_epoch_complete']
            or review['step'] != endpoint or review['new_updates'] != updates
            or completion['stop_report']['reason'] != 'step_limit'
            or completion['stop_report']['step'] != endpoint
            or completion['stop_report']['final_evaluation'] != 'complete'
            or audit['optimizer_updates'] != updates or audit['cursor'] != updates*accumulation
            or audit['presentations'] != updates*batch or not audit['planned_keys_matched']
            or not audit['unused_prefetch_excluded'] or not audit['actual_compute_groups_verified']
            or any(audit['source_counts'].get(k, 0) != n for k,n in SOURCES.items())
            or sum(audit['source_counts'].values()) != updates*batch
            or sum(audit['resolution_counts'].values()) != updates*batch
            or actual['presentations'] != updates*batch or not actual['actual_planned_keys_matched']
            or actual['source_counts'] != audit['source_counts']
            or not actual['synthetic_identity_and_no_repeats_verified']
            or coverage['step'] != endpoint or coverage['updates'] != updates
            or not coverage['full_epoch_completed'] or not coverage['unused_prefetch_excluded']
            or not coverage['matched_encoded_image_identity']
            or len(counts) != IMAGES or any(type(n) is not int or n != 1 for n in counts)
            or config['loss'] != dict(spatial=1., summary=1., cosine=1., summary_cosine=1.)
            or any(k in config for k in ('ordinary_geometry', 'geometry_supervision', 'feature_replay'))):
        raise ValueError('Closed epoch, actual image coverage or feature-only contract differs')
    expected = dict(expected=IMAGES, committed=IMAGES, unique=IMAGES, missing=0,
                    repeated=0, each_image_exactly_once=True)
    if (actual['synthetic_coverage'] != expected
            or any(coverage[k] != v for k,v in expected.items())):
        raise ValueError('Every synthetic image must have exactly one committed presentation')
    if (coverage['actual_inputs']['sha256'] != audit['records']['sha256']
            or actual['actual_input_prefix']['sha256'] != audit['records']['sha256']):
        raise ValueError('Coverage and final review must bind the same complete input audit')
    return endpoint


def verify(root):
    completion_ref = file_ref(root/'completed.json')
    completion = checked_json(completion_ref)
    plan = checked_json(file_ref(root/'plan.json'))
    config = checked_json(plan['config'])
    batch, updates, _ = epoch_shape(plan)
    accumulation = batch//4
    endpoint = plan['source_step']+updates
    review_ref = file_ref(root/f'validation-{endpoint}-review-01/report.json')
    review = checked_json(review_ref)
    audit_ref = file_ref(root/'training/actual-rgb-input-audit.json')
    audit = checked_json(audit_ref)
    coverage = checked_json(completion['coverage'])
    counts = checked_json(coverage['counts'])
    check_closure(plan, config, completion, review, audit, coverage, counts)
    provenance_ref = file_ref(root/'training/provenance.json')
    provenance = checked_json(provenance_ref)
    initial_proof = checked_json(file_ref(root/'initial-verification.json'))
    if (initial_proof['step'] != plan['source_step'] or initial_proof['optimizer_states'] != 100
            or initial_proof['cursor'] != 0 or initial_proof['real_stream_start_cursor'] != plan['source_cursor']):
        raise ValueError('Initial parent or continued original-data cursor changed')
    records = [json.loads(line) for line in (root/'training/train.jsonl').read_text().splitlines()]
    if [r['step'] for r in records] != list(range(plan['source_step']+1, endpoint+1)):
        raise ValueError('Missing or repeated optimizer updates')
    max_roundoff = 0.
    for number, row in enumerate(records, 1):
        if (any(not math.isfinite(v) for v in row.values() if isinstance(v, (int, float)))
                or row['effective_batch'] != batch or row['logical_data_cursor'] != number*accumulation
                or row['lr_used'] != 1e-5 or any(k.startswith('gt_') for k in row)):
            raise ValueError('Invalid update, cursor or unexpected geometry metric')
        if batch != 32 and (row.get('actual_forward_batch') != batch or row.get('model_forwards_per_update') != 1):
            raise ValueError('The actual model batch was split')
        total = sum(row[k] for k in ('spatial', 'summary', 'cosine', 'summary_cosine'))
        max_roundoff = max(max_roundoff, abs(total-row['total']))
        if not math.isclose(total, row['total'], rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError('Feature total differs beyond float32 summation rounding')
    source = Path(completion['final_checkpoint']['path'])
    before = source.stat()
    state = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    initial = torch.load(root/'training/initial.pt', map_location='cpu', weights_only=True, mmap=True)
    if (state['step'] != endpoint or state['cursor'] != updates*accumulation or state['config'] != config
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
            raise ValueError('Nonfinite final state: '+section)
    steps = sorted({int(v['step']) for v in state['optimizer']['state'].values()})
    if len(state['optimizer']['state']) != 100 or steps != [endpoint]:
        raise ValueError('Final Adam states or update counters differ')
    after = source.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('Final checkpoint changed during readback')
    output = root/'finish-verification-01'
    output.mkdir(exist_ok=False)
    fixed = output/f'checkpoint-step{endpoint}.pt'
    os.link(source, fixed)
    if not os.path.samefile(source, fixed):
        raise ValueError('Fixed checkpoint is not the closed final inode')
    report = dict(stage='full_room_feature_epoch_final_verified', step=endpoint,
        new_updates=updates, cursor=updates*accumulation, optimizer_states=100, optimizer_step_values=steps,
        all_adapter_and_adam_finite=True, fixed_statistics_unchanged=True,
        model_and_data_identities_unchanged=True, all_training_records_finite=True,
        feature_total_arithmetic_checked=True, maximum_logged_total_roundoff=max_roundoff,
        geometry_loss=False, actual_rgb_presentations=updates*batch, actual_synthetic_presentations=IMAGES,
        every_synthetic_image_exactly_once=True, coverage=completion['coverage'], actual_input_audit=audit_ref,
        completion=completion_ref, result_review=review_ref, provenance=provenance_ref,
        checkpoint=dict(path=str(fixed), sha256=completion['final_checkpoint']['sha256'],
            hardlink_to_closed_final=True, hash_reused_from_controller_after_final_close=True),
        model_inference=False, optimizer_updates_by_verifier=0, created_at=time.time(),
        verifier=file_ref(Path(__file__)))
    publish_json(output/'report.json', report)
    print(json.dumps(dict(report=file_ref(output/'report.json'), step=endpoint, cursor=updates*accumulation)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    verify(args.root.resolve())
