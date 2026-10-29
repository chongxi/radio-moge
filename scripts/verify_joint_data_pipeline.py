"""Audit the one-time data pipeline probe and subsequent real training, no weights read."""

import argparse
import json
import math
from pathlib import Path
import time

from monitor_joint_full_day import write
from render_joint_full_page import lines


def verify(root):
    output = root / 'data-pipeline-verification.json'
    if output.exists():
        assert json.loads(output.read_text())['stage'] == 'passed'
        print(json.dumps(dict(stage='already_verified')))
        return
    files = ['train/data-pipeline-benchmark.json', 'data-pipeline-transition/checkpoint-verification.json',
             'train/resume-state.json', 'train/performance.json', 'train/provenance.json']
    if any(not (root / f).exists() for f in files):
        print(json.dumps(dict(stage='pending', reason='Waiting for bounded pipeline probe')))
        return
    bench, checkpoint, resume, perf, provenance = [json.loads((root / f).read_text()) for f in files]
    if bench['stage'] != 'passed':
        print(json.dumps(dict(stage='pending', reason='Pipeline comparison still running')))
        return
    anchor = checkpoint['stopped']['step']
    rows = [r for r in lines(root / 'train/train.jsonl') if r['step'] > anchor]
    if not rows or rows[-1]['step'] < anchor + 20:
        print(json.dumps(dict(stage='pending', reason='Waiting for20real production updates')))
        return
    old = json.loads((root / 'train/throughput-benchmark.json').read_text())['selected']
    assert checkpoint['stage'] == 'passed'
    assert bench['adapter_restored_exactly'] and bench['optimizer_scheduler_rng_restored']
    assert bench['production_optimizer_updates'] == 0 and bench['disposable_updates'] == 144
    assert bench['input_equality']['exact_augmented_pixels_and_compute_order']
    assert bench['input_equality']['images'] == 128
    assert perf == bench['selected'] == provenance['performance'] == resume['performance']
    assert {k: v for k, v in perf.items() if k != 'data_pipeline'} == old
    assert resume['step'] == bench['anchor_step'] == anchor
    assert resume['cursor'] == bench['anchor_cursor'] == 10940 + 8 * (anchor - 2735)
    assert resume['config_and_manifest_match']
    assert resume['runtime_limits']['steps'] == 61367
    assert resume['runtime_limits']['max_seconds'] == 86400
    assert math.isclose(resume['original_absolute_deadline'], 1793317558.69, abs_tol=.001)
    assert provenance['data_digests'] == json.loads((root / 'startup-verification.json').read_text())['data_digests']
    identity = json.loads((root / 'train/wandb-run.json').read_text())
    assert identity['id'] == 'eadda6e3'
    aggregate = []
    for prepared in (False, True):
        trials = [r for r in bench['rows'] if r['prepared'] == prepared]
        assert len(trials) == 2
        assert all(r['stage'] == 'passed' and len(r['seconds']) == 30 for r in trials)
        speed = 32 * 60 / sum(sum(r['seconds']) for r in trials)
        assert math.isclose(speed, bench['paired_rounds'][int(prepared)]['images_per_second'])
        aggregate.append(speed)
    expected = (aggregate[1] > 1.03 * aggregate[0] and all(
        next(r for r in bench['rows'] if r['round'] == i and r['prepared'])['images_per_second'] >
        next(r for r in bench['rows'] if r['round'] == i and not r['prepared'])['images_per_second']
        for i in (0, 1)))
    assert perf['data_pipeline'] == ('prepared' if expected else 'legacy')
    cfg = json.loads((root / 'config.json').read_text())
    for r in rows:
        assert all(math.isfinite(v) for v in r.values() if type(v) in (int, float))
        assert r['effective_batch'] == 32 and r['compute_batch_limit'] == 16
        assert r['logical_data_cursor'] == 10940 + 8 * (r['step'] - 2735)
        assert r['schedule_position'] == 2735 + 2 * (r['step'] - 2735)
        assert r['gradient_norm'] > 0 and r.get('update_norm', 1) > 0
        assert math.isclose(r['total'], sum(r[k] * w for k, w in cfg['loss'].items()),
                            rel_tol=2e-5, abs_tol=2e-6)
    proof = dict(stage='passed', checked_at=time.time(), resume_step=anchor,
                 last_observed_step=rows[-1]['step'], baseline_images_per_second=aggregate[0],
                 prepared_images_per_second=aggregate[1], selected=perf['data_pipeline'],
                 exact_real_input_equality=True, original_schedule_batch_and_deadline_preserved=True,
                 same_wandb_id=identity['id'], checkpoint_sha256=checkpoint['checkpoint_sha256'],
                 optimizer_updates_discarded=144,
                 caveat='Throughput comparison does not establish decoder speed or accuracy gains')
    write(output, proof)
    print(json.dumps(proof))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    verify(p.parse_args().root)
