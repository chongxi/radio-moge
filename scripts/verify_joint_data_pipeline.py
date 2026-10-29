"""Audit the one-time data pipeline probe and subsequent real training, no weights read."""

import argparse
import json
import math
import netrc
from pathlib import Path
import shlex
import subprocess
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


def verify_wandb(root):
    local_path = root / 'data-pipeline-verification.json'
    if not local_path.exists():
        print(json.dumps(dict(stage='pending', reason='Local production verification not completed')))
        return
    proof = json.loads(local_path.read_text())
    assert proof['stage'] == 'passed'
    output = root / 'data-pipeline-wandb-verification.json'
    if output.exists():
        assert json.loads(output.read_text())['stage'] == 'passed'
        print(json.dumps(dict(stage='already_verified')))
        return
    config = json.loads((root / 'config.json').read_text())
    provenance = json.loads((root / 'train/provenance.json').read_text())
    rows = [r for r in lines(root / 'train/train.jsonl') if r['step'] > proof['resume_step']]
    code = '''import json,os,sys,time
p=json.load(sys.stdin);os.environ['WANDB_API_KEY']=p.pop('key');os.environ['WANDB_BASE_URL']='https://api.forge.coreweave.com'
import wandb
run=wandb.Api(timeout=60).run('brainsim/radio-moge/eadda6e3')
assert run.state=='running' and run.config['experiment']==p['config']
assert run.config['provenance']==p['provenance']
expected={r['step']:r for r in p['rows']}
keys=['optimizer_step','train/lr_used','train/effective_batch','train/logical_data_cursor']
for row in run.scan_history(keys=keys,page_size=100):
 if row.get('optimizer_step') not in expected or any(row.get(k) is None for k in keys):continue
 e=expected[row['optimizer_step']]
 assert all(row['train/'+k]==e[k] for k in ['lr_used','effective_batch','logical_data_cursor'])
 print(json.dumps(dict(stage='passed',checked_at=time.time(),id=run.id,state=run.state,matched_production_step=e['step'],runtime_provenance_exact=True)));break
else:raise AssertionError('No resumed production history row matched')
'''
    secret = netrc.netrc(str(Path.home() / '.netrc')).authenticators('api.wandb.ai')[2]
    result = subprocess.run(['ssh', '-F', '/home/hub/.ssh/radio-moge-config', 'radio-moge-gpu',
                             shlex.join(['/home/buildhub/radio-moge/.venv/bin/python', '-c', code])],
                            input=json.dumps(dict(key=secret, config=config, provenance=provenance,
                                                  rows=rows)), text=True, capture_output=True)
    del secret
    if result.returncode:
        raise RuntimeError('W&B resume verification failed; credential-bearing output withheld')
    value = json.loads(result.stdout)
    write(output, value)
    print(json.dumps(value))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--wandb', action='store_true')
    a = p.parse_args()
    (verify_wandb if a.wandb else verify)(a.root)
