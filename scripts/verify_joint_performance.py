"""Audit the actual throughput decision and resumed training, optionally W&B once."""

import argparse
import json
import math
import netrc
from pathlib import Path
import shlex
import subprocess
import time

from render_joint_full_page import lines
from monitor_joint_full_day import write


def verify(root, wandb=False):
    proof_path = root / ('performance-wandb-verification.json' if wandb else 'performance-verification.json')
    if proof_path.exists():
        assert json.loads(proof_path.read_text())['stage'] == 'passed'
        print(json.dumps(dict(stage='already_verified', proof=str(proof_path))))
        return
    names = ['train/performance.json','train/throughput-benchmark.json','train/resume-state.json',
             'performance-transition/checkpoint-verification.json']
    if any(not (root / n).exists() for n in names):
        print(json.dumps(dict(stage='pending', reason='Wait for completed bounded benchmark and resume')))
        return
    perf, bench, resume, checkpoint = [json.loads((root / n).read_text()) for n in names]
    cfg = json.loads((root / 'config.json').read_text())
    identity = json.loads((root / 'train/wandb-run.json').read_text())
    provenance = json.loads((root / 'train/provenance.json').read_text())
    rows = [r for r in lines(root / 'train/train.jsonl') if r['step'] > perf['anchor_step']]
    if not rows or rows[-1]['step'] < perf['anchor_step'] + 20:
        print(json.dumps(dict(stage='pending', reason='Wait for actual production updates after restored state')))
        return
    assert identity['id'] == 'eadda6e3' and identity['mode'] == 'online'
    assert bench['stage'] == checkpoint['stage'] == 'passed'
    assert bench['production_optimizer_updates'] == 0 and bench['adapter_restored_exactly']
    assert bench['optimizer_scheduler_rng_restored'] and bench['selected'] == perf
    assert len(bench['rows']) == 16
    assert resume['step'] == checkpoint['stopped']['step'] == perf['anchor_step']
    assert resume['cursor'] == perf['anchor_cursor'] == 4 * perf['anchor_step']
    assert resume['config_and_manifest_match'] and resume['performance'] == perf
    assert provenance['performance'] == perf
    assert provenance['data_digests'] == json.loads((root / 'startup-verification.json').read_text())['data_digests']
    assert perf['base_effective_batch'] == 16 and perf['schedule_ratio'] == perf['effective_batch'] / 16
    assert resume['runtime_limits']['max_seconds'] == 86400
    endpoint = resume['runtime_limits']['steps']
    def position(step):
        return perf['anchor_schedule_position'] + (step - perf['anchor_step']) * perf['schedule_ratio']
    assert position(endpoint) <= 120000 < position(endpoint + 1)
    assert math.isclose(resume['original_absolute_deadline'],perf['original_started_at'] + 86400)
    aggregate = bench['paired_rounds']
    fastest = max(r['images_per_second'] for r in aggregate)
    chosen = next(r for r in aggregate if all(r[k] == perf[k] for k in ('workers','compute_batch','effective_batch')))
    assert chosen['images_per_second'] >= .97 * fastest
    for row in rows:
        assert all(math.isfinite(v) for v in row.values() if type(v) in (int,float))
        assert row['effective_batch'] == perf['effective_batch'] and row['gradient_norm'] > 0
        if 'update_norm' in row:
            assert row['update_norm'] > 0
        assert row['logical_data_cursor'] == perf['anchor_cursor'] + (row['step']-perf['anchor_step']) * perf['effective_batch'] // 4
        assert row['schedule_position'] == position(row['step'])
        assert math.isclose(row['total'],sum(row[k]*w for k,w in cfg['loss'].items()),rel_tol=2e-5,abs_tol=2e-6)
    value = dict(stage='passed',checked_at=time.time(),anchor_step=perf['anchor_step'],
                 last_observed_step=rows[-1]['step'],profile=perf,benchmark_images_per_second=chosen['images_per_second'],
                 image_budget_preserved=True,absolute_deadline_preserved=True,production_rows=len(rows),
                 same_wandb_id=identity['id'],checkpoint_sha256=checkpoint['checkpoint_sha256'],
                 caveat='Throughput selection is not proof that a larger batch improves generalization')
    if wandb:
        assert (root / 'performance-verification.json').exists()
        val = next(r for r in lines(root / 'train/validation.jsonl') if r['step'] == 2000)
        keys = ['macro/spatial/cosine','macro/summary/cosine','macro/summary/norm_ratio','gt_student/abs_rel']
        remote='''import json,os,sys,time
p=json.load(sys.stdin);os.environ['WANDB_API_KEY']=p.pop('key');os.environ['WANDB_BASE_URL']='https://api.forge.coreweave.com'
import wandb
run=wandb.Api(timeout=60).run('brainsim/radio-moge/eadda6e3');s=dict(run.summary)
assert run.state=='running' and run.config['experiment']==p['config']
assert s['optimizer_step']>p['anchor'] and s['train/effective_batch']==p['batch']
expected={r['step']:r for r in p['rows']}
for r in run.scan_history(keys=['optimizer_step','train/lr_used','train/effective_batch','train/logical_data_cursor'],page_size=100):
 if r.get('optimizer_step') in expected and all(r.get(k) is not None for k in ['train/lr_used','train/effective_batch','train/logical_data_cursor']):
  e=expected[r['optimizer_step']];assert r['train/lr_used']==e['lr_used'] and r['train/logical_data_cursor']==e['logical_data_cursor'];break
else:raise AssertionError('No resumed history row matched')
for r in run.scan_history(keys=['optimizer_step']+['validation/'+k for k in p['keys']],page_size=100):
 if r.get('optimizer_step')==2000 and all(r.get('validation/'+k) is not None for k in p['keys']):
  error=max(abs(r['validation/'+k]-p['validation'][k]) for k in p['keys']);assert error<1e-10;break
else:raise AssertionError('Completed validation2000 missing')
print(json.dumps(dict(stage='passed',id=run.id,state=run.state,checked_at=time.time(),step=s['optimizer_step'],validation2000_max_error=error,resumed_history_matches=True)))
'''
        secret = netrc.netrc(str(Path.home()/'.netrc')).authenticators('api.wandb.ai')[2]
        result = subprocess.run(['ssh','-F','/home/hub/.ssh/radio-moge-config','radio-moge-gpu',
                                 shlex.join(['/home/buildhub/radio-moge/.venv/bin/python','-c',remote])],
                                input=json.dumps(dict(key=secret,config=cfg,anchor=perf['anchor_step'],
                                                      batch=perf['effective_batch'],rows=rows,validation=val,keys=keys)),
                                text=True,capture_output=True)
        del secret
        if result.returncode:
            raise RuntimeError('Read-only W&B verification failed; credential-bearing output withheld')
        value=json.loads(result.stdout)
    write(proof_path,value)
    print(json.dumps(value))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--wandb',action='store_true')
    args=p.parse_args()
    verify(args.root,args.wandb)
