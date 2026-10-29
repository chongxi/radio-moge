"""Verify the authorized continuation from saved metadata and actual updates, without weights I/O."""

import argparse
import json
import math
import netrc
from pathlib import Path
import shlex
import subprocess
import time

from monitor_joint_full_day import write
from render_joint_full_page import lines, read


def verify(root, wandb=False):
    destination = root/('startup-wandb-verification.json' if wandb else 'startup-verification.json')
    if destination.exists():
        assert read(destination, {})['stage'] == 'passed'
        print(json.dumps(dict(stage='already_verified')))
        return
    rows = lines(root/'train/train.jsonl')
    if not rows or rows[-1]['step'] < 61387:
        print(json.dumps(dict(stage='pending',reason='Waiting for at least 20 actual new updates')))
        return
    config = read(root/'config.json', {})
    saved = read(root/'train/config.json', {})
    plan = read(root/'plan.json', {})
    launch = read(root/'launch.json', {})
    provenance = read(root/'train/provenance.json', {})
    restored = read(root/'train/fork-state.json', {})
    identity = read(root/'train/wandb-run.json', {})
    parent = root.parent/'adapter-qknorm2048-full-s43-20261028'
    source = read(parent/'final-review/source-verification.json', {})['files'][0]
    assert config == saved and plan['authorized'] and plan['additional_updates'] == 30000
    assert launch['launch_count'] == 1
    assert restored['step'] == 61367 and restored['cursor'] == 479996
    assert restored['adam_steps'] == [61367] and restored['optimizer_lr'] == 1.5e-5
    assert restored['checkpoint'] == source['source']
    assert provenance['execution_revision'] == launch['revision']
    assert provenance['runtime_limits'] == dict(steps=91367, max_seconds=28800)
    assert provenance['data_digests'] == source['checkpoint']['data_digests']
    assert provenance['train_images'] == 796478 and provenance['validation_images'] == 1268
    perf = provenance['performance']
    assert all(perf[k] == v for k,v in dict(workers=4,effective_batch=32,compute_batch=16,
               compact_transport=True,data_pipeline='prepared',autotune=False).items())
    assert identity['mode'] == 'online' and identity['id'] != plan['parent_run']
    assert all(row['step'] > 61367 and row['lr_used'] == row['lr'] == 1.5e-5
               and row['logical_data_cursor'] == 479996+8*(row['step']-61367)
               and row['effective_batch'] == 32 and row['compute_batch_limit'] == 16
               and row['gradient_norm'] > 0 and row.get('update_norm', 0) > 0
               for row in rows)
    assert all(math.isfinite(v) for row in rows for v in row.values() if type(v) in (float,int))
    if wandb:
        assert read(root/'startup-verification.json', {})['stage'] == 'passed'
        code = '''import json,os,sys,time
p=json.load(sys.stdin);os.environ['WANDB_API_KEY']=p.pop('key');os.environ['WANDB_BASE_URL']='https://api.forge.coreweave.com'
import wandb
r=wandb.Api(timeout=60).run('/'.join(p['identity'][k] for k in ('entity','project','id')))
assert r.state in ('running','finished') and r.config['experiment']==p['config']
assert r.config['provenance']==p['provenance']
keys=['optimizer_step','train/step','train/lr_used','train/logical_data_cursor','train/total']
for row in r.scan_history(keys=keys,page_size=1000):
 if row.get('optimizer_step')!=p['row']['step']:continue
 assert all(abs(row['train/'+k]-p['row'][k])<1e-10 for k in ('step','lr_used','logical_data_cursor','total'))
 print(json.dumps(dict(stage='passed',checked_at=time.time(),id=r.id,state=r.state,observed_step=row['optimizer_step'],config_exact=True,provenance_exact=True,train_row_exact=True,url=p['identity']['url'])))
 break
else:raise AssertionError('Actual update not yet present in online history')
'''
        secret = netrc.netrc(str(Path.home()/'.netrc')).authenticators('api.wandb.ai')[2]
        payload = dict(key=secret,identity=identity,config=config,provenance=provenance,row=rows[0])
        result = subprocess.run(['ssh','-F','/home/hub/.ssh/radio-moge-config','radio-moge-gpu',
            shlex.join(['/home/buildhub/radio-moge/.venv/bin/python','-c',code])],
            input=json.dumps(payload),text=True,capture_output=True)
        del secret,payload
        if result.returncode:
            raise RuntimeError('Read-only W&B check failed; credential-bearing output withheld')
        proof = json.loads(result.stdout)
    else:
        proof = dict(stage='passed',checked_at=time.time(),observed_step=rows[-1]['step'],
                     restored_step=61367,restored_cursor=479996,restored_adam_steps=[61367],
                     first_logged_step=rows[0]['step'],logged_rows=len(rows),
                     all_logged_finite=True,nonzero_updates=True,actual_constant_lr=1.5e-5,
                     runtime_limits=provenance['runtime_limits'],data_digests=provenance['data_digests'],
                     performance=perf,new_wandb=identity,parent_source_identity=plan['parent_identity'],
                     execution_revision=launch['revision'],no_weights_read=True)
    write(destination,proof)
    print(json.dumps(proof))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--wandb',action='store_true')
    args = parser.parse_args()
    verify(args.root,args.wandb)
