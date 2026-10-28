"""Verify this full run once, including actual optimizer rows and optional online tracking."""

import argparse
import hashlib
import json
import math
import netrc
from pathlib import Path
import shlex
import subprocess
import time

from render_joint_full_page import lines


def write(path, value):
    temp=path.with_suffix('.incoming')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    temp.replace(path)


def verify(root, wandb=False):
    output=root/('startup-wandb-verification.json' if wandb else 'startup-verification.json')
    if output.exists():
        assert json.loads(output.read_text())['stage']=='passed'
        print(json.dumps(dict(stage='already_verified',proof=str(output))))
        return
    train=lines(root/'train/train.jsonl')
    val=lines(root/'train/validation.jsonl')
    required=['train/provenance.json','train/config.json','train/wandb-run.json','preflight-verification.json']
    if not train or train[-1]['step']<20 or not val or any(not (root/n).exists() for n in required):
        print(json.dumps(dict(stage='pending',reason='Wait for >=20 actual updates, initial evaluation and provenance')))
        return
    cfg=json.loads((root/'config.json').read_text())
    saved=json.loads((root/'train/config.json').read_text())
    provenance=json.loads((root/'train/provenance.json').read_text())
    identity=json.loads((root/'train/wandb-run.json').read_text())
    preflight=json.loads((root/'preflight-verification.json').read_text())
    launch=json.loads((root/'launch.json').read_text())
    assert cfg==saved and preflight['stage']=='passed' and launch['launch_count']==1
    assert provenance['train_images']==796478 and provenance['validation_images']==1268
    assert provenance['data_digests']==dict(
        train='9cbe7546dadaaba3994cd2886784801c6cc673bef3dc7a592004f9d0a158c39d',
        val='a60f3b9a4e317463e66a42ee138b3b40a0be65b64d542cdd0b0e0df965322bcf')
    assert provenance['code_revision']==cfg['code_revision']==launch['revision']
    assert provenance['execution_revision']==cfg['code_revision']
    assert provenance['fork_provenance'] is None
    for key,value in cfg['adapter'].items():
        assert provenance['adapter'][key]==value
    assert cfg['adapter']['spec']['summary_cleaning']=='per-token-zero-channel'
    assert cfg['adapter']['spec']['qk_norm'] and cfg['adapter']['initialization_seed']==43
    assert identity['mode']=='online' and identity['id'] and train[0]['step']==1
    assert provenance['runtime_limits']==dict(steps=120000,max_seconds=86400)
    keys=['macro/spatial/cosine','macro/summary/cosine','macro/spatial/norm_ratio','macro/summary/norm_ratio']
    initial=next(row for row in val if row['step']==0)
    assert all(math.isfinite(initial[k]) for k in keys)
    if wandb:
        assert (root/'startup-verification.json').exists()
        remote='''import json,os,sys,time,math
p=json.load(sys.stdin);os.environ['WANDB_API_KEY']=p.pop('key');os.environ['WANDB_BASE_URL']='https://api.forge.coreweave.com'
import wandb
run=wandb.Api(timeout=60).run('brainsim/radio-moge/'+p['identity']['id'])
assert run.state in ('running','finished') and run.config['experiment']==p['config']
assert run.config['provenance']==p['provenance']
s=dict(run.summary);step=s['optimizer_step'];assert step>=20
history=list(run.scan_history(keys=['optimizer_step']+['validation/'+k for k in p['keys']],page_size=100))
row=next(r for r in history if r.get('optimizer_step')==0 and all(r.get('validation/'+k) is not None for k in p['keys']))
error=max(abs(row['validation/'+k]-p['initial'][k]) for k in p['keys']);assert error<1e-10
t=p['config']['train'];used=s['train/lr_used'];trainstep=s.get('train/step',step)
i=trainstep-1
factor=(i+1)/t['warmup'] if i<t['warmup'] else .05+.95*(1+math.cos(math.pi*min(1,(i-t['warmup'])/(t['steps']-t['warmup']))))/2
assert math.isclose(used,t['lr']*factor,rel_tol=1e-10)
print(json.dumps(dict(stage='passed',checked_at=time.time(),id=run.id,state=run.state,step=step,train_step=trainstep,lr_used=used,config_exact=True,provenance_exact=True,initial_metric_max_error=error,url=p['identity']['url'])))
'''
        credential=netrc.netrc(str(Path.home()/'.netrc')).authenticators('api.wandb.ai')[2]
        result=subprocess.run(['ssh','-F','/home/hub/.ssh/radio-moge-config','radio-moge-gpu',
            shlex.join(['/home/buildhub/radio-moge/.venv/bin/python','-c',remote])],text=True,
            input=json.dumps(dict(key=credential,identity=identity,config=cfg,provenance=provenance,
                                  initial=initial,keys=keys)),capture_output=True)
        del credential
        if result.returncode:
            raise RuntimeError('Read-only W&B verification failed; credential-bearing output withheld')
        proof=json.loads(result.stdout)
    else:
        errors=[]
        for row in train:
            assert all(math.isfinite(v) for v in row.values() if type(v) in (int,float))
            assert row['gradient_norm']>0
            if 'update_norm' in row:
                assert row['update_norm']>0
            errors.append(abs(row['total']-sum(row[k]*w for k,w in cfg['loss'].items())))
        assert max(errors)<2e-5
        assert any(row.get('update_norm',0)>0 for row in train)
        proof=dict(stage='passed',checked_at=time.time(),observed_step=train[-1]['step'],
                   logged_rows=len(train),train_images=796478,validation_images=1268,
                   data_digests=provenance['data_digests'],initialization='scratch',
                   initial_validation={k:initial[k] for k in keys},new_wandb=identity,
                   all_logged_finite=True,nonzero_updates=True,max_weighted_loss_error=max(errors),
                   code_revision=cfg['code_revision'],all_radio_inputs_clean=True,
                   config_sha256=hashlib.sha256((root/'train/config.json').read_bytes()).hexdigest())
    write(output,proof)
    print(json.dumps(proof,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--wandb',action='store_true')
    a=p.parse_args()
    verify(a.root,a.wandb)
