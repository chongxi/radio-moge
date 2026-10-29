"""Read-only remote monitor; copies small evidence, never weights or process environments."""

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess

from render_joint_full_page import render, lines


def write(path, value):
    temp=path.with_suffix('.incoming')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    temp.replace(path)


def monitor(root, remote):
    known={}
    for p in (root/'train').glob('eval-*/*.json'):
        known[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
    code='''import base64,hashlib,json,subprocess,sys,time
from pathlib import Path
payload=json.load(sys.stdin);r=Path(payload['root']);known=payload['known']
state=json.loads((r/'status.json').read_text()) if (r/'status.json').exists() else {}
names=['status.json','dataset-verification.json','preflight.json','preflight-console.log','training-console.log','train/config.json','train/provenance.json','train/wandb-run.json','train/train.jsonl','train/validation.jsonl','train/stop-report.json','train/resume-state.json','train/performance.json','train/throughput-benchmark.json','performance-transition/status.json','performance-transition/resume-console.log','performance-transition/launch.json','performance-transition/controller-launch.json','performance-transition/controller-console.log','performance-transition/checkpoint-verification.json','performance-transition/request.json']
names += ['train/data-pipeline-benchmark.json']
names += ['data-pipeline-transition/'+n for n in ['status.json','resume-console.log','launch.json','controller-launch.json','controller-console.log','checkpoint-verification.json','request.json']]
names += [str(p.relative_to(r)) for p in (r/'train').glob('eval-*/*.json')]
files={}
for name in names:
 p=r/name
 if not p.exists():continue
 if name in known:continue
 for _ in range(3):
  a=p.stat();data=p.read_bytes();b=p.stat()
  if (a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns):break
 else:continue
 if len(data)>24000000:raise ValueError('Unexpected oversized small artifact')
 sha=hashlib.sha256(data).hexdigest()
 files[name]=dict(data=base64.b64encode(data).decode(),sha256=sha,bytes=len(data),mtime_ns=b.st_mtime_ns)
pids=[str(state[k]) for k in ('pid','child_pid') if state.get(k)]
processes=subprocess.run(['ps','-p',','.join(pids),'-o','pid,ppid,stat,etime,time,args'],capture_output=True,text=True).stdout if pids else ''
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True)
print(json.dumps(dict(checked_at=time.time(),files=files,processes=processes,gpu=gpu,state=state)))
'''
    result=subprocess.run(['ssh','-F','/home/hub/.ssh/radio-moge-config','radio-moge-gpu',
                           shlex.join(['/opt/conda/bin/python3','-c',code])],
                          input=json.dumps(dict(root=remote,known=known)),text=True,capture_output=True,check=True)
    result=json.loads(result.stdout)
    receipts={}
    for name,row in result.pop('files').items():
        data=base64.b64decode(row['data'])
        assert hashlib.sha256(data).hexdigest()==row['sha256']
        p=root/name
        p.parent.mkdir(parents=True,exist_ok=True)
        t=p.with_suffix(p.suffix+'.incoming')
        t.write_bytes(data)
        t.replace(p)
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
        receipts[name]={k:v for k,v in row.items() if k!='data'}
    config=json.loads((root/'config.json').read_text())
    rows=lines(root/'train/train.jsonl')
    for row in rows:
        assert all(math.isfinite(v) for v in row.values() if type(v) in (int,float))
        step=row['step']
        t=config['train']
        def factor(s):
            if s<t['warmup']:
                return (s+1)/t['warmup']
            p=min(1,max(0,(s-t['warmup'])/(t['steps']-t['warmup'])))
            return .05+.95*(1+math.cos(math.pi*p))/2
        schedule = row.get('schedule_position', step)
        ratio = row.get('effective_batch', t['batch_size']*t['accumulation']) / (t['batch_size']*t['accumulation'])
        assert math.isclose(row['lr_used'],t['lr']*factor(schedule-ratio),rel_tol=1e-10)
        assert math.isclose(row['lr'],t['lr']*factor(schedule),rel_tol=1e-10)
        assert row['gradient_norm']>0
        if 'update_norm' in row:
            assert row['update_norm']>0
        assert math.isclose(row['total'],sum(row[k]*w for k,w in config['loss'].items()),rel_tol=2e-5,abs_tol=2e-6)
    result.update(logged_rows=len(rows),last_step=rows[-1]['step'] if rows else 0,
                  all_logged_finite=True,actual_lr_verified=bool(rows),copies=receipts)
    write(root/'monitor.json',result)
    handoff_path=root.parent/'handoff.json'
    handoff=json.loads(handoff_path.read_text())
    handoff['active_joint_full_training']=dict(root=str(root),remote_root=remote,
        stage=result['state'].get('stage','preparing'),last_step=result['last_step'],
        checked_at=result['checked_at'],summary_cleaning='per-token-zero-channel',
        remaining=['monitor bounded 24h training','audit completed intermediate evaluations',
                   'final evaluation and verified serial checkpoint backup'])
    write(handoff_path,handoff)
    render(root,Path('.kanna/visualizations/teacher-plane-pilot.html'))
    print(json.dumps({k:v for k,v in result.items() if k not in ('copies','state')},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('remote_root')
    a=p.parse_args()
    monitor(a.root,a.remote_root)
