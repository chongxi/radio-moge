"""Atomically synchronize local review metadata; never overwrite trainer-owned files."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


def sync(root, remote_base):
    if (root/'final-review/launch.json').exists():
        proof=root/'final-review/backup-verification.json'
        assert proof.exists() and json.loads(proof.read_text())['stage']=='passed'
    names=['continuation.json','code-deployment.json','configuration-verification.json',
           'initialization-recovery.json','initialization-failure-review.json',
           'preflight-verification.json',
           'startup-verification.json','startup-wandb-verification.json','monitor.json',
           'browser-startup-verification.json','experiment.md','completion-verification.json',
           'performance-plan.json','performance-code-deployment.json','performance-verification.json',
           'performance-wandb-verification.json','performance-utilization-verification.json',
           'throughput-benchmark-verification.json','performance-page-verification.json',
           'utilization-observation.json','batch-literature.md',
           'data-pipeline-plan.json','data-pipeline-code-deployment.json',
           'data-pipeline-verification.json','batch-fetch-analysis.json',
           'data-pipeline-wandb-verification.json','data-pipeline-utilization-verification.json',
           'data-pipeline-page-verification.json',
           'data-pipeline-recovery-plan.json','compact-transport-code-deployment.json',
           'data-pipeline-failure-review.json']
    paths=[root.parent/'handoff.json']+[root/n for n in names]
    paths += [root/'final-review'/name for name in
              ('source-verification.json','plan.json','launch.json',
               'prepared-dataset-backup-status.json','backup-verification.json')]
    paths+=list(root.glob('eval-*-verification.json'))
    download=root.parent/'scannetpp-download-20261028'
    paths += [download/name for name in ('plan.json','launch.json','status.json','monitor.json',
              'continuation.json','deployment-verification.json','monitor-copy-verification.json',
              'page-verification.json')]
    files={}
    for p in paths:
        if p.exists():
            data=p.read_bytes()
            files[str(p.relative_to(root.parent))]=dict(data=base64.b64encode(data).decode(),
                sha256=hashlib.sha256(data).hexdigest())
    code='''import base64,hashlib,json,sys
from pathlib import Path
p=json.load(sys.stdin);b=Path(p['base'])
for relative,row in p['files'].items():
 f=b/relative;f.parent.mkdir(parents=True,exist_ok=True);d=base64.b64decode(row['data'])
 assert hashlib.sha256(d).hexdigest()==row['sha256']
 t=f.with_suffix('.incoming');t.write_bytes(d);t.replace(f)
 assert hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256']
print(json.dumps(dict(stage='passed',files=len(p['files']),sha256={k:v['sha256'] for k,v in p['files'].items()})))
'''
    x=subprocess.run(['ssh','-F','/home/hub/.ssh/radio-moge-config','radio-moge-gpu',
                      shlex.join(['/opt/conda/bin/python3','-c',code])],
                     input=json.dumps(dict(base=remote_base,files=files)),
                     text=True,capture_output=True,check=True)
    proof=json.loads(x.stdout)
    proof['checked_at']=time.time()
    p=root/'metadata-remote-sync.json'
    t=p.with_suffix('.incoming')
    t.write_text(json.dumps(proof,indent=2)+'\n')
    t.replace(p)
    print(json.dumps(dict(stage='passed',files=proof['files'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('remote_base')
    a=p.parse_args()
    sync(a.root,a.remote_base)
