"""Start one serial, verified backup of the completed plane comparison."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.incoming')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def start(base, remote_base, name='teacher-plane-subset-pilot-20261028'):
    root, remote = base / name, remote_base / name
    review = root / 'final-review'
    if (review / 'launch.json').exists():
        raise FileExistsError('Backup already launched; inspect its status instead')
    review.mkdir(exist_ok=True)
    source = review / 'source-verification.json'
    if not source.exists():
        code = '''from pathlib import Path
import hashlib,json,time,torch
r=Path(REMOTE_ROOT)
status=json.loads((r/'status.json').read_text())
assert status['stage']=='complete' and status['child_pid'] is None
def identity(p):
 s=p.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
paths=list(r.glob('**/geometry-*.npz'))
paths+=list(r.glob('*.json'))+list(r.glob('*.jsonl'))+list(r.glob('*-console.log'))
for arm in ('plane','control'):
 paths+=list((r/arm).glob('*.json'))+list((r/arm).glob('*.jsonl'))
 paths+=list((r/arm).glob('eval-*/*.json'))
 paths+=list((r/arm).glob('train-probe-*/*.json'))
paths+=list((r/'assessment').glob('*/*/*.json'))
paths=sorted(set(paths),key=lambda p:(p.suffix!='.npz',str(p)))
paths+=[r/arm/file for arm in ('plane','control') for file in ('latest.pt','best.pt')]
files=[]
for p in paths:
 before=identity(p)
 with p.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
 assert identity(p)==before
 item=dict(relative=str(p.relative_to(r)),source=str(p),bytes=p.stat().st_size,sha256=digest,identity=before)
 if p.suffix=='.pt':
  state=torch.load(p,map_location='cpu',weights_only=True)
  counters=sorted({int(s['step'].item()) for s in state['optimizer']['state'].values() if 'step' in s})
  item['checkpoint']=dict(step=state['step'],cursor=state['cursor'],optimizer_lr=state['optimizer']['param_groups'][0]['lr'],adam_steps=counters,best=state['best'],execution_revision=state['execution_revision'],data_digests=state['data_digests'])
  assert state['cursor']==state['step']*2
  if p.name=='latest.pt':
   assert state['step']==status['matched_steps'] and counters==[state['step']]
  assert state['optimizer']['param_groups'][0]['lr']==1e-5
  del state
 assert identity(p)==before
 files.append(item)
print(json.dumps(dict(stage='passed',checked_at=time.time(),status=status,files=files)))
'''.replace('REMOTE_ROOT', repr(str(remote)))
        command = ['/home/buildhub/radio-moge/.venv/bin/python', '-c', code]
        result = subprocess.run(['ssh', '-F', '/home/hub/.ssh/radio-moge-config', 'radio-moge-gpu',
                                 'OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 '
                                 + shlex.join(command)], capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stderr.decode())
        atomic(source, json.loads(result.stdout))
    verified = json.loads(source.read_text())
    staging = review / 'transfer-staging'
    staging.mkdir(exist_ok=True)
    items = []
    for item in verified['files']:
        relative = Path(item['relative'])
        if relative.suffix == '.npz':
            destination = root / relative
        elif relative.suffix == '.pt':
            destination = review / relative.parent / (
                'last.pt' if relative.name == 'latest.pt' else 'selected.pt')
        else:
            destination = review / 'artifacts' / relative
        items.append(dict(source=item['source'], destination=str(destination),
                          bytes=item['bytes'], sha256=item['sha256']))
    plan = dict(purpose='Completed plane/control pilot backup; no inference or training.',
                staging=str(staging), datasets=[], bwlimit_kib=20480,
                ssh_command='ssh -F /home/hub/.ssh/radio-moge-config', ssh_host='radio-moge-gpu',
                source_root=str(remote), checkpoints_first=items)
    atomic(review / 'plan.json', plan)
    script = base / 'code/2bb0342/mirror_attention_datasets.py'
    assert hashlib.sha256(script.read_bytes()).hexdigest() == (
        'd1efdc6cffde534a53e715f659fbd47a418fd169aa910865983019701b9569d8')
    command = ['nice', '-n', '15', '/usr/bin/python3', '-u', str(script),
               '--plan', str(review / 'plan.json')]
    with (review / 'console.log').open('w') as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    launch = dict(pid=child.pid, command=command, started_at=time.time(),
                  files=len(items), total_bytes=sum(i['bytes'] for i in items))
    atomic(review / 'launch.json', launch)
    print(json.dumps(launch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--remote-base', type=Path, required=True)
    parser.add_argument('--name', default='teacher-plane-subset-pilot-20261028')
    args = parser.parse_args()
    start(args.base, args.remote_base, args.name)
