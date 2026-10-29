"""Back up the completed joint run once, using the existing serial copy controller."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from backup_architecture_screen import atomic


REMOTE = r'''
from pathlib import Path
import hashlib,json,math,sys,tarfile,time,torch
torch.set_num_threads(1)
r=Path(json.load(sys.stdin)['root'])
read=lambda p:json.loads(p.read_text())
status=read(r/'status.json');stop=read(r/'train/stop-report.json')
assert status['stage']=='training_complete_review_pending'
assert status['last_exit_code']==0 and status['child_pid'] is None
assert not any(Path('/proc/'+str(pid)).exists() for pid in (41704,41709))
assert stop['reason']=='step_limit' and stop['step']==61367
assert stop['final_evaluation']=='complete' and stop['elapsed_seconds']<86400
wb=read(r/'eval-0061367-wandb-verification.json')
assert wb['stage']=='passed' and wb['state']=='finished' and wb['verified_metrics']==165
assert read(r/'eval-0061367-verification.json')['stage']=='passed'
config=read(r/'config.json');provenance=read(r/'train/provenance.json')
rows=[json.loads(x) for x in (r/'train/validation.jsonl').read_text().splitlines()]
assert len(rows)==61 and len({row['step'] for row in rows})==61
final=next(row for row in rows if row['step']==61367)
assert final['logical_data_cursor']==479996 and final['schedule_position']==119999
selection=config['evaluation']['selection_metric']
selected=min((row for row in rows if row['postprocess_failed']==0),key=lambda row:row[selection])
assert config['evaluation']['selection_metric']=='macro/teacher_depth/abs_rel'
def identity(p):
 s=p.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def digest(p):
 with p.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
checkpoint_names=['train/latest.pt','train/best.pt','train/initial.pt',
 'performance-transition/before/resume.pt','data-pipeline-transition/before/resume.pt']
assert {str(p.relative_to(r)) for p in r.rglob('*.pt')}==set(checkpoint_names)
files=[]
for name in checkpoint_names:
 p=r/name;before=identity(p)
 item=dict(relative=name,source=str(p),bytes=before[2],identity=before)
 if name.endswith('before/resume.pt'):
  prior=read(p.parent.parent/'checkpoint-verification.json');stat=prior['source_stat']
  assert prior['stage']=='passed' and prior['snapshot']==str(p)
  assert (stat['inode'],stat['size'],stat['mtime_ns'])==(before[1],before[2],before[3])
  item.update(sha256=prior['checkpoint_sha256'],prior_source_hash_reused=True,
              previously_verified_step=prior['stopped']['step'])
 else:
  state=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
  assert state['config']==config and state['data_digests']==provenance['data_digests']
  assert state['adapter_config']==provenance['adapter']
  for group in ('adapter','statistics'):
   assert all(torch.isfinite(value).all() for value in state[group].values())
  counters=sorted({int(s['step']) for s in state['optimizer']['state'].values()})
  if name=='train/initial.pt':
   assert state['step']==state['cursor']==0 and counters==[]
  else:
   step=61367 if name=='train/latest.pt' else selected['step']
   assert state['step']==step and counters==[step]
   assert state['cursor']==10940+8*(step-2735)
   assert state['scheduler']['last_epoch']==step
   assert state['execution_revision']=='ac92de9bd38450f8ac41217d6aac565d1b09a9cf'
   assert state['runtime_limits']['steps']==61367 and state['runtime_limits']['max_seconds']==86400
   assert state['performance']['effective_batch']==32 and state['performance']['compute_batch']==16
   assert state['performance']['original_started_at']+86400==1793317558.69
   position=2735+2*(step-2735)
   expected_lr=.0003*(.05+.95*(1+math.cos(math.pi*(position-1000)/119000))/2)
   assert math.isclose(state['optimizer']['param_groups'][0]['lr'],expected_lr,rel_tol=1e-10)
   assert state['best']==selected[selection]
   for moment in state['optimizer']['state'].values():
    assert all(torch.isfinite(value).all() for value in moment.values() if isinstance(value,torch.Tensor))
  item['checkpoint']=dict(step=state['step'],cursor=state['cursor'],adam_steps=counters,
    optimizer_lr=state['optimizer']['param_groups'][0]['lr'],best=state['best'],
    execution_revision=state['execution_revision'],all_adapter_statistics_optimizer_values_finite=True,
    data_digests=state['data_digests'])
  del state
  item['sha256']=digest(p)
 assert identity(p)==before
 files.append(item)
# Keep all run evidence and PNG previews. Cloud W&B internals, event files and locks
# are redundant with the audited JSON history and are intentionally excluded.
metadata=[]
for p in sorted(r.rglob('*')):
 relative=p.relative_to(r)
 if not p.is_file() or any(part.startswith('.') for part in relative.parts):continue
 if 'wandb' in relative.parts or 'final-review' in relative.parts:continue
 if p.suffix not in ('.json','.jsonl','.md','.log','.png'):continue
 assert not p.is_symlink()
 before=identity(p);sha=digest(p);assert identity(p)==before
 metadata.append(dict(relative=str(relative),source=str(p),bytes=before[2],sha256=sha,identity=before))
archive=r/'backup-metadata.tar';temporary=r/'backup-metadata.tar.incoming'
assert not archive.exists() and not temporary.exists(), 'Inspect existing archive; do not restart blindly'
with tarfile.open(temporary,'w') as tar:
 for item in metadata:
  p=Path(item['source']);assert identity(p)==item['identity']
  tar.add(p,arcname=item['relative'],recursive=False)
  assert identity(p)==item['identity']
temporary.replace(archive);before=identity(archive);sha=digest(archive)
assert identity(archive)==before
files+=metadata
files.append(dict(relative=archive.name,source=str(archive),bytes=before[2],sha256=sha,
                  identity=before,metadata_bundle=True))
print(json.dumps(dict(stage='passed',checked_at=time.time(),files=files,
 bundled_metadata=[x['relative'] for x in metadata],stop=stop,
 actual_step=61367,logical_data_cursor=479996,schedule_position=119999,
 image_presentations=1919984,equivalent_epochs=1919984/796478,
 selection_metric=selection,selected_step=selected['step'],selected_score=selected[selection],
 wandb_state=wb['state'],training_processes_absent=True,no_inference=True,
 excluded='W&B internal event/log copies and temporary locks; audited history retained. Datasets and pretrained teachers are outside this run.')))
'''


def start(root, remote):
    review = root / 'final-review'
    if (review / 'launch.json').exists():
        raise FileExistsError('Backup already launched; inspect status, never duplicate it')
    proof = json.loads((root / 'eval-0061367-wandb-verification.json').read_text())
    assert proof['stage'] == 'passed' and proof['state'] == 'finished'
    review.mkdir(exist_ok=True)
    source = review / 'source-verification.json'
    if not source.exists():
        command = ['ssh', '-F', '/home/hub/.ssh/radio-moge-config', 'radio-moge-gpu',
                   'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '
                   + shlex.join(['/home/buildhub/radio-moge/.venv/bin/python', '-c', REMOTE])]
        response = subprocess.run(command, input=json.dumps(dict(root=remote)),
                                  text=True, capture_output=True, check=True)
        atomic(source, json.loads(response.stdout))
    verified = json.loads(source.read_text())
    assert verified['stage'] == 'passed'
    staging = review / 'transfer-staging'
    staging.mkdir(exist_ok=True)
    items = []
    for item in verified['files']:
        if item['relative'] in verified['bundled_metadata']:
            continue
        relative = Path(item['relative'])
        if relative == Path('train/latest.pt'):
            destination = review / 'last.pt'
        elif relative == Path('train/best.pt'):
            destination = review / 'selected.pt'
        else:
            destination = review / relative
        items.append(dict(source=item['source'], destination=str(destination),
                          bytes=item['bytes'], sha256=item['sha256']))
    total_bytes = sum(item['bytes'] for item in items)
    assert shutil.disk_usage(review).free > total_bytes + 1024**3
    plan = dict(purpose='Completed joint full-data run; final, selected, initial and preserved resume checkpoints plus evidence.',
                staging=str(staging), datasets=[], bwlimit_kib=20480,
                ssh_command='ssh -F /home/hub/.ssh/radio-moge-config', ssh_host='radio-moge-gpu',
                source_root=remote, checkpoints_first=items)
    atomic(review / 'plan.json', plan)
    script = root.parent / 'code/2bb0342/mirror_attention_datasets.py'
    assert hashlib.sha256(script.read_bytes()).hexdigest() == (
        'd1efdc6cffde534a53e715f659fbd47a418fd169aa910865983019701b9569d8')
    command = ['nice', '-n', '15', '/usr/bin/python3', '-u', str(script),
               '--plan', str(review / 'plan.json')]
    with (review / 'console.log').open('w') as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    launch = dict(pid=child.pid, command=command, started_at=time.time(),
                  files=len(items), total_bytes=total_bytes)
    atomic(review / 'launch.json', launch)
    print(json.dumps(launch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('remote_root')
    args = parser.parse_args()
    start(args.root, args.remote_root)
