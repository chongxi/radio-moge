"""Back up the stopped 30k continuation, preserving all independent best checkpoints."""

import argparse
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from backup_architecture_screen import atomic
from finalize_architecture_backup import finish as finish_copies


def selections_at(specs, rows, step):
    """Recompute each historical best, keeping the earlier checkpoint on ties."""
    best = {name: dict(value=s['baseline'], step=61367, metric=s['metric'], mode=s['mode'])
            for name, s in specs.items()}
    for row in rows:
        if row['step'] > step or row['postprocess_failed'] != 0:
            continue
        for name, spec in specs.items():
            value, previous = row[spec['metric']], best[name]['value']
            if value < previous if spec['mode'] == 'min' else value > previous:
                best[name] = dict(value=value, step=row['step'],
                                  metric=spec['metric'], mode=spec['mode'])
    return best


REMOTE = r'''
from pathlib import Path
import hashlib,json,sys,tarfile,time,torch
torch.set_num_threads(1)
r=Path(json.load(sys.stdin)['root'])
read=lambda p:json.loads(p.read_text())
status=read(r/'status.json');stop=read(r/'train/stop-report.json')
assert status['stage']=='training_complete_review_pending'
assert status['last_exit_code']==0 and status['child_pid'] is None
assert status['parent_identity_unchanged']
assert not any(Path('/proc/'+str(pid)).exists() for pid in (200609,200615))
assert stop['reason']=='step_limit' and stop['step']==91367
assert stop['final_evaluation']=='complete' and stop['elapsed_seconds']<28800
assert read(r/'completion-verification.json')['stage']=='passed'
for step in range(62367,91368,1000):
 audit=read(r/f'eval-{step:07d}-verification.json')
 wb=read(r/f'eval-{step:07d}-wandb-verification.json')
 assert audit['stage']=='passed' and audit['step']==step and audit['images']==1268
 assert audit['nyu_sensor_rows']==116 and audit['postprocess_failures']==0
 assert wb['stage']=='passed' and wb['id']=='d382894f' and wb['step']==step
 assert wb['verified_metrics']==165 and wb['max_metric_difference']==0
assert wb['state']=='finished'
config=read(r/'config.json');provenance=read(r/'train/provenance.json')
rows=[json.loads(x) for x in (r/'train/validation.jsonl').read_text().splitlines()]
assert [row['step'] for row in rows]==list(range(62367,91368,1000))
final=rows[-1]
assert final['logical_data_cursor']==719996 and final['schedule_position']==179999
specs=config['evaluation']['checkpoint_metrics']
assert set(specs)=={'spatial','summary','teacher-depth','nyu-absrel','nyu-rmse','nyu-delta1'}
selections=selections_at(specs,rows,91367)
assert selections==read(r/'train/checkpoint-selection.json')
assert config['evaluation']['selection_metric']=='macro/teacher_depth/abs_rel'
expected={'train/latest.pt':91367,'train/initial.pt':61367,
          'train/best.pt':selections['teacher-depth']['step']}
expected.update({f'train/best-{name}.pt':item['step'] for name,item in selections.items()})
assert {str(p.relative_to(r)) for p in r.rglob('*.pt')}==set(expected)
def identity(p):
 s=p.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def digest(p):
 with p.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
files=[];inodes={};aliases={}
for name,step in expected.items():
 p=r/name;assert not p.is_symlink();before=identity(p)
 item=dict(relative=name,source=str(p),bytes=before[2],identity=before)
 key=tuple(before[:2])
 if key in inodes:
  prior=inodes[key]
  assert before==prior['identity'] and prior['checkpoint']['step']==step
  item.update(sha256=prior['sha256'],checkpoint=prior['checkpoint'],
              hardlink_of=prior['relative'])
  aliases[name]=prior['relative']
 else:
  state=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
  assert state['config']==config and state['data_digests']==provenance['data_digests']
  assert state['adapter_config']==provenance['adapter']
  assert state['revisions']==provenance['revisions']
  assert state['fork_provenance']==provenance['fork_provenance']
  fork=state['fork_provenance']
  assert fork['step']==61367 and fork['cursor']==479996 and fork['adam_steps']==[61367]
  assert fork['budget_started_at']==1793288222.1457853
  assert fork['lr_schedule']=='constant' and fork['optimizer_lr']==1.5e-5
  assert state['step']==step and state['cursor']==479996+8*(step-61367)
  counters=sorted({int(s['step']) for s in state['optimizer']['state'].values()})
  assert counters==[step] and state['scheduler']['last_epoch']==step-61367
  assert all(g['lr']==1.5e-5 for g in state['optimizer']['param_groups'])
  assert state['execution_revision']=='b7aed80a4fec5b992df8185c6b5199311d277c01'
  assert state['runtime_limits']==dict(steps=91367,max_seconds=28800)
  assert state['performance']['effective_batch']==32 and state['performance']['compute_batch']==16
  historical=selections_at(specs,rows,step)
  assert state['best_metrics']==historical and state['best']==historical['teacher-depth']['value']
  for group in ('adapter','statistics'):
   assert all(torch.isfinite(value).all() for value in state[group].values())
  for moment in state['optimizer']['state'].values():
   assert all(torch.isfinite(v).all() for v in moment.values() if isinstance(v,torch.Tensor))
  item['checkpoint']=dict(step=step,cursor=state['cursor'],adam_steps=counters,
   optimizer_lr=state['optimizer']['param_groups'][0]['lr'],best=state['best'],
   best_metrics=historical,scheduler_epoch=state['scheduler']['last_epoch'],
   execution_revision=state['execution_revision'],data_digests=state['data_digests'],
   all_adapter_statistics_optimizer_values_finite=True,restored_parent_state_verified=True)
  del state
  item['sha256']=digest(p)
  inodes[key]=item
 assert identity(p)==before
 files.append(item)
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
assert not archive.exists() and not temporary.exists(),'Inspect existing archive; never restart blindly'
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
 aliases=aliases,unique_checkpoint_files=len(inodes),checkpoint_names=len(expected),
 bundled_metadata=[x['relative'] for x in metadata],stop=stop,
 actual_step=91367,logical_data_cursor=719996,schedule_position=179999,
 new_updates=30000,new_image_presentations=960000,image_presentations=2879984,
 equivalent_epochs=2879984/796478,independent_selections=selections,
 selection_metric=config['evaluation']['selection_metric'],
 selected_step=selections['teacher-depth']['step'],
 selected_score=selections['teacher-depth']['value'],
 wandb_state=wb['state'],training_processes_absent=True,no_inference=True,
 excluded='W&B internal event/log copies and temporary locks; audited history retained. Datasets and pretrained teachers are outside this run.')))
'''


def transfer_items(source, review):
    by_name = {item['relative']: item for item in source['files']}
    aliases = source['aliases']
    for name, primary in aliases.items():
        assert primary not in aliases and name != primary
        for key in ('identity', 'sha256', 'bytes', 'checkpoint'):
            assert by_name[name][key] == by_name[primary][key]
    items = []
    for name, item in by_name.items():
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        if name in source['bundled_metadata'] or name in aliases:
            continue
        items.append(dict(source=item['source'], destination=str(review / relative),
                          bytes=item['bytes'], sha256=item['sha256']))
    return items


def start(root, remote, *, remote_verifier=None, expected_step=91367, purpose=None):
    review = root / 'final-review'
    review.mkdir(exist_ok=True)
    with (review / '.backup-start.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (review / 'launch.json').exists():
            raise FileExistsError('Backup already launched; inspect it, never duplicate it')
        assert json.loads((root / 'completion-verification.json').read_text())['stage'] == 'passed'
        source = review / 'source-verification.json'
        if not source.exists():
            code = (inspect.getsource(selections_at) + '\n' + REMOTE
                    if remote_verifier is None else remote_verifier)
            command = ['ssh', '-F', '/home/hub/.ssh/radio-moge-config', 'radio-moge-gpu',
                       'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '
                       + shlex.join(['/home/buildhub/radio-moge/.venv/bin/python', '-c', code])]
            response = subprocess.run(command, input=json.dumps(dict(root=remote)),
                                      text=True, capture_output=True)
            if response.returncode:
                raise RuntimeError(response.stderr)
            atomic(source, json.loads(response.stdout))
        verified = json.loads(source.read_text())
        assert verified['stage'] == 'passed' and verified['actual_step'] == expected_step
        items = transfer_items(verified, review)
        total = sum(item['bytes'] for item in items)
        assert shutil.disk_usage(review).free > total + 1024**3
        staging = review / 'transfer-staging'
        staging.mkdir(exist_ok=True)
        plan = dict(purpose=purpose or 'Completed 30000-update continuation: final, initial and all six independent best checkpoints, with audited evidence.',
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
                      files=len(items), total_bytes=total,
                      checkpoint_names=verified['checkpoint_names'],
                      unique_checkpoint_files=verified['unique_checkpoint_files'])
        atomic(review / 'launch.json', launch)
        print(json.dumps(launch))


def restore_aliases(source, review, receipts):
    expected = transfer_items(source, review)
    assert receipts == expected
    by_name = {item['relative']: item for item in source['files']}
    for name, primary in source['aliases'].items():
        destination, original = review / name, review / primary
        assert original.stat().st_size == by_name[name]['bytes']
        assert not original.is_symlink() and not destination.is_symlink()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            assert destination.samefile(original), 'Unexpected alias destination; never overwrite'
        else:
            os.link(original, destination)
        assert destination.samefile(original)
    return dict(stage='passed', aliases=source['aliases'],
                all_aliases_share_verified_primary_inodes=True)


def finish(root):
    review = root / 'final-review'
    proof = review / 'backup-verification.json'
    if proof.exists():
        assert json.loads(proof.read_text())['stage'] == 'passed'
        assert json.loads((review / 'alias-verification.json').read_text())['stage'] == 'passed'
        print('{"stage":"already_verified"}')
        return
    state = json.loads((review / 'prepared-dataset-backup-status.json').read_text())
    if state['stage'] != 'all_copies_finished_pending_verification':
        print(json.dumps(dict(stage=state['stage'], verified=len(state['verified_checkpoints']))))
        return
    assert state['child_pid'] is None
    source = json.loads((review / 'source-verification.json').read_text())
    aliases = restore_aliases(source, review, state['verified_checkpoints'])
    atomic(review / 'alias-verification.json', dict(aliases, checked_at=time.time()))
    finish_copies(root)
    report = json.loads(proof.read_text())
    report.update(alias_verification=aliases, checkpoint_names=source['checkpoint_names'],
                  unique_checkpoint_files=source['unique_checkpoint_files'])
    atomic(proof, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'finish'])
    parser.add_argument('root', type=Path)
    parser.add_argument('remote_root', nargs='?')
    args = parser.parse_args()
    if args.action == 'start':
        assert args.remote_root
        start(args.root, args.remote_root)
    else:
        finish(args.root)
