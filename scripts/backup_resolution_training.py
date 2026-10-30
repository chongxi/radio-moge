"""Verify and back up the stopped 3k mixed-resolution trial exactly once."""

import argparse
import inspect
import math
from pathlib import Path

from backup_joint_continuation import finish, start


def selections_at(specs, rows, step, before_initial=False):
    """Independent history, including native metrics first established at baseline."""
    best = {name: dict(value=s['baseline'], step=91367, metric=s['metric'], mode=s['mode'])
            for name, s in specs.items() if 'baseline' in s}
    for row in rows:
        if row['step'] > step or before_initial or row['postprocess_failed'] != 0:
            continue
        for name, spec in specs.items():
            metric = spec['metric']
            if metric.startswith('resolution/'):
                prefix = '/'.join(metric.split('/')[:2])
                if row[prefix + '/postprocess_failed'] != 0:
                    continue
            value = row[metric]
            assert math.isfinite(value)
            previous = best.get(name, {}).get('value')
            if previous is None or (value < previous if spec['mode'] == 'min' else value > previous):
                best[name] = dict(value=value, step=row['step'], metric=metric, mode=spec['mode'])
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
assert not any(Path('/proc/'+str(pid)).exists() for pid in (229,235,236))
assert stop['reason']=='step_limit' and stop['step']==94367
assert stop['final_evaluation']=='complete' and stop['elapsed_seconds']<7200
assert read(r/'completion-verification.json')['stage']=='passed'
assert read(r/'startup-verification.json')['stage']=='passed'
for step in range(91367,94368,1000):
 audit=read(r/f'eval-{step:07d}-resolution-verification.json')
 wb=read(r/f'eval-{step:07d}-wandb-verification.json')
 assert audit['stage']=='passed' and audit['step']==step
 assert set(audit['views'])=={'original','native-640x400'}
 for view in audit['views'].values():
  assert view['images']==1268 and view['nyu_sensor_rows']==116
  assert view['postprocess_failures']==0 and view['all_per_image_values_finite']
  assert view['exact_identity_order_source_match'] and view['max_frozen_teacher_difference']==0
 assert wb['stage']=='passed' and wb['id']=='dd5bc333' and wb['step']==step
 assert wb['verified_metrics']==325 and wb['max_metric_difference']==0
assert wb['state']=='finished'
config=read(r/'config.json');provenance=read(r/'train/provenance.json')
rows=[json.loads(x) for x in (r/'train/validation.jsonl').read_text().splitlines()]
assert [row['step'] for row in rows]==list(range(91367,94368,1000))
final=rows[-1]
assert final['logical_data_cursor']==743996 and final['schedule_position']==94367
specs=config['evaluation']['checkpoint_metrics']
assert set(specs)=={'spatial','summary','teacher-depth','nyu-absrel','nyu-rmse','nyu-delta1',
                   '640x400-spatial','640x400-nyu-absrel'}
selections=selections_at(specs,rows,94367)
assert selections==read(r/'train/checkpoint-selection.json')
assert config['evaluation']['selection_metric']=='macro/teacher_depth/abs_rel'
assert config['data']['buckets']==[[480,640],[640,480],[400,640]]
expected={'train/latest.pt':94367,'train/initial.pt':91367,
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
  item.update(sha256=prior['sha256'],checkpoint=prior['checkpoint'],hardlink_of=prior['relative'])
  aliases[name]=prior['relative']
 else:
  state=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
  assert state['config']==config and state['data_digests']==provenance['data_digests']
  assert state['adapter_config']==provenance['adapter']
  assert state['revisions']==provenance['revisions']
  assert state['fork_provenance']==provenance['fork_provenance']
  fork=state['fork_provenance']
  assert fork['step']==91367 and fork['cursor']==719996 and fork['adam_steps']==[91367]
  assert fork['budget_started_at']==1793334796.4264562
  assert fork['lr_schedule']=='constant' and fork['optimizer_lr']==1.5e-5
  assert fork['resolution_change']['parent_buckets']==[[480,640],[640,480]]
  assert fork['resolution_change']['buckets']==config['data']['buckets']
  assert state['step']==step and state['cursor']==719996+8*(step-91367)
  counters=sorted({int(s['step']) for s in state['optimizer']['state'].values()})
  assert counters==[step] and state['scheduler']['last_epoch']==step-91367
  assert all(g['lr']==1.5e-5 for g in state['optimizer']['param_groups'])
  assert state['execution_revision']=='8cd1adb1a9ed9f85c95054ecc5846d71736a7678'
  assert state['runtime_limits']==dict(steps=94367,max_seconds=7200)
  assert state['performance']==provenance['performance']
  assert state['performance']['effective_batch']==32 and state['performance']['compute_batch']==32
  historical=selections_at(specs,rows,step,before_initial=name=='train/initial.pt')
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
 actual_step=94367,logical_data_cursor=743996,schedule_position=94367,
 new_updates=3000,new_image_presentations=96000,
 independent_selections=selections,selection_metric=config['evaluation']['selection_metric'],
 selected_step=selections['teacher-depth']['step'],selected_score=selections['teacher-depth']['value'],
 wandb_state=wb['state'],training_processes_absent=True,no_inference=True,
 excluded='W&B internal event/log copies and temporary locks; audited history retained. Datasets and pretrained teachers are outside this run.')))
'''


def remote_verifier():
    return 'import math\n' + inspect.getsource(selections_at) + '\n' + REMOTE


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'finish'])
    parser.add_argument('root', type=Path)
    parser.add_argument('remote_root', nargs='?')
    args = parser.parse_args()
    if args.action == 'start':
        assert args.remote_root
        start(args.root, args.remote_root, remote_verifier=remote_verifier(), expected_step=94367,
              purpose='Completed 3000-update mixed-resolution trial: final, initial and all eight independent best checkpoints, with both validation views.')
    else:
        finish(args.root)
