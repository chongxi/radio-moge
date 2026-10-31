"""Verify a stopped data pilot and one qualified continuation checkpoint, without inference."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

from audit_data_evaluations import continuation_gate
from monitor_joint_full_day import write


def review(plan, rows, audits, wandb, selected_step):
    expected = list(range(plan['parent_step'], plan['final_step'] + 1, 1000))
    if plan['phase'] != 'pilot' or plan['additional_updates'] != 3000:
        raise ValueError('This verifier is only for the completed bounded 3000-update pilot')
    if [r['step'] for r in rows] != expected or selected_step not in expected[1:]:
        raise ValueError('Incomplete evaluation history or invalid selected source')
    for step in expected:
        audit, wb = audits[step], wandb[step]
        if audit['stage'] != 'passed' or audit['step'] != step:
            raise ValueError('Missing independent evaluation audit')
        if set(audit['views']) != {'original', 'native-640x400', 'dataset-scannetpp-dslr'}:
            raise ValueError('All three validation views are required')
        for name, view in audit['views'].items():
            count, gt = (128, 0) if name.startswith('dataset-') else (1268, 116)
            if (view['images'] != count or view['nyu_sensor_rows'] != gt
                    or view['postprocess_failures'] != 0
                    or not view['all_per_image_values_finite']
                    or not view['exact_identity_order_source_match']):
                raise ValueError('Invalid validation audit')
        if (wb['stage'] != 'passed' or wb['step'] != step
                or wb['verified_metrics'] != 403 or wb['max_metric_difference'] != 0
                or not wb['config_verified'] or wb['id'] != wandb[expected[0]]['id']):
            raise ValueError('Missing matching W&B verification')
    if wandb[expected[-1]]['state'] != 'finished':
        raise ValueError('Final W&B run must be finished')
    gates = {r['step']: continuation_gate(rows[0], r, plan['continuation_guardrails'])
             for r in rows[1:]}
    if gates[selected_step]['stage'] != 'passed':
        raise ValueError('Selected source fails unchanged continuation guardrails')
    return dict(selected_step=selected_step, gates=gates,
                final_checkpoint_qualified=gates[expected[-1]]['stage'] == 'passed',
                selected_checkpoint_qualified=True,
                failed_checkpoint_steps=[s for s, g in gates.items() if g['stage'] != 'passed'])


REMOTE = r'''
import fcntl,hashlib,json,math,pathlib,subprocess,sys,time
payload=json.load(sys.stdin);r=pathlib.Path(payload['root']);read=lambda p:json.loads(p.read_text())
plan=read(r/'plan.json');config=read(r/'config.json');status=read(r/'status.json')
assert plan==payload['plan'] and config==payload['config']
stop=read(r/'train/stop-report.json');provenance=read(r/'train/provenance.json')
rows=[json.loads(x) for x in (r/'train/validation.jsonl').read_text().splitlines()]
assert rows==payload['rows']
assert status['stage']=='training_complete_review_pending'
assert status['last_exit_code']==0 and status['child_pid'] is None
assert status['parent_identity_unchanged'] and status['execution_revision']==payload['revision']
assert stop['reason']=='step_limit' and stop['step']==plan['final_step']
assert stop['final_evaluation']=='complete' and stop['elapsed_seconds']<plan['max_seconds']
assert not any(pathlib.Path('/proc/'+str(pid)).exists() for pid in payload['pids'])
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
def identity(p):
 s=p.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
assert identity(pathlib.Path(plan['parent_checkpoint']))==plan['parent_identity']
relative=pathlib.PurePosixPath(payload['selected_relative'])
assert relative.parent==pathlib.PurePosixPath('train') and relative.suffix=='.pt'
assert not (r/relative).is_symlink()
step=payload['selected_step'];selected_row=next(x for x in rows if x['step']==step)
specs=config['evaluation']['checkpoint_metrics'];best={}
for row in rows:
 if row['step']>step:break
 for name,spec in specs.items():
  value=row[spec['metric']];assert math.isfinite(value)
  old=best.get(name,{}).get('value')
  if old is None or (value<old if spec['mode']=='min' else value>old):
   best[name]=dict(value=value,step=row['step'],metric=spec['metric'],mode=spec['mode'])
selection=read(r/'train/checkpoint-selection.json')
if str(relative)!='train/latest.pt':
 name=relative.stem.removeprefix('best-');assert selection[name]['step']==step
else:assert step==plan['final_step']
with pathlib.Path(plan['training_lock']).open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 import torch
 torch.set_num_threads(1)
 checkpoint=r/relative;before=identity(checkpoint)
 state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
 assert state['step']==step and state['cursor']==selected_row['logical_data_cursor']
 assert state['cursor']==(step-plan['parent_step'])*plan['effective_batch']//config['train']['batch_size']
 assert state['config']==config and state['data_digests']==provenance['data_digests']
 assert state['adapter_config']==provenance['adapter'] and state['revisions']==provenance['revisions']
 assert state['fork_provenance']==provenance['fork_provenance']
 fork=state['fork_provenance'];assert fork['step']==plan['parent_step'] and fork['cursor']==0
 assert fork['data_change']['parent_cursor']==plan['parent_cursor'] and fork['data_change']['new_cursor']==0
 assert fork['adam_steps']==[plan['parent_step']] and fork['checkpoint']==plan['parent_checkpoint']
 assert set(fork['restored'])=={'adapter','statistics','optimizer_moments','rng'}
 assert 'data_cursor_for_new_manifest' in fork['reset'] and fork['optimizer_lr']==plan['learning_rate']
 assert fork['budget_started_at']==status['started_at']
 adam=sorted({int(x['step']) for x in state['optimizer']['state'].values()});assert adam==[step]
 assert state['scheduler']['last_epoch']==step-plan['parent_step']
 assert all(g['lr']==plan['learning_rate'] for g in state['optimizer']['param_groups'])
 assert state['execution_revision']==payload['revision']
 assert state['runtime_limits']==dict(steps=plan['final_step'],max_seconds=plan['max_seconds'])
 assert state['performance']==provenance['performance']
 assert state['performance']['effective_batch']==32 and state['performance']['compute_batch']==32
 assert state['best_metrics']==best and state['best']==best['teacher-depth']['value']
 for group in ('adapter','statistics'):
  assert state[group] and all(torch.isfinite(v).all() for v in state[group].values())
 for moment in state['optimizer']['state'].values():
  assert all(torch.isfinite(v).all() for v in moment.values() if isinstance(v,torch.Tensor))
 assert state['rng']
 info=dict(step=step,cursor=state['cursor'],adam_steps=adam,
  scheduler_epoch=state['scheduler']['last_epoch'],optimizer_lr=plan['learning_rate'],
  best_metrics=best,data_digests=state['data_digests'],execution_revision=state['execution_revision'],
  all_adapter_statistics_optimizer_values_finite=True,parent_restore_metadata_verified=True)
 del state
 with checkpoint.open('rb') as handle:sha=hashlib.file_digest(handle,'sha256').hexdigest()
 assert identity(checkpoint)==before
 fcntl.flock(lock,fcntl.LOCK_UN)
print(json.dumps(dict(stage='passed',checked_at=time.time(),files=[dict(relative=str(relative),
 source=str(checkpoint),identity=before,bytes=before[2],sha256=sha,checkpoint=info)],
 stop=stop,status=status,training_processes_absent=True,gpu_compute_empty=True,training_lock_free=True,
 original_parent_identity_unchanged=True,no_inference=True,
 scope='Only the selected stopped pilot checkpoint was loaded and hashed; historical source was stat-only.')))
'''


def verify(root, remote, selected_relative, selected_step):
    def read(path):
        return json.loads(path.read_text())
    source_path = root / 'final-review/source-verification.json'
    completion_path = root / 'completion-verification.json'
    if source_path.exists() or completion_path.exists():
        raise FileExistsError('Existing proof: inspect instead of repeating checkpoint reads')
    plan, config = read(root/'plan.json'), read(root/'config.json')
    rows = [json.loads(s) for s in (root/'train/validation.jsonl').read_text().splitlines()]
    steps = [r['step'] for r in rows]
    audits = {s: read(root/f'eval-{s:07d}-data-verification.json') for s in steps}
    wandb = {s: read(root/f'eval-{s:07d}-wandb-verification.json') for s in steps}
    decision = review(plan, rows, audits, wandb, selected_step)
    continuation = read(root/'continuation.json')
    payload = dict(root=remote, plan=plan, config=config, rows=rows,
                   selected_relative=selected_relative, selected_step=selected_step,
                   revision=continuation['launch']['revision'],
                   pids=[continuation['launch']['pid'],
                         continuation['controller_state']['child_pid'], continuation['trainer_pid']])
    result = subprocess.run(['ssh', '-F', '/home/hub/.ssh/radio-moge-config', 'radio-moge-gpu',
                             shlex.join(['/home/buildhub/radio-moge/.venv/bin/python', '-c', REMOTE])],
                            input=json.dumps(payload), text=True, capture_output=True, check=True)
    source = json.loads(result.stdout)
    source_path.parent.mkdir(exist_ok=True)
    write(source_path, source)
    completion = dict(stage='passed', checked_at=time.time(), **decision,
                      actual_final_step=plan['final_step'], actual_updates=plan['additional_updates'],
                      selected_relative=selected_relative, source_verification=str(source_path),
                      all_evaluation_audits_passed=True, wandb_state='finished',
                      stop=source['stop'], processes_absent=True, parent_identity_unchanged=True,
                      scope='Clean pilot completion and qualified selected source. Does not assert '
                            'all pilot checkpoints passed continuation guardrails.')
    write(completion_path, completion)
    print(json.dumps(completion))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('remote_root')
    parser.add_argument('--selected-relative', required=True)
    parser.add_argument('--selected-step', type=int, required=True)
    args = parser.parse_args()
    verify(args.root, args.remote_root, args.selected_relative, args.selected_step)
