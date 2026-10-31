"""One user-authorized exact resume after changing the external review policy."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from run_data_continuation import atomic, identity


def validate(plan, proof, config, now, request=None):
    source_lr = plan['learning_rate'] if request is None else request['previous_lr']
    if (plan.get('authorized') is not True or plan.get('stop_on_metric_regression') is not False
            or plan.get('preserve_exact_training_state') is not True
            or proof.get('stage') != 'passed' or not proof.get('all_state_tensors_finite')
            or not proof.get('config_and_digests_match')
            or proof['identity'] != plan['source_identity']
            or proof['source_checkpoint'] != plan['source_checkpoint']
            or proof['step'] != plan['resume_step'] or proof['cursor'] != plan['resume_cursor']
            or proof['learning_rate'] != source_lr
            or proof['runtime_limits'] != {'steps': config['train']['steps'],
                                          'max_seconds': config['train']['max_seconds']}
            or proof['original_budget_started_at'] != plan['original_budget_started_at']):
        raise ValueError('Policy, exact state or budget differs from the reviewed resume')
    if request is not None:
        if (request.get('authorized') is not True or not request.get('id')
                or request['learning_rate'] != plan['learning_rate']
                or request['learning_rate'] == source_lr
                or not math.isfinite(request['learning_rate'])
                or not 0 < request['learning_rate'] <= 1e-3
                or request['expected_step'] != proof['step']
                or request['expected_cursor'] != proof['cursor']
                or request['runtime_limits'] != proof['runtime_limits']
                or request['budget_started_at'] != plan['original_budget_started_at']):
            raise ValueError('LR-only request differs from the reviewed exact source')
    start = plan['original_budget_started_at']
    deadline, outer = plan['original_deadline'], plan['outer_deadline']
    if (not all(math.isfinite(v) for v in [start, deadline, outer, now])
            or not 0 < start <= now < deadline
            or deadline != start + config['train']['max_seconds'] or outer != deadline + 600):
        raise ValueError('The original absolute deadline must remain unchanged and unexpired')
    return math.ceil(outer - now)


def validate_previous_stop(plan, old, interruption=None):
    if (old.get('last_exit_code') == 0 and old.get('stop_report', {}).get('reason') == 'signal'
            and old['stop_report']['step'] == plan['resume_step']):
        return
    if (not interruption or interruption.get('stage') != 'passed'
            or not interruption.get('no_new_optimizer_updates')
            or not interruption.get('interrupted_before_restore')
            or not interruption.get('processes_absent')
            or interruption.get('controller_pid') != old.get('pid')
            or old.get('last_exit_code') not in [-2, 130]
            or interruption.get('source_content_identity') != plan['source_identity'][:4]
            or interruption.get('resume_step') != plan['resume_step']
            or interruption.get('resume_cursor') != plan['resume_cursor']):
        raise ValueError('Expected a verified clean stop or reviewed pre-restore interruption')


def run(root, repo, revision, transition_name='cosine-priority-resume-20261031'):
    if Path(transition_name).name != transition_name:
        raise ValueError('Transition must be a direct child directory')
    transition = root / transition_name
    def read(path):
        return json.loads(path.read_text())
    plan = read(transition / 'plan.json')
    proof = read(transition / 'source-verification.json')
    config = read(root / 'config.json')
    request = read(transition / 'request.json') if (transition / 'request.json').exists() else None
    validate(plan, proof, config, time.time(), request)
    source = Path(plan['source_checkpoint'])
    with Path(plan['training_lock']).open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (transition / 'status.json').exists():
            raise FileExistsError('Already launched; no duplicate or automatic retry')
        if any(Path('/proc/' + str(pid)).exists() for pid in plan['previous_pids']):
            raise RuntimeError('Previous processes must have exited')
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPU already has a compute process')
        if identity(source) != plan['source_identity']:
            raise ValueError('Reviewed stopped source changed')
        old = read(root / 'status.json')
        interruption_path = transition / 'interrupted-initialization-verification.json'
        interruption = read(interruption_path) if interruption_path.exists() else None
        validate_previous_stop(plan, old, interruption)
        atomic(transition / 'previous-status.json', old)
        remaining = validate(plan, proof, config, time.time(), request)
        command = ['timeout', '--kill-after=120', str(remaining), sys.executable, '-u',
                   '-m', 'radio_moge.train', '--config', str(root / 'config.json'),
                   '--resume', str(source), '--execution-revision', revision]
        if request is not None:
            command.extend(['--lr-adjustment', str(transition / 'request.json')])
        state = dict(old, stage='cosine_priority_resume_starting', pid=os.getpid(),
                     child_pid=None, resumed_at=time.time(), command=command,
                     previous_controller_pid=old['pid'], execution_revision=revision,
                     current_learning_rate=plan['learning_rate'],
                     review_policy=str(transition / 'plan.json'), policy_resume_count=1)
        if request is not None:
            state['lr_adjustment_id'] = request['id']
        for key in ['last_exit_code', 'finished_at', 'stop_report']:
            state.pop(key, None)

        def record(**changes):
            state.update(changes, updated_at=time.time())
            atomic(transition / 'status.json', state)
            atomic(root / 'status.json', state)

        record()
        env = dict(os.environ, PYTHONPATH=str(repo / 'src'), OMP_NUM_THREADS='2',
                   MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', WANDB_BASE_URL='https://api.forge.coreweave.com')
        with (transition / 'resume-console.log').open('x') as log:
            child = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            record(stage='training', child_pid=child.pid)
            while child.poll() is None:
                record()
                time.sleep(5)
        record(child_pid=None, last_exit_code=child.returncode, finished_at=time.time(),
               resume_source_content_identity_unchanged=
               identity(source)[:4] == plan['source_identity'][:4])
        if child.returncode:
            record(stage='needs_inspection')
            return
        record(stage='training_complete_review_pending', stop_report=read(root / 'train/stop-report.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('repo', type=Path)
    parser.add_argument('revision')
    parser.add_argument('--transition', default='cosine-priority-resume-20261031')
    args = parser.parse_args()
    run(args.root, args.repo, args.revision, args.transition)
