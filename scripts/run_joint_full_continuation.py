"""Launch one authorized 30,000-update continuation of the completed joint run."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from radio_moge.checkpoint import atomic_json
from radio_moge.config import MOGE_COMMIT, RADIO_COMMIT
from radio_moge.upstream import source_path


def identity(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def validate_plan(plan, config):
    if plan.get('authorized') is not True or plan['additional_updates'] != 30000:
        raise ValueError('Explicit 30,000-update authorization required')
    if plan['parent_step'] != 61367 or plan['parent_cursor'] != 479996:
        raise ValueError('Unexpected parent checkpoint')
    t = config['train']
    if (t['steps'] != 91367 or t['lr_schedule'] != 'constant' or t['lr'] != 1.5e-5
            or t['max_seconds'] != 28800 or t['eval_every'] != 1000
            or t['save_every'] != 500):
        raise ValueError('Unexpected continuation budget, LR or cadence')
    if config['output_dir'] != str(Path(plan['remote_root']) / 'train'):
        raise ValueError('Output must be the isolated continuation directory')


def inspect_initialization_failure(root):
    state = json.loads((root/'status.json').read_text())
    if (state.get('stage') != 'needs_inspection' or state.get('last_exit_code') != 1
            or state.get('child_pid') is not None):
        raise ValueError('Recovery requires a reviewed exited initialization failure')
    if not (root/'train').is_dir() or any((root/'train').iterdir()):
        raise ValueError('Recovery requires an empty train directory and zero updates')
    console = (root/'training-console.log').read_text()
    if 'FileNotFoundError: Missing ' not in console or '/vendor/MoGe.' not in console:
        raise ValueError('Recovery is limited to the identified missing upstream source')
    if (root/'initialization-recovery.json').exists():
        raise FileExistsError('A recovery was already attempted; inspect, never repeat')
    return state


def run(root, repo, revision, recover_initialization=False):
    plan = json.loads((root / 'plan.json').read_text())
    config = json.loads((root / 'config.json').read_text())
    validate_plan(plan, config)
    checkpoint = Path(plan['parent_checkpoint'])
    parent = checkpoint.parent.parent
    with Path(plan['training_lock']).open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not recover_initialization and ((root / 'status.json').exists() or (root / 'train').exists()):
            raise FileExistsError('Already launched or populated; never duplicate automatically')
        active = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True
        ).strip()
        if active:
            raise RuntimeError('GPU has an existing compute process; inspect without replacing it')
        # Fail before the large manifest scan if deployment omitted pinned vendor links.
        upstream = {name: str(source_path(name, commit, repo/'vendor'/name))
                    for name, commit in [('MoGe', MOGE_COMMIT), ('RADIO', RADIO_COMMIT)]}
        if identity(checkpoint) != plan['parent_identity']:
            raise ValueError('Verified parent source identity changed; no automatic rehash')
        completion = json.loads((parent / 'completion-verification.json').read_text())
        stop = json.loads((parent / 'train/stop-report.json').read_text())
        if completion['stage'] != 'passed' or stop['step'] != 61367 or stop['final_evaluation'] != 'complete':
            raise ValueError('Parent finalization is not verified')
        budget_started_at = time.time()
        launch_count = 1
        if recover_initialization:
            failed = inspect_initialization_failure(root)
            original_launch = json.loads((root/'launch.json').read_text())
            old_pids = [failed['pid'], 199676]
            if any(Path(f'/proc/{pid}').exists() for pid in old_pids):
                raise RuntimeError('Previous process still exists; never overlap')
            budget_started_at = failed['started_at']
            if time.time() - budget_started_at >= 28800:
                raise RuntimeError('Original stage wall-clock budget already exhausted')
            archive = root/'initialization-failure'
            archive.mkdir(exist_ok=False)
            for name in ('status.json','launch.json','controller-console.log','training-console.log','train'):
                (root/name).rename(archive/name)
            launch = json.loads((root/'recovery-launch.json').read_text())
            launch_count = 2
            launch.update(launch_count=launch_count, prior_failed_launch=original_launch,
                          original_budget_started_at=budget_started_at)
            atomic_json(launch,root/'launch.json')
            atomic_json(dict(stage='passed',checked_at=time.time(),zero_optimizer_updates=True,
                             old_processes_absent=old_pids,upstream_sources=upstream,
                             original_budget_started_at=budget_started_at,
                             absolute_deadline=budget_started_at+28800,
                             parent_identity_unchanged=True,attempt=1,
                             prior_status=failed,archive=str(archive)),
                        root/'initialization-recovery.json')
        command = [sys.executable, '-u', '-m', 'radio_moge.train', '--config',
                   str(root / 'config.json'), '--fork-from', str(checkpoint),
                   '--execution-revision', revision,
                   '--budget-started-at', str(budget_started_at)]
        state = dict(stage='starting', pid=os.getpid(), child_pid=None,
                     started_at=time.time(), original_started_at=budget_started_at,
                     execution_revision=revision, launch_count=launch_count,
                     parent_step=61367, parent_cursor=479996, final_step=91367,
                     max_seconds=28800, learning_rate=1.5e-5, command=command,
                     parent_identity_verified=True, parent_sha256_reused=plan['parent_sha256'])

        def record(**changes):
            state.update(changes, updated_at=time.time())
            atomic_json(state, root / 'status.json')

        record()
        env = dict(os.environ, PYTHONPATH=str(repo / 'src'), OMP_NUM_THREADS='2',
                   MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        with (root / 'training-console.log').open('x') as log:
            child = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            record(stage='training', child_pid=child.pid)
            while child.poll() is None:
                record()
                time.sleep(5)
        record(child_pid=None, last_exit_code=child.returncode)
        if child.returncode:
            record(stage='needs_inspection')
            return
        stop = json.loads((root / 'train/stop-report.json').read_text())
        unchanged = identity(checkpoint) == plan['parent_identity']
        record(stage='training_complete_review_pending' if unchanged and
               stop['final_evaluation'] == 'complete' else 'needs_inspection',
               stop_report=stop, parent_identity_unchanged=unchanged, finished_at=time.time())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('repo', type=Path)
    parser.add_argument('revision')
    parser.add_argument('--recover-initialization',action='store_true')
    args = parser.parse_args()
    run(args.root.resolve(), args.repo.resolve(), args.revision, args.recover_initialization)
