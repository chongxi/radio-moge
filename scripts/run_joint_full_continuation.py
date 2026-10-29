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


def run(root, repo, revision):
    plan = json.loads((root / 'plan.json').read_text())
    config = json.loads((root / 'config.json').read_text())
    validate_plan(plan, config)
    checkpoint = Path(plan['parent_checkpoint'])
    parent = checkpoint.parent.parent
    with Path(plan['training_lock']).open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'status.json').exists() or (root / 'train').exists():
            raise FileExistsError('Already launched or populated; never duplicate automatically')
        active = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True
        ).strip()
        if active:
            raise RuntimeError('GPU has an existing compute process; inspect without replacing it')
        if identity(checkpoint) != plan['parent_identity']:
            raise ValueError('Verified parent source identity changed; no automatic rehash')
        completion = json.loads((parent / 'completion-verification.json').read_text())
        stop = json.loads((parent / 'train/stop-report.json').read_text())
        if completion['stage'] != 'passed' or stop['step'] != 61367 or stop['final_evaluation'] != 'complete':
            raise ValueError('Parent finalization is not verified')
        command = [sys.executable, '-u', '-m', 'radio_moge.train', '--config',
                   str(root / 'config.json'), '--fork-from', str(checkpoint),
                   '--execution-revision', revision]
        state = dict(stage='starting', pid=os.getpid(), child_pid=None,
                     started_at=time.time(), execution_revision=revision, launch_count=1,
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
    args = parser.parse_args()
    run(args.root.resolve(), args.repo.resolve(), args.revision)
