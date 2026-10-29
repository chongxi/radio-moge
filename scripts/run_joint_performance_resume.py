"""One controlled checkpoint transition, bounded throughput probe, then exact-state continuation."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from radio_moge.checkpoint import atomic_json


def process(pid):
    path = Path(f'/proc/{pid}')
    if not path.exists():
        return None
    stat = (path / 'stat').read_text().split(') ', 1)[1].split()
    return dict(state=stat[0], start_ticks=int(stat[19]),
                command=(path / 'cmdline').read_bytes().replace(b'\0', b' ').decode())


def active(pid):
    p = process(pid)
    return p is not None and p['state'] != 'Z'


def run(root, repo, revision):
    plan = json.loads((root / 'performance-plan.json').read_text())
    assert plan['authorized'] is True and plan['no_budget_extension'] is True
    transition = root / 'performance-transition'
    transition.mkdir(exist_ok=True)
    own_lock = (transition / '.lock').open('a+')
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (transition / 'status.json').exists():
        raise FileExistsError('Existing transition; inspect, never relaunch')
    prior = json.loads((root / 'status.json').read_text())
    assert prior['stage'] == 'training'
    trainer, wrapper = prior['child_pid'], prior['pid']
    assert trainer == plan['original_trainer_pid'] and wrapper == plan['original_wrapper_pid']
    live = process(trainer)
    assert live and '-m radio_moge.train' in live['command'] and str(root / 'config.json') in live['command']
    boot = int(next(line.split()[1] for line in Path('/proc/stat').read_text().splitlines()
                    if line.startswith('btime ')))
    started_at = boot + live['start_ticks'] / os.sysconf('SC_CLK_TCK')
    cfg = json.loads((root / 'config.json').read_text())
    assert cfg['train']['max_seconds'] == 86400 and cfg['train']['batch_size'] == 4
    evals = [json.loads(s) for s in (root / 'train/validation.jsonl').read_text().splitlines()]
    assert any(v['step'] == 2000 for v in evals)
    state = dict(stage='saving_for_performance_transition', pid=os.getpid(), child_pid=None,
                 original_trainer_pid=trainer, original_wrapper_pid=wrapper,
                 original_started_at=started_at, original_deadline=started_at + 86400,
                 started_at=time.time(), revision=revision, no_automatic_extension=True)

    def record(**updates):
        state.update(updates, updated_at=time.time())
        atomic_json(state, transition / 'status.json')
        atomic_json(state, root / 'status.json')

    atomic_json(prior, transition / 'original-wrapper-status.json')
    record()
    try:
        # Trainer handles SIGTERM by completing its update and atomically saving all state.
        os.kill(trainer, signal.SIGTERM)
        record(signal_sent_at=time.time())
        limit = time.monotonic() + 240
        while active(trainer) or active(wrapper):
            if time.monotonic() > limit:
                raise TimeoutError('Original trainer still stopping; no forced kill or duplicate')
            time.sleep(1)
        # Original wrapper may have written its last status; take ownership after it exits.
        lock = Path(json.loads((root / 'plan.json').read_text())['global_training_lock']).open('a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        local_lock = (root / '.training.lock').open('a+')
        fcntl.flock(local_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                    '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPU still occupied after controlled stop')
        stop = json.loads((root / 'train/stop-report.json').read_text())
        assert stop['reason'] == 'signal' and stop['step'] >= 2000
        snapshot = transition / 'before'
        snapshot.mkdir()
        for name in ('config.json', 'provenance.json', 'wandb-run.json', 'train.jsonl',
                     'validation.jsonl', 'stop-report.json'):
            shutil.copy2(root / 'train' / name, snapshot / name)
        # latest.pt is atomically replaced by the trainer, so this hardlink is immutable thereafter.
        parent = snapshot / 'resume.pt'
        os.link(root / 'train/latest.pt', parent)
        stat = parent.stat()
        with parent.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert (stat.st_size, stat.st_mtime_ns, stat.st_ino) == (
            parent.stat().st_size, parent.stat().st_mtime_ns, parent.stat().st_ino)
        atomic_json(dict(stage='passed', stopped=stop, checkpoint_bytes=stat.st_size,
                         checkpoint_sha256=digest, snapshot=str(parent), original_processes_absent=True,
                         source_stat=dict(inode=stat.st_ino, size=stat.st_size, mtime_ns=stat.st_mtime_ns)),
                    transition / 'checkpoint-verification.json')
        request = dict(plan['benchmark'], authorized=True, autotune=True,
                       original_started_at=started_at)
        atomic_json(request, transition / 'request.json')
        env = dict(os.environ, PYTHONPATH=str(repo / 'src'), OMP_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
        cmd = [sys.executable, '-u', '-m', 'radio_moge.train', '--config', str(root / 'config.json'),
               '--resume', str(parent), '--performance-config', str(transition / 'request.json'),
               '--execution-revision', revision]
        with (transition / 'resume-console.log').open('x') as log:
            child = subprocess.Popen(cmd, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            record(stage='performance_resume_initializing', child_pid=child.pid,
                   resume_from_step=stop['step'], command=cmd)
            atomic_json(dict(pid=os.getpid(), child_pid=child.pid, launched_at=time.time(),
                             revision=revision, command=cmd, launch_count=1), transition / 'launch.json')
            while child.poll() is None:
                path = root / 'train/throughput-benchmark.json'
                if path.exists():
                    proof = json.loads(path.read_text())
                    stage = 'training' if proof['stage'] == 'passed' else 'throughput_benchmark'
                    if state['stage'] != stage:
                        record(stage=stage)
                time.sleep(5)
            code = child.returncode
        record(child_pid=None, last_exit_code=code)
        if code:
            raise RuntimeError(f'Performance resume exit {code}; inspect, never blind retry')
        record(stage='training_complete_review_pending',
               stop=json.loads((root / 'train/stop-report.json').read_text()), finished_at=time.time())
    except BaseException as error:
        record(stage='needs_inspection', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--revision', required=True)
    args = p.parse_args()
    run(args.root, args.repo, args.revision)
