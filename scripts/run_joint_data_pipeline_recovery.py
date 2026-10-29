"""Resume the inspected zero-update SHM failure with validated compact RGB transport."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from radio_moge.checkpoint import atomic_json


def run(root, repo, revision):
    plan = json.loads((root / 'data-pipeline-recovery-plan.json').read_text())
    assert plan['authorized'] and plan['no_budget_extension']
    directory = root / 'data-pipeline-recovery'
    directory.mkdir(exist_ok=True)
    own = (directory / '.lock').open('a+')
    fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (directory / 'status.json').exists():
        raise FileExistsError('Existing recovery; never duplicate')
    prior = json.loads((root / 'data-pipeline-transition/status.json').read_text())
    assert prior['stage'] == 'needs_inspection' and prior['last_exit_code'] == 1
    assert prior['child_pid'] is None
    failed = json.loads((root / 'train/data-pipeline-benchmark.json').read_text())
    assert failed['disposable_updates'] == 0 and not failed['rows']
    for pid in (41047, 41065):
        proc = Path(f'/proc/{pid}/stat')
        assert not proc.exists() or proc.read_text().split(') ', 1)[1].split()[0] == 'Z'
    global_path = json.loads((root / 'plan.json').read_text())['global_training_lock']
    global_lock = Path(global_path).open('a+')
    fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    local_lock = (root / '.training.lock').open('a+')
    fcntl.flock(local_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                        '--format=csv,noheader'], text=True).strip()
    preflight = json.loads((root / 'preflight-compact-transport.json').read_text())
    assert preflight['stage'] == 'passed' and preflight['images_per_pipeline'] == 256
    assert all(r['exact_pixels'] for r in preflight['results'])
    check = json.loads((root / 'data-pipeline-transition/checkpoint-verification.json').read_text())
    parent = Path(check['snapshot'])
    stat = parent.stat()
    assert check['source_stat'] == dict(inode=stat.st_ino, size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    original = json.loads((root / 'data-pipeline-transition/before/performance.json').read_text())
    assert original['original_started_at'] + 86400 == plan['original_deadline']
    request = dict(original, autotune=False, data_autotune=True, compact_transport=True,
                   data_probe_filename='data-pipeline-benchmark-v2.json')
    atomic_json(request, directory / 'request.json')
    atomic_json(dict(stage='passed', source_identity_unchanged=True,
                     reused_checkpoint_sha256=check['checkpoint_sha256'], no_rehash=True,
                     resume_step=9150, resume_cursor=62260, failed_updates=0,
                     original_processes_absent=True), directory / 'recovery-verification.json')
    state = dict(stage='data_pipeline_recovery_initializing', pid=os.getpid(), child_pid=None,
                 revision=revision, original_deadline=plan['original_deadline'],
                 started_at=time.time(), no_automatic_extension=True, resume_from_step=9150)

    def record(**changes):
        state.update(changes, updated_at=time.time())
        atomic_json(state, directory / 'status.json')
        atomic_json(state, root / 'status.json')

    record()
    try:
        env = dict(os.environ, PYTHONPATH=str(repo / 'src'), OMP_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
        cmd = [sys.executable, '-u', '-m', 'radio_moge.train', '--config', str(root / 'config.json'),
               '--resume', str(parent), '--performance-config', str(directory / 'request.json'),
               '--execution-revision', revision]
        with (directory / 'resume-console.log').open('x') as log:
            child = subprocess.Popen(cmd, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            record(child_pid=child.pid, command=cmd)
            atomic_json(dict(pid=os.getpid(), child_pid=child.pid, launched_at=time.time(),
                             revision=revision, command=cmd, launch_count=1), directory / 'launch.json')
            while child.poll() is None:
                path = root / 'train/data-pipeline-benchmark-v2.json'
                if path.exists():
                    proof = json.loads(path.read_text())
                    stage = 'training' if proof['stage'] == 'passed' else 'data_pipeline_benchmark'
                    if state['stage'] != stage:
                        record(stage=stage)
                time.sleep(5)
            code = child.returncode
        record(child_pid=None, last_exit_code=code)
        if code:
            raise RuntimeError(f'Inspected recovery exit {code}; no automatic retry')
        record(stage='training_complete_review_pending',
               stop=json.loads((root / 'train/stop-report.json').read_text()), finished_at=time.time())
    except BaseException as error:
        record(stage='needs_inspection', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    run(args.root, args.repo, args.revision)
