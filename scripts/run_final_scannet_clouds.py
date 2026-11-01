"""One bounded evaluation-only export from the completed epoch; never train or retry."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def atomic(path, value):
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
        temporary = f.name
    os.replace(temporary, path)


def identity(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def run(root, runtime, training_root, reference_export):
    checkpoint = training_root / 'train/latest.pt'
    with (training_root.parent / 'attention-24h/.training.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (root / 'controller.json').exists(), 'Inference already launched'
        assert not (root / 'inference-attempt.json').exists()
        assert not (root / 'data.json').exists()
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                            '--format=csv,noheader'], text=True).strip()
        proof = json.loads((training_root / 'full-epoch-completion-verification.json').read_text())
        assert proof['all_checks_passed'] and proof['final_step'] == 232288
        assert identity(checkpoint) == proof['latest_checkpoint_metadata_only']['identity']
        state = dict(stage='starting', controller_pid=os.getpid(), started_at=time.time(),
                     source_identity_before=identity(checkpoint), production_updates=0,
                     backend='pytorch', max_seconds=1200, launch_count=1,
                     parent_teacher_reused=True, training_enabled=False)

        def record(**changes):
            state.update(changes, updated_at=time.time())
            atomic(root / 'controller.json', state)

        record()
        command = ['timeout', '--kill-after=30', '1200',
                   '/home/buildhub/radio-moge/.venv/bin/python', '-u',
                   str(Path(__file__).with_name('export_final_scannet_clouds.py')),
                   '--checkpoint', str(checkpoint), '--training-root', str(training_root),
                   '--reference-export', str(reference_export), '--output', str(root)]
        env = dict(os.environ, PYTHONPATH=str(runtime / 'src'), OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
        with (root / 'console.log').open('x') as log:
            child = subprocess.Popen(command, cwd=runtime, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            record(stage='running', child_pid=child.pid)
            while child.poll() is None:
                record()
                time.sleep(5)
        after = identity(checkpoint)
        record(stage='finished' if child.returncode == 0 else 'failed',
               returncode=child.returncode, child_pid=None, finished_at=time.time(),
               source_identity_after=after,
               source_identity_unchanged=after == state['source_identity_before'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'runtime', 'training_root', 'reference_export'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.root, args.runtime, args.training_root, args.reference_export)
