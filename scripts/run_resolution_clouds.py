"""One post-backup fixed20 inference job; frozen teacher/parent results are reused."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from run_resolution_continuation import atomic, identity


def run(root, repo, training_root, parent_export):
    checkpoint = training_root/'train/best-spatial.pt'
    with (training_root.parent/'attention-24h/.training.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (root/'controller.json').exists() and not (root/'data.json').exists(), 'Do not repeat a launched inference'
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPU has an existing compute process')
        backup = json.loads((training_root/'final-review/backup-verification.json').read_text())
        source = json.loads((training_root/'final-review/source-verification.json').read_text())
        assert backup['stage'] == 'passed' and backup['lock_free']
        assert backup['all_plan_files_serially_copied_and_sha256_readback_verified']
        expected = next(x for x in source['files'] if x['relative'] == 'train/best-spatial.pt')
        assert identity(checkpoint) == expected['identity']
        assert expected['checkpoint']['step'] == 94367
        state = dict(stage='starting', controller_pid=os.getpid(), started_at=time.time(),
                     source_identity_before=identity(checkpoint), production_updates=0,
                     backend='pytorch', max_seconds=600, parent_teacher_reused=True)

        def record(**changes):
            state.update(changes, updated_at=time.time())
            atomic(root/'controller.json', state)

        record()
        command = ['timeout', '--kill-after=30', '600', '/home/buildhub/radio-moge/.venv/bin/python',
                   '-u', str(repo/'scripts/export_resolution_clouds.py'), '--checkpoint', str(checkpoint),
                   '--training-root', str(training_root), '--parent-export', str(parent_export), '--output', str(root)]
        env = dict(os.environ, PYTHONPATH=str(repo/'src'), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
        with (root/'console.log').open('x') as log:
            child = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            record(stage='running', child_pid=child.pid)
            while child.poll() is None:
                record()
                time.sleep(5)
        record(stage='finished' if child.returncode == 0 else 'failed', returncode=child.returncode,
               child_pid=None, finished_at=time.time(), source_identity_after=identity(checkpoint))
        record(source_identity_unchanged=state['source_identity_after'] == state['source_identity_before'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'repo', 'training_root', 'parent_export'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.root, args.repo, args.training_root, args.parent_export)
