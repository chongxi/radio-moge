"""One serialized, time-limited frozen encoder TensorRT probe; no training updates."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from run_batch_benchmark import atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'repo', 'python', 'deps', 'reference_report'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=2400)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 3600:
        raise ValueError('Probe must remain bounded')
    status_path = args.root / 'controller.json'
    with status_path.open('x') as handle:
        json.dump(dict(stage='starting', pid=os.getpid(), started_at=time.time()), handle)
    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], text=True).strip()
    if busy:
        atomic(status_path, dict(stage='refused_busy_gpu', pid=os.getpid(), finished_at=time.time()))
        raise RuntimeError('GPU already has a compute process')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(args.deps), str(args.repo / 'src'))),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               WANDB_MODE='disabled', TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='0')
    command = ['timeout', '--signal=TERM', '--kill-after=30', str(args.max_seconds + 30),
               str(args.python), '-u', '-m', 'radio_moge.trt_encoder_probe',
               '--root', str(args.root), '--reference-report', str(args.reference_report),
               '--max-seconds', str(args.max_seconds)]
    with (args.root / 'console.log').open('x') as log:
        child = subprocess.Popen(command, cwd=args.repo, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT)
        state = dict(stage='running', pid=os.getpid(), child_pid=child.pid,
                     started_at=time.time(), max_seconds=args.max_seconds,
                     production_updates=0, command=command)
        atomic(status_path, state)
        code = child.wait()
        state.update(stage='finished' if code == 0 else 'failed', returncode=code,
                     child_pid=None, finished_at=time.time())
        atomic(status_path, state)


if __name__ == '__main__':
    main()
