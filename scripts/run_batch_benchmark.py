"""Run one bounded batch benchmark and sample GPU telemetry without retaining weights."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def run(args):
    root = args.root
    if (root / 'controller.json').exists() or (root / 'results').exists():
        raise FileExistsError('Benchmark already started; inspect existing state before retry')
    # A launch claim prevents two controllers from starting the same benchmark.
    with (root / 'controller.json').open('x') as handle:
        json.dump(dict(stage='starting', pid=os.getpid(), started_at=time.time()), handle)
    current = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                       '--format=csv,noheader,nounits'], text=True).strip()
    if current:
        raise RuntimeError('GPU already has a compute process; no benchmark launched')
    plan = json.loads((root / 'plan.json').read_text())
    assert 0 < plan['max_seconds'] <= 1200
    env = dict(os.environ, PYTHONPATH=str(args.repo / 'src'), OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled',
               TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='0')
    command = ['timeout', '--signal=TERM', '--kill-after=30', str(plan['max_seconds'] + 60),
               str(args.python), '-u', '-m', 'radio_moge.batch_benchmark',
               '--checkpoint', str(args.checkpoint), '--plan', str(root / 'plan.json'),
               '--output', str(root / 'results')]
    status = dict(stage='running', pid=os.getpid(), started_at=time.time(), command=command)
    with (root / 'console.log').open('w') as log, (root / 'gpu.jsonl').open('w') as gpu:
        process = subprocess.Popen(command, cwd=args.repo, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
        status['child_pid'] = process.pid
        atomic(root / 'controller.json', status)
        while process.poll() is None:
            sample = subprocess.run(['nvidia-smi',
                '--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,temperature.gpu',
                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
            fields = sample.stdout.strip().split(',')
            if sample.returncode == 0 and len(fields) == 6:
                row = dict(checked_at=time.time())
                for key, value in zip(('gpu_percent', 'memory_percent', 'memory_mib', 'power_w',
                                       'sm_clock_mhz', 'temperature_c'), fields, strict=True):
                    try:
                        row[key] = float(value.strip())
                    except ValueError:
                        row[key] = None
                gpu.write(json.dumps(row) + '\n')
                gpu.flush()
            time.sleep(1)
        status.update(stage='finished' if process.returncode == 0 else 'failed',
                      returncode=process.returncode, child_pid=None, finished_at=time.time())
        atomic(root / 'controller.json', status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'repo', 'checkpoint', 'python'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(parser.parse_args())
