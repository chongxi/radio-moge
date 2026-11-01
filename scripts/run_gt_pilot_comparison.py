"""Single bounded post-pilot evaluation; holds the training lock, no optimizer."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from radio_moge.checkpoint import atomic_json
from export_gt_pilot_comparison import identity


def run(args):
    plan = json.loads((args.output / 'plan.json').read_text())
    assert plan['authorized'] and plan['production_updates'] == 0
    assert plan['max_seconds'] == 1200 and not plan['automatic_retry']
    pilot_plan = json.loads((args.pilot / 'plan.json').read_text())
    proof = json.loads((args.pilot / 'completion-verification.json').read_text())
    assert proof['stage'] == 'passed' and proof['final_step'] == 1000
    checkpoint = args.pilot / 'train/latest.pt'
    before = identity(checkpoint)
    assert before == proof['checkpoint_identity']
    with Path(pilot_plan['training_lock']).open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not any((args.output / p).exists() for p in ('controller.json', 'inference-attempt.json'))
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                            '--format=csv,noheader'], text=True).strip()
        state = dict(stage='starting', controller_pid=os.getpid(), started_at=time.time(),
                     max_seconds=1200, launch_count=1, production_updates=0,
                     automatic_retry=False, source_identity_before=before)
        atomic_json(state, args.output / 'controller.json')
        command = ['timeout', '--kill-after=30', '1200', sys.executable, '-u',
                   str(Path(__file__).with_name('export_gt_pilot_comparison.py'))]
        for name in ('pilot', 'reference_export', 'archive_proof', 'output'):
            command.extend(['--' + name.replace('_', '-'), str(getattr(args, name))])
        if plan.get('completed_fixed20'):
            completed = Path(plan['completed_fixed20'])
            prior = json.loads((completed / 'data.json').read_text())
            assert prior['status'] == 'complete' and prior['checkpoint_step'] == 1000
            command.extend(['--completed-fixed20', str(completed)])
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        with (args.output / 'console.log').open('x') as log:
            child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            state.update(stage='running', child_pid=child.pid)
            atomic_json(state, args.output / 'controller.json')
            returncode = child.wait()
        after = identity(checkpoint)
        state.update(stage='finished' if returncode == 0 and before == after else 'failed',
                     returncode=returncode, finished_at=time.time(),
                     source_identity_after=after, source_identity_unchanged=before == after)
        atomic_json(state, args.output / 'controller.json')
        if state['stage'] != 'finished':
            raise RuntimeError('Evaluation failed; inspect evidence, never automatically repeat inference')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot', 'reference-export', 'archive-proof', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(parser.parse_args())
