"""Bounded, sequential 2x2 GT-objective/feature-weight experiment on one GPU.

Prepare/run on the GPU host. All arms start from the same immutable archive and
fresh Adam; the source, data, camera protocol, augmentation and evaluation stay
fixed. Never auto-resume a failed arm or overwrite an earlier experiment.
"""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from radio_moge.checkpoint import atomic_json
from radio_moge.config import ROOT
from radio_moge.official_gt import OFFICIAL_GT_WEIGHTS
from radio_moge.train import validate_config


FEATURE_KEYS = ('spatial', 'summary', 'cosine', 'summary_cosine')
EXPERIMENT = Path('/home/buildhub/shared/radio-moge/chongxi/data-expansion-attention-20261022')
SOURCE = EXPERIMENT / 'model-archive-feature-final232288-20261101/final.pt'
ARMS = [('custom-f01', 'metric_custom', .1), ('custom-f1', 'metric_custom', 1.),
        ('official-f01', 'moge2_core', .1), ('official-f1', 'moge2_core', 1.)]


def digest(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def prepare(root, revision, steps):
    if (root / 'plan.json').exists():
        raise FileExistsError('Experiment already prepared; do not overwrite its plan')
    assert 1 <= steps <= 250
    template = json.loads((ROOT / 'configs/scannetpp-gt-geometry.json').read_text())
    archive_proof = json.loads((SOURCE.parent / 'archive-verification.json').read_text())
    if digest(SOURCE) != archive_proof['sha256']:
        raise ValueError('Archive digest differs from verification')
    configurations = {}
    for index, (name, objective, feature_weight) in enumerate(ARMS):
        config = deepcopy(template)
        config['output_dir'] = str(root / name / 'train')
        config['train'].update(steps=steps, max_seconds=3600, eval_at_start=index == 0,
                               eval_every=steps, save_every=steps, log_every=10,
                               tensorboard=False, update_norm_every=50)
        config['geometry_supervision']['objective'] = objective
        if objective == 'moge2_core':
            config['loss'] = dict(OFFICIAL_GT_WEIGHTS)
        config['loss'].update({key: feature_weight for key in FEATURE_KEYS})
        config['tracking'] = {'enabled': True, 'mode': 'offline', 'project': 'radio-moge',
                              'entity': 'brainsim', 'name': root.name + '-' + name,
                              'group': root.name, 'tags': ['loss-weight-2x2', objective]}
        config['evaluation']['visualizations'] = 2
        validate_config(config)
        path = root / name / 'config.json'
        atomic_json(config, path)
        configurations[name] = {'path': str(path), 'sha256': digest(path),
                                'objective': objective, 'feature_weight': feature_weight}
    plan = dict(authorized=True, source=str(SOURCE), source_sha256=archive_proof['sha256'],
                source_step=232288, revision=revision, steps_per_arm=steps, seed=43,
                maximum_seconds_per_arm=3600, training_lock=str(EXPERIMENT / 'attention-24h/.training.lock'),
                arms=configurations, prepared_at=time.time(), automatic_restart=False,
                changed_factors=['geometry objective', 'common multiplier of four feature losses'],
                unchanged=['archive adapter and feature RMS statistics', 'fresh AdamW', 'LR 1e-5 constant',
                           '81 training views', 'seed, paired augmentation, effective batch 16',
                           '400x640 training bucket', 'frozen RADIO and MoGe encoder/decoder',
                           'fixed validation views and geometry reconstruction'],
                limits=['single training seed, short optimization budget',
                        'official-core arm uses exact upstream global/local/edge/scale losses, '
                        'not normal-head/mask losses or upstream data refinement',
                        'validation is repeatedly used, not an untouched benchmark test'],
                conditional_next='Test feature weight 3 only after weight 1 improves retention; separate new arm.')
    atomic_json(plan, root / 'plan.json')
    print(json.dumps(plan, indent=2))


def run(root):
    plan = json.loads((root / 'plan.json').read_text())
    if (root / 'controller.json').exists():
        raise FileExistsError('Controller already ran; inspect it instead of auto-restarting')
    before = SOURCE.stat()
    with Path(plan['training_lock']).open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = dict(stage='running', pid=os.getpid(), started_at=time.time(), completed=[])
        atomic_json(status, root / 'controller.json')
        for name, arm in plan['arms'].items():
            if digest(arm['path']) != arm['sha256']:
                raise ValueError('Prepared configuration changed')
            config = json.loads(Path(arm['path']).read_text())
            if Path(config['output_dir']).exists():
                raise FileExistsError('Arm already has output; refusing overwrite/resume')
            status.update(current_arm=name, arm_started_at=time.time())
            command = ['timeout', '--signal=TERM', '--kill-after=120', '4200', sys.executable,
                       '-u', '-m', 'radio_moge.train', '--config', arm['path'], '--init',
                       plan['source'], '--execution-revision', plan['revision']]
            with (root / name / 'console.log').open('x') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL)
                status['child_pid'] = child.pid
                atomic_json(status, root / 'controller.json')
                returncode = child.wait()
            stop_path = Path(config['output_dir']) / 'stop-report.json'
            stop = json.loads(stop_path.read_text()) if stop_path.exists() else {}
            record = dict(name=name, returncode=returncode, stop=stop, finished_at=time.time())
            status['completed'].append(record)
            if returncode or stop.get('step') != plan['steps_per_arm']:
                status.update(stage='failed', reason='arm failed or did not reach matched update count')
                atomic_json(status, root / 'controller.json')
                raise RuntimeError(f'{name} failed; no automatic restart')
            atomic_json(status, root / 'controller.json')
        after = SOURCE.stat()
        unchanged = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        status.update(stage='complete' if unchanged else 'failed', finished_at=time.time(),
                      source_unchanged=unchanged)
        atomic_json(status, root / 'controller.json')
        if not unchanged:
            raise RuntimeError('Source checkpoint was modified')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--revision')
    parser.add_argument('--steps', type=int, default=250)
    args = parser.parse_args()
    if args.mode == 'prepare':
        if not args.revision:
            parser.error('--revision is required for prepare')
        prepare(args.root, args.revision, args.steps)
    else:
        run(args.root)


if __name__ == '__main__':
    main()
