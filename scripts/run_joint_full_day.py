"""Single guarded 24-hour scratch joint-adapter run; never extends or stops the GPU."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from radio_moge.checkpoint import atomic_json
from radio_moge.train import validate_config


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ready(plan, config):
    if plan.get('authorized') is not True:
        raise ValueError('Full training authorization missing')
    validate_config(config)
    expected = plan['expected_config']
    if config != expected:
        raise ValueError('Config differs from reviewed plan')
    a, t, loss = config['adapter'], config['train'], config['loss']
    if a['kind'] != 'joint_attention' or a['initialization_seed'] != 43:
        raise ValueError('Expected fresh joint adapter seed43')
    spec = a['spec']
    if (spec['width'], spec['blocks'], spec['heads']) != (2048, 4, 32):
        raise ValueError('Unexpected architecture')
    if not spec['qk_norm'] or spec.get('summary_cleaning') != 'per-token-zero-channel':
        raise ValueError('QK norm and all-input cleaning are required')
    if t['max_seconds'] != 86400 or t['steps'] != 120000:
        raise ValueError('Unexpected authorized budget')
    if loss != dict(spatial=1., summary=1., cosine=1., summary_cosine=1.):
        raise ValueError('Unexpected objective')
    dataset = Path(plan['prepared_dataset'])
    info_path = dataset / 'dataset_info.json'
    if digest(info_path) != plan['dataset_info_sha256']:
        raise ValueError('Prepared dataset proof changed')
    info = json.loads(info_path.read_text())
    if info['stage'] != 'ready' or info['train_images'] != 796478 or info['selection_images'] != 1268:
        raise ValueError('Full dataset is not ready')
    if not all(info['verification'][k] for k in (
        'all_paths_exist', 'split_groups_paths_hashes_disjoint', 'old_holdouts_byte_identical'
    )):
        raise ValueError('Dataset checks incomplete')
    if config['data']['mixture_weights'] != info['mixture_weights']:
        raise ValueError('Source mixture changed')
    for key, name in [('train_manifest', 'train.jsonl'), ('val_manifest', 'val-selection.jsonl')]:
        if Path(config['data'][key]).resolve() != (dataset / name).resolve():
            raise ValueError('Unexpected data manifest')
    for name, sha in info['manifest_sha256'].items():
        if digest(dataset / name) != sha:
            raise ValueError(f'Manifest changed: {name}')
    for split, sha in info['preserved_heldout_sha256'].items():
        if digest(Path(plan['base_dataset']) / f'{split}.jsonl') != sha:
            raise ValueError(f'Original heldout changed: {split}')
    return info


def run(root, repo, revision):
    plan = json.loads((root / 'plan.json').read_text())
    config = json.loads((root / 'config.json').read_text())
    global_lock = Path(plan['global_training_lock']).open('a+')
    fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    local_lock = (root / '.training.lock').open('a+')
    fcntl.flock(local_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'status.json').exists() or (root / 'train').exists():
        raise FileExistsError('Existing run: inspect, never duplicate')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU occupied: inspect before starting')
    state = dict(stage='checking_data', pid=os.getpid(), child_pid=None,
                 started_at=time.time(), revision=revision, no_automatic_extension=True)

    def record(**changes):
        state.update(changes, updated_at=time.time())
        atomic_json(state, root / 'status.json')

    record()
    try:
        if config['code_revision'] != revision:
            raise ValueError('Execution revision mismatch')
        info = ready(plan, config)
        atomic_json(dict(stage='passed', checked_at=time.time(),
                         train_images=info['train_images'], validation_images=info['selection_images'],
                         manifest_sha256=info['manifest_sha256'],
                         dataset_info_sha256=plan['dataset_info_sha256'],
                         original_data_unchanged=True, initialization='scratch'),
                    root / 'dataset-verification.json')
        env = dict(os.environ, PYTHONPATH=str(repo / 'src'), OMP_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
        for stage, command in [
            ('preflight', [sys.executable, '-u', str(repo / 'scripts/preflight_joint_full.py'),
                           '--root', str(root)]),
            ('training', [sys.executable, '-u', '-m', 'radio_moge.train', '--config',
                          str(root / 'config.json'), '--execution-revision', revision]),
        ]:
            if stage == 'training':
                proof = json.loads((root / 'preflight.json').read_text())
                if proof['stage'] != 'passed' or proof['disposable_optimizer_updates'] != 3:
                    raise RuntimeError('Preflight not passed')
                if json.loads((root / 'plan.json').read_text()) != plan:
                    raise RuntimeError('Plan changed during preflight')
                if json.loads((root / 'config.json').read_text()) != config:
                    raise RuntimeError('Config changed during preflight')
            with (root / f'{stage}-console.log').open('x') as log:
                child = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                record(stage=stage, child_pid=child.pid, command=command)
                code = child.wait()
            record(child_pid=None, last_exit_code=code)
            if code:
                raise RuntimeError(f'{stage} failed with exit {code}; no retry')
        stop = json.loads((root / 'train/stop-report.json').read_text())
        record(stage='training_complete_review_pending', stop=stop, finished_at=time.time())
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
