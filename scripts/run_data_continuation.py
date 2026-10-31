"""One bounded data fork with verified snapshot and immutable source checkpoint."""

import argparse
import fcntl
import hashlib
import math
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def atomic(path, value):
    temporary = path.with_suffix('.incoming')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def identity(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def validate_plan(plan, config, performance):
    if plan.get('authorized') is not True or not 1 <= plan['additional_updates'] <= 100000:
        raise ValueError('A bounded authorized data experiment is required')
    if config['data']['buckets'] != [[480, 640], [640, 480], [400, 640]]:
        raise ValueError('Preserve all three existing resolution buckets')
    if (config['train']['steps'] != plan['parent_step'] + plan['additional_updates']
            or config['train']['lr'] != plan['learning_rate']
            or config['train']['lr_schedule'] != 'constant'
            or config['train']['max_seconds'] != plan['max_seconds']
            or performance['effective_batch'] != plan['effective_batch']):
        raise ValueError('Budget, learning rate or effective batch differs from the plan')
    if config['output_dir'] != str(Path(plan['remote_root']) / 'train'):
        raise ValueError('Training output must be isolated')


def run(root, repo, revision):
    plan = json.loads((root / 'plan.json').read_text())
    config = json.loads((root / 'config.json').read_text())
    performance = json.loads((root / 'performance-request.json').read_text())
    validate_plan(plan, config, performance)
    checkpoint = Path(plan['parent_checkpoint'])
    with Path(plan['training_lock']).open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'status.json').exists() or (root / 'train').exists():
            raise FileExistsError('Already launched; never automatically duplicate or restart')
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPU already has a compute process')
        if identity(checkpoint) != plan['parent_identity']:
            raise ValueError('Source checkpoint identity changed')
        parent = checkpoint.parent.parent
        proof = json.loads((parent / 'completion-verification.json').read_text())
        if proof['stage'] != 'passed':
            raise ValueError('Source stage is not complete and verified')
        source = json.loads((parent / 'final-review/source-verification.json').read_text())
        selected = [f for f in source['files'] if f.get('relative') ==
                    str(checkpoint.relative_to(parent))]
        if (source['stage'] != 'passed' or len(selected) != 1
                or selected[0]['identity'] != plan['parent_identity']
                or selected[0]['checkpoint']['step'] != plan['parent_step']):
            raise ValueError('Selected checkpoint is not the verified source')
        prepared = json.loads(Path(plan['snapshot_verification']).read_text())
        if prepared['stage'] != 'passed' or not prepared['decoded_all_accepted_images']:
            raise ValueError('Snapshot preparation has not passed')
        manifest = Path(config['data']['train_manifest'])
        def digest(path):
            value = hashlib.sha256()
            with path.open('rb') as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b''):
                    value.update(chunk)
            return value.hexdigest()
        if digest(manifest) != prepared['manifest_sha256']['train.jsonl']:
            raise ValueError('Frozen training manifest changed after preparation')
        if plan['phase'] == 'pilot':
            parent_config = json.loads((parent / 'config.json').read_text())
            if digest(Path(parent_config['data']['train_manifest'])) != prepared['original_train_sha256']:
                raise ValueError('Snapshot does not preserve the source training manifest')
        for view in config['evaluation']['dataset_variants'].values():
            if digest(Path(view['manifest'])) != prepared['manifest_sha256']['val-scannetpp.jsonl']:
                raise ValueError('Frozen new validation manifest changed')
        if plan['phase'] == 'coverage':
            count = prepared['counts']['original_train'] + prepared['counts']['scannetpp_train']
            if (plan['additional_updates'] != math.ceil(count / plan['effective_batch'])
                    or config['data'].get('coverage_epochs') != 1
                    or config['data'].get('mixture_weights')):
                raise ValueError('Coverage phase must visit every frozen sample')
        elif plan['phase'] != 'pilot' or plan['additional_updates'] != 3000:
            raise ValueError('Expected the authorized 3000-update pilot')
        for name, expected in plan['upstream_sources'].items():
            actual = subprocess.check_output(['git', '-C', str(repo / 'vendor' / name),
                                              'rev-parse', 'HEAD'], text=True).strip()
            if actual != expected:
                raise ValueError('Pinned upstream source differs')
        start = time.time()
        performance['original_started_at'] = start
        atomic(root / 'performance-request.json', performance)
        command = ['timeout', '--kill-after=120', str(plan['max_seconds'] + 600),
                   sys.executable, '-u', '-m', 'radio_moge.train', '--config',
                   str(root / 'config.json'), '--fork-from', str(checkpoint),
                   '--data-fork', '--performance-config', str(root / 'performance-request.json'),
                   '--execution-revision', revision, '--budget-started-at', str(start)]
        state = dict(stage='starting', pid=os.getpid(), child_pid=None, started_at=start,
                     parent_step=plan['parent_step'], parent_cursor=plan['parent_cursor'],
                     final_step=config['train']['steps'], max_seconds=plan['max_seconds'],
                     execution_revision=revision, command=command, launch_count=1)

        def record(**changes):
            state.update(changes, updated_at=time.time())
            atomic(root / 'status.json', state)

        record()
        env = dict(os.environ, PYTHONPATH=str(repo / 'src'), OMP_NUM_THREADS='2',
                   MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', WANDB_BASE_URL='https://api.forge.coreweave.com')
        with (root / 'training-console.log').open('x') as log:
            child = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            record(stage='training', child_pid=child.pid)
            while child.poll() is None:
                record()
                time.sleep(5)
        record(child_pid=None, last_exit_code=child.returncode, finished_at=time.time(),
               parent_identity_unchanged=identity(checkpoint) == plan['parent_identity'])
        if child.returncode:
            record(stage='needs_inspection')
            return
        stop = json.loads((root / 'train/stop-report.json').read_text())
        record(stage='training_complete_review_pending', stop_report=stop)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('repo', type=Path)
    p.add_argument('revision')
    a = p.parse_args()
    run(a.root, a.repo, a.revision)
