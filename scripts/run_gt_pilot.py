"""Once-only bounded controller for a verified measured-geometry pilot."""
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


def run(root):
    plan = json.loads((root / 'plan.json').read_text())
    config_path = root / 'config.json'
    config = json.loads(config_path.read_text())
    validate_config(config)
    assert plan['authorized'] and plan['source_step'] == 232288
    assert config['geometry_supervision']['target'] == 'ground_truth'
    assert config['train']['lr'] == 1e-5 and config['train']['lr_schedule'] == 'constant'
    assert config['train']['steps'] == plan['max_updates'] and config['train']['max_seconds'] == plan['max_seconds']
    assert 1 <= plan['max_updates'] <= 1000 and 0 < plan['max_seconds'] <= 7200
    assert config['output_dir'] == str(root / 'train')
    archive = json.loads(Path(plan['archive_proof']).read_text())
    assert archive['stage'] == 'verified' and archive['source_step'] == plan['source_step']
    source = Path(plan['source_checkpoint'])
    assert str(source) == archive['archive'] and source.stat().st_size == archive['bytes']
    source_stat = source.stat()
    backward = json.loads(Path(plan['backward_proof']).read_text())
    assert backward['stage'] == 'passed' and backward['optimizer_updates'] == 0
    for split, key in [('train', 'train_manifest'), ('val', 'geometry_val_manifest')]:
        path = Path(config['data'][key])
        proof = json.loads((path.parent / 'data-verification.json').read_text())
        assert proof['stage'] == 'passed' and proof['split'] == split
        with path.open('rb') as file:
            assert hashlib.file_digest(file, 'sha256').hexdigest() == proof['manifest_sha256']
    with Path(plan['training_lock']).open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not any((root / name).exists() for name in ('status.json', 'train'))
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        started = time.time()
        state = {'stage': 'running', 'started_at': started, 'controller_pid': os.getpid(),
                 'source_step': plan['source_step'], 'maximum_updates': plan['max_updates'],
                 'maximum_seconds': plan['max_seconds'], 'launch_count': 1, 'automatic_restart': False}
        atomic_json(state, root / 'status.json')
        command = ['timeout', '--signal=TERM', '--kill-after=120', str(plan['max_seconds'] + 600),
                   sys.executable, '-u', '-m', 'radio_moge.train', '--config', str(config_path),
                   '--init', str(source), '--execution-revision', plan['revision']]
        with (root / 'console.log').open('xb') as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            state['timeout_pid'] = child.pid
            atomic_json(state, root / 'status.json')
            result = child.wait()
        after = source.stat()
        unchanged = (after.st_size, after.st_mtime_ns) == (source_stat.st_size, source_stat.st_mtime_ns)
        state.update(stage='finished' if result == 0 else 'failed', finished_at=time.time(),
                     returncode=result, archive_unchanged=unchanged)
        atomic_json(state, root / 'status.json')
        if not unchanged:
            raise RuntimeError('Archived source changed during the new stage')
        if result:
            raise RuntimeError(f'Bounded GT pilot exited with code {result}; no automatic restart')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    run(parser.parse_args().root)
