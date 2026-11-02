"""Fetch only missing scanner registration metadata into a NEW evidence root.

Uses the already authorized private official downloader configuration in memory.
Never prints credentials, request URLs, response bodies or exception messages.
Original dataset/configuration files are read-only. No training is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

import numpy as np
import requests
import yaml

from indoor_download import atomic_write, write_json


class SafeDownloadError(RuntimeError):
    pass


def eligible_scenes(dataset):
    train = set((dataset / 'splits/nvs_sem_train.txt').read_text().split())
    forbidden = set()
    for split in ('nvs_sem_val', 'nvs_test', 'sem_test', 'nvs_test_iphone'):
        forbidden.update((dataset / f'splits/{split}.txt').read_text().split())
    if train & forbidden:
        raise ValueError('Official scene split overlap')
    return sorted(scene for scene in train if int(hashlib.sha256(
        ('scannetpp-internal-val-v1:' + scene).encode()).hexdigest()[:8], 16) % 20 != 0)


def check_poses(value, required_ids):
    poses = np.asarray(value, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not required_ids:
        raise ValueError('Expected scanner-indexed 4x4 transforms and raw panorama IDs')
    if min(required_ids) < 0 or max(required_ids) >= len(poses):
        raise ValueError('A raw panorama has no corresponding scanner pose')
    needed = poses[required_ids]
    if not np.isfinite(needed).all() or not np.allclose(needed[:, 3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError('Nonfinite or nonhomogeneous scanner transform')
    rotation = needed[:, :3, :3]
    error = np.max(np.abs(rotation @ rotation.transpose(0, 2, 1) - np.eye(3)))
    determinant_error = np.max(np.abs(np.linalg.det(rotation) - 1))
    if error > 1e-3 or determinant_error > 1e-3:
        raise ValueError('Scanner transforms are not rigid proper rotations')
    return dict(poses=len(poses), required_scan_ids=required_ids,
                rotation_orthogonality_max_abs=float(error),
                rotation_determinant_max_error=float(determinant_error))


def fetch(session, url):
    """Bounded transport; all exceptions are scrubbed before leaving this call."""
    try:
        for attempt in range(4):
            with session.get(url, timeout=(15, 45), stream=True) as response:
                code = response.status_code
                if code in (429, 500, 502, 503, 504) and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                if code != 200:
                    raise SafeDownloadError(f'Official server HTTP {code}')
                contents = response.raw.read(2 * 1024**2 + 1, decode_content=True)
                if len(contents) > 2 * 1024**2:
                    raise SafeDownloadError('Scanner metadata exceeds size bound')
                return contents
    except SafeDownloadError:
        raise
    except Exception:
        raise SafeDownloadError('Scanner metadata transport failed; private details suppressed') from None


def run(dataset, output, private_config):
    if output.exists():
        raise FileExistsError('Supplement exists; inspect before any recovery')
    scenes = eligible_scenes(dataset)
    output.mkdir(parents=True)
    state = dict(stage='downloading_metadata', expected_scenes=len(scenes), complete=0,
                 source_asset='scan_transformed_poses_path', training_started=False,
                 original_files_modified=False, scenes=[], started_at=time.time())
    write_json(output / 'status.json', state)
    try:
        cfg = yaml.safe_load(private_config.read_text())
        template, token = cfg['root_url'], cfg['token']
        if (urlsplit(template).scheme != 'https' or 'TOKEN' not in template
                or 'FILEPATH' not in template or not isinstance(token, str) or not token):
            raise ValueError('Invalid authorized downloader configuration')
        with requests.Session() as session:
            for scene in scenes:
                required = sorted(int(p.stem) for p in (dataset / 'data' / scene /
                                                        'panocam/images').glob('*.jpg'))
                relative = f'data/{scene}/scans/scanner_poses.json'
                state['active_scene'] = scene
                write_json(output / 'status.json', state)
                contents = fetch(session, template.replace('TOKEN', token).replace('FILEPATH', relative))
                check = check_poses(json.loads(contents), required)
                target = output / relative
                atomic_write(target, contents)
                state['scenes'].append(dict(scene=scene, path=relative, bytes=len(contents),
                    sha256=hashlib.sha256(contents).hexdigest(), **check))
                state['complete'] += 1
                if state['complete'] % 25 == 0:
                    print(json.dumps(dict(complete=state['complete'], expected=len(scenes))), flush=True)
                time.sleep(.2)  # Preserve the publisher's downloader pacing.
        state.update(stage='complete', active_scene=None, completed_at=time.time())
        write_json(output / 'status.json', state)
        return state
    except BaseException as error:
        state.update(stage='failed', error_type=type(error).__name__,
            error_message=str(error) if isinstance(error, SafeDownloadError) else
                          'Stopped for metadata/configuration validation; details suppressed')
        write_json(output / 'status.json', state)
        print(json.dumps(dict(stage='failed', error_type=state['error_type'],
                              message=state['error_message'])), flush=True)
        return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--private-config', type=Path, required=True)
    args = parser.parse_args()
    sys.exit(0 if run(args.dataset, args.output, args.private_config)['stage'] == 'complete' else 1)
