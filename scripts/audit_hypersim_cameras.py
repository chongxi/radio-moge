"""Census every frozen training-frame camera; quarantine non-rigid poses.

No geometry repair, no depth fitting, and no optimizer. A passed census means
all rows are accounted for, NOT that every camera or the corpus is accepted.
"""
import argparse
import json
from pathlib import Path
import sqlite3
import time

import numpy as np

from cache_hypersim_geometry import NativeAssets, camera_assets, source_rows
from indoor_download import write_json
from radio_moge.geometry_cache import publish_json, sha_file
from radio_moge.geometry_sources import hypersim_pose_rotation


def pose_record(matrix):
    matrix = np.asarray(matrix, np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return dict(usable=False, reason='nonfinite_or_wrong_shape')
    scale = float(np.sqrt(np.trace(matrix.T @ matrix) / 3))
    result = dict(uniform_scale_candidate=scale, source_matrix=matrix.tolist())
    if scale > 1e-12:
        rotation = matrix / scale
        result.update(orthogonality_residual=float(np.abs(rotation.T @ rotation - np.eye(3)).max()),
                      determinant=float(np.linalg.det(rotation)),
                      singular_values=np.linalg.svd(matrix)[1].tolist())
    try:
        hypersim_pose_rotation(matrix)
    except ValueError as error:
        result.update(usable=False, reason=str(error))
    else:
        result.update(usable=True)
    return result


def run(config, attempt):
    root = Path(config['root'])
    inventory = json.loads((root / 'inventory.json').read_text())
    if (inventory['manifest_sha256'] != config['manifest_sha256']
            or sha_file(root / 'inventory.sqlite') != inventory['inventory_sha256']):
        raise ValueError('Frozen inventory changed')
    output = root / attempt
    output.mkdir(exist_ok=False)
    groups = sorted(x['group'] for x in inventory['source_groups'] if x['dataset'] == 'hypersim')
    assets, cameras, rejected = NativeAssets(root / 'native-source/hypersim'), {}, []
    status = dict(stage='auditing', started_at=time.time(), frames=0, pose_rejected=0,
                  cameras=0, training_admission=False, optimizer_updates=0)
    try:
        with sqlite3.connect(f'file:{root / "inventory.sqlite"}?mode=ro', uri=True) as db:
            db.execute('PRAGMA cache_size=-32768')
            active, data = None, None
            for row in source_rows(db, groups):
                scene, camera = row['group'].split(':')[1], row['camera']
                key = f'{scene}/{camera}'
                if key != active:
                    data, proofs = camera_assets(assets, row)
                    active = key
                    if key not in cameras:
                        cameras[key] = dict(scene=scene, camera=camera, source_proofs=proofs,
                                            frames=0, rejected_frames=[])
                indices = data['camera_keyframe_frame_indices.hdf5'].reshape(-1)
                match = np.flatnonzero(indices == row['frame_id'])
                if len(match) != 1:
                    raise ValueError('Training frame is missing or ambiguous in source poses')
                index = match.item()
                if (not np.isfinite(data['camera_keyframe_positions.hdf5'][index]).all()
                        or not np.isfinite(data['scale']) or data['scale'] <= 0):
                    raise ValueError('Invalid native position or scene units')
                diagnostic = pose_record(data['camera_keyframe_orientations.hdf5'][index])
                if not diagnostic['usable']:
                    cameras[key]['rejected_frames'].append(row['frame_id'])
                    rejected.append(dict(image=row, camera_key=key, diagnostic=diagnostic))
                    status['pose_rejected'] += 1
                cameras[key]['frames'] += 1
                status['frames'] += 1
                if status['frames'] % 100 == 0:
                    status.update(cameras=len(cameras), updated_at=time.time())
                    write_json(output / 'status.json', status)
                    print(json.dumps(status), flush=True)
        if status['frames'] != config['expected_counts']['hypersim']:
            raise ValueError('Camera census does not cover the frozen training set')
        report = dict(stage='complete', manifest_sha256=config['manifest_sha256'],
            inventory_sha256=inventory['inventory_sha256'], frames=status['frames'],
            pose_rejected=status['pose_rejected'], cameras=cameras, rejections=rejected,
            completed_at=time.time(), training_admission=False, optimizer_updates=0,
            policy='No repair of shear/anisotropic scale/reflection; explicit native quarantine',
            pose_code_sha256=sha_file(Path(__file__).parents[1] / 'src/radio_moge/geometry_sources.py'))
        publish_json(output / 'report.json', report)
        status.update(stage='complete', completed_at=time.time(), cameras=len(cameras))
        write_json(output / 'status.json', status)
        print(json.dumps(status), flush=True)
    except BaseException as error:
        status.update(stage='failed', error=type(error).__name__, message=str(error)[:1000])
        write_json(output / 'status.json', status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if Path(args.attempt).name != args.attempt:
        parser.error('Use a new attempt basename')
    run(json.loads(args.config.read_text()), args.attempt)
