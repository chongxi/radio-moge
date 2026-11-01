"""Prepare original-resolution laser pano RGB/GT as calibrated perspective pairs.

Only official training scenes outside the permanent internal holdout are admitted.
Inputs are immutable. Outputs live in a NEW directory on the dataset/shared disk.
An explicit scene/scan limit is a preparation verification subset, not full data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from radio_moge.checkpoint import atomic_json
from radio_moge.gt_planes import measured_planes
from radio_moge.scannetpp_panorama import PanoramaProjector


def sha_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def heldout(scene):
    return int(hashlib.sha256(('scannetpp-internal-val-v1:' + scene).encode())
               .hexdigest()[:8], 16) % 20 == 0


def plane_targets(depth, valid, intrinsics):
    h, w = depth.shape
    u, v = np.meshgrid((np.arange(w) + .5) / w, (np.arange(h) + .5) / h)
    points = np.stack(((u - intrinsics[0, 2]) / intrinsics[0, 0],
                       (v - intrinsics[1, 2]) / intrinsics[1, 1], np.ones_like(u)), -1)
    points *= depth[..., None]
    labels, planes = np.zeros((h, w), np.uint8), np.zeros((6, 4), np.float32)
    patches = measured_planes(points, valid)
    for i, patch in enumerate(patches):
        labels[patch.mask] = i + 1
        planes[i] = np.r_[patch.normal, patch.offset]
    return labels, planes, [patch.metadata() for patch in patches]


def prepare(dataset, output, scenes=None, scan_limit=None, height=400, width=640,
            split='train', min_gt_fraction=.2):
    if output.exists():
        raise FileExistsError('Use a new output directory; no implicit retry or overwrite')
    train = set((dataset / 'splits/nvs_sem_train.txt').read_text().split())
    forbidden = set()
    for name in ('nvs_sem_val', 'nvs_test', 'sem_test', 'nvs_test_iphone'):
        forbidden.update((dataset / f'splits/{name}.txt').read_text().split())
    if train & forbidden:
        raise ValueError('Official scene splits overlap')
    if split not in ('train', 'val') or not .2 <= min_gt_fraction <= 1:
        raise ValueError('Invalid split or minimum measured support')
    eligible = sorted(s for s in train if heldout(s) == (split == 'val'))
    selected = sorted(scenes) if scenes else eligible
    if (not selected or len(set(selected)) != len(selected)
            or not set(selected) <= set(eligible) or (scan_limit is not None and scan_limit < 1)):
        raise ValueError('Scenes must match the official train and permanent train/val split')
    output.mkdir(parents=True, mode=0o700)
    (output / 'rgb').mkdir()
    (output / 'geometry').mkdir()
    state = {'stage': 'preparing', 'started_at': time.time(), 'source': str(dataset),
             'source_modality': 'original panocam RGB and laser radial range',
             'eligible_training_scenes': len(eligible), 'selected_scenes': selected,
             'verification_subset': bool(scenes or scan_limit), 'scan_limit': scan_limit,
             'split': split, 'min_gt_fraction': min_gt_fraction,
             'plane_method': 'sparse measured GT RANSAC; no hole depth or label filling',
             'records': 0, 'rejected': [], 'input_proofs': [], 'training_started': False,
             'source_files_modified': False, 'all_splits_preserved': True}
    rows = []
    try:
        for scene in selected:
            pano = dataset / 'data' / scene / 'panocam'
            images = sorted((pano / 'images').glob('*.jpg'), key=lambda p: int(p.stem))
            if scan_limit is not None:
                images = images[:scan_limit]
            if not images:
                raise ValueError(f'No original panorama RGB: {scene}')
            for image_path in images:
                source_paths = {'rgb': image_path}
                for key, folder in [('range', 'depth'), ('azimuth', 'azim'),
                                    ('polar', 'elev'), ('mask', 'anon_mask')]:
                    source_paths[key] = pano / folder / (image_path.stem + '.png')
                arrays, proof = {}, {'scene': scene, 'scan': image_path.stem, 'assets': {}}
                for key, path in source_paths.items():
                    before = path.stat()
                    with Image.open(path) as file:
                        if file.getexif().get(274, 1) != 1:
                            raise ValueError('Unexpected source orientation')
                        array = np.array(file.convert('RGB') if key == 'rgb' else file)
                    if key in ('range', 'azimuth', 'polar'):
                        if array.dtype != np.uint16:
                            raise ValueError('Expected original uint16 millimetre/milliradian data')
                        array = array.astype(np.float32) / 1000
                    arrays[key] = array
                    proof['assets'][key] = {'path': str(path), 'bytes': before.st_size,
                                            'sha256': sha_file(path), 'shape': list(array.shape)}
                    after = path.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise ValueError('Source changed while preparing GT')
                projector = PanoramaProjector(arrays['azimuth'], arrays['polar'])
                for yaw in (0, 60, 120, 180, 240, 300):
                    rgb, depth, valid, intrinsics, report, source_indices = projector.project(
                        arrays['rgb'], arrays['range'], arrays['mask'],
                        height=height, width=width, yaw=yaw, return_source_indices=True)
                    name = f'{scene}-{image_path.stem}-yaw{yaw:03d}'
                    if report['rgb_valid_fraction'] < .98 or report['valid_fraction'] < min_gt_fraction:
                        state['rejected'].append({'id': name, 'projection': report})
                        continue
                    labels, planes, regions = plane_targets(depth, valid, intrinsics)
                    rgb_path, gt_path = output / 'rgb' / (name + '.png'), output / 'geometry' / (name + '.npz')
                    temporary = rgb_path.with_suffix('.png.incoming')
                    Image.fromarray(rgb).save(temporary, format='PNG')
                    temporary.replace(rgb_path)
                    temporary = gt_path.with_suffix('.npz.incoming')
                    with temporary.open('xb') as file:
                        np.savez_compressed(file, depth=depth, valid=valid, intrinsics=intrinsics,
                                            plane_labels=labels, planes=planes,
                                            source_indices=source_indices)
                    temporary.replace(gt_path)
                    # Independent readback before admitting a published record.
                    with Image.open(rgb_path) as file:
                        if not np.array_equal(np.array(file), rgb):
                            raise ValueError('RGB roundtrip mismatch')
                    with np.load(gt_path, allow_pickle=False) as packed:
                        if not np.array_equal(packed['depth'], depth) or not np.array_equal(packed['valid'], valid):
                            raise ValueError('Geometry roundtrip mismatch')
                    rows.append({'image': str(rgb_path.relative_to(output)),
                                 'geometry': str(gt_path.relative_to(output)),
                                 'sha256': sha_file(rgb_path), 'geometry_sha256': sha_file(gt_path),
                                 'dataset': 'scannetpp_panocam', 'supervision': 'scannetpp_panocam_laser',
                                 'group': 'scannetpp:' + scene, 'scene': scene, 'id': name,
                                 'split': split, 'official_split': 'nvs_sem_train',
                                 'source_scan': image_path.stem, 'projection': report,
                                 'planes': regions, 'width': width, 'height': height})
                    state['records'] = len(rows)
                    state['updated_at'] = time.time()
                    atomic_json(state, output / 'preparation.json')
                    print(json.dumps({'id': name, 'valid_fraction': report['valid_fraction'],
                                      'plane_regions': len(regions)}), flush=True)
                state['input_proofs'].append(proof)
                del projector, arrays
        if not rows:
            raise ValueError('No view passed the calibrated GT admission checks')
        temporary = output / (split + '.jsonl.incoming')
        temporary.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
        temporary.replace(output / (split + '.jsonl'))
        state.update(stage='prepared', finished_at=time.time(), manifest_sha256=sha_file(output / (split + '.jsonl')))
        atomic_json(state, output / 'preparation.json')
    except BaseException as error:
        state.update(stage='failed', failed_at=time.time(), error=type(error).__name__ + ': ' + str(error))
        atomic_json(state, output / 'preparation.json')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--scenes', nargs='+')
    parser.add_argument('--scan-limit', type=int)
    parser.add_argument('--split', choices=['train', 'val'], default='train')
    parser.add_argument('--min-gt-fraction', type=float, default=.2)
    args = parser.parse_args()
    prepare(args.dataset, args.output, args.scenes, args.scan_limit,
            split=args.split, min_gt_fraction=args.min_gt_fraction)
