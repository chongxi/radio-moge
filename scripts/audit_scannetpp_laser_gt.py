"""Independently verify prepared laser pairs against their original source pixels."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from radio_moge.checkpoint import atomic_json
from radio_moge.data import read_manifest
from radio_moge.gt_data import validate_gt_records


def digest(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def audit(root):
    state = json.loads((root / 'preparation.json').read_text())
    assert state['stage'] == 'prepared'
    split = state['split']
    manifest = root / (split + '.jsonl')
    assert digest(manifest) == state['manifest_sha256']
    rows = read_manifest(manifest)
    validate_gt_records(rows, split=split)
    assert len({r['id'] for r in rows}) == len(rows)
    assert len({r['sha256'] for r in rows}) == len(rows), 'Duplicate RGB output'
    reports = []
    for proof in state['input_proofs']:
        selected = [r for r in rows if r['scene'] == proof['scene'] and r['source_scan'] == proof['scan']]
        if not selected:
            continue
        arrays = {}
        for key, asset in proof['assets'].items():
            path = Path(asset['path'])
            assert path.stat().st_size == asset['bytes'] and digest(path) == asset['sha256']
            with Image.open(path) as image:
                arrays[key] = np.array(image.convert('RGB') if key == 'rgb' else image)
        for row in selected:
            assert digest(row['image']) == row['sha256'] and digest(row['geometry']) == row['geometry_sha256']
            with Image.open(row['image']) as image:
                rgb = np.array(image)
            with np.load(row['geometry'], allow_pickle=False) as packed:
                depth, valid, k = packed['depth'], packed['valid'], packed['intrinsics']
                ij, labels, planes = packed['source_indices'], packed['plane_labels'], packed['planes']
            h, w = depth.shape
            yy, xx = ij[..., 0], ij[..., 1]
            assert yy.min() >= 0 and yy.max() < arrays['range'].shape[0]
            assert xx.min() >= 0 and xx.max() < arrays['range'].shape[1]
            assert np.array_equal(rgb[valid], arrays['rgb'][yy[valid], xx[valid]])
            assert (arrays['mask'][yy[valid], xx[valid]] == 255).all()
            az = arrays['azimuth'][yy, xx].astype(np.float64) * .001
            el = arrays['polar'][yy, xx].astype(np.float64) * .001
            scanner = np.stack((np.sin(el) * np.cos(az), np.sin(el) * np.sin(az), np.cos(el)), -1)
            rotation = np.asarray(row['projection']['original_camera_to_scanner_rotation'])
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            assert abs(np.linalg.det(rotation) - 1) < 1e-6
            camera = scanner @ rotation
            z = arrays['range'][yy, xx].astype(np.float64) * .001 * camera[..., 2]
            np.testing.assert_allclose(depth[valid], z[valid], rtol=1e-6, atol=3e-6)
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            pu = (camera[..., 0] / np.maximum(camera[..., 2], 1e-8) * k[0, 0] + k[0, 2]) * w - .5
            pv = (camera[..., 1] / np.maximum(camera[..., 2], 1e-8) * k[1, 1] + k[1, 2]) * h - .5
            pixel_error = np.hypot(pu - u, pv - v)
            assert pixel_error[valid].max() <= .5002
            # Verify original 3x3 masks, holes and relative-range discontinuities on ALL accepted pixels.
            y, x = yy[valid], xx[valid]
            ranges = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    sy = np.clip(y + dy, 0, arrays['range'].shape[0] - 1)
                    sx = np.clip(x + dx, 0, arrays['range'].shape[1] - 1)
                    ranges.append(arrays['range'][sy, sx].astype(np.float64))
                    assert (arrays['mask'][sy, sx] == 255).all()
            lo, hi = np.min(ranges, axis=0), np.max(ranges, axis=0)
            assert (lo > 0).all() and (hi < 65535).all()
            assert ((hi - lo) <= .03001 * lo).all()
            rays = np.stack(((u + .5) / w, (v + .5) / h, np.ones_like(u)), -1) @ np.linalg.inv(k).T
            points = rays * depth[..., None]
            rms = []
            for label in np.unique(labels[valid & (labels > 0)]):
                mask = valid & (labels == label)
                equation = planes[int(label) - 1]
                residual = points[mask] @ equation[:3] + equation[3]
                measured = float(np.sqrt(np.mean(residual ** 2)))
                assert abs(measured - row['planes'][int(label) - 1]['rms']) < 3e-6
                rms.append(measured)
            assert valid.mean() >= state['min_gt_fraction']
            reports.append({'id': row['id'], 'valid_pixels': int(valid.sum()),
                            'valid_fraction': float(valid.mean()), 'max_pixel_error': float(pixel_error[valid].max()),
                            'max_z_difference_m': float(np.max(np.abs(depth[valid] - z[valid]))),
                            'plane_rms_m': rms})
    assert len(reports) == len(rows)
    result = {'stage': 'passed', 'checked_at': time.time(), 'records': len(rows),
              'scenes': len({r['scene'] for r in rows}), 'split': split,
              'manifest_sha256': digest(manifest), 'independent_original_pixel_checks': True,
              'all_output_hashes_verified': True, 'official_permanent_splits_preserved': True,
              'source_not_modified': True, 'rows': reports}
    atomic_json(result, root / 'data-verification.json')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
