"""Detect teacher planes in an immutable viewer export and compare student flatness.

This is an illustrative analysis of existing predictions, not model evaluation or
training. Detector thresholds are fixed before inspecting a sample's results.
"""

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from radio_moge.checkpoint import atomic_json
from radio_moge.planes import PlaneConfig, detect_planes, fit_plane, plane_losses, plane_config_dict
from radio_moge.pointcloud import unproject


def atomic_bytes(path, data):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def plane_metrics(points, mask, teacher_normal):
    selected = points[mask]
    if not np.isfinite(selected).all() or (selected[:, 2] <= 0).any():
        raise ValueError("Invalid prediction inside fixed teacher mask; do not drop student outliers")
    n, d, values = fit_plane(selected)
    extent = float(np.sqrt(values[1:].sum()))
    residual = np.abs(selected @ n + d)
    if extent < 1e-8:
        raise ValueError("Collapsed surface")
    return {"rms_m": float(np.sqrt(np.mean(residual ** 2))),
            "normalized_rms": float(np.sqrt(np.mean(residual ** 2)) / extent),
            "p95_normalized": float(np.quantile(residual, .95) / extent),
            "normal_angle_to_teacher_deg": float(np.rad2deg(np.arccos(
                np.clip(abs(n @ teacher_normal), 0, 1)))),
            "extent_m": extent, "normal": n.tolist(), "offset": d}, residual / extent


def run(viewer, sample_id, output):
    started = time.monotonic()
    if output.exists():
        raise FileExistsError(f"Use a fresh report directory: {output}")
    output.mkdir(parents=True)
    manifest = json.loads((viewer / "manifest.json").read_text())
    geometry = json.loads((viewer / "geometry/manifest.json").read_text())
    assert geometry['depth_manifest_sha256'] == hashlib.sha256(
        (viewer / 'manifest.json').read_bytes()).hexdigest()
    index = next(i for i, row in enumerate(manifest['samples']) if row['id'] == sample_id)
    sample, camera = manifest['samples'][index], geometry['samples'][index]
    assert sample['id'] == camera['id']
    cfg = PlaneConfig()
    atomic_json({'stage': 'prepared', 'sample': sample, 'source': str(viewer),
                 'detector': plane_config_dict(), 'training': False,
                 'selection': 'User screenshot match by scene appearance and both rounded FoVs; '
                 'illustrative held-out image, not threshold tuning or a new benchmark'},
                output / 'plan.json')
    inputs = [viewer / 'manifest.json', viewer / 'geometry/manifest.json', viewer / sample['depth'],
              viewer / sample['rgb'], viewer / camera['teacher']['mask'],
              viewer / camera['after']['mask']]
    proofs = [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
               'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in inputs]
    depths = np.frombuffer(gzip.decompress((viewer / sample['depth']).read_bytes()), dtype='<f4')
    depths = depths.reshape(len(manifest['channels']), sample['height'], sample['width'])
    points = {name: unproject(depths[manifest['channels'].index(name)], camera[name]['intrinsics'])
              for name in ('teacher', 'after')}
    valid = np.asarray(Image.open(viewer / camera['teacher']['mask'])) > 0
    patches = detect_planes(points['teacher'], valid, cfg)
    masks = np.stack([p.mask for p in patches]) if patches else np.zeros((0, *valid.shape), bool)
    buf = io.BytesIO()
    np.savez_compressed(buf, masks=masks)
    atomic_bytes(output / 'teacher-plane-masks.npz', buf.getvalue())
    rows, residuals = [], {k: np.full(valid.shape, np.nan) for k in points}
    for i, patch in enumerate(patches):
        row = {'plane_id': i + 1, 'teacher_fit': patch.metadata()}
        for key in points:
            metrics, distances = plane_metrics(points[key], patch.mask, patch.normal)
            row[key] = metrics
            residuals[key][patch.mask] = distances
        row['student_teacher_flatness_ratio'] = (
            row['after']['normalized_rms'] / max(row['teacher']['normalized_rms'], 1e-12))
        rows.append(row)
    # Validate autograd on these exact cached surfaces, without updating a model.
    teacher = torch.tensor(points['teacher'][None], dtype=torch.float32)
    student = torch.tensor(points['after'][None], dtype=torch.float32, requires_grad=True)
    losses = plane_losses(student, teacher, [patches])
    sum(losses.values()).backward()
    assert torch.isfinite(student.grad).all()
    union = masks.any(0)
    outside = student.grad[0][~torch.tensor(union)]
    assert not outside.numel() or outside.abs().max() == 0
    if patches:
        assert student.grad.abs().sum() > 0
    for proof in proofs:
        p = Path(proof['path'])
        assert p.stat().st_mtime_ns == proof['mtime_ns'] and p.stat().st_size == proof['bytes']
    result = {'stage': 'passed', 'sample_id': sample_id, 'source': str(viewer),
              'checkpoint_step': manifest['run']['steps'], 'plane_count': len(patches),
              'coverage_fraction': float(union.mean()), 'planes': rows,
              'losses': {k: v.item() for k, v in losses.items()},
              'gradient_finite': True, 'gradient_outside_teacher_masks_zero': True,
              'gradient_l2': float(student.grad.norm()), 'inputs': proofs,
              'elapsed_seconds': time.monotonic() - started,
              'limits': 'Geometric teacher pseudo-labels, not semantic segmentation or ground truth. '
              'Illustrative screenshot analysis only; no training, smoothing or deployed edits. '
              'Flatness uses each surface own best-fit plane/extent; pose is reported separately.'}
    atomic_json(result, output / 'report.json')
    rgb = np.asarray(Image.open(viewer / sample['rgb']).convert('RGB')) / 255.
    overlay = rgb.copy()
    colors = plt.get_cmap('tab10')
    for i, patch in enumerate(patches):
        overlay[patch.mask] = .45 * overlay[patch.mask] + .55 * np.array(colors(i)[:3])
    buf = io.BytesIO()
    Image.fromarray((overlay * 255).astype('uint8')).save(buf, format='PNG')
    atomic_bytes(output / 'plane-overlay.png', buf.getvalue())
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"Original: {sample_id}, {sample['scene']}")
    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title(f'Teacher-selected planes: {len(patches)}, coverage {union.mean():.1%}')
    for i, patch in enumerate(patches):
        y, x = np.argwhere(patch.mask).mean(0)
        axes[0, 1].text(x, y, str(i + 1), color='white', fontsize=13,
                        bbox={'facecolor': 'black', 'alpha': .65, 'pad': 2})
    nonempty = [r[np.isfinite(r)] for r in residuals.values() if np.isfinite(r).any()]
    vmax = max(.2, float(np.quantile(np.concatenate(nonempty), .98)) * 100) if nonempty else 1
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('#17232c')
    for ax, key, title in zip(axes[1], ('teacher', 'after'), ('MoGe-2', 'Trained model')):
        im = ax.imshow(residuals[key] * 100, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(title + ': distance to own best-fit plane / extent')
    fig.colorbar(im, ax=list(axes[1]), label='Normalized plane residual (%)', shrink=.8)
    for ax in axes.flat:
        ax.axis('off')
    fig.suptitle('Fixed teacher masks; same pixels; no global-scale advantage', fontsize=14)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=145)
    atomic_bytes(output / 'plane-diagnostic.png', buf.getvalue())
    plt.close(fig)
    print(json.dumps({k: v for k, v in result.items() if k != 'inputs'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer', type=Path, required=True)
    parser.add_argument('--sample', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.viewer, args.sample, args.output)
