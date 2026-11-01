"""Fit fixed planes to measured sparse GT without inventing depth in holes."""
import numpy as np
from scipy import ndimage

from .planes import PlaneConfig, PlanePatch, fit_plane


def measured_planes(points, valid, config=None):
    cfg = config or PlaneConfig()
    p = np.asarray(points, dtype=np.float64)
    available = np.asarray(valid, dtype=bool).copy() & np.isfinite(p).all(-1) & (p[..., 2] > 0)
    if available.sum() < 3:
        return []
    scale = float(np.median(p[..., 2][available]))
    threshold = cfg.distance_ratio * scale
    min_pixels = max(32, int(np.ceil(available.size * cfg.min_area_ratio)))
    rng, patches = np.random.default_rng(cfg.seed), []
    for _ in range(cfg.max_planes * 3):
        ids = np.flatnonzero(available)
        if len(ids) < min_pixels or len(patches) == cfg.max_planes:
            break
        if len(ids) > cfg.max_samples:
            ids = rng.choice(ids, cfg.max_samples, replace=False)
        samples = p.reshape(-1, 3)[ids]
        triples = samples[rng.integers(len(samples), size=(cfg.iterations, 3))]
        normals = np.cross(triples[:, 1] - triples[:, 0], triples[:, 2] - triples[:, 0])
        length = np.linalg.norm(normals, axis=1)
        keep = length > scale * scale * 1e-8
        normals = normals[keep] / length[keep, None]
        offsets = -(normals * triples[keep, 0]).sum(1)
        best, score = None, 0
        for start in range(0, len(normals), 32):
            support = np.abs(samples @ normals[start:start + 32].T + offsets[start:start + 32]) < threshold
            counts = support.sum(0)
            i = int(counts.argmax())
            if counts[i] > score:
                best, score = support[:, i], int(counts[i])
        if best is None or score < 3:
            break
        normal, offset, _ = fit_plane(samples[best])
        candidates = available & (np.abs(p @ normal + offset) < threshold)
        available[candidates] = False
        # Closing is ONLY for connected-region discovery. No missing pixel gains
        # a depth target or a plane label; final support is intersected with GT.
        regions = ndimage.binary_closing(candidates, structure=np.ones((5, 5)))
        if cfg.erosion:
            regions = ndimage.binary_erosion(regions, iterations=cfg.erosion)
        labels, _ = ndimage.label(regions)
        counts = np.bincount(labels[candidates].ravel())
        for index in np.flatnonzero(counts >= min_pixels):
            if index == 0:
                continue
            mask = candidates & (labels == index)
            normal, offset, _ = fit_plane(p[mask])
            mask &= np.abs(p @ normal + offset) < threshold
            if mask.sum() < min_pixels:
                continue
            normal, offset, values = fit_plane(p[mask])
            rms = float(np.sqrt(values[0]))
            if rms > cfg.max_rms_ratio * scale or np.sqrt(values[1]) < cfg.min_extent_ratio * scale:
                continue
            patches.append(PlanePatch(mask, normal, offset, rms, float(np.sqrt(values.sum())),
                                      float(np.clip(1 - rms / threshold, 0, 1))))
            if len(patches) == cfg.max_planes:
                break
    return patches
