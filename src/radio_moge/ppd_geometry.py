"""Cached PPD log-depth calibration and GPU camera-space point maps.

Calibration is explicitly supplied by a keyframe policy. This module does not
assume that a single calibration remains accurate for later video frames.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class LogDepthCalibration:
    a: float
    b: float
    max_depth: float
    min_depth: float = 1e-3

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.a, self.b, self.max_depth, self.min_depth)):
            raise ValueError("Calibration must be finite")
        if self.a <= 0 or not 0 < self.min_depth <= self.max_depth:
            raise ValueError("Calibration must have positive slope and valid depth bounds")


def fit_log_depth(relative, reference, mask, seed):
    """Use the official PPD degree-1 RANSAC objective; no GT is required.

    The random state is local and reproducible. Unlike upstream's silent fitting
    failure fallback, failed or degenerate inputs raise instead of replacing a
    previously valid calibration. The nonpositive-slope fallback follows PPD.
    """
    from sklearn.linear_model import RANSACRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures

    relative = np.asarray(relative, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if relative.shape != reference.shape or mask.shape != reference.shape:
        raise ValueError("Depth maps and mask must have the same shape")
    mask = mask & np.isfinite(relative) & np.isfinite(reference) & (reference > 0)
    if mask.sum() < 16:
        raise ValueError("Too few valid pixels to calibrate")
    x, y = relative[mask], np.log(reference[mask] + 1.0)
    if np.ptp(x) < 1e-6:
        raise ValueError("Relative depth has no usable variation")
    model = make_pipeline(PolynomialFeatures(degree=1, include_bias=False),
                          RANSACRegressor(max_trials=1000, random_state=seed))
    model.fit(x[:, None], y[:, None])
    estimator = model.named_steps['ransacregressor'].estimator_
    a, b = float(estimator.coef_.item()), float(estimator.intercept_.item())
    if a <= 0:
        a, b = float(y.mean() / x.mean()), 0.0
    return LogDepthCalibration(a, b, float(reference[mask].max()))


def apply_log_depth_numpy(relative, calibration):
    """Reference arithmetic, including PPD's positive-depth and upper clipping."""
    relative = np.asarray(relative, dtype=np.float32)
    return np.clip(np.exp(calibration.a * relative + calibration.b) - 1.0,
                   calibration.min_depth, calibration.max_depth)


class CachedPPDGeometry(nn.Module):
    """Fixed normalized intrinsics, pixel-center rays, replaceable calibration.

    Returns an organized H×W×3 camera-space point map and a fresh validity mask.
    RGB may remain in its existing GPU image buffer. There is no CPU compaction,
    data readback, rendering, or world-pose estimation in forward().
    """

    def __init__(self, height, width, intrinsics, calibration, device='cuda'):
        super().__init__()
        k = np.asarray(intrinsics, dtype=np.float64)
        if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0
                or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1])):
            raise ValueError("Invalid normalized camera intrinsics")
        u, v = np.meshgrid((np.arange(width) + .5) / width,
                           (np.arange(height) + .5) / height)
        rays = np.stack((u, v, np.ones_like(u)), -1) @ np.linalg.inv(k).T
        self.height, self.width = height, width
        self.register_buffer('rays', torch.from_numpy(rays.astype(np.float32)).to(device))
        self.register_buffer('parameters_ab_range', torch.empty(4, device=device))
        self.update_calibration(calibration)

    @torch.no_grad()
    def update_calibration(self, calibration):
        values = [calibration.a, calibration.b, calibration.min_depth, calibration.max_depth]
        self.parameters_ab_range.copy_(torch.tensor(values, device=self.rays.device))

    def forward(self, relative, current_mask):
        assert relative.shape == current_mask.shape == (self.height, self.width)
        a, b, lower, upper = self.parameters_ab_range.unbind()
        depth = (relative.float() * a + b).exp() - 1.0
        depth = torch.minimum(torch.maximum(depth, lower), upper)
        valid = current_mask & torch.isfinite(relative) & torch.isfinite(depth)
        points = depth[..., None] * self.rays
        return depth, points, valid


class GeometryGraph:
    """Single-stream fixed-shape graph; outputs are overwritten by next replay."""

    def __init__(self, geometry, relative, compile_graph=True):
        self.geometry = geometry
        self.relative = relative.clone().contiguous()
        self.all_valid = torch.ones_like(self.relative, dtype=torch.bool)
        self.current_mask = self.all_valid.clone()
        forward = geometry.forward
        if compile_graph:
            forward = torch.compile(forward, fullgraph=True, dynamic=False,
                                    options={'max_autotune': False, 'triton.cudagraphs': False})
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(3):
                self.outputs = forward(self.relative, self.current_mask)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.inference_mode():
            self.outputs = forward(self.relative, self.current_mask)
        self.graph.replay()
        torch.cuda.synchronize()

    def __call__(self, relative, current_mask=None):
        assert relative.shape == self.relative.shape and relative.device == self.relative.device
        self.relative.copy_(relative)
        # Never retain the previous frame's learned mask implicitly.
        self.current_mask.copy_(self.all_valid if current_mask is None else current_mask)
        self.graph.replay()
        return self.outputs
