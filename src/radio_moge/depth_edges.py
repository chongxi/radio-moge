"""MoGe CLI relative-depth edge masking, without changing surviving depths.

Equivalent to utils3d_moge.np.depth_map_edge(depth, rtol=threshold), using
its default 3 x 3 neighborhood and ignoring NaN neighbors. The validity mask
is applied AFTER edge detection, as in MoGe's export script.
"""

import numpy as np
from scipy.ndimage import maximum_filter


def relative_depth_edge_score(depth):
    """Return (local maximum - local minimum) / center depth for a 2D map."""
    depth = np.asarray(depth)
    if depth.ndim != 2 or depth.dtype.kind != "f":
        raise ValueError("Expected a two-dimensional floating point depth map")
    high = maximum_filter(np.where(np.isnan(depth), -np.inf, depth),
                          size=3, mode="constant", cval=-np.inf)
    low_negative = maximum_filter(np.where(np.isnan(depth), -np.inf, -depth),
                                  size=3, mode="constant", cval=-np.inf)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (high + low_negative) / depth


def depth_edge_keep_mask(depth, valid, threshold=0.04):
    """Keep finite, positive valid pixels not flagged by the relative edge test."""
    depth, valid = np.asarray(depth), np.asarray(valid, dtype=bool)
    if valid.shape != depth.shape or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Invalid validity mask or positive finite threshold")
    return (valid & np.isfinite(depth) & (depth > 0)
            & ~(relative_depth_edge_score(depth) > threshold))
