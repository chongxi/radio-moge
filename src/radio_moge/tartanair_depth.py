"""Conservative TartanAir V1 left-camera Z, never radial depth or sky GT.

Publisher calibration: 640x480, fx=fy=320, cx=320, cy=240, baseline=.25m.
The finite sky sphere is not a real distant surface. Our 0.1--200m task domain
is an explicit training selection, NOT a publisher invalid-depth definition.
All consistency checks use the released right eye; they are not an independent
physical measurement of this synthetic world.
"""
import numpy as np
from scipy.ndimage import binary_erosion, map_coordinates, maximum_filter, minimum_filter

from radio_moge.geometry_sources import normalized_opencv_intrinsics


def _support(depth, maximum_depth):
    native = np.isfinite(depth) & (depth > 0)
    task = native & (depth >= .1) & (depth <= maximum_depth)
    safe = np.where(task, depth, 0)
    stable = binary_erosion(task, iterations=1)
    stable &= maximum_filter(safe, 3) - minimum_filter(safe, 3) <= .05 + .02 * safe
    return native, task, stable


def convert_tartanair(left, right, left_rgb, right_rgb, maximum_depth=200.):
    left, right, left_rgb, right_rgb = map(np.asarray, (left, right, left_rgb, right_rgb))
    if (left.shape != (480, 640) or right.shape != left.shape
            or left.dtype.kind != 'f' or right.dtype.kind != 'f'
            or left_rgb.shape != (*left.shape, 3) or right_rgb.shape != left_rgb.shape
            or left_rgb.dtype != np.uint8 or right_rgb.dtype != np.uint8
            or not np.isfinite(maximum_depth) or not 1 <= maximum_depth <= 200):
        raise ValueError('Require released V1 float depth and uint8 RGB on native 480x640 grids')
    native, task, stable_left = _support(left, maximum_depth)
    _, _, stable_right = _support(right, maximum_depth)
    ys, xs = np.indices(left.shape, dtype=float)
    # Left pixel's point projects LEFT in the right image, without a y shift.
    disparity = np.zeros(left.shape, float)
    np.divide(80., left, out=disparity, where=task)
    rx = xs - disparity
    coords = [ys, rx]
    in_frame = (rx >= 0) & (rx <= 639)
    right_safe = np.where(stable_right, right, 0).astype(float)
    projected_z = map_coordinates(right_safe, coords, order=1, mode='constant', cval=0)
    right_support = map_coordinates(stable_right.astype(float), coords, order=1,
                                    mode='constant', cval=0) >= 1 - 1e-9
    geometric = stable_left & in_frame & right_support
    error_z = np.abs(projected_z - np.where(task, left, 0))
    geometric &= error_z <= .01 + .005 * np.where(task, left, 0)
    warped_rgb = np.stack([map_coordinates(right_rgb[..., c].astype(float) / 255, coords,
        order=1, mode='constant', cval=0) for c in range(3)], -1)
    photometric_error = np.abs(warped_rgb - left_rgb.astype(float) / 255).mean(-1)
    valid = geometric & (photometric_error <= .08)
    depth = np.where(valid, left, 0).astype(np.float32)
    quality = dict(native_finite_positive_fraction=float(native.mean()),
        task_range_fraction=float(task.mean()), task_range_m=[.1, float(maximum_depth)],
        task_range_is_not_publisher_invalid_mask=True,
        stable_left_fraction=float(stable_left.mean()),
        stereo_geometric_fraction=float(geometric.mean()), valid_fraction=float(valid.mean()),
        rgb_stereo_mae_on_geometric=float(photometric_error[geometric].mean()) if geometric.any() else None,
        stereo_depth_max_error_m=float(error_z[valid].max()) if valid.any() else None,
        source_needs_review=bool(valid.mean() < .05),
        independent_physical_depth_accuracy_checked=False)
    k = normalized_opencv_intrinsics(640, 480, 320, 320, 320, 240)
    return depth, valid, k, quality
