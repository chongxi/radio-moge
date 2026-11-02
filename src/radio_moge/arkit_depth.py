"""ARKit native-resolution support and exact camera-grid registration.

Do not resize sparse depth into a dense RGB label. The native grid and its own K
are retained; its rays map into the RGB grid through the two calibrated cameras.
Quarter turns rotate both cameras and measurements without interpolation.
"""
import numpy as np
from scipy.ndimage import binary_erosion, maximum_filter, minimum_filter

from .geometry_sources import normalized_opencv_intrinsics, pinhole_rays


def rotate_camera(camera, rotation_cw):
    if type(rotation_cw) is not int or rotation_cw not in (0, 90, 180, 270):
        raise ValueError('Only exact quarter turns are permitted')
    width, height, fx, fy, cx, cy = map(float, camera)
    for _ in range(rotation_cw // 90):
        width, height, fx, fy, cx, cy = height, width, fy, fx, height - 1 - cy, cx
    return [width, height, fx, fy, cx, cy]


def registration_uv(depth_camera, rgb_camera):
    """RGB normalized half-center coordinates for each ORIGINAL depth pixel."""
    depth_camera, rgb_camera = np.asarray(depth_camera), np.asarray(rgb_camera)
    if depth_camera.shape != (6,) or rgb_camera.shape != (6,):
        raise ValueError('Each native camera must include width/height/fx/fy/cx/cy')
    kd = normalized_opencv_intrinsics(*depth_camera)
    kr = normalized_opencv_intrinsics(*rgb_camera)
    rays = pinhole_rays(kd, int(depth_camera[1]), int(depth_camera[0]))
    projected = rays @ kr.T
    return projected[..., :2] / projected[..., 2:], kd, kr


def native_depth(raw, confidence, camera, rotation_cw):
    raw, confidence = np.asarray(raw), np.asarray(confidence)
    if (raw.dtype != np.uint16 or raw.ndim != 2 or confidence.dtype != np.uint8
            or raw.shape != confidence.shape or raw.shape != (int(camera[1]), int(camera[0]))
            or not np.isin(confidence, [0, 1, 2]).all()):
        raise ValueError('Unexpected native ARKit depth/confidence/camera encoding')
    # Official source: uint16 millimeters and confidence 2 (high). Saturation,
    # holes and depth boundaries remain excluded. This is not a physical error bound.
    support = (raw > 0) & (raw < 65535) & (confidence == 2)
    depth = raw.astype(np.float32) / 1000
    valid = binary_erosion(support, structure=np.ones((3, 3)), border_value=0)
    valid &= maximum_filter(depth, 3) - minimum_filter(depth, 3) <= .05 + .02 * depth
    depth[~valid] = 0
    output_camera = rotate_camera(camera, rotation_cw)
    return (np.rot90(depth, -rotation_cw // 90).copy(),
            np.rot90(valid, -rotation_cw // 90).copy(), output_camera)
