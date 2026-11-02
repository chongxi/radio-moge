"""Explicit camera conventions for offline depth preparation (NumPy only)."""
import numpy as np


def scannetpp_nerfstudio_to_colmap_c2w(transform):
    """Undo BOTH world and camera changes in official common/utils/nerfstudio.py."""
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError('Expected finite 4x4 camera-to-world transform')
    world = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1.]])
    camera = np.diag([1., -1., -1., 1.])
    return world @ transform @ camera


def normalized_opencv_intrinsics(width, height, fx, fy, cx, cy):
    """OpenCV integer pixel-center coordinates -> normalized half-center grid."""
    if min(width, height, fx, fy) <= 0 or not np.isfinite([width, height, fx, fy, cx, cy]).all():
        raise ValueError('Invalid pinhole calibration')
    return np.array([[fx / width, 0, (cx + .5) / width],
                     [0, fy / height, (cy + .5) / height], [0, 0, 1.]], dtype=np.float64)


def pinhole_rays(intrinsics, height, width):
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if (intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()
            or not np.allclose(intrinsics[2], [0, 0, 1])
            or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0):
        raise ValueError('Finite pinhole camera required')
    x, y = np.meshgrid((np.arange(width) + .5) / width, (np.arange(height) + .5) / height)
    rays = np.stack((x, y, np.ones_like(x)), -1) @ np.linalg.inv(intrinsics).T
    return rays


def range_to_z(radial, rays, valid):
    radial, rays, valid = np.asarray(radial), np.asarray(rays), np.asarray(valid, dtype=bool)
    if rays.shape != (*radial.shape, 3) or valid.shape != radial.shape:
        raise ValueError('Range, rays and mask dimensions differ')
    norm = np.linalg.norm(rays, axis=-1)
    support = valid & np.isfinite(radial) & (radial > 0)
    support &= np.isfinite(rays).all(-1) & (norm > 0) & (rays[..., 2] > 0)
    depth = np.zeros(radial.shape, np.float32)
    depth[support] = radial[support] * rays[..., 2][support] / norm[support]
    return depth, support
