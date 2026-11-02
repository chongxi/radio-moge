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


def hypersim_pinhole_frame(cam_from_uv):
    """Exact RQ camera frame for Hypersim tilt-shift matrices, WITHOUT RGB warp.

    The publisher's matrix maps (2u-1, 1-2v, 1) into OpenGL camera rays. A rigid
    rotation of the camera coordinate frame yields a pinhole K, preserving every
    image ray and radial distance. Never substitute one universal focal length.
    Returns normalized K and the rotation from OpenCV-converted publisher axes.
    """
    from scipy.linalg import rq

    matrix = np.asarray(cam_from_uv, np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError('Hypersim camera matrix must be finite 3x3')
    grid = np.array([[2, 0, -1], [0, -2, 1], [0, 0, 1.]])
    original_rays = np.diag([1., -1., -1.]) @ matrix @ grid
    if np.linalg.det(original_rays) <= 1e-10:
        raise ValueError('Unsupported reflected or singular camera')
    k, rotation = rq(np.linalg.inv(original_rays))
    signs = np.diag(np.where(np.diag(k) < 0, -1., 1.))
    k, rotation = k @ signs, signs @ rotation
    k /= k[2, 2]
    if (np.linalg.det(rotation) < .999999 or not np.allclose(rotation @ rotation.T, np.eye(3))
            or not np.allclose(k[2], [0, 0, 1]) or k[0, 0] <= 0 or k[1, 1] <= 0):
        raise ValueError('Unable to construct a proper pinhole camera frame')
    check = k @ rotation @ original_rays
    if check[2, 2] <= 0 or not np.allclose(check / check[2, 2], np.eye(3), atol=1e-9):
        raise ValueError('RQ camera does not preserve the original pixel rays')
    return k, rotation
