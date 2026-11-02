"""Literal NYUv2 publisher-toolbox projection for source reproducibility audits.

Reference: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html (toolbox).
Deliberately retains its mixed pixel origins, negative-axis extrinsic and sensor
Z output. This is NOT an accepted/corrected RGB-camera-Z training conversion.
"""
import numpy as np


RGB_F = np.array([5.1885790117450188e2, 5.1946961112127485e2])
RGB_C = np.array([3.2558244941119034e2, 2.5373616633400465e2])
RGB_D = np.array([.20796615318809061, -.58613825163911781,
                  .00072231363135888329, .0010479627195765181, .49856986684705107])
DEPTH_F = np.array([5.8262448167737955e2, 5.8269103270988637e2])
DEPTH_C = np.array([3.1304475870804731e2, 2.3844389626620386e2])
DEPTH_D = np.array([-.099897236553084481, .39065324602765344,
                    .0019290592870229277, -.0019422022475975055, -.51031725053400578])
_R = -np.array([.99997798940829263, .0050518419386157446, .0043011152014118693,
                -.0050359919480810989, .99998051861143999, -.0036879781309514218,
                -.0043196624923060242, .0036662365748484798, .99998394948385538]).reshape(3, 3, order='F')
ROTATION = np.linalg.inv(_R.T)
TRANSLATION = np.array([.025031875059141302, -.00029342312935846411, .00066238747008330102])


def undistort(image, focal, center, distortion, noise_mask=None):
    """Publisher's rect/undistort: zero-origin rays, explicit bilinear weights."""
    image = np.asarray(image, float)
    h, w = image.shape[:2]
    yy, xx = np.indices((h, w), dtype=float)
    x, y = (xx - center[0]) / focal[0], (yy - center[1]) / focal[1]
    k1, k2, p1, p2, k3 = distortion
    radius2 = x * x + y * y
    radial = 1 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
    sx = focal[0] * (x * radial + 2 * p1 * x * y + p2 * (radius2 + 2 * x * x)) + center[0]
    sy = focal[1] * (y * radial + p1 * (radius2 + 2 * y * y) + 2 * p2 * x * y) + center[1]
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    good = (x0 >= 0) & (x0 <= w - 2) & (y0 >= 0) & (y0 <= h - 2)
    a, b = sx[good] - x0[good], sy[good] - y0[good]
    x0, y0 = x0[good], y0[good]
    weights = np.stack(((1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b))
    ys, xs = np.stack((y0, y0, y0 + 1, y0 + 1)), np.stack((x0, x0 + 1, x0, x0 + 1))
    if noise_mask is not None:
        weights *= ~np.asarray(noise_mask, bool)[ys, xs]
        total = weights.sum(0)
        weights = np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)
    output = np.full_like(image, 255. if noise_mask is None else 0.)
    values = image[ys, xs]
    if image.ndim == 3:
        weights = weights[..., None]
    output[good] = (values * weights).sum(0)
    return output


def reproduce_projected_depth(raw_codes):
    """Already-byte-swapped Kinect codes -> published *sensor* Z on RGB array."""
    raw_codes = np.asarray(raw_codes, float)
    if raw_codes.shape != (480, 640):
        raise ValueError('The publisher toolbox requires 480x640 Kinect frames')
    noise = undistort(255. * (raw_codes == raw_codes.max()), DEPTH_F, DEPTH_C, DEPTH_D) > 0
    raw = undistort(raw_codes, DEPTH_F, DEPTH_C, DEPTH_D, noise)
    raw[(raw < 600) | noise] = 2047
    depth = 351.3 / (1092.5 - raw)
    depth = np.where(depth < 0, 0, np.minimum(depth, 10))
    yy, xx = np.indices(depth.shape, dtype=float)
    # Exact publisher one-origin depth_plane2depth_world (not silently fixed).
    points = np.stack(((xx + 1 - DEPTH_C[0]) / DEPTH_F[0] * depth,
                       (yy + 1 - DEPTH_C[1]) / DEPTH_F[1] * depth, depth), -1)
    transformed = points @ ROTATION.T + TRANSLATION
    projected = transformed[..., :2] / transformed[..., 2:] * RGB_F + RGB_C
    xy = np.floor(projected + .5).astype(int)  # Only positive in-frame indices survive.
    good = (xy[..., 0] > 0) & (xy[..., 0] < 640) & (xy[..., 1] > 0) & (xy[..., 1] < 480)
    indices = (xy[..., 1][good] - 1) * 640 + xy[..., 0][good] - 1
    output = np.full(480 * 640, np.inf)
    # Equivalent to descending sensor-depth sorted overwrite; no RGB Z correction.
    np.minimum.at(output, indices, depth[good])
    output[~np.isfinite(output)] = 0
    return output.reshape(480, 640)


def reproduce_undistorted_rgb(rgb):
    value = undistort(rgb, RGB_F, RGB_C, RGB_D)
    return np.floor(np.clip(value, 0, 255) + .5).astype(np.uint8)
