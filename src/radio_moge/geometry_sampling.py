"""Camera-aware, one-observation-per-cell candidate sampling (NumPy only).

RGB may be filtered/warped. Geometry NEVER is: a loss cell contains one exact
native observation, its exact projected UV, and its original integer index.
Sampling predictions at that UV is not interpolation of a measured label.
This is our data adapter, not the upstream MoGe label-refinement pipeline.
"""
import io
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry_cache import LABEL_KINDS, read_geometry_member
from .geometry_sources import pinhole_rays


def camera_matrix(value):
    k = np.asarray(value, np.float64)
    if (k.shape != (3, 3) or not np.isfinite(k).all()
            or not np.allclose(k[2], [0, 0, 1], atol=1e-10, rtol=0)
            or k[0, 0] <= 0 or k[1, 1] <= 0 or abs(np.linalg.det(k)) < 1e-12):
        raise ValueError('Invalid normalized pinhole camera')
    return k


def centered_view(k_rgb, rgb_shape, output_shape, center=(.5, .5), zoom=1., flip=False):
    """Rotate the chosen RGB ray onto +Z; fit a centered square-pixel viewport.

    The new pose is obtained ONLY from calibration, never from a fitted depth.
    Bisection fits the complete inverse-homography quadrilateral inside the RGB
    pixel-center rectangle; this also supports off-image principal points.
    """
    k_rgb = camera_matrix(k_rgb)
    ih, iw = rgb_shape
    oh, ow = output_shape
    if (min(ih, iw, oh, ow) < 4 or not 1 <= zoom <= 4
            or len(center) != 2 or not np.isfinite(center).all()
            or not all(.05 <= x <= .95 for x in center)):
        raise ValueError('Invalid viewport request')
    z = np.linalg.solve(k_rgb, [*center, 1.])
    z /= np.linalg.norm(z)
    # Camera right remains horizontal with respect to the original camera up.
    x = np.cross([0., 1., 0.], z)
    if np.linalg.norm(x) < 1e-8:
        raise ValueError('Degenerate camera up direction')
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rotation = np.stack((x, y, z))
    aspect = ow / oh
    corners = np.array([[-aspect, -1], [aspect, -1], [aspect, 1], [-aspect, 1.]])
    lower, upper = np.array([.5 / iw, .5 / ih]), 1 - np.array([.5 / iw, .5 / ih])

    def fits(half_height):
        rays = np.column_stack((corners * half_height, np.ones(4)))
        src = rays @ (k_rgb @ rotation.T).T
        if np.any(src[:, 2] <= 1e-10):
            return False
        uv = src[:, :2] / src[:, 2:]
        return bool(np.all((uv >= lower) & (uv <= upper)))

    lo, hi = 0., 1.
    if not fits(lo):
        raise ValueError('Selected center is outside the RGB camera')
    while fits(hi) and hi < 1024:
        hi *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    half_height = lo / zoom
    if half_height < 1e-8:
        raise ValueError('No usable centered camera viewport')
    k = np.array([[1 / (2 * aspect * half_height), 0, .5],
                  [0, 1 / (2 * half_height), .5], [0, 0, 1.]])
    if flip:
        rotation = np.diag([-1., 1., 1.]) @ rotation
    # A horizontal flip is explicitly a camera-coordinate reflection, not a pose.
    transform = k @ rotation @ np.linalg.inv(k_rgb)
    return dict(intrinsics=k, camera_transform=rotation,
                rgb_to_view_uv=transform, output_shape=(oh, ow), flipped=bool(flip))


def warp_rgb(rgb, view):
    """Antialiased RGB-only warp. No depth or validity array enters this code."""
    import cv2

    ih, iw = rgb.shape[:2]
    oh, ow = view['output_shape']
    inverse = np.linalg.inv(view['rgb_to_view_uv'])
    # Conservative prefilter: use the minimum local reduction at 9 locations,
    # so that a perspective crop is not blurred merely because its source is big.
    reductions = []
    for v in (.05, .5, .95):
        for u in (.05, .5, .95):
            q = inverse @ [u, v, 1.]
            jac = (inverse[:2, :2] * q[2] - q[:2, None] * inverse[2, :2]) / q[2]**2
            jac = np.diag([iw, ih]) @ jac @ np.diag([1 / ow, 1 / oh])
            reductions.append(np.linalg.svd(jac, compute_uv=False).min())
    factor = min(1., 1.25 / max(min(reductions), 1e-8))
    if factor < .8:
        iw, ih = max(4, round(iw * factor)), max(4, round(ih * factor))
        rgb = np.array(Image.fromarray(rgb).resize((iw, ih), Image.Resampling.LANCZOS))
    u, v = np.meshgrid((np.arange(ow) + .5) / ow, (np.arange(oh) + .5) / oh)
    q = np.stack((u, v, np.ones_like(u)), -1) @ inverse.T
    uv = q[..., :2] / q[..., 2:]
    # A resized RGB retains the same normalized camera; borders only extend its
    # filtering footprint. The viewport itself was checked in original pixels.
    return cv2.remap(rgb, (uv[..., 0] * iw - .5).astype(np.float32),
                     (uv[..., 1] * ih - .5).astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def select_observations(depth, valid, k_depth, view):
    """Project exact native points and select the closest UV to each loss cell.

    This is angular decimation, not a z-buffer or label refinement. Pure camera
    rotation is one-to-one in ray space. Ties use native row-major index, never
    prediction error or measured depth. Loss resolution cannot exceed either
    original depth dimension or the model RGB size. Invalid pixels are ignored
    before any arithmetic, including NaN/Inf in their storage.
    """
    depth, valid = np.asarray(depth), np.asarray(valid)
    if depth.ndim != 2 or valid.shape != depth.shape or valid.dtype != np.bool_:
        raise ValueError('Expected native depth and explicit Boolean support')
    if not np.isfinite(depth[valid]).all() or not (depth[valid] > 0).all():
        raise ValueError('Nonfinite/nonpositive supported native depth')
    dh, dw = depth.shape
    oh, ow = view['output_shape']
    kd = camera_matrix(k_depth)
    # A crop/zoom can contain far fewer native rays than the full source grid.
    # Cap the loss resolution by its actual calibrated angular sampling density,
    # not only by the original array dimensions. Otherwise a zoom creates a
    # checkerboard of empty cells and destroys adjacent edge pairs. This only
    # DECIMATES observations; it never fills those empty cells or duplicates GT.
    inverse = kd @ np.linalg.inv(view['camera_transform']) @ np.linalg.inv(view['intrinsics'])
    densities = []
    for v in (0., .5, 1.):
        for u in (0., .5, 1.):
            q = inverse @ [u, v, 1.]
            jac = (inverse[:2, :2] * q[2] - q[:2, None] * inverse[2, :2]) / q[2]**2
            jac = np.diag([dw, dh]) @ jac @ np.diag([1 / ow, 1 / oh])
            densities.append(np.linalg.svd(jac, compute_uv=False).min())
    scale = min(1., dh / oh, dw / ow, min(densities))
    lh, lw = max(2, int(np.floor(oh * scale))), max(2, int(np.floor(ow * scale)))
    rays = pinhole_rays(kd, dh, dw)
    ids = np.flatnonzero(valid)
    points = (rays.reshape(-1, 3)[ids] * depth.ravel()[ids, None]) @ view['camera_transform'].T
    projected = points @ view['intrinsics'].T
    positive = projected[:, 2] > 1e-10
    uv = np.zeros((len(points), 2), np.float64)
    uv[positive] = projected[positive, :2] / projected[positive, 2:]
    # No border padding or extrapolation when sampling the predicted point map.
    lower, upper = np.array([.5 / ow, .5 / oh]), 1 - np.array([.5 / ow, .5 / oh])
    support = positive & np.all((uv >= lower) & (uv <= upper), axis=1)
    points, uv, ids = points[support], uv[support], ids[support]
    ij = np.floor(uv * [lw, lh]).astype(np.int64)
    cells = ij[:, 1] * lw + ij[:, 0]
    distance = ((uv * [lw, lh] - ij - .5)**2).sum(1)
    order = np.lexsort((ids, distance, cells))
    chosen = order[np.r_[True, np.diff(cells[order]) != 0]] if len(order) else order
    cells, ids = cells[chosen], ids[chosen]
    target = dict(points=np.ones((lh * lw, 3), np.float32),
                  query_uv=np.full((lh * lw, 2), .5, np.float32),
                  valid=np.zeros(lh * lw, bool), native_index=np.full(lh * lw, -1, np.int64))
    target['points'][cells] = points[chosen]
    target['query_uv'][cells] = uv[chosen]
    target['valid'][cells] = True
    target['native_index'][cells] = ids
    target = {key: value.reshape(lh, lw, *value.shape[1:]) for key, value in target.items()}
    target.update(intrinsics=view['intrinsics'].copy(), output_shape=(oh, ow),
                  native_shape=(dh, dw), native_valid_count=int(valid.sum()),
                  visible_native_count=int(support.sum()))
    # Never silently turn an ordinary crop into the legacy centered contract.
    for key in ('camera_contract', 'camera_pixel_convention'):
        if key in view:
            target[key] = view[key]
    if len(ids) != len(np.unique(ids)) or len(ids) > valid.sum():
        raise AssertionError('Geometry observations were duplicated')
    return target


def source_supervision(metadata):
    """No estimated teacher scale is allowed into the metric-scale objective."""
    kind = metadata.get('label_kind')
    if kind not in LABEL_KINDS:
        raise ValueError('Unknown label provenance')
    if metadata.get('source_needs_review', False):
        raise ValueError('Unresolved source-quality flag')
    if kind == 'teacher_pseudo':
        if (metadata.get('metric_calibrated') is not False
                or metadata.get('camera_kind') != 'teacher_estimated'
                or not metadata.get('teacher_revision')):
            raise ValueError('Teacher labels cannot claim calibrated metric supervision')
        return 'relative_pseudo'
    if (metadata.get('metric_calibrated') is not True
            or metadata.get('camera_kind') not in ('calibrated_sensor', 'synthetic_camera')):
        raise ValueError('Native metric supervision requires a calibrated source camera')
    return 'metric_native'


def load_candidate(pack_path, entry, output_shape=(400, 640), seed=0, augment=False):
    """Read/check one candidate for diagnostics. This is NOT training admission.

    Caller must separately enforce the final manifest-bound acceptance gate.
    Do not wire this function to a trainer just because a cache exists.
    """
    depth, valid, kd, metadata = read_geometry_member(pack_path, entry)
    row = entry['image']
    if (row.get('split') != 'train' or row['sha256'] != metadata['image_sha256']
            or row['dataset'] != metadata['dataset']):
        raise ValueError('Cache/image provenance mismatch')
    mode = source_supervision(metadata)
    rgb_bytes = Path(row['image']).read_bytes()
    if hashlib.sha256(rgb_bytes).hexdigest() != row['sha256']:
        raise ValueError('Frozen RGB bytes changed')
    with Image.open(io.BytesIO(rgb_bytes)) as file:
        if file.getexif().get(274, 1) != 1:
            raise ValueError('Undocumented RGB EXIF transformation')
        image = file.convert('RGB')
    rotation = row.get('image_rotation_cw', 0)
    if type(rotation) is not int or rotation not in (0, 90, 180, 270):
        raise ValueError('Undocumented RGB rotation')
    if rotation != metadata.get('image_rotation_cw', 0):
        raise ValueError('RGB rotation not bound to geometry provenance')
    if rotation:
        image = image.transpose({90: Image.Transpose.ROTATE_270,
                                 180: Image.Transpose.ROTATE_180,
                                 270: Image.Transpose.ROTATE_90}[rotation])
    rgb = np.array(image)
    rgb_shape = tuple(rgb.shape[:2])
    # Some original NYU rows have no dimension fields. The bytes are still
    # checksum-bound; derive dimensions from those exact bytes, never from a
    # guessed resize or a native depth array. If either frozen field exists,
    # both must match. Explicit multigrid metadata is checked independently below.
    if (('height' in row or 'width' in row)
            and rgb_shape != (row.get('height'), row.get('width'))):
        raise ValueError('Frozen RGB dimensions differ after orientation')
    if 'rgb_intrinsics_normalized' in metadata:
        kr = camera_matrix(metadata['rgb_intrinsics_normalized'])
        if (list(rgb_shape) != metadata.get('rgb_shape')
                or not np.allclose(metadata.get('depth_to_rgb_uv_transform'),
                                   kr @ np.linalg.inv(kd), rtol=0, atol=1e-8)):
            raise ValueError('Unverified native-to-RGB camera mapping')
    else:
        if depth.shape != rgb_shape:
            raise ValueError('Unequal RGB/depth grids need explicit calibrated mapping')
        kr = camera_matrix(kd)
    rng = np.random.default_rng(seed)
    center = tuple(rng.uniform(.4, .6, 2)) if augment else (.5, .5)
    zoom = float(rng.uniform(1., 1.2)) if augment else 1.
    flip = bool(rng.random() < .5) if augment else False
    view = centered_view(kr, rgb_shape, output_shape, center, zoom, flip)
    target = select_observations(depth, valid, kd, view)
    target.update(supervision=mode, label_kind=metadata['label_kind'],
                  image_sha256=row['sha256'], dataset=row['dataset'], cache_key=entry['key'])
    return warp_rgb(rgb, view), target, view
