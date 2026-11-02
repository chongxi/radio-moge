"""Conservative native Structured3D perspective support, never depth completion.

The release's uint16 depth is camera Z in mm (verified against layout planes).
Normal images use camera x-right/y-up/z-forward; flip Y for OpenCV. Layouts
describe room structure, not furniture, so only matching structural semantics
and normals may be compared to a layout plane. All other surfaces still need
local depth/normal consistency. These checks are source consistency, not an
independent measurement of a rendered scene's physical accuracy.
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, maximum_filter, minimum_filter

from .geometry_sources import normalized_opencv_intrinsics, pinhole_rays


def convert_structured3d(raw, normal_rgb, semantic, camera, layout, alpha):
    raw, normal_rgb, semantic = np.asarray(raw), np.asarray(normal_rgb), np.asarray(semantic)
    camera, alpha = np.asarray(camera, np.float64), np.asarray(alpha)
    if (raw.ndim != 2 or raw.dtype != np.uint16 or normal_rgb.dtype != np.uint8
            or normal_rgb.shape != (*raw.shape, 3) or semantic.shape != raw.shape
            or semantic.dtype != np.uint8 or alpha.shape != raw.shape
            or camera.shape != (12,) or not np.isfinite(camera).all()
            or not (0 < camera[9] < np.pi / 2 and 0 < camera[10] < np.pi / 2)):
        raise ValueError('Unexpected Structured3D native encoding or camera')
    look, up = camera[3:6], camera[6:9]
    if (abs(np.linalg.norm(look) - 1) > 1e-4 or abs(np.linalg.norm(up) - 1) > 1e-4
            or abs(look @ up) > 1e-4):
        raise ValueError('Structured3D look/up axes are not orthonormal')
    height, width = raw.shape
    k = normalized_opencv_intrinsics(width, height, width / 2 / np.tan(camera[9]),
                                     height / 2 / np.tan(camera[10]), width / 2, height / 2)
    depth = raw.astype(np.float32) / 1000
    native = (raw > 0) & (raw < 65535) & (alpha == 255)
    points = pinhole_rays(k, height, width) * depth[..., None]
    # Exclude holes, discontinuities and the image border before finite differences.
    stable = binary_erosion(native, structure=np.ones((5, 5)), border_value=0)
    stable &= maximum_filter(depth, size=5) - minimum_filter(depth, size=5) <= .05 + .02 * depth
    dx = np.roll(points, -2, axis=1) - np.roll(points, 2, axis=1)
    dy = np.roll(points, -2, axis=0) - np.roll(points, 2, axis=0)
    geometric_normal = np.cross(dx, dy)
    length = np.linalg.norm(geometric_normal, axis=-1)
    geometric_normal /= np.maximum(length[..., None], 1e-12)
    normal = (normal_rgb.astype(np.float64) / 128 - 1) * [1, -1, 1]
    normal_length = np.linalg.norm(normal, axis=-1)
    normal /= np.maximum(normal_length[..., None], 1e-12)
    cosine = np.abs(np.sum(geometric_normal * normal, axis=-1))
    supported = stable & (length > 1e-12) & (normal_length > .9) & (normal_length < 1.1)
    # Semantic 9/19 (window/mirror) are conservatively excluded, not filled.
    supported &= (semantic >= 1) & (semantic <= 40) & ~np.isin(semantic, [9, 19])
    supported &= cosine >= np.cos(np.deg2rad(15))
    checked, agrees = np.zeros(raw.shape, bool), np.zeros(raw.shape, bool)
    plane_stats = []
    for plane in layout['planes']:
        class_id = {'wall': 1, 'floor': 2, 'ceiling': 22}.get(plane['type'])
        if class_id is None:
            raise ValueError('Unknown layout plane type')
        pnormal = np.asarray(plane['normal'], np.float64) * [1, -1, 1]
        if abs(np.linalg.norm(pnormal) - 1) > 1e-4 or not np.isfinite(plane['offset']):
            raise ValueError('Layout plane is not normalized/finite')
        polygon_image = Image.new('1', (width, height))
        draw = ImageDraw.Draw(polygon_image)
        for polygon in plane['visible_mask']:
            if len(polygon) >= 3:
                draw.polygon([tuple(layout['junctions'][i]['coordinate']) for i in polygon], fill=1)
        candidates = binary_erosion(np.asarray(polygon_image), iterations=3) & native
        candidates &= (semantic == class_id) & (np.abs(normal @ pnormal) >= np.cos(np.deg2rad(5)))
        error = np.abs(points @ pnormal + float(plane['offset']) / 1000)
        matches = candidates & (error <= .005)
        checked |= candidates
        agrees |= matches
        plane_stats.append(dict(type=plane['type'], tested_pixels=int(candidates.sum()),
                                matching_pixels=int(matches.sum())))
    conflict = checked & ~agrees
    valid = supported & ~conflict
    stats = dict(native_supported_pixels=int(native.sum()), native_valid_fraction=float(native.mean()),
        stable_fraction=float(stable.mean()), valid_fraction=float(valid.mean()),
        normal_consistent_pixels=int(supported.sum()), layout_checked_pixels=int(checked.sum()),
        layout_verified_pixels=int((agrees & supported).sum()),
        layout_conflicting_pixels=int(conflict.sum()), planes=plane_stats,
        source_needs_review=bool((agrees & supported).sum() < 100),
        normal_cosine_quantiles=(np.quantile(cosine[stable], [.01, .5, .99]).tolist()
                                 if stable.any() else None))
    depth[~valid] = 0
    return depth, valid, k, stats


def parse_errata(text):
    """Conservatively quarantine entire listed scenes/rooms, including layout errata."""
    import re
    entries = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if not re.fullmatch(r'scene_\d{5}(?:_room_\d+)?', line):
            raise ValueError('Unknown upstream errata entry')
        entries.add(line)
    if not entries:
        raise ValueError('Empty upstream errata file')
    return entries


def is_erratum(row, entries):
    return (f'scene_{row["scene"]}' in entries
            or f'scene_{row["scene"]}_room_{row["room"]}' in entries)
