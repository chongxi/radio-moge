"""Perspective RGB/GT pairs from ScanNet++'s explicit scanner ray maps.

No equirectangular assumption, mesh reconstruction, or depth hole filling. RGB,
range and angles are sampled from the SAME original pixel. Ray-to-output-pixel
error is bounded explicitly. Work is serial and refinement is chunked.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def scanner_rays(azimuth, polar):
    sine = np.sin(polar)
    return np.stack((sine * np.cos(azimuth), sine * np.sin(azimuth), np.cos(polar)), -1)


def perspective_camera(height, width, yaw, pitch=0., fov_degrees=90.):
    if min(height, width) < 16 or not 30 <= fov_degrees <= 110 or abs(pitch) > 60:
        raise ValueError('Invalid bounded perspective camera')
    yaw, pitch = np.deg2rad([yaw, pitch])
    forward = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    right = np.array([np.sin(yaw), -np.cos(yaw), 0.])
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward), axis=1).astype(np.float32)
    focal = width / (2 * np.tan(np.deg2rad(fov_degrees) / 2))
    intrinsics = np.array([[focal / width, 0, .5], [0, focal / height, .5], [0, 0, 1]],
                          dtype=np.float32)
    u, v = np.meshgrid((np.arange(width) + .5) / width, (np.arange(height) + .5) / height)
    rays = np.stack(((u - .5) / intrinsics[0, 0], (v - .5) / intrinsics[1, 1], np.ones_like(u)), -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return intrinsics, rotation, (rays @ rotation.T).astype(np.float32)


class PanoramaProjector:
    def __init__(self, azimuth, polar, stride=8):
        if azimuth.shape != polar.shape or azimuth.ndim != 2 or not 1 <= stride <= 16:
            raise ValueError('Expected paired angular maps and bounded search stride')
        if (not np.isfinite(azimuth).all() or not np.isfinite(polar).all()
                or azimuth.min() < 0 or azimuth.max() > 2 * np.pi + .001
                or polar.min() < 0 or polar.max() > np.pi + .001):
            raise ValueError('Angular maps must be finite radians in the official polar convention')
        self.azimuth, self.polar, self.stride = azimuth, polar, stride
        h, w = azimuth.shape
        ys, xs = np.meshgrid(np.arange(0, h, stride), np.arange(0, w, stride), indexing='ij')
        self.coarse_y, self.coarse_x = ys.ravel(), xs.ravel()
        coarse = scanner_rays(azimuth[::stride, ::stride], polar[::stride, ::stride])
        self.tree = cKDTree(coarse.reshape(-1, 3))

    def indices(self, wanted):
        _, nearest = self.tree.query(wanted.reshape(-1, 3), workers=1)
        cy, cx = self.coarse_y[nearest], self.coarse_x[nearest]
        oy, ox = np.meshgrid(np.arange(-self.stride, self.stride + 1),
                             np.arange(-self.stride, self.stride + 1), indexing='ij')
        result_y, result_x = np.empty_like(cy), np.empty_like(cx)
        h, w = self.azimuth.shape
        target = wanted.reshape(-1, 3)
        for start in range(0, len(cx), 1024):
            end = min(start + 1024, len(cx))
            ys = np.clip(cy[start:end, None] + oy.ravel(), 0, h - 1)
            xs = np.clip(cx[start:end, None] + ox.ravel(), 0, w - 1)
            rays = scanner_rays(self.azimuth[ys, xs], self.polar[ys, xs])
            error = ((rays - target[start:end, None]) ** 2).sum(-1)
            chosen = error.argmin(-1)
            result_y[start:end], result_x[start:end] = ys[np.arange(end - start), chosen], xs[np.arange(end - start), chosen]
        return result_y.reshape(wanted.shape[:2]), result_x.reshape(wanted.shape[:2])

    def project(self, rgb, distance, mask, height=400, width=640, yaw=0., pitch=0.,
                fov_degrees=90., max_pixel_error=.5, return_source_indices=False):
        if (rgb.shape != (*self.azimuth.shape, 3) or distance.shape != self.azimuth.shape
                or mask.shape != self.azimuth.shape or not 0 < max_pixel_error <= 1):
            raise ValueError('All original panorama modalities must have identical dimensions')
        intrinsics, rotation, wanted = perspective_camera(height, width, yaw, pitch, fov_degrees)
        ys, xs = self.indices(wanted)
        rays = scanner_rays(self.azimuth[ys, xs], self.polar[ys, xs]) @ rotation
        u = (rays[..., 0] / np.maximum(rays[..., 2], 1e-6) * intrinsics[0, 0] + .5) * width - .5
        v = (rays[..., 1] / np.maximum(rays[..., 2], 1e-6) * intrinsics[1, 1] + .5) * height - .5
        target_u, target_v = np.meshgrid(np.arange(width), np.arange(height))
        pixel_error = np.hypot(u - target_u, v - target_v)
        angular_support = (rays[..., 2] > 0) & (pixel_error <= max_pixel_error)
        anonymous_ok = ndimage.binary_erosion(mask == 255, structure=np.ones((3, 3)),
                                             iterations=1, border_value=0)[ys, xs]
        # Reject a depth stencil if its native source pixels cross a hole or edge.
        native_valid = np.isfinite(distance) & (distance > 0) & (distance < 65.535)
        clean = np.where(native_valid, distance, 0)
        lo = ndimage.minimum_filter(clean, size=3, mode='constant', cval=0)
        hi = ndimage.maximum_filter(clean, size=3, mode='constant', cval=0)
        depth_support = native_valid[ys, xs] & (lo[ys, xs] > 0)
        depth_support &= (hi[ys, xs] - lo[ys, xs]) <= .03 * lo[ys, xs]
        valid = angular_support & anonymous_ok & depth_support
        # Released pano depth is radial range, not camera z.
        depth = distance[ys, xs] * rays[..., 2]
        depth = np.where(valid, depth, 0).astype(np.float32)
        output_rgb = rgb[ys, xs].copy()
        output_rgb[~(angular_support & anonymous_ok)] = 127
        result = output_rgb, depth, valid, intrinsics, {
            'yaw_degrees': yaw, 'pitch_degrees': pitch, 'fov_degrees': fov_degrees,
            'source_shape': list(distance.shape), 'output_shape': [height, width],
            'valid_fraction': float(valid.mean()),
            'rgb_valid_fraction': float((angular_support & anonymous_ok).mean()),
            'max_allowed_pixel_error': max_pixel_error,
            'valid_pixel_error_p50_p95_max': (np.quantile(pixel_error[valid], [.5, .95, 1]).tolist()
                                             if valid.any() else []),
            'same_original_pixel_for_rgb_range_angles': True,
            'range_to_z_conversion': True, 'depth_holes_filled': False,
            'original_camera_to_scanner_rotation': rotation.tolist(),
        }
        if return_source_indices:
            return (*result, np.stack((ys, xs), -1).astype(np.int32))
        return result
