"""Versioned camera-Z clipping for new, unaccepted mesh-derived candidates.

Numerical visibility agreement is not physical accuracy of the source mesh.
Historical original-camera queries remain in mesh_depth.py unchanged.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

from radio_moge.geometry_sources import pinhole_rays


@dataclass(frozen=True)
class MeshClipContract:
    near_m: float = .05
    far_m: float = 20.
    boundary_guard_m: float = .0001
    schema_version: int = 2
    depth_convention: str = 'camera_z_meters'
    visibility: str = 'first_hit_inside_clip_interval_double_sided'
    boundary_policy: str = 'reject_guard_band_no_surface_skipping'
    maximum_gpu_cpu_error_m: float = .002

    def __post_init__(self):
        values = (self.near_m, self.far_m, self.boundary_guard_m,
                  self.maximum_gpu_cpu_error_m)
        if (any(isinstance(v, bool) or not isinstance(v, (int, float))
                or not math.isfinite(v) for v in values)
                or not 0 < 2*self.boundary_guard_m < self.near_m
                or self.far_m-self.near_m <= 4*self.boundary_guard_m
                or self.schema_version != 2 or type(self.schema_version) is not int
                or self.depth_convention != 'camera_z_meters'
                or self.visibility != 'first_hit_inside_clip_interval_double_sided'
                or self.boundary_policy != 'reject_guard_band_no_surface_skipping'
                or self.maximum_gpu_cpu_error_m != .002):
            raise ValueError('Explicit finite camera-Z clipping v2 contract required')

    def as_dict(self):
        return asdict(self)

    def interior(self, z):
        z = np.asarray(z, dtype=np.float64)
        return (np.isfinite(z) & (z > self.near_m+self.boundary_guard_m)
                & (z < self.far_m-self.boundary_guard_m))


def checked_camera(intrinsics, c2w, shape):
    k, pose = np.asarray(intrinsics, np.float64), np.asarray(c2w, np.float64)
    if (k.shape != (3, 3) or not np.isfinite(k).all()
            or not np.array_equal(k[2], [0, 0, 1]) or k[0, 0] <= 0 or k[1, 1] <= 0
            or abs(np.linalg.det(k)) < 1e-12 or len(shape) != 2 or min(shape) < 1
            or pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.array_equal(pose[3], [0, 0, 0, 1])
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), rtol=0, atol=1e-5)
            or abs(np.linalg.det(pose[:3, :3])-1) > 1e-5):
        raise ValueError('Finite calibrated pinhole camera and rigid pose required')
    return k, pose


def clipped_cpu_depth(intersector, intrinsics, c2w, selected, contract):
    """Query before and at near; never enumerate a capped hit list.

    The 2*guard look-behind puts an exact near-plane triangle in front of the
    query origin. Any returned hit in/before the near guard rejects the ray;
    it is never skipped to manufacture a farther usable surface. The guard is
    a declared numerical exclusion band, not a physical depth tolerance. An
    interior hit must also agree with a second closest query at the near plane.
    """
    if not isinstance(contract, MeshClipContract):
        raise ValueError('Explicit MeshClipContract required')
    selected = np.asarray(selected)
    if selected.ndim != 2 or selected.dtype != np.bool_:
        raise ValueError('Boolean selected ray grid required')
    k, pose = checked_camera(intrinsics, c2w, selected.shape)
    indices = np.flatnonzero(selected)
    depth = np.zeros(selected.size, np.float32)
    # 0=not queried, 1=missing, 2=near guard, 3=far guard/outside, 4=valid hit.
    reason = np.zeros(selected.size, np.uint8)
    reason[indices] = 1
    if len(indices):
        camera_rays = pinhole_rays(k, *selected.shape).reshape(-1, 3)[indices]
        if not np.allclose(camera_rays[:, 2], 1, rtol=0, atol=1e-12):
            raise ValueError('Camera ray Z must equal one, not unit Euclidean length')
        rays = camera_rays @ pose[:3, :3].T
        origins = pose[:3, 3]+(contract.near_m-2*contract.boundary_guard_m)*rays
        points, hits, _ = intersector.intersects_location(origins, rays, multiple_hits=False)
        points, hits = np.asarray(points, np.float64), np.asarray(hits)
        if (hits.ndim != 1 or hits.dtype.kind not in 'iu'
                or points.shape != (len(hits), 3) or not np.isfinite(points).all()
                or (len(hits) and (hits.min() < 0 or hits.max() >= len(indices)))
                or len(np.unique(hits)) != len(hits)):
            raise ValueError('Malformed or duplicate clipped nearest-hit result')
        # Use inverse, not an assumed transpose, for faithful rounded poses.
        camera_points = (points-pose[:3, 3]) @ np.linalg.inv(pose[:3, :3]).T
        z = camera_points[:, 2]
        expected = camera_rays[hits]*z[:, None]
        if (not np.allclose(camera_points, expected, rtol=0, atol=1e-7)
                or np.any(z < contract.near_m-3*contract.boundary_guard_m)):
            raise ValueError('CPU intersections do not lie on forward clipped camera rays')
        positions = indices[hits]
        # The saved float32 value must satisfy the same collar as the exact hit.
        # A float64 interior value can round onto/into an excluded boundary.
        with np.errstate(over='ignore'):
            saved_z = z.astype(np.float32).astype(np.float64)
        near = ((z <= contract.near_m+contract.boundary_guard_m)
                | (saved_z <= contract.near_m+contract.boundary_guard_m))
        far = ((z >= contract.far_m-contract.boundary_guard_m)
               | (saved_z >= contract.far_m-contract.boundary_guard_m))
        good = ~(near | far)
        reason[positions[near]] = 2
        reason[positions[far]] = 3
        reason[positions[good]] = 4
        depth[positions[good]] = z[good]
        if good.any():
            slots = hits[good]
            check_origins = pose[:3, 3]+contract.near_m*rays[slots]
            check_points, check_hits, _ = intersector.intersects_location(
                check_origins, rays[slots], multiple_hits=False)
            check_points, check_hits = np.asarray(check_points, np.float64), np.asarray(check_hits)
            if (check_hits.ndim != 1 or check_hits.dtype.kind not in 'iu'
                    or check_points.shape != (len(check_hits), 3)
                    or not np.isfinite(check_points).all()
                    or (len(check_hits) and (check_hits.min() < 0 or check_hits.max() >= len(slots)))
                    or len(np.unique(check_hits)) != len(check_hits)):
                raise ValueError('Malformed near-plane cross-check result')
            agree = np.zeros(len(slots), bool)
            # Float64 plane intersections should retain the same actual point;
            # 1 micrometer is a numerical consistency check, not a source bound.
            agree[check_hits] = np.max(np.abs(check_points-points[good][check_hits]), axis=1) <= 1e-6
            reason[positions[good][~agree]] = 7
            depth[positions[good][~agree]] = 0
    return depth.reshape(selected.shape), reason.reshape(selected.shape)


def verify_clipped_cpu_intersections(intersector, intrinsics, c2w, depth, selected, contract):
    """Return CPU Z, final mask, exact per-pixel reason codes and accounting."""
    depth, selected = np.asarray(depth), np.asarray(selected)
    if (depth.ndim != 2 or selected.shape != depth.shape or selected.dtype != np.bool_
            or not np.isfinite(depth[selected]).all() or np.any(depth[selected] <= 0)):
        raise ValueError('Finite positive candidate depth and boolean support required')
    cpu, reason = clipped_cpu_depth(intersector, intrinsics, c2w, selected, contract)
    hit = reason == 4
    error = np.abs(cpu.astype(np.float64)-depth)
    # 5=numerical parity failure; 6=GPU clip-boundary ambiguity.
    reason[hit & (error > contract.maximum_gpu_cpu_error_m)] = 5
    reason[hit & ~contract.interior(depth)] = 6
    support = reason == 4
    counts = {str(i): int(np.count_nonzero(reason == i)) for i in range(8)}
    stats = dict(clip_contract=contract.as_dict(), cpu_rays_checked=int(selected.sum()),
        reason_counts=counts, reason_codes={'0': 'not_selected', '1': 'missing_clipped_hit',
            '2': 'near_boundary_guard', '3': 'far_boundary_or_outside', '4': 'verified',
            '5': 'gpu_cpu_parity', '6': 'gpu_boundary_guard', '7': 'query_origin_instability'},
        verified_pixels=int(support.sum()),
        maximum_accepted_gpu_cpu_error_m=float(error[support].max()) if support.any() else None,
        final_depth='CPU clipped first-hit original-camera Z',
        independent_physical_depth_accuracy_checked=False)
    return np.where(support, cpu, 0), support, reason, stats
