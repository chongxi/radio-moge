import numpy as np
import pytest

from radio_moge.geometry_sources import pinhole_rays
from radio_moge.mesh_clip import (MeshClipContract, clipped_cpu_depth,
                                  verify_clipped_cpu_intersections)
from radio_moge.mesh_depth import CudaMeshDepth


class PlaneIntersector:
    """Exact intersections of world-space planes; independent analytical oracle."""
    def __init__(self, depths, pose=None):
        pose = np.eye(4) if pose is None else pose
        self.point = [pose[:3, 3]+pose[:3, :3] @ np.array([0., 0., z]) for z in depths]
        self.normal = pose[:3, :3] @ np.array([0., 0., 1.])
        self.calls = 0

    def intersects_location(self, origins, rays, multiple_hits):
        assert multiple_hits is False
        self.calls += 1
        best = np.full(len(rays), np.inf)
        for point in self.point:
            t = ((point-origins) @ self.normal)/(rays @ self.normal)
            best = np.minimum(best, np.where(t >= 0, t, np.inf))
        hits = np.flatnonzero(np.isfinite(best))
        return origins[hits]+best[hits, None]*rays[hits], hits, np.zeros(len(hits), int)


K = np.array([[.7, .03, .4], [0, .9, .6], [0, 0, 1.]])


@pytest.mark.parametrize('depths,expected,reason', [
    ([.02, 2], 2, 4), ([.02], 0, 1), ([25], 0, 3), ([.08, 2], .08, 4),
    ([], 0, 1), ([.05, 2], 0, 2), ([.04995, 2], 0, 2), ([.05005, 2], 0, 2),
    ([.0505, 2], .0505, 4), ([19.99995], 0, 3), ([20.00005], 0, 3),
    ([19.9995], 19.9995, 4), ([.04985, 2], 0, 2),
])
def test_clip_before_visibility_and_boundary_guard(depths, expected, reason):
    selected = np.ones((4, 6), bool)
    d, codes = clipped_cpu_depth(PlaneIntersector(depths), K, np.eye(4), selected,
                                 MeshClipContract())
    assert (codes == reason).all()
    np.testing.assert_allclose(d, expected, rtol=0, atol=1e-6)


def test_offaxis_range_is_not_z_and_depth_is_not_measured_from_shifted_origin():
    k = np.array([[.05, 0, .1], [0, .05, .1], [0, 0, 1.]])
    rays = pinhole_rays(k, 4, 6)
    assert (np.linalg.norm(rays, axis=-1)*.02 > .05).any()
    angle = .4
    pose = np.eye(4)
    pose[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)]]
    pose[:3, 3] = [13, -7, 5]
    d, reason = clipped_cpu_depth(PlaneIntersector([.02, 2], pose), k, pose,
                                  np.ones((4, 6), bool), MeshClipContract())
    np.testing.assert_allclose(d, 2, atol=1e-6, rtol=0)
    assert (reason == 4).all()


def test_mismatch_and_unstable_query_are_rejected_not_filled():
    selected = np.ones((4, 6), bool)
    selected[0, 0] = False
    candidate = np.full((4, 6), 2., np.float32)
    candidate[2, 3] = 2.01
    depth, mask, reasons, stats = verify_clipped_cpu_intersections(
        PlaneIntersector([.02, 2]), K, np.eye(4), candidate, selected, MeshClipContract())
    assert reasons[2, 3] == 5 and reasons[0, 0] == 0
    assert stats['cpu_rays_checked'] == 23 and stats['verified_pixels'] == 22
    assert (depth[mask] == 2).all() and not depth[~mask].any()

    class Unstable(PlaneIntersector):
        def intersects_location(self, *args, **kwargs):
            points, hits, ids = super().intersects_location(*args, **kwargs)
            if self.calls == 2:
                points += .01
            return points, hits, ids
    depth, reason = clipped_cpu_depth(Unstable([2]), K, np.eye(4), selected,
                                      MeshClipContract())
    assert (reason[selected] == 7).all() and not depth.any()


def test_empty_support_does_not_query_and_malformed_hits_fail():
    engine = PlaneIntersector([2])
    d, reason = clipped_cpu_depth(engine, K, np.eye(4), np.zeros((4, 6), bool), MeshClipContract())
    assert not d.any() and not reason.any() and engine.calls == 0
    class Duplicate:
        def intersects_location(self, *args, **kwargs):
            return np.ones((2, 3)), np.array([0, 0]), np.array([0, 0])
    with pytest.raises(ValueError, match='duplicate'):
        clipped_cpu_depth(Duplicate(), K, np.eye(4), np.ones((4, 6), bool), MeshClipContract())


@pytest.mark.parametrize('change', [dict(near_m=0), dict(far_m=.05), dict(near_m=float('nan')),
    dict(far_m=float('inf')), dict(boundary_guard_m=0), dict(boundary_guard_m=.03),
    dict(schema_version=True), dict(visibility='original_camera_then_mask'),
    dict(maximum_gpu_cpu_error_m=.02), dict(near_m=True)])
def test_invalid_contracts_fail(change):
    with pytest.raises(ValueError):
        MeshClipContract(**change)


def test_contract_cannot_be_combined_with_separate_gpu_bounds_before_gpu_import():
    with pytest.raises(ValueError, match='one explicit'):
        CudaMeshDepth([], [], near=.01, clip_contract=MeshClipContract())


@pytest.mark.parametrize('front', [.02, .05, .04995, .05005, .0505, .08])
def test_actual_embree_front_and_rear_planes(front):
    trimesh = pytest.importorskip('trimesh')
    ray = pytest.importorskip('trimesh.ray.ray_pyembree')
    xy = np.array([[-10., -10], [10, -10], [10, 10], [-10, 10]])
    vertices = np.concatenate([np.column_stack([xy, np.full(4, z)]) for z in [front, 2]])
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    d, reason = clipped_cpu_depth(ray.RayMeshIntersector(mesh), K, np.eye(4),
                                  np.ones((4, 6), bool), MeshClipContract())
    if abs(front-.05) < .0001:
        assert (reason == 2).all() and not d.any()
    else:
        assert (reason == 4).all()
        np.testing.assert_allclose(d, 2 if front < .05 else front, atol=1e-6, rtol=0)
