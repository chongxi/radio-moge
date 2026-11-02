import numpy as np
import pytest
import torch

from radio_moge.geometry_sources import pinhole_rays
from radio_moge.mesh_depth import CudaMeshDepth, conservative_visibility, verify_cpu_intersections


@pytest.fixture
def rasterizer():
    pytest.importorskip('nvdiffrast.torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA rasterizer test requires a GPU')
    # Non-frontoparallel plane z = 3 + .2*x - .1*y tests perspective interpolation.
    vertices = np.array([[-10, -10, 2], [10, -10, 6], [10, 10, 4], [-10, 10, 0]], float)
    return CudaMeshDepth(vertices, [[0, 1, 2], [0, 2, 3]])


def test_cuda_depth_pixel_centers_and_top_down_orientation(rasterizer):
    k = np.array([[.9, .02, .4], [0, 1.1, .6], [0, 0, 1]])
    pose = np.eye(4)
    depth, valid, _ = rasterizer.render(k, pose, 48, 64)
    rays = pinhole_rays(k, 48, 64)
    truth = 3 / (1 - .2 * rays[..., 0] + .1 * rays[..., 1])
    assert valid.all()
    # One triangle crosses the near plane. Float32 clipping/interpolation can
    # lose a few tens of micrometers; the required max error is 0.1 mm, with a
    # tighter 99th-percentile check so this does not hide a convention error.
    np.testing.assert_allclose(depth, truth, atol=1e-4)
    assert np.quantile(np.abs(depth - truth), .99) < 2e-5


def test_cuda_holes_and_occlusions_do_not_create_measured_support():
    pytest.importorskip('nvdiffrast.torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA rasterizer test requires a GPU')
    vertices = np.array([[-1, -1, 4], [1, -1, 4], [0, 1, 4],
                         [-.5, -.5, 2], [.5, -.5, 2], [0, .5, 2]])
    renderer = CudaMeshDepth(vertices, [[0, 1, 2], [3, 4, 5]])
    depth, valid, _ = renderer.render(np.array([[1, 0, .5], [0, 1, .5], [0, 0, 1]]),
                                       np.eye(4), 32, 48)
    assert 0 < valid.mean() < .5
    np.testing.assert_allclose(depth[valid], 2, atol=1e-6)
    assert np.all(depth[~valid] == 0)


def test_conservative_support_only_removes_holes_edges_and_unstable_rays():
    depth = np.full((32, 48), 3., np.float32)
    valid = np.ones_like(depth, bool)
    valid[10, 10] = False
    depth[10, 10] = 0
    depth[:, 30:] = 6
    shifted = [(depth.copy(), valid.copy()) for _ in range(4)]
    shifted[0][0][20, 20] = 8
    support = conservative_visibility(depth, valid, shifted)
    assert support.sum() > 500 and not np.any(support & ~valid)
    assert not support[8:13, 8:13].any()
    assert not support[:, 28:32].any()
    assert not support[20, 20]


def test_cuda_conservative_visibility_preserves_planar_depth(rasterizer):
    k = np.array([[.9, .02, .4], [0, 1.1, .6], [0, 0, 1]])
    original, valid, _ = rasterizer.render(k, np.eye(4), 48, 64)
    depth, support, _, quality = rasterizer.render_conservative(k, np.eye(4), 48, 64)
    assert quality['initial_valid_pixels'] == int(valid.sum())
    assert support.mean() > .7
    np.testing.assert_array_equal(original[support], depth[support])
    assert np.all(depth[~support] == 0)


def test_all_candidate_rays_checked_and_mismatch_never_repaired():
    import trimesh
    from trimesh.ray.ray_pyembree import RayMeshIntersector

    mesh = trimesh.Trimesh(vertices=[[-10, -10, 3], [10, -10, 3], [10, 10, 3], [-10, 10, 3]],
                          faces=[[0, 1, 2], [0, 2, 3]], process=False)
    k = np.array([[1, 0, .5], [0, 1, .5], [0, 0, 1]])
    original = np.full((16, 24), 3., np.float32)
    original[8, 9] = 3.01
    candidate = np.ones_like(original, bool)
    candidate[2, 3] = False
    depth, support, stats = verify_cpu_intersections(RayMeshIntersector(mesh), k, np.eye(4),
                                                     original, candidate)
    assert stats['cpu_rays_checked'] == candidate.sum()
    assert stats['rejected_inconsistent_hits'] == 1
    assert not support[8, 9] and not support[2, 3]
    assert np.all(depth[support] == 3) and np.all(depth[~support] == 0)
    empty_depth, empty_mask, empty = verify_cpu_intersections(RayMeshIntersector(mesh), k,
        np.eye(4), original, np.zeros_like(candidate))
    assert empty['cpu_rays_checked'] == 0 and not empty_mask.any() and not empty_depth.any()
