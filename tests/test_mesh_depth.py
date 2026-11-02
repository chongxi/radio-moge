import numpy as np
import pytest
import torch

from radio_moge.geometry_sources import pinhole_rays
from radio_moge.mesh_depth import CudaMeshDepth


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
