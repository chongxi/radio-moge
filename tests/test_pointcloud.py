import numpy as np
import pytest
import torch

from radio_moge.pointcloud import unproject, verify_geometry


def test_projection_pixel_centers_and_official_point_map():
    # Test off-center, non-square camera calibration, not just a centered square image.
    depth = np.array([[1, 2, 3], [2, 3, 4]], dtype=np.float32)
    k = np.array([[.8, 0, .4], [0, 1.2, .6], [0, 0, 1]], dtype=np.float32)
    utils = pytest.importorskip('utils3d_moge')
    points = utils.pt.depth_map_to_point_map(torch.tensor(depth)[None],
                                            intrinsics=torch.tensor(k)[None])
    pred = dict(depth=torch.tensor(depth)[None], points=points,
                intrinsics=torch.tensor(k)[None],
                mask=torch.tensor([[[True, False, True], [True, True, True]]]))
    camera, mask, error = verify_geometry(depth, pred)
    assert error < 1e-6
    assert not mask[0, 1]
    np.testing.assert_allclose(camera, k)
    np.testing.assert_allclose(unproject(depth, k)[..., 2], depth)
    with pytest.raises(AssertionError, match='existing evaluation'):
        verify_geometry(depth * 1.1, pred)
    with pytest.raises(AssertionError, match='official MoGe'):
        verify_geometry(depth, {**pred, 'points': points * 1.1})


def test_invalid_intrinsics_are_rejected():
    with pytest.raises(ValueError, match='intrinsics'):
        unproject(np.ones((2, 3)), [[0, 0, .5], [0, 1, .5], [0, 0, 1]])
