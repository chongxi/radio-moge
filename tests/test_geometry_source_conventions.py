import numpy as np

from radio_moge.geometry_sources import (normalized_opencv_intrinsics, pinhole_rays,
    range_to_z, scannetpp_nerfstudio_to_colmap_c2w)


def test_opencv_half_pixel_normalization_and_resize():
    k = normalized_opencv_intrinsics(640, 480, 500, 501, 319.2, 240.1)
    rays = pinhole_rays(k, 480, 640)
    np.testing.assert_allclose(rays[0, 0], [-319.2 / 500, -240.1 / 501, 1])
    small = pinhole_rays(k, 240, 320)
    np.testing.assert_allclose(small[0, 0], [(0.5 - 319.2) / 500, (0.5 - 240.1) / 501, 1])


def test_scannetpp_undoes_world_axes_not_only_camera_axes():
    original = np.eye(4)
    original[:3, 3] = [2, 3, 4]
    nerfstudio = original.copy()
    nerfstudio[:3, 1:3] *= -1
    nerfstudio = nerfstudio[[1, 0, 2, 3], :]
    nerfstudio[2] *= -1
    np.testing.assert_array_equal(scannetpp_nerfstudio_to_colmap_c2w(nerfstudio), original)


def test_radial_range_conversion_never_fills_invalid_depth():
    rays = np.array([[[0., 0, 1], [1, 0, 1]], [[0, 1, 1], [0, 0, -1]]])
    radial = np.array([[3., 3 * np.sqrt(2)], [np.nan, 3]])
    depth, valid = range_to_z(radial, rays, np.ones((2, 2), bool))
    np.testing.assert_allclose(depth, [[3, 3], [0, 0]])
    np.testing.assert_array_equal(valid, [[True, True], [False, False]])
