import numpy as np

from radio_moge.geometry_sources import (normalized_opencv_intrinsics, pinhole_rays,
    range_to_z, scannetpp_nerfstudio_to_colmap_c2w, hypersim_pinhole_frame)


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


def test_tilt_shift_camera_preserves_rays_without_resampling_rgb():
    angle = .25
    rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]])
    k = np.array([[.8, .01, .4], [0, 1.2, .6], [0, 0, 1.]])
    grid = np.array([[2, 0, -1], [0, -2, 1], [0, 0, 1.]])
    matrix = np.diag([1., -1., -1.]) @ rotation.T @ np.linalg.inv(k) @ np.linalg.inv(grid)
    actual_k, actual_rotation = hypersim_pinhole_frame(matrix * 2)
    np.testing.assert_allclose(actual_k, k, atol=1e-12)
    np.testing.assert_allclose(actual_rotation, rotation, atol=1e-12)
    rays = pinhole_rays(actual_k, 48, 64)
    depth, valid = range_to_z(np.ones((48, 64)) * 3, rays, np.ones((48, 64), bool))
    assert valid.all()
    np.testing.assert_allclose(np.linalg.norm(rays * depth[..., None], axis=-1), 3, atol=2e-7)
