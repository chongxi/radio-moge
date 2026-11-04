import numpy as np
import pytest

from probe_scannetpp_triangle_scan_support import (
    angle_sensitivity,
    compare_triangle,
    directions,
    nearest_angle_pixel,
    range_neighborhood,
)


def test_angle_sensitivity_distinguishes_robust_interior_from_thin_nominal_hit():
    pose = np.eye(4)
    measure = range_neighborhood(
        np.full((3, 3), 3000, np.uint16), np.full((3, 3), 255, np.uint8), 1, 1
    )
    # Azimuth=0, polar=1571 mrad points nearly along +X. Plane lies at X=2.
    broad = np.array([[2, -1, -1], [2, 1, -1], [2, 0, 1]], float)
    result = angle_sensitivity(broad, pose, 0, 1571, measure)
    assert result["all_nine_rays_hit_exact_triangle"]
    assert result["minimum_measured_patch_minus_maximum_tested_triangle_m"] > 0.99
    center = directions(0, 1571) * 2
    thin = center + np.array([[0, -0.0001, -0.0001], [0, 0.0001, -0.0001], [0, 0, 0.0001]])
    assert compare_triangle(thin, pose, directions(0, 1571), measure)["intersects_exact_triangle"]
    result = angle_sensitivity(thin, pose, 0, 1571, measure)
    assert not result["all_nine_rays_hit_exact_triangle"]
    assert result["minimum_measured_patch_minus_maximum_tested_triangle_m"] is None


def test_actual_angles_not_assumed_equirectangular_coordinates_and_seam():
    az = np.tile(np.linspace(6200, 80, 33, dtype=np.uint16), (17, 1))
    polar = np.tile(np.linspace(500, 2600, 17, dtype=np.uint16)[:, None], (1, 33))
    for y, x in ((0, 0), (8, 16), (16, 32), (7, 11)):
        target = directions(az[y, x], polar[y, x])
        a, b, d, error = nearest_angle_pixel(az, polar, target, stride=4)
        assert (a, b) == (y, x) and error < 1e-14
        np.testing.assert_allclose(d, target, rtol=0, atol=0)


def test_nearest_pixel_selected_by_angles_not_favorable_depth_or_mask():
    az = np.array([[0, 1000], [2000, 3000]], dtype=np.uint16)
    polar = np.full_like(az, 1571)
    y, x, _, _ = nearest_angle_pixel(az, polar, directions(1000, 1571), stride=1)
    assert (y, x) == (0, 1)


def test_neighbor_range_mask_saturation_and_boundaries():
    depth = np.full((5, 5), 2000, dtype=np.uint16)
    mask = np.full((5, 5), 255, dtype=np.uint8)
    assert range_neighborhood(depth, mask, 2, 2)["stable_supported"]
    depth[1, 1] = 1000
    assert not range_neighborhood(depth, mask, 2, 2)["stable_supported"]
    depth[:] = 2000
    mask[1, 1] = 0
    assert not range_neighborhood(depth, mask, 2, 2)["stable_supported"]
    mask[:] = 255
    depth[:] = 65535
    assert not range_neighborhood(depth, mask, 2, 2)["stable_supported"]
    assert not range_neighborhood(depth, mask, 0, 0)["stable_supported"]


@pytest.mark.parametrize(
    "depth,diagnostic",
    [
        (3000, "measurement_beyond_triangle"),
        (1000, "measurement_before_triangle"),
        (2000, "measurement_near_triangle"),
    ],
)
def test_exact_triangle_measurement_relation_is_radial_under_camera_transform(depth, diagnostic):
    pose = np.eye(4)
    pose[:3, :3] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    pose[:3, 3] = [10, 20, 30]
    verts = np.array([[-1, -1, 2], [1, -1, 2], [0, 1, 2]]) @ pose[:3, :3].T + pose[:3, 3]
    measurement = range_neighborhood(
        np.full((3, 3), depth, np.uint16), np.full((3, 3), 255, np.uint8), 1, 1
    )
    result = compare_triangle(verts, pose, [0, 0, 1], measurement)
    assert result["intersects_exact_triangle"] and result["triangle_range_m"] == pytest.approx(2)
    assert result["diagnostic"] == diagnostic


def test_missing_exact_triangle_or_mask_does_not_become_a_contradiction():
    verts = np.array([[-1, -1, 2], [1, -1, 2], [0, 1, 2]], float)
    measurement = dict(stable_supported=False)
    assert (
        compare_triangle(verts, np.eye(4), [0, 0, 1], measurement)["diagnostic"]
        == "unsupported_or_boundary_measurement"
    )
    assert (
        compare_triangle(verts, np.eye(4), [2, 0, 1], measurement)["diagnostic"]
        == "stored_ray_misses_exact_triangle"
    )
