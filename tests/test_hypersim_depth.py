import numpy as np

from radio_moge.geometry_sources import pinhole_rays
from radio_moge.hypersim_depth import convert_verified_range, half_ulp


def example():
    k = np.array([[1, 0, .5], [0, 1, .5], [0, 0, 1.]])
    grid = np.array([[2, 0, -1], [0, -2, 1], [0, 0, 1.]])
    matrix = np.diag([1., -1., -1.]) @ np.linalg.inv(k) @ np.linalg.inv(grid)
    points = pinhole_rays(k, 12, 16) * 3
    radial = np.linalg.norm(points, axis=-1).astype(np.float16)
    world = (points * np.array([1, -1, -1])).astype(np.float16)
    return radial, world, matrix


def test_fp16_world_quantization_and_uniform_pose_scale():
    radial, world, matrix = example()
    depth, valid, k, stats = convert_verified_range(radial, world, np.eye(3) * .25,
                                                    np.zeros(3), 1., matrix)
    assert valid.all() and stats['inherited_pose_uniform_scale'] == .25
    np.testing.assert_allclose(depth, 3, atol=.002)
    assert k[0, 0] == 1
    assert half_ulp(np.array([1], np.float16))[0] == 2**-11


def test_wrong_units_and_mispaired_world_positions_are_not_admitted():
    radial, world, matrix = example()
    _, valid, _, _ = convert_verified_range(radial, world, np.eye(3), np.zeros(3), .1, matrix)
    assert not valid.any()
    world[..., 0] += 1
    _, valid, _, _ = convert_verified_range(radial, world, np.eye(3), np.zeros(3), 1., matrix)
    assert not valid.any()


def test_source_holes_stay_invalid():
    radial, world, matrix = example()
    radial[0, 0], world[1, 1] = np.nan, np.nan
    depth, valid, _, _ = convert_verified_range(radial, world, np.eye(3), np.zeros(3), 1., matrix)
    assert not valid[0, 0] and not valid[1, 1]
    assert depth[0, 0] == depth[1, 1] == 0
    assert valid[2:, 2:].all()
