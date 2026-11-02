import numpy as np
import pytest

from radio_moge.structured3d_depth import (convert_structured3d, parse_errata, is_erratum,
                                         perspective_camera)


def example():
    raw = np.full((64, 96), 3000, np.uint16)
    normal = np.empty((*raw.shape, 3), np.uint8)
    normal[:] = [128, 128, 0]
    semantic = np.ones(raw.shape, np.uint8)
    camera = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, .7, .5, 1.])
    layout = dict(junctions=[dict(coordinate=[x, y]) for x, y in
                            [(0, 0), (95, 0), (95, 63), (0, 63)]],
        planes=[dict(type='wall', normal=[0, 0, -1], offset=3000, visible_mask=[[0, 1, 2, 3]])])
    return raw, normal, semantic, camera, layout, np.full(raw.shape, 255, np.uint8)


def test_native_z_units_and_invalid_regions_stay_invalid():
    args = list(example())
    args[0][10, 10], args[0][15, 15] = 0, 65535
    args[2][20, 20], args[5][25, 25] = 19, 0
    depth, valid, k, stats = convert_structured3d(*args)
    assert stats['layout_verified_pixels'] > 100 and not stats['source_needs_review']
    assert valid.mean() > .7 and k[0, 2] == 48.5 / 96
    for point in ((10, 10), (15, 15), (20, 20), (25, 25), (0, 0)):
        assert not valid[point] and depth[point] == 0
    np.testing.assert_array_equal(depth[valid], 3)


def test_wrong_units_and_normals_fail_support():
    args = list(example())
    args[0][:] = 300  # Tenfold unit error, despite local normal consistency.
    _, valid, _, stats = convert_structured3d(*args)
    assert not valid.any() and stats['source_needs_review']
    args = list(example())
    args[1][:] = [255, 128, 128]
    assert not convert_structured3d(*args)[1].any()


def test_upstream_errata_scene_and_room_scope():
    entries = parse_errata('# invalid scene\nscene_01155\n# layout\nscene_00010_room_846619\n')
    assert is_erratum(dict(scene='01155', room='0'), entries)
    assert is_erratum(dict(scene='00010', room='846619'), entries)
    assert not is_erratum(dict(scene='00010', room='123'), entries)
    with pytest.raises(ValueError):
        parse_errata('silently changed format')


def test_publisher_directions_are_not_a_precomputed_rotation_matrix():
    args = list(example())
    original = convert_structured3d(*args)
    # Exact failed scene_01181/room_2163/view_0 directions. The directions are
    # unit length, but dot=-0.000112038; the upstream parser builds axes itself.
    args[3][3:9] = [-.81234837, .58293471, .016650991,
                    .013599993, -.009799995, .99985949]
    result = convert_structured3d(*args)
    for old, new in zip(original[:3], result[:3]):
        np.testing.assert_array_equal(old, new)
    assert abs(result[3]['camera_directions']['normalized_look_up_dot']) > 1e-4
    assert not result[3]['camera_directions']['world_rotation_applied_to_depth']
    args[3][3:6] *= 7
    args[3][6:9] *= 3  # Upstream normalizes directions; length is not scene scale.
    for old, new in zip(original[:3], convert_structured3d(*args)[:3]):
        np.testing.assert_array_equal(old, new)


@pytest.mark.parametrize('directions', [[0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 2]])
def test_degenerate_directions_still_fail(directions):
    args = list(example())
    args[3][3:9] = directions
    with pytest.raises(ValueError, match='directions'):
        convert_structured3d(*args)


def test_float_intrinsics_use_half_angle_and_integer_pixel_origin():
    camera = example()[3]
    k, _ = perspective_camera(camera, 720, 1280)
    assert k[0, 0] == pytest.approx(.5 / np.tan(.7))
    assert k[1, 1] == pytest.approx(.5 / np.tan(.5))
    assert k[0, 2] == pytest.approx(640.5 / 1280)
    assert k[1, 2] == pytest.approx(360.5 / 720)
    assert k[0, 0] * 1280 != int(k[0, 0] * 1280)
