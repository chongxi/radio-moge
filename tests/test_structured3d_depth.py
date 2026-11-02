import numpy as np
import pytest

from radio_moge.structured3d_depth import convert_structured3d, parse_errata, is_erratum


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
