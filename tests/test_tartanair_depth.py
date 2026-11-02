import numpy as np
import pytest

from radio_moge.tartanair_depth import convert_tartanair


def pair():
    rng = np.random.default_rng(43)
    rgb = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    # Z=4 -> disparity20: right[x]=left[x+20].
    right = np.zeros_like(rgb)
    right[:, :-20] = rgb[:, 20:]
    return np.full((480, 640), 4, np.float32), rgb, right


def test_metric_z_and_eye_sign_with_textured_plane():
    z, left, right = pair()
    depth, valid, k, q = convert_tartanair(z, z, left, right)
    assert valid.mean() > .95
    assert np.all(depth[valid] == 4) and np.all(depth[~valid] == 0)
    assert q['rgb_stereo_mae_on_geometric'] == 0
    np.testing.assert_allclose(k, [[.5, 0, 320.5 / 640], [0, 320 / 480, 240.5 / 480], [0, 0, 1]])
    _, bad_eye, _, bad = convert_tartanair(z, z, right, left)
    assert bad_eye.mean() < .03 and bad['source_needs_review']


def test_native_holes_finite_sky_occlusions_and_no_input_mutation():
    z, left, right = pair()
    z[20:40, 100:120] = 0
    z[50:70, 100:120] = np.nan
    z[80:100, 100:120] = 10000
    z[110:130, 100:120] = np.inf
    original = z.copy()
    depth, valid, _, quality = convert_tartanair(z, np.full_like(z, 4), left, right)
    assert not valid[19:41, 99:121].any()
    assert not valid[50:70, 100:120].any()
    assert not valid[80:100, 100:120].any()
    assert not valid[110:130, 100:120].any()
    assert np.isfinite(depth).all() and (depth[~valid] == 0).all()
    assert quality['task_range_is_not_publisher_invalid_mask']
    np.testing.assert_array_equal(z, original)
    _, wrong_metric, _, _ = convert_tartanair(np.full_like(z, 4), np.full_like(z, 8), left, right)
    assert not wrong_metric.any()


def test_no_blind_native_resizing_or_integer_depth():
    z, left, right = pair()
    with pytest.raises(ValueError):
        convert_tartanair(z[::2], z[::2], left[::2], right[::2])
    with pytest.raises(ValueError):
        convert_tartanair(z.astype(np.uint16), z, left, right)
