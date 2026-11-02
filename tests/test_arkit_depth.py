import numpy as np

from radio_moge.arkit_depth import native_depth, registration_uv, rotate_camera


def test_registration_is_not_blind_normalized_grid_resize():
    low = [256, 192, 213.594, 213.594, 127.384, 96.5393]
    rgb = [640, 480, 533.985, 533.985, 318.46, 241.34825]
    uv, _, _ = registration_uv(low, rgb)
    # Integer pixel (0,0) maps to RGB integer pixel (0,0), not pixel (.75,.75).
    np.testing.assert_allclose(uv[0, 0], [.5 / 640, .5 / 480], atol=1e-12)
    np.testing.assert_allclose(uv[10, 20], [(50 + .5) / 640, (25 + .5) / 480], atol=1e-12)


def test_quarter_turns_preserve_grid_registration_and_sample_count():
    low, rgb = [16, 12, 15, 15, 7.5, 5.5], [40, 30, 37.5, 37.5, 18.75, 13.75]
    original_uv, _, _ = registration_uv(low, rgb)
    raw, conf = np.full((12, 16), 2000, np.uint16), np.full((12, 16), 2, np.uint8)
    raw[5, 6], conf[7, 8] = 0, 1
    first = native_depth(raw, conf, low, 0)
    assert first[0].shape == raw.shape and not first[1][5, 6] and not first[1][7, 8]
    for rotation in (90, 180, 270):
        depth, valid, kd = native_depth(raw, conf, low, rotation)
        uv, _, _ = registration_uv(kd, rotate_camera(rgb, rotation))
        expected = original_uv.copy()
        for _ in range(rotation // 90):
            expected = np.stack((1 - expected[..., 1], expected[..., 0]), -1)
            expected = np.rot90(expected, -1)
        np.testing.assert_allclose(uv, expected, atol=1e-12)
        assert valid.sum() == first[1].sum() and depth.size == raw.size
        np.testing.assert_array_equal(depth, np.rot90(first[0], -rotation // 90))
