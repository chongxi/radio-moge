import numpy as np
import pytest

from radio_moge.nyu_projection import reproduce_projected_depth, undistort


def test_literal_undistort_pixel_origin_and_out_of_frame_fill():
    value = np.arange(48).reshape(6, 8).astype(float)
    result = undistort(value, [2, 2], [3, 2], np.zeros(5))
    np.testing.assert_array_equal(result[:-1, :-1], value[:-1, :-1])
    assert (result[-1] == 255).all() and (result[:, -1] == 255).all()
    mask = np.zeros_like(value, bool)
    mask[2, 3] = True
    depth = undistort(value, [2, 2], [3, 2], np.zeros(5), mask)
    assert depth[2, 3] == 0 and not depth[-1].any()


def test_projection_invalid_sensor_codes_do_not_produce_fake_depth():
    assert not reproduce_projected_depth(np.full((480, 640), 2047)).any()
    with pytest.raises(ValueError):
        reproduce_projected_depth(np.full((240, 320), 1000))
