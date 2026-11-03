import numpy as np
import pytest

from probe_arkit_exact_faro_camera import exact_camera, rgb_correspondence
from radio_moge.arkit_depth import rotate_camera
from radio_moge.geometry_sources import normalized_opencv_intrinsics


@pytest.mark.parametrize('turn', [0, 90, 180, 270])
def test_released_camera_projects_rgb_at_native_observations_under_quarter_turns(turn):
    low = np.stack([np.arange(48, dtype=np.uint8).reshape(6, 8) * 4] * 3, -1)
    wide = low.repeat(2, axis=0).repeat(2, axis=1)
    a = [8, 6, 10, 10, 3.5, 2.5]
    b = exact_camera([16, 12, 20, 20, 7.5, 5.5], (12, 16))
    kd = normalized_opencv_intrinsics(*rotate_camera(a, turn))
    low = np.rot90(low, -turn // 90).copy()
    wide = np.rot90(wide, -turn // 90).copy()
    valid = np.ones(low.shape[:2], bool)
    valid[0] = False
    error, mask, stats = rgb_correspondence(kd, valid.shape, rotate_camera(b, turn), wide, low, valid)
    assert np.array_equal(mask, valid) and stats['compared_native_pixels'] == int(valid.sum())
    assert error[mask].max() < 1e-14
    changed = rotate_camera(b, turn)
    changed[4] += 2
    _, _, shifted = rgb_correspondence(kd, valid.shape, changed, wide, low, valid)
    assert shifted['mean_absolute_rgb_difference'] > .005


@pytest.mark.parametrize('camera', [
    [640, 480, 1000, 1000, 960, 720],
    [1920, 1440, -1000, 1000, 960, 720],
    [1920, 1440, 1000, 1000, float('nan'), 720],
    [1920, 1440, 1000, 1000, 1920, 720],
])
def test_rejects_wrong_grid_or_invalid_camera_instead_of_guessing(camera):
    with pytest.raises(ValueError, match='released FARO grid'):
        exact_camera(camera, (1440, 1920))
