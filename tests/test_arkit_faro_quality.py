import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from probe_arkit_faro_quality import compare  # noqa: E402
from radio_moge.arkit_depth import rotate_camera  # noqa: E402
from radio_moge.geometry_sources import normalized_opencv_intrinsics  # noqa: E402


def test_known_sensor_bias_is_reported_without_fitting_or_depth_interpolation():
    camera = [640, 480, 500, 500, 320, 240]
    kd = normalized_opencv_intrinsics(256, 192, 200, 200, 128, 96)
    depth = np.full((192, 256), 2.1, np.float32)
    faro = np.full((1440, 1920), 2000, np.uint16)
    result = compare(depth, np.ones(depth.shape, bool), kd, faro, camera, 0, False)
    assert result['compared_pixels'] > 40000
    assert abs(result['median_signed_difference_m'] - .1) < 1e-6
    assert result['fraction_within_5cm'] == 0
    assert not result['registration_calibration_accepted']


def test_holes_and_unstable_faro_boundaries_never_become_reference_measurements():
    camera = [640, 480, 500, 500, 320, 240]
    kd = normalized_opencv_intrinsics(*rotate_camera([256, 192, 200, 200, 128, 96], 90))
    depth = np.full((256, 192), 2., np.float32)
    faro = np.full((1440, 1920), 2000, np.uint16)
    faro[:, :960] = 0
    result = compare(depth, np.ones(depth.shape, bool), kd, faro, camera, 90, True)
    assert 10000 < result['compared_pixels'] < depth.size // 2
    assert result['fraction_within_5cm'] == 1
