from copy import deepcopy

import numpy as np
import pytest

from visualize_arkit_outcomes import aligned_display, require_array_parity, sample_rgb


@pytest.mark.parametrize('turn', [0, 90, 180, 270])
def test_calibrated_sampling_survives_all_turns_without_half_pixel_resize_error(turn):
    low, camera = [16, 12, 13, 14, 7, 5], [40, 30, 32.5, 35, 17.5, 12.5]
    y, x = np.mgrid[:30, :40]
    rgb = np.stack((x * 5, y * 7, x + y * 2), -1).astype(np.uint8)
    sampled, _, _, _ = sample_rgb(rgb, low, camera)
    # Integer pixel (4, 6) maps to (10, 15), unlike a normalized resize.
    np.testing.assert_allclose(sampled[4, 6], rgb[10, 15] / 255, atol=1e-12)
    low_rgb = np.round(sampled * 255).astype(np.uint8)
    errors = np.abs(sampled - low_rgb / 255).mean(-1)
    quality = dict(paired_rgb_mae=float(errors.mean()), paired_rgb_error_p90=float(np.quantile(errors, .9)))
    original = rgb.copy()
    upright, native, actual, _, _, _ = aligned_display(rgb, low_rgb, low, camera, turn, quality)
    np.testing.assert_allclose(actual, np.rot90(sampled, -turn // 90), atol=1e-12)
    np.testing.assert_array_equal(upright, np.rot90(rgb, -turn // 90))
    np.testing.assert_array_equal(native, np.rot90(low_rgb, -turn // 90))
    np.testing.assert_array_equal(rgb, original)
    wrong_camera = list(camera)
    wrong_camera[4] += 1
    with pytest.raises(ValueError, match='producer statistics'):
        aligned_display(rgb, low_rgb, low, wrong_camera, turn, quality)


def test_reproduction_cannot_change_depth_mask_camera_or_claim_new_source_provenance():
    depth, valid, k = np.ones((3, 4), np.float32), np.ones((3, 4), bool), np.eye(3)
    quality, proofs = {'stat': .2}, [{'source': 'fixture'}]
    meta = dict(rgb_intrinsics_normalized=k.tolist(), depth_to_rgb_uv_transform=k.tolist(),
                source_quality=quality, source_proofs=proofs)
    cached = (depth, valid, k, meta)
    reproduced = (depth.copy(), valid.copy(), k.copy(), k.copy(), proofs, quality)
    require_array_parity(cached, reproduced)
    for index in range(6):
        changed = list(deepcopy(reproduced))
        if index < 4:
            changed[index].flat[0] = 0
        else:
            changed[index] = [] if index == 4 else {}
        with pytest.raises(ValueError, match='reproduce cached'):
            require_array_parity(cached, changed)
