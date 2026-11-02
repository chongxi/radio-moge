import hashlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from cache_arkit_geometry import convert  # noqa: E402
from radio_moge.arkit_depth import rotate_camera  # noqa: E402


def fixture(tmp_path, rotation):
    rgb = np.zeros((24, 32, 3), np.uint8)
    rgb[..., 0] = np.arange(32) * 7
    rgb[..., 1] = np.arange(24)[:, None] * 10
    Image.fromarray(rgb).save(tmp_path / 'rgb.png')
    Image.fromarray(rgb[::2, ::2]).save(tmp_path / 'lowres_wide.png')
    Image.fromarray(np.full((12, 16), 2000, np.uint16)).save(tmp_path / 'lowres_depth.png')
    Image.fromarray(np.full((12, 16), 2, np.uint8)).save(tmp_path / 'confidence.png')
    low, large = [16, 12, 15, 15, 7.5, 5.5], [32, 24, 30, 30, 15, 11]
    for kind, camera in (('lowres_wide_intrinsics', low), ('vga_wide_intrinsics', large)):
        (tmp_path / (kind + '.pincam')).write_text(' '.join(map(str, camera)))
    class Assets:
        def get(self, base, kind, stem):
            path = tmp_path / (kind + ('.pincam' if kind.endswith('intrinsics') else '.png'))
            return path, dict(source='fixture:' + kind, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    row = dict(video_id='123', timestamp=1.234, image_rotation_cw=rotation,
        source_member='vga_wide/123_1.234.png', source_url='https://example.invalid/vga_wide.zip',
        image=str(tmp_path / 'rgb.png'), sha256=hashlib.sha256((tmp_path / 'rgb.png').read_bytes()).hexdigest(),
        native_camera_width_height_fx_fy_cx_cy=large,
        camera_width_height_fx_fy_cx_cy=rotate_camera(large, rotation))
    return row, Assets()


def test_cache_preserves_sparse_native_grid_and_exact_cameras(tmp_path):
    row, assets = fixture(tmp_path, 90)
    depth, valid, kd, kr, proofs, stats = convert(row, assets)
    assert depth.shape == (16, 12) and valid.sum() == 140
    assert stats['paired_rgb_mae'] < 1e-14 and stats['image_registration_pass']
    np.testing.assert_array_equal(depth[valid], 2)
    assert len(proofs) == 5 and not np.array_equal(kd, kr)


def test_mispaired_rgb_and_wrong_timestamp_are_not_admitted(tmp_path):
    row, assets = fixture(tmp_path, 0)
    with Image.open(tmp_path / 'lowres_wide.png') as file:
        bad = np.asarray(file)[:, ::-1].copy()
    Image.fromarray(bad).save(tmp_path / 'lowres_wide.png')
    assert not convert(row, assets)[1].any()
    row['timestamp'] = 1.235
    with pytest.raises(ValueError, match='timestamp'):
        convert(row, assets)
