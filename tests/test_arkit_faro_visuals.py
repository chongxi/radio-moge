import hashlib
import json
import zlib

import numpy as np
import pytest

from visualize_arkit_faro_probe import RetainedBundle, display_fields, error_colors
from radio_moge.arkit_depth import rotate_camera
from radio_moge.geometry_sources import normalized_opencv_intrinsics


def test_retained_source_requires_exact_receipt_bytes_crc_and_never_fetches(tmp_path):
    base = 'https://docs-assets.developer.apple.com/test/Training/123'
    url, member, data = base + '/lowres_depth.zip', 'lowres_depth/123_1.png', b'fixture'
    proof = dict(source=url, member=member, sha256=hashlib.sha256(data).hexdigest(),
                 bytes=len(data), crc32=zlib.crc32(data))
    path = tmp_path / hashlib.sha256(url.encode()).hexdigest()[:16] / member
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    receipt = path.with_name(path.name + '.receipt.json')
    receipt.write_text(json.dumps(proof))
    bundle = RetainedBundle(tmp_path, [proof])
    assert bundle.get(base, 'lowres_depth', '123_1') == (path, proof)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        bundle.get(base, 'lowres_depth', '123_1')
    path.write_bytes(data)
    changed = dict(proof, crc32=0)
    receipt.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='changed'):
        RetainedBundle(tmp_path, [changed]).get(base, 'lowres_depth', '123_1')
    path.unlink()
    with pytest.raises(FileNotFoundError):
        bundle.get(base, 'lowres_depth', '123_1')


@pytest.mark.parametrize('turn', [0, 90, 180, 270])
def test_display_maps_reproduce_unfitted_bias_with_holes_for_both_hypotheses(turn):
    camera = [640, 480, 500, 500, 320, 240]
    native = rotate_camera([256, 192, 200, 200, 128, 96], turn)
    kd = normalized_opencv_intrinsics(*native)
    depth = np.full((int(native[1]), int(native[0])), 2.1, np.float32)
    valid = np.ones_like(depth, bool)
    faro = np.full((1440, 1920), 2000, np.uint16)
    faro[:, :960] = 0
    originals = [x.copy() for x in (depth, valid, faro)]
    for half_center in (False, True):
        _, _, difference, mask, stats = display_fields(depth, valid, kd, faro, camera, turn, half_center)
        assert 10000 < mask.sum() < depth.size // 2
        assert abs(float(np.median(difference[mask])) - .1) < 1e-6
        assert not stats['registration_calibration_accepted']
        assert (error_colors(difference, mask)[~mask] == 0).all()
    for actual, expected in zip((depth, valid, faro), originals):
        np.testing.assert_array_equal(actual, expected)
