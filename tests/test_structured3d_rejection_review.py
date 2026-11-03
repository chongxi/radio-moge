from copy import deepcopy
import hashlib
import io
import json
import zipfile

import numpy as np
from PIL import Image
import pytest

from review_structured3d_rejections import read_members, reproduce
from radio_moge.geometry_cache import sha_file
from radio_moge.structured3d_depth import convert_structured3d


def fixture(root):
    raw = np.full((32, 48), 3000, np.uint16)
    rgb = np.full((32, 48, 4), 200, np.uint8)
    rgb[..., 3] = 255
    normal = np.broadcast_to(np.array([128, 128, 0], np.uint8), (*raw.shape, 3))
    semantic = np.ones(raw.shape, np.uint8)
    camera = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, .7, .5, 1.])
    layout = dict(junctions=[], planes=[])
    _, _, _, stats = convert_structured3d(raw, normal, semantic, camera, layout, rgb[..., 3])
    native = {}
    for name, array in [('rgb_rawlight.png', rgb), ('depth.png', raw),
                        ('normal.png', normal), ('semantic.png', semantic)]:
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format='PNG')
        native[name] = buffer.getvalue()
    native['camera_pose.txt'] = ' '.join(str(x) for x in camera).encode()
    native['layout.json'] = json.dumps(layout).encode()
    path = root / 'rgb.png'
    path.write_bytes(native['rgb_rawlight.png'])
    prefix = 'Structured3D/scene_00001/2D_rendering/1/perspective/empty/0/'
    archive_path = root / 'source.zip'
    with zipfile.ZipFile(archive_path, 'w') as archive:
        for name, data in native.items():
            archive.writestr(prefix + name, data)
    archive_sha = sha_file(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        proofs = [dict(source=str(archive_path), archive_sha256=archive_sha,
            member=prefix + name, sha256=hashlib.sha256(data).hexdigest(),
            crc32=archive.getinfo(prefix + name).CRC) for name, data in native.items()]
    record = dict(image=dict(image=str(path), sha256=sha_file(path), dataset='structured3d',
        split='train', height=32, width=48, source_archive=archive_path.name,
        source_member=prefix + 'rgb_rawlight.png'), source_proofs=proofs, stats=stats,
        reason='insufficient_verified_support')
    return record, native, archive_path, archive_sha


def test_layout_quarantine_reproduces_without_filling_or_changing_source(tmp_path):
    record, native, path, checksum = fixture(tmp_path)
    original = deepcopy(record)
    with zipfile.ZipFile(path) as archive:
        assert read_members(record, archive, path, checksum) == native
    _, raw, _, valid, stats = reproduce(record, native)
    assert not valid.any() and stats['source_needs_review']
    assert stats['candidate_fraction_before_image_gate'] > .5
    assert (raw == 3000).all() and record == original and sha_file(path) == checksum


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'sha', 'crc', 'archive', 'member'])
def test_reproduction_rejects_missing_ambiguous_or_changed_member_proofs(tmp_path, change):
    record, _, path, checksum = fixture(tmp_path)
    if change == 'missing':
        record['source_proofs'].pop()
    elif change == 'duplicate':
        record['source_proofs'].append(record['source_proofs'][0])
    elif change == 'sha':
        record['source_proofs'][0]['sha256'] = '0' * 64
    elif change == 'crc':
        record['source_proofs'][0]['crc32'] += 1
    elif change == 'archive':
        record['source_proofs'][0]['archive_sha256'] = '0' * 64
    else:
        record['image']['source_member'] = '../rgb_rawlight.png'
    with zipfile.ZipFile(path) as archive, pytest.raises(ValueError, match='proof|changed|identity'):
        read_members(record, archive, path, checksum)


def test_changed_frozen_rgb_and_reported_rejection_cannot_pass(tmp_path):
    record, native, _, _ = fixture(tmp_path)
    bad = deepcopy(record)
    bad['stats']['layout_verified_pixels'] = 100
    with pytest.raises(ValueError, match='statistics differ'):
        reproduce(bad, native)
    bad = deepcopy(record)
    bad['image']['image_rotation_cw'] = 90
    with pytest.raises(ValueError, match='original training RGB'):
        reproduce(bad, native)
    with open(record['image']['image'], 'ab') as file:
        file.write(b'changed')
    with pytest.raises(ValueError, match='original training RGB'):
        reproduce(record, native)
