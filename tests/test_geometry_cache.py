import hashlib
import json

import numpy as np
import pytest

from radio_moge.geometry_cache import (GeometryPackWriter, encode_geometry, read_geometry_member,
                                       sha_file)


def example(policy_hash='a' * 64):
    depth = np.array([[1, 2, np.nan], [3, 0, 5]], np.float32)
    valid = np.array([[1, 1, 0], [1, 0, 1]], bool)
    k = np.eye(3)
    provenance = dict(image_sha256='b' * 64, dataset='nyuv2', source_id='frame-1',
        label_kind='measured_sensor', camera_kind='calibrated_sensor', metric_calibrated=True,
        conversion='unfilled_z', policy_sha256=policy_hash,
        source_proofs=[dict(source='original/rawDepths', sha256='c' * 64)])
    row = dict(split='train', sha256='b' * 64, dataset='nyuv2')
    return row, depth, valid, k, provenance


def test_pack_roundtrip_mask_hashes_and_immutable_publication(tmp_path):
    writer = GeometryPackWriter(tmp_path, dict(name='test'), max_entries=1)
    row, depth, valid, k, proof = example(writer.policy_hash)
    writer.add(row, depth, valid, k, proof)
    writer.close()
    receipt = json.loads((tmp_path / 'pack-000000.json').read_text())
    assert not receipt['training_admission']
    assert receipt['sha256'] == sha_file(tmp_path / receipt['tar'])
    actual, mask, camera, _ = read_geometry_member(tmp_path / receipt['tar'], receipt['entries'][0])
    np.testing.assert_array_equal(mask, valid)
    np.testing.assert_array_equal(actual[valid], depth[valid])
    assert np.all(actual[~valid] == 0)
    np.testing.assert_array_equal(camera, k)
    with pytest.raises(FileExistsError):
        GeometryPackWriter(tmp_path, dict(name='test'))


def test_pseudo_cannot_be_claimed_as_metric_gt():
    _, d, v, k, p = example()
    p.update(label_kind='teacher_pseudo', teacher_revision='pinned', camera_kind='teacher_estimated')
    with pytest.raises(ValueError, match='calibrated metric'):
        encode_geometry(d, v, k, p)
    p['metric_calibrated'] = False
    key, payload, _ = encode_geometry(d, v, k, p)
    assert len(key) == 64 and len(hashlib.sha256(payload).digest()) == 32


def test_no_implicit_invalid_pixel_repair_or_zero_support():
    _, d, v, k, p = example()
    v[0, 2] = True  # A NaN cannot be promoted to measured support.
    with pytest.raises(ValueError):
        encode_geometry(d, v, k, p)
    with pytest.raises(ValueError):
        encode_geometry(d, np.zeros_like(v), k, p)


def test_training_binding_and_corruption_checks(tmp_path):
    writer = GeometryPackWriter(tmp_path, {}, max_entries=1)
    row, d, v, k, p = example(writer.policy_hash)
    with pytest.raises(ValueError):
        writer.add(dict(row, split='val'), d, v, k, p)
    writer.add(row, d, v, k, p)
    writer.close()
    receipt = json.loads((tmp_path / 'pack-000000.json').read_text())
    entry = dict(receipt['entries'][0], sha256='0' * 64)
    with pytest.raises(ValueError, match='hash mismatch'):
        read_geometry_member(tmp_path / receipt['tar'], entry)
