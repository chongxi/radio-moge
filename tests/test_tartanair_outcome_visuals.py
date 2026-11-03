from copy import deepcopy
import hashlib
import io
import json
import zlib

import numpy as np
from PIL import Image
import pytest

from audit_tartanair_outcomes import CONVERSION, candidate_details
from radio_moge.geometry_cache import GeometryPackWriter, canonical_json, sha_file
from radio_moge.tartanair_depth import convert_tartanair
from visualize_tartanair_outcomes import reproduce


def fixture(root, *, changed_depth=False, changed_camera=False, wrong_crc=False):
    rgb = np.full((480, 640, 3), 50, np.uint8)
    left = np.full((480, 640), 4., np.float32)
    path = root / 'left.png'
    Image.fromarray(rgb).save(path)
    row = dict(dataset='tartanair', split='train', width=640, height=480,
        image=str(path), sha256=sha_file(path), environment='gascola',
        group='tartanair:environment:gascola', trajectory='gascola/Hard/P002',
        source_url='https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_test/tartanair/gascola/Hard/image_left.zip',
        source_member='gascola/Hard/P002/image_left/001407_left.png')
    proofs = []
    for kind, data, suffix in [('depth_left', left, '_left_depth.npy'),
                              ('depth_right', left, '_right_depth.npy'), ('image_right', rgb, '_right.png')]:
        url = row['source_url'].rsplit('/', 1)[0] + '/' + kind + '.zip'
        member = row['source_member'].replace('/image_left/', '/' + kind + '/')
        member = member.removesuffix('_left.png') + suffix
        path = root / 'native' / hashlib.sha256(url.encode()).hexdigest()[:16] / member
        path.parent.mkdir(parents=True)
        buffer = io.BytesIO()
        if kind == 'image_right':
            Image.fromarray(data).save(buffer, format='PNG')
        else:
            np.save(buffer, data, allow_pickle=False)
        payload = buffer.getvalue()
        path.write_bytes(payload)
        proof = dict(source=url, member=member, sha256=sha_file(path), bytes=len(payload),
                     crc32=(zlib.crc32(payload) & 0xffffffff) ^ int(wrong_crc))
        path.with_name(path.name + '.receipt.json').write_text(json.dumps(proof))
        proofs.append(proof)
    depth, valid, camera, quality = convert_tartanair(left, left, rgb, rgb)
    policy = dict(fixture=True)
    checksum = hashlib.sha256(canonical_json(policy)).hexdigest()
    metadata = dict(image_sha256=row['sha256'], dataset='tartanair', source_id=row['source_member'],
        label_kind='synthetic_renderer', camera_kind='synthetic_camera', metric_calibrated=True,
        conversion=CONVERSION, policy_sha256=checksum, source_proofs=proofs,
        source_quality=quality, source_needs_review=False)
    if changed_depth:
        depth[valid] += 1
    if changed_camera:
        camera[0, 2] += .01
    writer = GeometryPackWriter(root / 'packs', policy)
    writer.add(row, depth, valid, camera, metadata)
    writer.close()
    receipt = json.loads((root / 'packs/pack-000000.json').read_text())
    entry = receipt['entries'][0]
    return entry, root / 'packs' / receipt['tar'], root / 'native', checksum, candidate_details(entry, checksum)


def test_actual_sources_reproduce_cached_arrays_without_changing_native_values(tmp_path):
    args = fixture(tmp_path)
    original = deepcopy(args[0])
    left, rgb, right, valid, quality, refs = reproduce(*args)
    assert (left == 4).all() and np.array_equal(rgb, right)
    assert .9 < valid.mean() < 1 and quality['valid_fraction'] == valid.mean()
    assert len(refs) == 3 and original == args[0]


@pytest.mark.parametrize('change', ['changed_depth', 'changed_camera'])
def test_self_consistent_cache_cannot_disagree_with_retained_source(tmp_path, change):
    args = fixture(tmp_path, **{change: True})
    with pytest.raises(ValueError, match='Reproduced native'):
        reproduce(*args)


def test_correct_sha_does_not_hide_wrong_publisher_crc(tmp_path):
    with pytest.raises(ValueError, match='CRC'):
        reproduce(*fixture(tmp_path, wrong_crc=True))


def test_audit_selection_must_match_exact_metrics_and_correct_eye(tmp_path):
    entry, pack, root, policy, metrics = fixture(tmp_path)
    with pytest.raises(ValueError, match='audited metrics'):
        reproduce(entry, pack, root, policy, dict(metrics, support=.5))
    entry['geometry_metadata']['source_proofs'][1]['member'] = entry['geometry_metadata']['source_proofs'][0]['member']
    with pytest.raises(ValueError, match='corresponding-eye'):
        reproduce(entry, pack, root, policy, metrics)


def test_changed_original_rgb_cannot_be_used_for_geometry_review(tmp_path):
    args = fixture(tmp_path)
    Image.fromarray(np.full((480, 640, 3), 51, np.uint8)).save(args[0]['image']['image'])
    with pytest.raises(ValueError, match='Evidence changed'):
        reproduce(*args)
