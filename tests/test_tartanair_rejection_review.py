from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from review_tartanair_rejections import reproduce, unreviewed_rejections
from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import publish_json, sha_file
from radio_moge.tartanair_depth import convert_tartanair


def fixture(root):
    rgb = np.full((480, 640, 3), 50, np.uint8)
    left, right = np.full((2, 480, 640), 10000, np.float32)
    left[:, 200:224] = 4
    right[:, 180:204] = 4
    path = root / 'rgb.png'
    Image.fromarray(rgb).save(path)
    row = dict(dataset='tartanair', split='train', width=640, height=480,
        image=str(path), sha256=sha_file(path),
        source_url='https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/test/gascola/Hard/image_left.zip',
        source_member='gascola/Hard/P002/image_left/001407_left.png')
    proofs, paths = [], []
    suffixes = {'depth_left': '_left_depth.npy', 'depth_right': '_right_depth.npy',
                'image_right': '_right.png'}
    for kind, data in [('depth_left', left), ('depth_right', right), ('image_right', rgb)]:
        url = row['source_url'].rsplit('/', 1)[0] + '/' + kind + '.zip'
        member = row['source_member'].replace('/image_left/', '/' + kind + '/')
        member = member.removesuffix('_left.png') + suffixes[kind]
        path = root / 'native' / hashlib.sha256(url.encode()).hexdigest()[:16] / member
        path.parent.mkdir(parents=True)
        buffer = io.BytesIO()
        if kind == 'image_right':
            Image.fromarray(data).save(buffer, format='PNG')
        else:
            np.save(buffer, data, allow_pickle=False)
        path.write_bytes(buffer.getvalue())
        proof = dict(source=url, member=member, sha256=sha_file(path), bytes=path.stat().st_size)
        path.with_name(path.name + '.receipt.json').write_text(json.dumps(proof))
        proofs.append(proof)
        paths.append(path)
    _, _, _, stats = convert_tartanair(left, right, rgb, rgb)
    record = dict(image=row, reason='insufficient_stereo_verified_support', source_proofs=proofs, stats=stats)
    return record, paths


def test_reproduce_low_support_without_modifying_native_depth_or_inputs(tmp_path):
    record, paths = fixture(tmp_path)
    original = deepcopy(record)
    before = [sha_file(p) for p in paths]
    left, _, _, valid, quality, refs = reproduce(record, tmp_path / 'native')
    assert 0 < valid.mean() < .05 and quality == record['stats']
    assert (left[~valid] > 0).all()  # Diagnostic retains native out-of-domain values.
    assert len(refs) == 3 and record == original
    assert before == [sha_file(p) for p in paths]


def test_mismatched_eye_or_frame_and_changed_source_bytes_are_rejected(tmp_path):
    original, paths = fixture(tmp_path)
    record = deepcopy(original)
    record['source_proofs'][1]['member'] = record['source_proofs'][0]['member']
    with pytest.raises(ValueError, match='corresponding-eye'):
        reproduce(record, tmp_path / 'native')
    record = deepcopy(original)
    record['source_proofs'][0]['member'] = record['source_proofs'][0]['member'].replace('001407', '001408')
    with pytest.raises(ValueError, match='corresponding-eye'):
        reproduce(record, tmp_path / 'native')
    with paths[0].open('ab') as file:
        file.write(b'changed')
    with pytest.raises(ValueError, match='Retained source'):
        reproduce(original, tmp_path / 'native')


def test_wrong_reported_support_and_changed_rgb_cannot_pass_reproduction(tmp_path):
    original, _ = fixture(tmp_path)
    record = deepcopy(original)
    record['stats']['valid_fraction'] = .001
    with pytest.raises(ValueError, match='Recomputed support'):
        reproduce(record, tmp_path / 'native')
    Image.fromarray(np.full((480, 640, 3), 51, np.uint8)).save(original['image']['image'])
    with pytest.raises(ValueError, match='Unchanged frozen'):
        reproduce(original, tmp_path / 'native')


def rejection_snapshot(root):
    references = []
    for i in range(2):
        path = root / f'{i}.json'
        publish_json(path, dict(image=dict(dataset='tartanair', image=f'/rgb/{i}.png',
                                          sha256=str(i) * 64)))
        references.append(file_ref(path))
    decision = dict(dataset='tartanair', image='/rgb/0.png', sha256='0' * 64,
                    decision='exclude', scope='geometry_supervision', rgb_replay_excluded=False,
                    reason='reviewed insufficient stereo support', evidence=[references[0]])
    return references, decision


def test_incremental_review_skips_only_exact_evidenced_prior_rejections(tmp_path):
    references, decision = rejection_snapshot(tmp_path)
    assert unreviewed_rejections(references, [decision]) == references[1:]
    assert unreviewed_rejections(references, [dict(decision, scope='geometry_only')]) == references[1:]
    assert unreviewed_rejections(references, []) == references
    assert unreviewed_rejections(references[:1], [decision]) == []


def test_incremental_review_rejects_changed_rgb_decision_or_rejection(tmp_path):
    references, decision = rejection_snapshot(tmp_path)
    for change in (dict(sha256='f' * 64), dict(evidence=[references[1]]),
                   dict(rgb_replay_excluded=True), dict(scope='rgb_replay'), dict(decision='accept')):
        with pytest.raises(ValueError):
            unreviewed_rejections(references, [dict(decision, **change)])
    with pytest.raises(ValueError, match='Unique'):
        unreviewed_rejections(references, [decision, decision])
    with pytest.raises(ValueError, match='Unique'):
        unreviewed_rejections(references + references[:1], [decision])
    Path(references[0]['path']).write_text('{}')
    with pytest.raises(ValueError, match='Evidence changed'):
        unreviewed_rejections(references, [decision])
