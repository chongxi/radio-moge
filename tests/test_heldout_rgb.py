from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from audit_corpus_rgb import run as train_census
from audit_heldout_rgb import compare, run
sys.path.pop(0)

from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import publish_json, sha_file
from radio_moge.heldout_rgb import identity, inspect_heldout_rgb, manifest_protocol


def test_heldout_legacy_nyu_is_never_coerced_to_encoded_sha_or_training(tmp_path):
    directory = tmp_path / 'nyuv2-internal-val-s43'
    directory.mkdir()
    image = Image.fromarray(np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3))
    path = directory / 'rgb.png'
    image.save(path)
    manifest = directory / 'val.jsonl'
    row = dict(id='nyuv2_0001', scene='office', group='office', image='rgb.png',
               sha256=hashlib.sha256(image.tobytes()).hexdigest(), depth='/must/not/open.npy')
    original = deepcopy(row)
    info = inspect_heldout_rgb(row, manifest, manifest_protocol(manifest))
    assert row == original and info['original_split'] is None
    assert info['original_sha256_kind'] == 'raw_rgb_bytes'
    assert info['group'] == 'nyuv2:office' and info['encoded_sha256'] == sha_file(path)
    with pytest.raises(ValueError, match='decoded RGB hash differs'):
        inspect_heldout_rgb(dict(row, sha256=sha_file(path)), manifest, manifest_protocol(manifest))
    with pytest.raises(ValueError, match='split differs'):
        inspect_heldout_rgb(dict(row, split='train'), manifest, manifest_protocol(manifest))
    with pytest.raises(ValueError, match='Unreviewed'):
        manifest_protocol(tmp_path / 'unknown.jsonl')


def test_heldout_only_uses_declared_orientation_and_size(tmp_path):
    directory = tmp_path / 'attention-full-mix-v1-20261022'
    directory.mkdir()
    image = Image.fromarray(np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3))
    path = directory / 'rgb.png'
    image.save(path)
    manifest = directory / 'val-full.jsonl'
    row = dict(image='rgb.png', dataset='arkitscenes', split='val-full', group='visit:1',
        sha256=sha_file(path), sha256_kind='encoded_file_bytes', image_rotation_cw=90, width=8, height=12)
    info = inspect_heldout_rgb(row, manifest, manifest_protocol(manifest))
    assert (info['width'], info['height']) == (8, 12)
    assert info['raw_canonical_rgb_sha256'] != info['canonical_rgb_sha256']
    for change, match in [({'width': 12}, 'dimensions'), ({'image_rotation_cw': True}, 'rotation'),
                          ({'split': 'train'}, 'split'), ({'sha256': '0' * 64}, 'encoded RGB')]:
        with pytest.raises(ValueError, match=match):
            inspect_heldout_rgb(dict(row, **change), manifest, manifest_protocol(manifest))


def audit_fixture(root):
    pixels = Image.fromarray(np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3))
    training = root / 'train.png'
    pixels.save(training, compress_level=0)
    train_row = dict(image=str(training), dataset='irs', group='irs:family:1', split='train',
                     sha256=sha_file(training), sha256_kind='encoded_file_bytes', width=16, height=12)
    with sqlite3.connect(root / 'inventory.sqlite') as db:
        db.execute('CREATE TABLE images(ordinal,image,row_json,dataset,scene_group)')
        db.execute('INSERT INTO images VALUES (?,?,?,?,?)',
                   (0, str(training), json.dumps(train_row), 'irs', train_row['group']))
    legacy = root / 'nyuv2-internal-val-s43'
    legacy.mkdir()
    reencoded = legacy / 'same-pixels.png'
    pixels.save(reencoded, compress_level=9)
    assert sha_file(training) != sha_file(reencoded)
    mixed = root / 'attention-full-mix-v1-20261022'
    mixed.mkdir()
    entries = [
        (legacy / 'val.jsonl', dict(image=str(reencoded), id='nyuv2_0001', scene='office', group='office',
            sha256=hashlib.sha256(pixels.tobytes()).hexdigest(), depth='/never/read/depth.npy'), True),
        (mixed / 'val.jsonl', dict(image=str(reencoded), dataset='nyuv2', split='val', group='nyuv2:office',
            sha256=sha_file(reencoded), sha256_kind='encoded_file_bytes'), True),
        (mixed / 'test-irs-official.jsonl', dict(image=str(reencoded), dataset='irs', split='official-test',
            official_split='test', group='irs:family:1', sha256=sha_file(reencoded),
            sha256_kind='encoded_file_bytes'), False),
    ]
    references = []
    for path, row, required in entries:
        path.write_text(json.dumps(row) + '\n')
        references.append(dict(**file_ref(path), rows=1, scene_disjoint_required=required))
    publish_json(root / 'inventory.json', dict(stage='inventory_complete', manifest_sha256='a' * 64,
        images=1, dataset_counts={'irs': 1}, heldout_checks=references,
        inventory_sha256=sha_file(root / 'inventory.sqlite')))
    return dict(root=str(root), manifest_sha256='a' * 64, expected_counts={'irs': 1})


def test_full_censuses_detect_reencoded_cross_source_leakage_and_preserve_memberships(tmp_path):
    config = audit_fixture(tmp_path)
    train_census(config, 'preflight', 'train-probe', 'unused')
    train_census(config, 'census', 'train-all', 'train-probe')
    probe = run(config, 'preflight', 'heldout-probe', 'unused')
    assert probe['processed'] == 3 and probe['failed_images'] == 0
    full = run(config, 'census', 'heldout-all', 'heldout-probe')
    assert full['processed'] == 3 and full['distinct_verified_paths'] == 1
    result = compare(tmp_path / 'train-all/report.json', tmp_path / 'heldout-all/report.json', tmp_path / 'comparison')
    assert result['exact_pixel_overlap_rows'] == 3 and result['forbidden_shared_scene_rows'] == 0
    assert result['per_manifest'][str(tmp_path / 'attention-full-mix-v1-20261022/test-irs-official.jsonl')][
        'shared_scene_rows'] == 1
    assert not result['training_admission'] and not result['near_duplicates_checked']
    assert result['automatic_exclusions'] == 0 and not result['evaluation_depth_read']
    with sqlite3.connect(tmp_path / 'comparison/comparison.sqlite') as db:
        assert db.execute('SELECT COUNT(*) FROM overlaps').fetchone()[0] == 3
    with pytest.raises(FileExistsError):
        run(config, 'preflight', 'heldout-probe', 'unused')


def test_bad_identity_remains_failure_and_prevents_full_census(tmp_path):
    config = audit_fixture(tmp_path)
    path = tmp_path / 'nyuv2-internal-val-s43/same-pixels.png'
    Image.new('RGB', (16, 12), 'red').save(path)
    report = run(config, 'preflight', 'failed-probe', 'unused')
    assert report['failed_images'] == 3
    with pytest.raises(ValueError, match='successful heldout RGB preflight'):
        run(config, 'census', 'blocked', 'failed-probe')


def test_changed_bound_manifest_and_official_test_protocol_are_rejected(tmp_path):
    config = audit_fixture(tmp_path)
    path = tmp_path / 'attention-full-mix-v1-20261022/test-irs-official.jsonl'
    row = json.loads(path.read_text())
    with pytest.raises(ValueError, match='official frame-audit'):
        identity(dict(row, official_split='train'), path, manifest_protocol(path))
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='Evidence changed'):
        run(config, 'preflight', 'changed-manifest', 'unused')


def test_partial_training_census_cannot_claim_complete_heldout_isolation(tmp_path):
    config = audit_fixture(tmp_path)
    train_census(config, 'preflight', 'train-probe', 'unused')
    run(config, 'preflight', 'heldout-probe', 'unused')
    run(config, 'census', 'heldout-all', 'heldout-probe')
    with pytest.raises(ValueError, match='Complete, error-free censuses'):
        compare(tmp_path / 'train-probe/report.json', tmp_path / 'heldout-all/report.json', tmp_path / 'blocked')
