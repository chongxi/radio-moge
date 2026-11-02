import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


scripts = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('gt_quality', scripts / 'audit_gt_source_quality.py')
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)
sys.path.remove(str(scripts))


def test_quality_diagnostics_do_not_claim_truth():
    depth = np.full((8, 8), 3., dtype=np.float32)
    valid = np.ones((8, 8), bool)
    valid[3:] = False
    depth[~valid] = 0
    result = quality.quality_stats(depth, valid, np.zeros((8, 8, 3), np.uint8))
    assert result['valid_fraction'] == .375
    assert 'less_than_half_measured_support' in result['review_flags']
    assert 'very_small_depth_spread_not_necessarily_wrong' in result['review_flags']
    assert 'accepted' not in result


def test_invalid_depth_must_not_be_filled_or_nonfinite():
    depth = np.ones((8, 8), np.float32)
    valid = np.ones((8, 8), bool)
    valid[0, 0] = False
    rgb = np.zeros((8, 8, 3), np.uint8)
    with pytest.raises(ValueError, match='not filled'):
        quality.quality_stats(depth, valid, rgb)
    depth[0, 0] = 0
    depth[1, 1] = np.nan
    with pytest.raises(ValueError, match='finite'):
        quality.quality_stats(depth, valid, rgb)


def test_inventory_is_split_safe_and_does_not_approve_available_files(tmp_path):
    train = next(str(i) for i in range(100) if not quality.heldout(str(i)))
    val = next(str(i) for i in range(100) if quality.heldout(str(i)))
    (tmp_path / 'splits').mkdir()
    for split in quality.SPLITS:
        (tmp_path / 'splits' / (split + '.txt')).write_text(f'{train}\n{val}\n' if split == 'nvs_sem_train' else '')
    records = {split: [dict(scene=scene, sha256=split)] for split, scene in [('train', train), ('val', val)]}
    for folder, suffix in quality.MODALITIES.values():
        path = tmp_path / 'data' / train / 'panocam' / folder
        path.mkdir(parents=True)
        (path / ('0' + suffix)).touch()
    result = quality.source_inventory(tmp_path, records)
    assert result['eligible_train_complete_scan_pairs'] == 1
    assert result['eligible_train_scenes_with_scanner_poses'] == 0
    assert 'NOT quality approval' in result['scope']
    (tmp_path / 'splits/nvs_sem_val.txt').write_text(train)
    with pytest.raises(ValueError, match='split overlap'):
        quality.source_inventory(tmp_path, records)


def test_audit_output_refuses_overwriting_prior_evidence(tmp_path):
    output = tmp_path / 'evidence.json'
    output.write_text('original')
    with pytest.raises(FileExistsError):
        quality.audit(tmp_path, output)
    assert output.read_text() == 'original'
