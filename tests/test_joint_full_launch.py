from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_joint_full_day import digest, ready


def prepared(tmp_path):
    config = json.loads(Path('configs/joint-qknorm-full-day.json').read_text())
    for name in ('train.jsonl', 'val-selection.jsonl', 'test.jsonl'):
        (tmp_path / name).write_text('fixture\n')
    config['data']['train_manifest'] = str(tmp_path / 'train.jsonl')
    config['data']['val_manifest'] = str(tmp_path / 'val-selection.jsonl')
    info = dict(stage='ready', train_images=796478, selection_images=1268,
                mixture_weights=config['data']['mixture_weights'],
                verification=dict(all_paths_exist=True, split_groups_paths_hashes_disjoint=True,
                                  old_holdouts_byte_identical=True),
                manifest_sha256={n:digest(tmp_path/n) for n in
                                 ('train.jsonl', 'val-selection.jsonl', 'test.jsonl')},
                preserved_heldout_sha256={'test':digest(tmp_path/'test.jsonl')})
    (tmp_path/'dataset_info.json').write_text(json.dumps(info))
    plan = dict(authorized=True, expected_config=deepcopy(config), prepared_dataset=str(tmp_path),
                base_dataset=str(tmp_path), dataset_info_sha256=digest(tmp_path/'dataset_info.json'))
    return plan, config


def test_guard_accepts_only_frozen_recipe_and_verified_manifests(tmp_path):
    plan, config = prepared(tmp_path)
    assert ready(plan, config)['train_images'] == 796478
    config['train']['seed'] = 42
    with pytest.raises(ValueError, match='reviewed plan'):
        ready(plan, config)
    config['train']['seed'] = 43
    (tmp_path/'train.jsonl').write_text('changed\n')
    with pytest.raises(ValueError, match='Manifest changed'):
        ready(plan, config)


def test_guard_rejects_budget_change_even_if_plan_config_both_changed(tmp_path):
    plan, config = prepared(tmp_path)
    config['train']['max_seconds'] = 172800
    plan['expected_config'] = deepcopy(config)
    with pytest.raises(ValueError, match='budget'):
        ready(plan, config)
