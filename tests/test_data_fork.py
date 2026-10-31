from copy import deepcopy
import hashlib
import json
import math

import numpy as np
from PIL import Image
import pytest
import torch

from radio_moge.data import DeterministicBatches
from radio_moge.data_fork import CoverageTracker, check_data_fork
from radio_moge.train import train
from test_performance import feature_config, profile
from test_training import tiny_system


@pytest.mark.parametrize('size', [1, 6, 31, 32, 33, 101])
def test_coverage_matches_consumed_sampler_and_exact_epoch_padding(size):
    batch, effective = 4, 32
    count = math.ceil(size / effective) * (effective // batch)
    actual = [key[0] for rows in DeterministicBatches(size, batch, [[32, 48]], 43, 0, count)
              for key in rows]
    tracker = CoverageTracker(size, batch, 43)
    tracker.advance(count // 2)
    assert tracker.report()['consumed_presentations'] == (count // 2) * batch
    tracker.advance(count)
    report = tracker.report()
    assert report['complete'] and report['unique_indices'] == len(set(actual)) == size
    assert report['repeated_presentations'] == len(actual) - size < effective
    assert report['consumed_index_sha256'] == hashlib.sha256(
        b''.join(i.to_bytes(8, 'little') for i in actual)).hexdigest()
    assert CoverageTracker(size, batch, 43, count).report() == report


def test_data_fork_preserves_adam_and_old_validation_resets_stream(tmp_path, monkeypatch):
    cfg = feature_config(tmp_path)
    cfg['train']['eval_at_start'] = True
    cfg['output_dir'] = str(tmp_path / 'parent')
    calls = []

    def evaluate(system, adapter, statistics, records, config, destination, limit):
        calls.append(destination.name)
        return {**{f'feature/{k}': 1. for k in cfg['loss']},
                'teacher_depth/abs_rel': .2, 'postprocess_failed': 0}

    monkeypatch.setattr('radio_moge.train.evaluate', evaluate)
    source = train(cfg, stop_after=2, system_adapter=tiny_system())
    parent = torch.load(source, weights_only=True)
    source_bytes = source.read_bytes()
    trial = deepcopy(cfg)
    trial['output_dir'] = str(tmp_path / 'trial')
    trial['train'].update(lr_schedule='constant', max_seconds=600, steps=3, eval_every=1)
    trial['data']['coverage_epochs'] = 1
    # Add one distinct new sample; keep the old training rows unchanged.
    image = tmp_path / 'new.png'
    Image.fromarray(np.full((40, 60, 3), 128, np.uint8)).save(image)
    manifest = tmp_path / 'expanded.jsonl'
    manifest.write_text(__import__('pathlib').Path(cfg['data']['train_manifest']).read_text() +
                        json.dumps(dict(image=str(image), group='new')) + '\n')
    trial['data']['train_manifest'] = str(manifest)
    val_image = tmp_path / 'extra-val.png'
    Image.fromarray(np.full((40, 60, 3), 64, np.uint8)).save(val_image)
    val = tmp_path / 'extra-val.jsonl'
    val.write_text(json.dumps(dict(image=str(val_image), group='heldout')) + '\n')
    trial['evaluation']['dataset_variants'] = {'new': dict(manifest=str(val), num_tokens=1200, max_side=48)}
    calls.clear()
    final_path = train(trial, fork_from=source, data_fork=True, performance=profile(2, 0),
                       system_adapter=tiny_system())
    initial = torch.load(tmp_path / 'trial/initial.pt', weights_only=True)
    assert initial['step'] == 2 and initial['cursor'] == 0
    assert initial['fork_provenance']['data_change']['parent_cursor'] == parent['cursor']
    for key, values in parent['optimizer']['state'].items():
        for name, value in values.items():
            torch.testing.assert_close(initial['optimizer']['state'][key][name], value, rtol=0, atol=0)
    for section in ['adapter', 'statistics']:
        for key, value in parent[section].items():
            torch.testing.assert_close(initial[section][key], value, rtol=0, atol=0)
    final = torch.load(final_path, weights_only=True)
    assert final['step'] == 3 and final['cursor'] == 4
    assert calls == ['eval-0000002', 'dataset-new', 'eval-0000003', 'dataset-new']
    assert json.loads((tmp_path / 'trial/coverage.json').read_text())['unique_indices'] == 7
    assert source.read_bytes() == source_bytes


@pytest.mark.parametrize('change', ['val', 'buckets', 'loss', 'augment', 'lr', 'weighted_epoch'])
def test_data_fork_rejects_unrelated_or_unverifiable_change(tmp_path, change):
    cfg = feature_config(tmp_path)
    trial = deepcopy(cfg)
    trial['train'].update(lr_schedule='constant', max_seconds=600, steps=5)
    digests = dict(train='new', val='same')
    if change == 'val':
        digests['val'] = 'changed'
    elif change == 'buckets':
        trial['data']['buckets'] = [[32, 32]]
    elif change == 'loss':
        trial['loss']['spatial'] = 2
    elif change == 'augment':
        trial['data']['augment'] = not cfg['data']['augment']
    elif change == 'lr':
        trial['train']['lr'] *= 2
    else:
        trial['data'].update(coverage_epochs=1, mixture_weights={'one': 1})
    with pytest.raises(ValueError):
        check_data_fork(trial, dict(config=cfg, step=2, data_digests=dict(train='old', val='same')), digests)
