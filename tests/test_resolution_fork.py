from copy import deepcopy
import json

import numpy as np
from PIL import Image
import pytest
import torch

from radio_moge.data import evaluation_image, TrainingImages, MixtureBatches
from radio_moge.selection import selection_winners
from radio_moge.train import check_resolution_fork, train
from test_performance import feature_config, profile
from test_training import tiny_system


def trial_config(config, output):
    trial = deepcopy(config)
    trial['output_dir'] = str(output)
    trial['data']['buckets'].append([32, 32])
    trial['train'].update(lr_schedule='constant', max_seconds=600, steps=5,
                          save_every=2, eval_every=2)
    trial['evaluation']['resolution_variants'] = {
        'square': dict(bucket=[32, 32], num_tokens=None)}
    return trial


def test_resolution_fork_restores_adam_and_evaluates_both_views(tmp_path, monkeypatch):
    cfg = feature_config(tmp_path)
    cfg['train']['eval_at_start'] = True
    cfg['output_dir'] = str(tmp_path / 'parent')
    calls = []

    def evaluate(system, adapter, statistics, records, config, destination, limit):
        calls.append((destination.name, config['evaluation'].get('bucket')))
        return {**{f'feature/{k}': 1. for k in cfg['loss']},
                'teacher_depth/abs_rel': .2, 'postprocess_failed': 0}

    monkeypatch.setattr('radio_moge.train.evaluate', evaluate)
    source = train(cfg, stop_after=2, system_adapter=tiny_system())
    parent = torch.load(source, weights_only=True)
    source_bytes = source.read_bytes()
    calls.clear()
    trial = trial_config(cfg, tmp_path / 'trial')
    perf = dict(profile(2, 8), data_pipeline='prepared', compact_transport=True)
    final = torch.load(train(trial, fork_from=source, resolution_fork=True,
                             performance=perf, system_adapter=tiny_system()), weights_only=True)
    initial = torch.load(tmp_path / 'trial/initial.pt', weights_only=True)
    assert initial['step'] == 2 and initial['cursor'] == 8
    for key, values in parent['optimizer']['state'].items():
        for name, value in values.items():
            torch.testing.assert_close(initial['optimizer']['state'][key][name], value, rtol=0, atol=0)
    for section in ['adapter', 'statistics']:
        for key, value in parent[section].items():
            torch.testing.assert_close(initial[section][key], value, rtol=0, atol=0)
    assert final['step'] == 5 and final['cursor'] == 20
    assert final['fork_provenance']['resolution_change']['parent_buckets'] == [[32, 48]]
    assert calls == [('eval-0000002', None), ('square', [32, 32]),
                     ('eval-0000004', None), ('square', [32, 32]),
                     ('eval-0000005', None), ('square', [32, 32])]
    rows = [json.loads(line) for line in (tmp_path / 'trial/validation.jsonl').read_text().splitlines()]
    assert all(r['resolution/square/teacher_depth/abs_rel'] == .2 for r in rows)
    assert source.read_bytes() == source_bytes


@pytest.mark.parametrize('change', ['remove', 'duplicate', 'lr', 'manifest', 'augment', 'batch', 'view'])
def test_resolution_fork_rejects_other_changes(tmp_path, change):
    cfg = feature_config(tmp_path)
    trial = trial_config(cfg, tmp_path / 'trial')
    if change == 'remove':
        trial['data']['buckets'] = [[32, 32]]
    elif change == 'duplicate':
        trial['data']['buckets'].append([32, 32])
    elif change == 'lr':
        trial['train']['lr'] *= 2
    elif change == 'manifest':
        trial['data']['train_manifest'] = 'different.jsonl'
    elif change == 'augment':
        trial['data']['augment'] = False
    elif change == 'batch':
        trial['train']['batch_size'] *= 2
    else:
        trial['evaluation']['resolution_variants']['square']['num_tokens'] = 1200
    with pytest.raises(ValueError):
        check_resolution_fork(trial, dict(config=cfg, data_digests={}, step=2), {})


def test_native_validation_rgb_depth_crop_is_aligned(tmp_path):
    values = np.arange(480, dtype=np.uint16)[:, None] * np.ones((1, 640), np.uint16)
    rgb = np.repeat((values % 256).astype(np.uint8)[..., None], 3, -1)
    image_path, depth_path = tmp_path / 'rgb.png', tmp_path / 'depth.npy'
    Image.fromarray(rgb).save(image_path)
    np.save(depth_path, values)
    image, depth = evaluation_image(dict(image=str(image_path), depth=str(depth_path)),
                                    max_side=640, bucket=[400, 640])
    np.testing.assert_array_equal(depth.numpy(), values[40:440])
    np.testing.assert_array_equal((image[0, 0].numpy() * 255).round(), values[40:440] % 256)
    assert image.shape[-2:] == (400, 640)
    dataset = TrainingImages([dict(image=str(image_path))], augment=True)
    crops = [dataset[(0, seed, 400, 640)] for seed in range(12)]
    assert all(c.shape == (3, 400, 640) for c in crops)
    assert len({int(c[0, 0, 0] * 255) for c in crops}) > 3
    torch.testing.assert_close(crops[0], dataset[(0, 0, 400, 640)], atol=0, rtol=0)


def test_three_buckets_are_sampled_and_native_selection_rejects_failed_view():
    buckets = [[480, 640], [640, 480], [400, 640]]
    batches = list(MixtureBatches([dict(dataset='one', group='scene')], {'one': 1.},
                                 4, buckets, 43, 719996, 300))
    assert {tuple(batch[0][-2:]) for batch in batches} == {tuple(x) for x in buckets}
    specs = {'new': dict(metric='resolution/new/macro/spatial/cosine', mode='max')}
    metrics = {'postprocess_failed': 0, 'resolution/new/postprocess_failed': .1,
               'resolution/new/macro/spatial/cosine': .99}
    assert selection_winners(specs, {}, metrics, 1) == []
    metrics['resolution/new/postprocess_failed'] = 0
    assert selection_winners(specs, {}, metrics, 1) == ['new']
