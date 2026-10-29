from copy import deepcopy
import json

import pytest
import torch

from radio_moge.performance import grouped_window, schedule_position, transition_limits, choose_profile
from radio_moge.train import train, make_loader
from radio_moge.data import read_manifest
from test_training import tiny_system, config_for


def profile(step, cursor, effective=16, compute=8, workers=0):
    import time
    return dict(authorized=True, autotune=False, original_started_at=time.time(),
                effective_batch=effective, compute_batch=compute, workers=workers,
                anchor_step=step, anchor_cursor=cursor, anchor_schedule_position=float(step),
                base_effective_batch=16, schedule_ratio=effective / 16)


def test_grouping_retains_every_tensor_and_separates_shapes():
    images = [torch.full((4, 3, h, w), i) for i, (h, w) in enumerate(
        [(32, 48), (48, 32), (32, 48), (32, 48)])]
    grouped = list(grouped_window(iter(images), 4, 8))
    assert [len(x) for x in grouped] == [8, 4, 4]
    values = sorted(int(x[0, 0, 0]) for group in grouped for x in group)
    assert values == sorted(i for i in range(4) for _ in range(4))


def test_image_based_lr_and_limits_are_continuous():
    perf = profile(2100, 8400, 64, 32)
    assert schedule_position(2100, perf) == 2100
    assert schedule_position(2101, perf) == 2104
    limits = transition_limits(dict(steps=120000, max_seconds=86400), dict(steps=120000), perf)
    assert limits['steps'] == 31575
    assert limits['max_seconds'] == 86400
    assert schedule_position(limits['steps'], perf) == 120000


def test_selection_excludes_oom_and_prefers_smaller_batch_near_tie():
    rows = [dict(stage='passed', effective_batch=b, peak_memory_gib=b,
                 images_per_second=s) for b, s in [(16, 60), (32, 62), (64, 63)]]
    rows += [dict(stage='oom', images_per_second=1000)]
    assert choose_profile(rows)['effective_batch'] == 32


def feature_config(tmp_path):
    cfg = config_for(tmp_path)
    cfg['loss'] = dict(spatial=1., summary=1., cosine=1., summary_cosine=1.)
    cfg['train'].update(batch_size=4, accumulation=4, steps=8, save_every=100)
    return cfg


def test_prefetch_preserves_augmentation_and_sampler(tmp_path):
    cfg = feature_config(tmp_path)
    cfg['data']['buckets'] = [[32, 48], [48, 32]]
    records = read_manifest(cfg['data']['train_manifest'])
    plain = list(make_loader(records, cfg, 7, 6))
    cfg['data']['workers'] = 2
    prefetched = list(make_loader(records, cfg, 7, 6))
    for a, b in zip(plain, prefetched, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_grouped_resume_matches_loss_gradient_and_preserves_checkpoint(tmp_path):
    cfg = feature_config(tmp_path)
    cfg['output_dir'] = str(tmp_path / 'base')
    checkpoint = train(cfg, stop_after=2, system_adapter=tiny_system())
    parent = torch.load(checkpoint, weights_only=True)
    normal = deepcopy(cfg)
    normal['output_dir'] = str(tmp_path / 'plain')
    plain = torch.load(train(normal, resume=checkpoint, stop_after=3,
                             system_adapter=tiny_system()), weights_only=True)
    grouped = deepcopy(cfg)
    grouped['output_dir'] = str(tmp_path / 'grouped')
    tuned = torch.load(train(grouped, resume=checkpoint, stop_after=3,
                            system_adapter=tiny_system(), performance=profile(2, 8)), weights_only=True)
    assert tuned['cursor'] == plain['cursor'] == 12
    assert tuned['scheduler']['last_epoch'] == plain['scheduler']['last_epoch'] == 3
    for k, v in plain['adapter'].items():
        torch.testing.assert_close(v, tuned['adapter'][k], rtol=2e-5, atol=3e-7)
    assert all(torch.equal(v, torch.load(checkpoint, weights_only=True)['adapter'][k])
               for k, v in parent['adapter'].items())


def test_larger_batch_consumes_exact_next_samples_and_lr_progress(tmp_path):
    cfg = feature_config(tmp_path)
    cfg['output_dir'] = str(tmp_path / 'base')
    checkpoint = train(cfg, stop_after=2, system_adapter=tiny_system())
    cfg['output_dir'] = str(tmp_path / 'larger')
    tuned = torch.load(train(cfg, resume=checkpoint, stop_after=3,
                            system_adapter=tiny_system(), performance=profile(2, 8, 64, 16)),
                       weights_only=True)
    assert tuned['step'] == 3 and tuned['cursor'] == 24
    assert tuned['runtime_limits']['steps'] == 3  # floor: never exceed original image budget
    row = json.loads((tmp_path / 'larger/train.jsonl').read_text().splitlines()[-1])
    assert row['effective_batch'] == 64 and row['schedule_position'] == 6


def test_runtime_change_rejects_geometry_and_unreviewed_profiles(tmp_path):
    from radio_moge.performance import validate_performance
    cfg = config_for(tmp_path)
    with pytest.raises(ValueError, match='feature-only'):
        validate_performance(profile(0, 0), cfg)


def test_evaluation_cadence_follows_images_after_batch_change():
    from radio_moge.performance import interval_due
    perf = profile(2113, 8452, 64, 32)
    assert not interval_due(2584, 2000, perf)
    assert interval_due(2585, 2000, perf)  # virtual position4001 crosses4000
    assert not interval_due(2586, 2000, perf)
