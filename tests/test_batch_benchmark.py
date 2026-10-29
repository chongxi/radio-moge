from copy import deepcopy
from types import SimpleNamespace
import time

import pytest
import torch

from radio_moge.batch_benchmark import validate_plan, reset_state, run_profile, aggregate
from radio_moge.checkpoint import rng_state
from radio_moge.data import read_manifest
from radio_moge.losses import FeatureStatistics
from test_training import tiny_system
from test_performance import feature_config


def test_plan_requires_equal_image_budgets_and_bounded_resources():
    plan = dict(profiles=[dict(effective_batch=32, compute_batch=16, workers=4),
                          dict(effective_batch=256, compute_batch=128, workers=4)],
                warmup_images=256, measured_images=1024, rounds=2, max_seconds=900)
    validate_plan(plan, 4)
    for key, value in [('warmup_images', 32), ('measured_images', 1056),
                       ('rounds', 1), ('max_seconds', 1300)]:
        with pytest.raises(ValueError):
            validate_plan(dict(plan, **{key: value}), 4)
    bad = deepcopy(plan)
    bad['profiles'][1]['workers'] = 8
    with pytest.raises(ValueError):
        validate_plan(bad, 4)


def test_isolated_updates_reset_adam_rng_and_weights_without_writing_checkpoints(tmp_path):
    config = feature_config(tmp_path)
    config['data']['buckets'] = [[32, 48], [48, 32]]
    system, adapter = tiny_system()
    stats = FeatureStatistics(6, 6)
    opt = torch.optim.AdamW(adapter.parameters(), lr=config['train']['lr'])
    # Populate Adam moments just as a real continuation checkpoint does.
    sum(p.square().sum() for p in adapter.parameters()).backward()
    opt.step()
    state = deepcopy(dict(adapter=adapter.state_dict(), statistics=stats.state_dict(),
                          optimizer=opt.state_dict(), rng=rng_state(), cursor=7))
    accelerator = SimpleNamespace(device=torch.device('cpu'), backward=lambda x: x.backward(),
                                  clip_grad_norm_=torch.nn.utils.clip_grad_norm_)
    records = read_manifest(config['data']['train_manifest'])
    before = sorted(str(p) for p in tmp_path.rglob('*.pt'))
    rows = []
    for effective, compute in [(16, 8), (32, 16)]:
        reset_state(adapter, stats, opt, state)
        row = run_profile(system, adapter, stats, opt, accelerator, records, config, state,
                          dict(effective_batch=effective, compute_batch=compute, workers=0),
                          dict(warmup_images=32, measured_images=64), time.monotonic() + 30)
        rows.append(row)
        assert row['end_cursor'] == 31 and row['measured_images'] == 64
        assert sum(int(key.split('x')[0]) * n for key, n in row['actual_compute_batches'].items()) == 64
        assert all(s['gradient_norm'] > 0 for s in row['steps'])
    reset_state(adapter, stats, opt, state)
    for key, value in adapter.state_dict().items():
        assert torch.equal(value, state['adapter'][key])
    for i, saved in state['optimizer']['state'].items():
        for key, value in saved.items():
            assert torch.equal(opt.state_dict()['state'][i][key], value)
    assert torch.equal(rng_state()['torch'], state['rng']['torch'])
    assert sorted(str(p) for p in tmp_path.rglob('*.pt')) == before
    assert aggregate(rows, [dict(effective_batch=16, compute_batch=8, workers=0)]) == []
