from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from radio_moge.checkpoint import atomic_save
from radio_moge.selection import selection_winners, validate_selections
from radio_moge.train import train
from test_performance import feature_config, profile
from test_training import tiny_system


def test_constant_fork_inherits_prepared_batch_but_not_old_caps(tmp_path, monkeypatch):
    cfg = feature_config(tmp_path)
    cfg['output_dir'] = str(tmp_path/'parent')
    checkpoint = train(cfg, stop_after=2, system_adapter=tiny_system())
    state = torch.load(checkpoint, weights_only=True)
    state['performance'] = dict(profile(2, 8, 32, 16), original_started_at=1.,
                               compact_transport=True, data_pipeline='prepared')
    state['runtime_limits'] = dict(steps=2, max_seconds=1)
    atomic_save(state, checkpoint)
    parent_bytes = checkpoint.read_bytes()
    calls = []

    def evaluate(system, adapter, statistics, validation, config, destination, limit):
        step = int(destination.name.split('-')[-1])
        calls.append(step)
        return {'feature/spatial': 1., 'feature/summary': 1., 'feature/cosine': .1,
                'feature/summary_cosine': .1,
                'teacher_depth/abs_rel': 1./step, 'spatial': step/10,
                'postprocess_failed': 0}

    monkeypatch.setattr('radio_moge.train.evaluate', evaluate)
    fork = deepcopy(cfg)
    fork['train'].update(steps=6, lr=1.5e-5, lr_schedule='constant', max_seconds=600,
                         eval_every=2, save_every=2)
    fork['evaluation']['checkpoint_metrics'] = {
        'spatial': dict(metric='spatial', mode='max', baseline=.2),
        'depth': dict(metric='teacher_depth/abs_rel', mode='min', baseline=.5)}
    fork['output_dir'] = str(tmp_path/'full')
    full_path = train(fork, fork_from=checkpoint, system_adapter=tiny_system())
    full = torch.load(full_path, weights_only=True)
    initial = torch.load(tmp_path/'full/initial.pt', weights_only=True)
    assert calls == [4, 6]
    assert initial['step'] == 2 and initial['cursor'] == 8
    for key, row in state['optimizer']['state'].items():
        for name, value in row.items():
            torch.testing.assert_close(initial['optimizer']['state'][key][name], value, rtol=0, atol=0)
    for section in ('adapter', 'statistics'):
        for key, value in state[section].items():
            torch.testing.assert_close(initial[section][key], value, rtol=0, atol=0)
    torch.testing.assert_close(initial['rng']['torch'], state['rng']['torch'], rtol=0, atol=0)
    assert full['step'] == 6 and full['cursor'] == 40
    assert full['runtime_limits'] == dict(steps=6, max_seconds=600)
    assert all(row['step'].item() == 6 for row in full['optimizer']['state'].values())
    assert all(json.loads(row)['lr_used'] == 1.5e-5
               for row in (tmp_path/'full/train.jsonl').read_text().splitlines())
    assert full['best_metrics']['spatial']['step'] == 6
    selected = torch.load(tmp_path/'full/best-spatial.pt', weights_only=True)
    assert selected['step'] == 6
    assert (tmp_path/'full/best-spatial.pt').stat().st_ino == (tmp_path/'full/best-depth.pt').stat().st_ino
    fork['output_dir'] = str(tmp_path/'split')
    part = train(fork, fork_from=checkpoint, stop_after=4, system_adapter=tiny_system())
    resumed = torch.load(train(fork, resume=part, system_adapter=tiny_system()), weights_only=True)
    assert resumed['cursor'] == full['cursor']
    assert resumed['best_metrics'] == full['best_metrics']
    for key in full['adapter']:
        torch.testing.assert_close(full['adapter'][key], resumed['adapter'][key], rtol=0, atol=0)
    assert checkpoint.read_bytes() == parent_bytes


def test_selections_ignore_failed_and_nonfinite_evaluations():
    selection = dict(depth=dict(metric='depth', mode='min', baseline=.2))
    validate_selections(selection)
    best = dict(depth=dict(value=.2, step=1))
    for metrics in [dict(depth=.1, postprocess_failed=1),
                    dict(depth=float('nan'), postprocess_failed=0),
                    dict(depth=.3, postprocess_failed=0)]:
        assert selection_winners(selection, best, metrics, 2) == []
    assert best['depth']['step'] == 1
    with pytest.raises(ValueError, match='name'):
        validate_selections({'../escape': selection['depth']})


def test_continuation_plan_rejects_unbounded_or_wrong_stage():
    import importlib.util
    spec = importlib.util.spec_from_file_location('continuation', Path(__file__).parents[1]/
                                                'scripts/run_joint_full_continuation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = dict(authorized=True, additional_updates=30000, parent_step=61367,
                parent_cursor=479996, remote_root='/new-stage')
    config = dict(output_dir='/new-stage/train', train=dict(steps=91367, lr_schedule='constant',
                  lr=1.5e-5, max_seconds=28800, eval_every=1000, save_every=500))
    module.validate_plan(plan, config)
    for key, value in [('steps', 1000000), ('lr', .1), ('max_seconds', 0)]:
        changed = deepcopy(config)
        changed['train'][key] = value
        with pytest.raises(ValueError):
            module.validate_plan(plan, changed)
