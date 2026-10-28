import importlib.util
import json
from pathlib import Path

import pytest
import torch

from radio_moge.adapters import EncoderFeatures
from radio_moge.losses import FeatureStatistics, feature_loss_total, weighted_loss


def test_summary_direction_is_scale_invariant_but_mse_preserves_amplitude():
    target = EncoderFeatures(torch.tensor([[[[1.]], [[0.]]]]), torch.tensor([[1., 0.]]))
    statistics = FeatureStatistics(2, 2)
    prediction = EncoderFeatures(target.spatial * 2, target.summary * 2)
    loss = statistics.loss(prediction, target)
    assert loss['cosine'].item() == pytest.approx(0)
    assert loss['summary_cosine'].item() == pytest.approx(0)
    assert loss['spatial'] > 0 and loss['summary'] > 0
    wrong = statistics.loss(EncoderFeatures(target.spatial, -target.summary), target)
    assert wrong['summary_cosine'].item() == pytest.approx(2)


def test_summary_angular_gradient_rotates_toward_teacher_and_old_objective_unchanged():
    target = EncoderFeatures(torch.randn(2, 3, 2, 2), torch.tensor([[1., 0., 0.]] * 2))
    summary = torch.tensor([[0., 1., 0.]] * 2, requires_grad=True)
    statistics = FeatureStatistics(3, 3)
    loss = statistics.loss(EncoderFeatures(target.spatial, summary), target)
    grad, = torch.autograd.grad(loss['summary_cosine'], summary, retain_graph=True)
    after = statistics.loss(EncoderFeatures(target.spatial, summary - .1 * grad), target)
    assert after['summary_cosine'] < loss['summary_cosine']
    weights = dict(spatial=1., summary=1., cosine=.1)
    old = loss['spatial'] + loss['summary'] + .1 * loss['cosine']
    torch.testing.assert_close(weighted_loss(loss, weights), old, rtol=0, atol=0)
    metrics = {f'feature/{k}': v.item() for k, v in loss.items()}
    weights.update(cosine=1., summary_cosine=1.)
    assert feature_loss_total(metrics, weights) == pytest.approx(weighted_loss(loss, weights).item())
    del metrics['feature/summary_cosine']
    with pytest.raises(KeyError):
        feature_loss_total(metrics, weights)


def test_paired_protocol_changes_objective_only(tmp_path):
    spec = importlib.util.spec_from_file_location('dual', Path(__file__).parents[1] / 'scripts/run_dual_cosine_pilot.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from radio_moge.config import load_config
    cfg = load_config(Path(__file__).parents[1] / 'configs/attention-full-day.json')
    old = tmp_path / 'adapter-joint-cosine512-pilot-20261028'
    old.mkdir()
    (old / 'completion-verification.json').write_text('{"stage":"passed"}')
    (old / 'control-config.json').write_text(json.dumps(cfg))
    for name in ('train', 'validation', 'train-probe', 'screenshot'):
        (old / f'{name}.jsonl').write_text('{"id":"fixture"}\n')
    original = tmp_path / 'attention-24h'
    original.mkdir()
    (original / 'config.json').write_text(json.dumps(cfg))
    root = tmp_path / 'new'
    plan = module.prepare(tmp_path, root, 'test')
    configs = [json.loads((root / f'{a}-config.json').read_text()) for a in module.ARMS]
    assert configs[0]['loss']['cosine'] == .1 and configs[0]['loss']['summary_cosine'] == 0
    assert configs[1]['loss']['cosine'] == configs[1]['loss']['summary_cosine'] == 1
    for config in configs:
        assert config['train']['steps'] == 2000 and config['train']['max_seconds'] == 1200
        assert config['loss']['spatial'] == config['loss']['summary'] == 1
        for key in ('loss', 'output_dir', 'tracking'):
            config.pop(key)
    assert configs[0] == configs[1]
    assert plan['target_cosine'] == .98 and plan['evaluation_steps'] == list(range(0, 2001, 200))
    with pytest.raises(FileExistsError):
        module.prepare(tmp_path, root, 'test')
