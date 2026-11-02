from copy import deepcopy

import pytest
import torch

from radio_moge.decoder import camera_points
from radio_moge.official_gt import (OFFICIAL_GT_KEYS, OFFICIAL_GT_WEIGHTS,
                                    official_functions, official_ground_truth_losses)
from radio_moge.train import validate_config
from test_gt_geometry import gt_config, plane_target, raw_for


def test_official_config_does_not_silently_mix_objectives(tmp_path):
    config = gt_config(tmp_path)
    config['geometry_supervision']['objective'] = 'moge2_core'
    config['loss'] = {k: .1 for k in ('spatial', 'summary', 'cosine', 'summary_cosine')}
    config['loss'].update(OFFICIAL_GT_WEIGHTS)
    validate_config(config)
    bad = deepcopy(config)
    bad['loss']['gt_depth'] = 1
    with pytest.raises(ValueError, match='must not be mixed'):
        validate_config(bad)
    bad = deepcopy(config)
    bad['loss']['gt_scale'] = 1
    with pytest.raises(ValueError, match='pinned official'):
        validate_config(bad)


@pytest.mark.upstream
def test_official_core_parity_scale_decoupling_and_rng(monkeypatch):
    # Smaller alignment sample for a bounded CPU unit fixture. Training always
    # reads the unmodified 48/24/12 official settings from the pinned config.
    actual = official_functions()
    small = {key: (fn, dict(params)) for key, (fn, params) in actual.items()}
    for key in ('gt_global', 'gt_local4', 'gt_local16'):
        small[key][1]['align_resolution'] = 4
        if key != 'gt_global':
            small[key][1]['num_patches'] = 2
    monkeypatch.setattr('radio_moge.official_gt.official_functions', lambda _: small)
    target = plane_target()
    raw = raw_for(target, scale=2., shift=.7)
    raw['metric_scale'] = torch.tensor([3.], requires_grad=True)
    before = torch.get_rng_state().clone()
    result = official_ground_truth_losses(raw, target, seed=19)
    assert torch.equal(before, torch.get_rng_state())
    assert result['gt_global'].abs() < 1e-6
    assert result['gt_scale'].item() == pytest.approx(torch.log(torch.tensor(1.5)).square().item())
    assert torch.autograd.grad(result['gt_scale'], raw['points'], allow_unused=True,
                               retain_graph=True)[0] is None
    assert all(torch.isfinite(result[key]) for key in OFFICIAL_GT_KEYS)
    truth = camera_points(target['depth'], target['intrinsics'])
    fn, params = small['gt_global']
    expected = fn(raw['points'][0], truth[0], target['valid'][0], **params)[0]
    torch.testing.assert_close(result['gt_global'], expected)
    raw['mask'].fill_(1)
    repeat = official_ground_truth_losses(raw, target, seed=19)
    for key in OFFICIAL_GT_KEYS:
        torch.testing.assert_close(result[key], repeat[key], rtol=0, atol=0)
