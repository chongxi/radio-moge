"""No optimizer steps: exercise the loss boundary on deliberately corrupted holes."""
from copy import deepcopy

import pytest
import torch

from radio_moge.gt_geometry import GT_LOSS_KEYS, ground_truth_losses
from radio_moge.official_gt import (OFFICIAL_GT_KEYS, official_functions,
                                    official_ground_truth_losses)
from test_gt_geometry import plane_target, raw_for


def masked_fixture():
    target = plane_target()
    raw = raw_for(target)
    with torch.no_grad():
        raw['points'][..., 2] += .2 * torch.linspace(-1, 1, 48).square()
    target['valid'][:, 9:23, 13:35] = False
    target['depth'][~target['valid']] = float('nan')
    return raw, target


def assert_mask_contract(function, keys):
    raw, target = masked_fixture()
    result = function(raw, target)
    assert all(torch.isfinite(result[key]) for key in keys)
    first = sum(result[key] for key in keys)
    first.backward()
    before_gradient = raw['points'].grad.clone()
    assert torch.isfinite(before_gradient).all()
    assert before_gradient[target['valid']].abs().sum() > 0
    assert before_gradient[~target['valid']].abs().max() == 0
    changed, target_changed = deepcopy(raw), deepcopy(target)
    changed['points'].grad = None
    with torch.no_grad():
        changed['points'][~target['valid']] = torch.tensor([10000., -20000., 30000.])
        target_changed['depth'][~target['valid']] = float('inf')
        changed['mask'].fill_(1)
    repeat = function(changed, target_changed)
    for key in keys:
        torch.testing.assert_close(result[key], repeat[key], rtol=0, atol=0)
    sum(repeat[key] for key in keys).backward()
    torch.testing.assert_close(before_gradient, changed['points'].grad, rtol=0, atol=0)
    target_changed['valid'].fill_(False)
    with pytest.raises(ValueError, match='Every GT image'):
        function(changed, target_changed)


def test_custom_loss_has_no_invalid_pixel_loss_or_gradient():
    assert_mask_contract(ground_truth_losses, GT_LOSS_KEYS)


@pytest.mark.upstream
def test_official_core_has_no_invalid_pixel_loss_or_gradient(monkeypatch):
    actual = official_functions()
    bounded = {key: (fn, dict(params)) for key, (fn, params) in actual.items()}
    for key in ('gt_global', 'gt_local4', 'gt_local16'):
        bounded[key][1]['align_resolution'] = 4
        if key != 'gt_global':
            bounded[key][1]['num_patches'] = 2
    monkeypatch.setattr('radio_moge.official_gt.official_functions', lambda _: bounded)
    assert_mask_contract(lambda raw, target: official_ground_truth_losses(raw, target, seed=43),
                         OFFICIAL_GT_KEYS)
