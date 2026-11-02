from copy import deepcopy

import pytest
import torch

from radio_moge.corpus_losses import (corpus_geometry_losses, sample_predictions,
                                      stable_edge_support)
from radio_moge.decoder import camera_points
from radio_moge.official_gt import OFFICIAL_GT_KEYS, official_functions


def fixture(mode='metric_native', device='cpu'):
    # Binary-exact center UVs make this a test of missing-label masking, not
    # grid_sample's non-binary subpixel rounding. Separate test covers subpixels.
    h, w = 32, 32
    k = torch.tensor([[.75, 0, .5], [0, 1., .5], [0, 0, 1.]], device=device)
    v, u = torch.meshgrid((torch.arange(h, device=device) + .5) / h,
                          (torch.arange(w, device=device) + .5) / w, indexing='ij')
    depth = 3 + .2 * u + .1 * u.square()
    points = camera_points(depth[None], k[None])[0]
    target = dict(points=points, query_uv=torch.stack((u, v), -1),
                  valid=torch.ones((h, w), dtype=torch.bool, device=device),
                  intrinsics=k, native_index=torch.arange(h * w, device=device).reshape(h, w),
                  output_shape=(h, w), supervision=mode,
                  label_kind='teacher_pseudo' if mode == 'relative_pseudo' else 'measured_sensor')
    raw = dict(points=(points + torch.stack((u * 0, u * 0, .05 * u.square()), -1))[None].requires_grad_(),
               metric_scale=torch.tensor([1.2], device=device, requires_grad=True),
               mask=torch.zeros((1, h, w), device=device))
    return raw, target


@pytest.fixture
def small_official(monkeypatch):
    actual = official_functions()
    small = {key: (fn, dict(params)) for key, (fn, params) in actual.items()}
    for key in ('gt_global', 'gt_local4', 'gt_local16'):
        small[key][1]['align_resolution'] = 4
        if key != 'gt_global':
            small[key][1]['num_patches'] = 2
    monkeypatch.setattr('radio_moge.corpus_losses.official_functions', lambda _: small)


@pytest.mark.upstream
def test_invalid_targets_predictions_and_predicted_mask_do_not_change_loss_or_gradient(small_official):
    raw, target = fixture()
    target['valid'][:6] = False
    target['valid'][12:15, 8:12] = False
    before = torch.get_rng_state().clone()
    a = corpus_geometry_losses(raw, [target], seed=37)
    sum(a[k] for k in OFFICIAL_GT_KEYS).backward()
    grad = raw['points'].grad.clone()
    assert torch.equal(before, torch.get_rng_state())
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert torch.all(grad[0][~target['valid']] == 0)
    raw2, target2 = deepcopy(raw), deepcopy(target)
    raw2['points'] = raw['points'].detach().clone().requires_grad_()
    raw2['mask'].fill_(1)
    target2['points'][~target2['valid']] = float('nan')
    target2['query_uv'][~target2['valid']] = float('inf')
    with torch.no_grad():
        raw2['points'][0][~target2['valid']] = 1e8
    b = corpus_geometry_losses(raw2, [target2], seed=37)
    sum(b[k] for k in OFFICIAL_GT_KEYS).backward()
    for key in OFFICIAL_GT_KEYS:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    torch.testing.assert_close(grad, raw2['points'].grad, rtol=0, atol=0)


@pytest.mark.upstream
def test_pseudo_has_no_scale_gradient_and_ignores_arbitrary_label_scale(small_official):
    raw, target = fixture(mode='relative_pseudo')
    a = corpus_geometry_losses(raw, [target], seed=13)
    assert a['gt_scale'] == 0 and a['gt_metric_images'] == 0
    sum(a[k] for k in OFFICIAL_GT_KEYS).backward()
    assert raw['metric_scale'].grad is None
    assert torch.isfinite(raw['points'].grad).all()
    target['points'] *= 8
    b = corpus_geometry_losses(raw, [target], seed=13)
    for key in OFFICIAL_GT_KEYS:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


@pytest.mark.upstream
def test_native_scale_uses_upstream_detached_optimal_scale(small_official):
    raw, target = fixture()
    result = corpus_geometry_losses(raw, [target], seed=7)
    assert result['gt_scale'] > 0 and result['gt_metric_images'] == 1
    result['gt_scale'].backward()
    assert raw['points'].grad is None
    assert torch.isfinite(raw['metric_scale'].grad).all()


@pytest.mark.upstream
@pytest.mark.parametrize('n', [0, 1, 31])
def test_empty_or_insufficient_support_is_explicit_finite_zero(small_official, n):
    raw, target = fixture()
    target['valid'].fill_(False)
    target['valid'].reshape(-1)[:n] = True
    target['points'][~target['valid']] = float('nan')
    result = corpus_geometry_losses(raw, [target])
    assert result['gt_skipped_sparse_images'] == 1
    sum(result[k] for k in OFFICIAL_GT_KEYS).backward()
    assert torch.all(raw['points'].grad == 0)
    assert all(result[k] == 0 for k in OFFICIAL_GT_KEYS)


def test_edges_never_cross_depth_steps_or_holes():
    _, target = fixture()
    target['points'][:, 16:, 2] += 3
    target['valid'][4:8, 4:8] = False
    edge = stable_edge_support(target['points'], target['valid'])
    assert not edge[:, 15:17].any()
    assert not edge[4:8, 4:8].any()
    assert edge[10, 10]


def test_queries_retain_subpixel_location_and_no_extrapolation():
    raw, target = fixture()
    target['query_uv'][5, 5, 0] += .2 / 32
    predicted = sample_predictions(raw['points'][0], target)
    expected = .8 * raw['points'][0, 5, 5] + .2 * raw['points'][0, 5, 6]
    torch.testing.assert_close(predicted[5, 5], expected, atol=3e-7, rtol=1e-6)
    target['query_uv'][5, 5] = 0
    with pytest.raises(ValueError, match='extrapolation'):
        sample_predictions(raw['points'][0], target)


@pytest.mark.upstream
def test_native_id_duplication_and_pseudo_metric_routing_rejected(small_official):
    raw, target = fixture()
    target['native_index'][1, 1] = target['native_index'][0, 0]
    with pytest.raises(ValueError, match='Repeated'):
        corpus_geometry_losses(raw, [target])
    target['label_kind'] = 'teacher_pseudo'
    with pytest.raises(ValueError, match='routing'):
        corpus_geometry_losses(raw, [target])
