from copy import deepcopy

import pytest
import torch

from radio_moge.corpus_losses import corpus_geometry_losses, ordinary_geometry_losses
from radio_moge.geometry_loss_camera import loss_focal
from radio_moge.ordinary_geometry import ORDINARY_CAMERA_CONTRACT, ORDINARY_LOCAL_RADIUS
from radio_moge.official_gt import OFFICIAL_GT_KEYS, official_functions


def fixture(k=None, pseudo=False, h=96, w=128):
    if k is None:
        k = torch.tensor([[.8, .08, -.2], [0., 1.3, .7], [0., 0., 1.]])
    v, u = torch.meshgrid((torch.arange(h)+.5)/h, (torch.arange(w)+.5)/w, indexing='ij')
    uv = torch.stack((u, v), -1)
    rays = torch.cat((uv, torch.ones((h, w, 1))), -1) @ torch.linalg.inv(k).T
    depth = 3 + .15*torch.sin(u*13) + .1*torch.cos(v*9)
    truth = rays*depth[..., None]
    perturbation = torch.stack((.06*torch.sin(u*21), .05*torch.cos(v*17), .08*u*v), -1)
    target = dict(camera_contract=ORDINARY_CAMERA_CONTRACT,
        camera_pixel_convention='normalized_half_center', local_radius_contract=ORDINARY_LOCAL_RADIUS,
        intrinsics=k, points=truth, query_uv=uv, valid=torch.ones((h, w), dtype=torch.bool),
        native_index=torch.arange(h*w).reshape(h, w), native_shape=(h, w), output_shape=(h, w),
        supervision='relative_pseudo' if pseudo else 'metric_native',
        label_kind='teacher_pseudo' if pseudo else 'synthetic_renderer')
    raw = dict(points=(truth+perturbation)[None].requires_grad_(),
               metric_scale=torch.tensor([1.2], requires_grad=True))
    return raw, target


@pytest.fixture
def small_official(monkeypatch):
    functions = {k: (fn, dict(params)) for k, (fn, params) in official_functions().items()}
    for key in ('gt_global', 'gt_local4', 'gt_local16'):
        functions[key][1]['align_resolution'] = 8
        if key != 'gt_global':
            functions[key][1]['num_patches'] = 8
    monkeypatch.setattr('radio_moge.corpus_losses.official_functions', lambda _: functions)


@pytest.fixture
def single_thread_cpu():
    # conftest sets two threads, overriding OMP_NUM_THREADS. Overlapping local
    # patch backward can differ in last bits even for legacy-versus-legacy at
    # two threads. Make this strict arithmetic comparison actually single-thread.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_focal_exact_zero_skew_and_independent_ray_diagonal_definition():
    k = torch.tensor([[.7, 0., -.4], [0., 1.3, 1.2], [0., 0., 1.]])
    _, target = fixture(k)
    expected = (k[0, 0].reciprocal().square()+k[1, 1].reciprocal().square()).rsqrt()
    assert torch.equal(loss_focal(target, ordinary=True), expected)
    k[0, 1] = .13
    _, target = fixture(k)
    focal = loss_focal(target, ordinary=True)
    j = torch.linalg.inv(k[:2, :2])
    # Independently evaluate actual ray displacements across the two diagonals.
    diagonals = torch.tensor([[1., 1.], [1., -1.]]) @ j.T
    expected_inverse_square = diagonals.square().sum(1).mean()
    torch.testing.assert_close(focal.reciprocal().square(), expected_inverse_square)
    changed = k.clone()
    changed[:2, 2] = torch.tensor([4., -3.])
    _, other = fixture(changed)
    torch.testing.assert_close(loss_focal(other, ordinary=True), focal, rtol=0, atol=0)
    reflected = k.clone()
    reflected[0, 1] *= -1
    reflected[0, 2] = 1-k[0, 2]
    _, other = fixture(reflected)
    torch.testing.assert_close(loss_focal(other, ordinary=True), focal, rtol=0, atol=0)


@pytest.mark.upstream
def test_explicit_dispatch_rejects_even_centered_or_sparse_ordinary(small_official):
    raw, target = fixture(torch.tensor([[.8, 0., .5], [0., 1.3, .5], [0., 0., 1.]]))
    with pytest.raises(ValueError, match='centered camera'):
        corpus_geometry_losses(raw, [target])
    for key in ('camera_contract', 'camera_pixel_convention', 'local_radius_contract'):
        bad = deepcopy(target)
        bad.pop(key)
        bad['valid'].fill_(False)
        with pytest.raises(ValueError, match='explicit pinhole'):
            ordinary_geometry_losses(raw, [bad])
    bad = deepcopy(target)
    bad['query_uv'][0, 0, 0] += .01
    with pytest.raises(ValueError, match='rays differ'):
        ordinary_geometry_losses(raw, [bad])
    bad = deepcopy(target)
    bad['valid'] = bad['valid'].float()
    with pytest.raises(ValueError, match='Boolean'):
        ordinary_geometry_losses(raw, [bad])


@pytest.mark.upstream
def test_zero_skew_centered_special_case_matches_legacy_loss_gradients_rng(small_official, single_thread_cpu):
    raw, target = fixture(torch.tensor([[.8, 0., .5], [0., 1.3, .5], [0., 0., 1.]]))
    legacy = {k: v for k, v in target.items() if k not in
              ('camera_contract', 'camera_pixel_convention', 'local_radius_contract')}
    before = torch.get_rng_state().clone()
    a = corpus_geometry_losses(raw, [legacy], seed=47)
    b = ordinary_geometry_losses(raw, [target], seed=47)
    for key in OFFICIAL_GT_KEYS:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        parameter = raw['metric_scale'] if key == 'gt_scale' else raw['points']
        x = torch.autograd.grad(a[key], parameter, retain_graph=True)[0]
        y = torch.autograd.grad(b[key], parameter, retain_graph=True)[0]
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    # Require the sum as well as every component to match; do not hide a
    # possible path difference behind a floating-point tolerance.
    grad_a = torch.autograd.grad(sum(a[k] for k in OFFICIAL_GT_KEYS),
                                 (raw['points'], raw['metric_scale']))
    grad_b = torch.autograd.grad(sum(b[k] for k in OFFICIAL_GT_KEYS),
                                 (raw['points'], raw['metric_scale']))
    for x, y in zip(grad_a, grad_b):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.upstream
def test_offcenter_skew_has_nonzero_finite_gradient_for_each_upstream_component(small_official):
    raw, target = fixture()
    before = torch.get_rng_state().clone()
    losses = ordinary_geometry_losses(raw, [target], seed=43)
    for key in OFFICIAL_GT_KEYS:
        parameter = raw['metric_scale'] if key == 'gt_scale' else raw['points']
        gradient = torch.autograd.grad(losses[key], parameter, retain_graph=True)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0, key
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.upstream
def test_relative_pseudo_scale_invariance_and_no_metric_scale_graph(small_official):
    raw, target = fixture(pseudo=True)
    a = ordinary_geometry_losses(raw, [target], seed=43)
    target['points'] *= 8
    b = ordinary_geometry_losses(raw, [target], seed=43)
    for key in OFFICIAL_GT_KEYS:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    sum(b[k] for k in OFFICIAL_GT_KEYS).backward()
    assert raw['metric_scale'].grad is None and b['gt_metric_images'] == 0
    assert b['gt_scale'] == 0 and torch.isfinite(raw['points'].grad).all()


@pytest.mark.upstream
def test_missing_sparse_zero_and_full_group_denominator(small_official):
    raw, target = fixture()
    a = ordinary_geometry_losses(raw, [target], seed=43)
    sparse = deepcopy(target)
    sparse['valid'].fill_(False)
    sparse['valid'][0, 0] = True
    sparse['points'][~sparse['valid']] = float('nan')
    sparse['query_uv'][~sparse['valid']] = float('inf')
    points = torch.cat((raw['points'].detach(), torch.full_like(raw['points'], float('nan')),
                        torch.full_like(raw['points'], float('nan')))).requires_grad_()
    scale = torch.tensor([1.2, float('nan'), float('nan')], requires_grad=True)
    b = ordinary_geometry_losses(dict(points=points, metric_scale=scale), [target, None, sparse], seed=43)
    for key in OFFICIAL_GT_KEYS:
        torch.testing.assert_close(a[key]/3, b[key], rtol=0, atol=0)
    assert (b['gt_supervised_images'], b['gt_missing_target_images'], b['gt_skipped_sparse_images']) == (1, 1, 1)
    assert b['gt_missing_target_images'].dtype == torch.int64
    sum(b[k] for k in OFFICIAL_GT_KEYS).backward()
    assert torch.isfinite(points.grad).all() and (points.grad[1:] == 0).all()
    assert torch.isfinite(scale.grad).all() and (scale.grad[1:] == 0).all()
    c = ordinary_geometry_losses(dict(points=points, metric_scale=scale), [None]*3)
    assert all(c[k] == 0 for k in OFFICIAL_GT_KEYS) and c['gt_missing_target_images'] == 3
    assert c['gt_skipped_sparse_images'] == 0


@pytest.mark.upstream
def test_invalid_storage_outside_valid_query_footprints_has_zero_gradient(small_official):
    # Binary-exact center queries isolate true invalid storage from the bilinear
    # footprint of a valid subpixel query. Neighboring pixels in that footprint
    # cannot be treated as irrelevant simply because their target cell is empty.
    raw, target = fixture(h=128, w=128)
    target['valid'][:20] = False
    a = ordinary_geometry_losses(raw, [target], seed=43)
    grad_a = torch.autograd.grad(sum(a[k] for k in OFFICIAL_GT_KEYS), raw['points'])[0]
    target['points'][~target['valid']] = float('nan')
    target['query_uv'][~target['valid']] = float('inf')
    raw['points'] = raw['points'].detach().clone().requires_grad_()
    with torch.no_grad():
        raw['points'][0, :20] = 1e8
    b = ordinary_geometry_losses(raw, [target], seed=43)
    grad_b = torch.autograd.grad(sum(b[k] for k in OFFICIAL_GT_KEYS), raw['points'])[0]
    for key in OFFICIAL_GT_KEYS:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    torch.testing.assert_close(grad_a, grad_b, rtol=0, atol=0)


@pytest.mark.upstream
def test_sparse_invalid_rays_and_native_ids_are_rejected_before_skip(small_official):
    raw, target = fixture()
    target['valid'].fill_(False)
    target['valid'][0, 0] = True
    target['query_uv'][0, 0] = torch.tensor([-.3, .5])
    ray = torch.tensor([-.3, .5, 1.]) @ torch.linalg.inv(target['intrinsics']).T
    target['points'][0, 0] = ray*2
    with pytest.raises(ValueError, match='extrapolation'):
        ordinary_geometry_losses(raw, [target])
    raw, target = fixture()
    target['valid'].fill_(False)
    target['valid'][0, 0] = True
    target['native_index'][0, 0] = 10**9
    with pytest.raises(ValueError, match='native grid'):
        ordinary_geometry_losses(raw, [target])


def test_unrepresentable_focal_rejected_even_zero_skew():
    _, target = fixture()
    target['valid'].fill_(False)
    target['intrinsics'] = torch.diag(torch.tensor([1e30, 1e30, 1.]))
    with pytest.raises(ValueError, match='local-radius'):
        loss_focal(target, ordinary=True)


@pytest.mark.upstream
def test_outer_resolution_group_weights_keep_whole_update_denominator_32(small_official):
    raw, target = fixture()
    point = raw['points'].detach()
    points = point.repeat(32, 1, 1, 1).requires_grad_()
    scales = torch.full((32,), 1.2, requires_grad=True)
    targets = [None]*32
    targets[14] = target
    a = ordinary_geometry_losses(dict(points=points, metric_scale=scales), targets, seed=43)
    # The original preparer may split the update into different resolution
    # groups. Both sums count feature-only slots, including a fully empty group.
    groups = [(0, 12), (12, 32)]
    values = [ordinary_geometry_losses(dict(points=points[left:right], metric_scale=scales[left:right]),
                targets[left:right], seed=43) for left, right in groups]
    b = {key: sum(value[key]*(right-left)/32 for value, (left, right) in zip(values, groups))
         for key in OFFICIAL_GT_KEYS}
    for key in OFFICIAL_GT_KEYS:
        torch.testing.assert_close(a[key], b[key], rtol=1e-6, atol=1e-10)
    ga = torch.autograd.grad(sum(a[k] for k in OFFICIAL_GT_KEYS), points)[0]
    gb = torch.autograd.grad(sum(b[k] for k in OFFICIAL_GT_KEYS), points)[0]
    torch.testing.assert_close(ga, gb, rtol=1e-6, atol=1e-10)
    assert ga[14].abs().sum() > 0 and (ga[:14] == 0).all() and (ga[15:] == 0).all()
    assert sum(int(v['gt_missing_target_images']) for v in values) == 31
