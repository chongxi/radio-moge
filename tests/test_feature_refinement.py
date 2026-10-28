import pytest
import torch

from radio_moge.adapters import EncoderFeatures, GeometryAdapter, RadioFeatures
from radio_moge.feature_refinement import FeatureRefinement


MODES = ['pointwise', 'multiscale', 'joint_attention']


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('shape', [(3, 5), (1, 1)])
def test_identity_initialization_and_roundtrip(mode, shape):
    torch.manual_seed(42)
    base = GeometryAdapter('attention', 8, 12, 6, 6, indices=[0, 1, 2, 3],
                           spatial_width=16, attention_heads=4, summary_width=8, summary_heads=2)
    extended = GeometryAdapter(**base.config, refinement=mode, refinement_width=16,
                               refinement_heads=4)
    missing, unexpected = extended.load_state_dict(base.state_dict(), strict=False)
    assert missing and all(k.startswith('refinement.') for k in missing) and not unexpected
    maps = tuple(torch.randn(2, 8, *shape) for _ in range(4))
    inputs = RadioFeatures(maps[-1], torch.randn(2, 12), maps)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        before, after = base(inputs), extended(inputs)
    for key in ('spatial', 'summary'):
        torch.testing.assert_close(getattr(before, key), getattr(after, key), rtol=0, atol=0)
    loaded = GeometryAdapter(**extended.config)
    loaded.load_state_dict(extended.state_dict(), strict=True)
    torch.testing.assert_close(loaded(inputs).spatial, extended(inputs).spatial, rtol=0, atol=0)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('activation', ['gelu', 'swiglu'])
def test_zero_head_receives_gradient_then_internal_prior_learns(mode, activation):
    torch.manual_seed(1)
    module = FeatureRefinement(mode, 8, 6, width=16, heads=4, activation=activation)
    optimizer = torch.optim.SGD(module.parameters(), lr=.01)
    inputs = EncoderFeatures(torch.randn(2, 8, 3, 5), torch.randn(2, 6))
    for step in range(2):
        optimizer.zero_grad()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            spatial, summary = module(inputs.spatial, inputs.summary)
        loss = spatial.float().square().mean() + summary.float().square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert module.spatial_out.weight.grad.abs().sum() > 0
        assert module.summary_out.weight.grad.abs().sum() > 0
        if step == 1:
            assert module.spatial_in.weight.grad.abs().sum() > 0
            assert next(module.blocks[0].ffn.parameters()).grad.abs().sum() > 0
        optimizer.step()
    assert not torch.equal(spatial, inputs.spatial)


@pytest.mark.parametrize('mode', MODES)
def test_prior_receptive_field_and_summary_feedback(mode):
    torch.manual_seed(2)
    module = FeatureRefinement(mode, 8, 6, width=16, heads=4)
    torch.nn.init.normal_(module.spatial_out.weight, std=.05)
    spatial = torch.randn(1, 8, 5, 7, requires_grad=True)
    summary = torch.randn(1, 6, requires_grad=True)
    result, _ = module(spatial, summary)
    gs, gc = torch.autograd.grad(result[0, 0, 2, 3], (spatial, summary), allow_unused=True)
    gs[:, :, 2, 3] = 0
    if mode == 'pointwise':
        assert gs.abs().sum() == 0 and gc.abs().sum() == 0
    else:
        assert gs.abs().sum() > 0
    if mode == 'joint_attention':
        assert gc.abs().sum() > 0
    else:
        assert gc.abs().sum() == 0


def test_default_keeps_legacy_adapter_config_and_state_names():
    adapter = GeometryAdapter('linear', 8, 12, 6, 6)
    assert 'refinement' not in adapter.config
    assert not any(key.startswith('refinement.') for key in adapter.state_dict())
    with pytest.raises(ValueError):
        GeometryAdapter(refinement='unregistered')
