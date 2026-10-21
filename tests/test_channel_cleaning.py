from copy import deepcopy

import pytest
import torch

from radio_moge.adapters import GeometryAdapter, RadioFeatures, zero_dominant_channels


def test_cleaning_is_per_image_preserves_inputs_and_uses_mean_absolute_value():
    x = torch.ones(2, 12, 2, 2)
    x[0, 2] = torch.tensor([[20.0, -20.0], [20.0, -20.0]])
    x[1, 7] = -30
    x[:, 4] = 8  # Strict > threshold, not >=.
    original = x.clone()
    actual = zero_dominant_channels(x)
    expected = original.clone()
    expected[0, 2] = 0
    expected[1, 7] = 0
    assert torch.equal(actual, expected)
    assert torch.equal(x, original)
    assert torch.equal(actual, torch.cat([zero_dominant_channels(row[None]) for row in x]))
    assert torch.equal(zero_dominant_channels(torch.zeros_like(x)), torch.zeros_like(x))


@pytest.mark.parametrize("kind", ["linear", "mlp", "fusion"])
def test_adapter_cleaning_matches_manual_inputs_and_keeps_summary(kind):
    raw = GeometryAdapter(kind, 12, 8, 6, 6, hidden_dim=16, fusion_dim=8, indices=[0, 1])
    clean = GeometryAdapter(**raw.config, channel_cleaning="zero-channel")
    clean.load_state_dict(raw.state_dict())
    x = torch.ones(2, 12, 2, 3)
    x[0, 2] = 20
    x[1, 7] = -30
    tap = x.roll(1, dims=1)
    features = RadioFeatures(x, torch.randn(2, 8), (x, tap))
    untouched = deepcopy(features)
    manual = RadioFeatures(
        zero_dominant_channels(x),
        features.summary,
        tuple(zero_dominant_channels(v) for v in features.intermediates),
    )
    expected, actual = raw(manual), clean(features)
    torch.testing.assert_close(actual.spatial, expected.spatial)
    torch.testing.assert_close(actual.summary, raw(features).summary, rtol=0, atol=0)
    assert torch.equal(features.spatial, untouched.spatial)
    for a, b in zip(features.intermediates, untouched.intermediates):
        assert torch.equal(a, b)
    (actual.spatial.square().mean() + actual.summary.square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in clean.parameters())
    reloaded = GeometryAdapter(**clean.config)
    reloaded.load_state_dict(clean.state_dict())
    torch.testing.assert_close(reloaded(features).spatial, actual.spatial)
    assert "channel_cleaning" not in raw.config  # Legacy checkpoint config stays identical.


@pytest.mark.parametrize("threshold", [0, -1, float("nan"), float("inf")])
def test_reject_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        GeometryAdapter(channel_cleaning="zero-channel", channel_threshold=threshold)
