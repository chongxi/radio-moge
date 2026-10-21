import numpy as np
import pytest
import torch

from radio_moge.adapters import EncoderFeatures, GeometryAdapter, RadioFeatures
from radio_moge.checkpoint import atomic_save, restore_rng, rng_state
from radio_moge.config import load_config
from radio_moge.data import DeterministicBatches, check_split
from radio_moge.losses import FeatureStatistics, geometry_losses
from radio_moge.metrics import depth_metrics


def test_json_learning_rate_scientific_notation_is_numeric(tmp_path):
    path = tmp_path / 'refine.json'
    path.write_text('{"train": {"lr": 3e-05}}')
    assert load_config(path)['train']['lr'] == pytest.approx(0.00003)


@pytest.mark.parametrize("kind", ["linear", "mlp", "fusion"])
def test_adapter_gradients_and_shapes(kind):
    adapter = GeometryAdapter(kind, 8, 16, 6, 6, hidden_dim=12, fusion_dim=4, indices=[0, 2])
    source = RadioFeatures(
        torch.randn(2, 8, 3, 4),
        torch.randn(2, 16),
        (torch.randn(2, 8, 3, 4), torch.randn(2, 8, 3, 4)),
    )
    value = adapter(source)
    assert value.spatial.shape == (2, 6, 3, 4)
    assert value.summary.shape == (2, 6)
    (value.spatial.square().mean() + value.summary.square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())


def test_statistics_preserve_absolute_magnitude():
    target = EncoderFeatures(torch.full((2, 3, 2, 2), 10.0), torch.full((2, 3), 2.0))
    predicted = EncoderFeatures(target.spatial * 2, target.summary * 2)
    stats = FeatureStatistics(3, 3)
    stats.fit([target])
    assert stats.loss(predicted, target)["spatial"].item() == pytest.approx(1)
    assert stats.loss(predicted, target)["summary"].item() == pytest.approx(1)
    assert stats.loss(target, target)["spatial"].item() == 0
    # Cosine alone cannot detect amplitude errors seen by a frozen decoder.
    assert stats.loss(predicted, target)["cosine"].item() == pytest.approx(0, abs=1e-6)


def test_geometry_invalid_teacher_and_empty_mask_have_finite_gradients():
    for mask_value in (0.0, 1.0):
        target = {
            "points": torch.randn(1, 3, 4, 3),
            "mask": torch.full((1, 3, 4), mask_value),
            "metric_scale": torch.ones(1),
        }
        target["points"][0, 0, 0] = float("nan")
        prediction = {
            "points": torch.randn(1, 3, 4, 3, requires_grad=True),
            "mask": torch.full((1, 3, 4), 0.6, requires_grad=True),
            "metric_scale": torch.full((1,), 1.5, requires_grad=True),
        }
        losses = geometry_losses(prediction, target)
        sum(losses.values()).backward()
        assert all(torch.isfinite(v) for v in losses.values())
        assert all(torch.isfinite(value.grad).all() for value in prediction.values())


def test_resume_sampler_does_not_depend_on_prefetch():
    all_batches = list(DeterministicBatches(13, 3, [[32, 48], [48, 32]], 42, 0, 12))
    resumed = list(DeterministicBatches(13, 3, [[32, 48], [48, 32]], 42, 5, 7))
    assert resumed == all_batches[5:]


def test_split_rejects_sequence_leakage():
    with pytest.raises(ValueError, match="leakage"):
        check_split([{"image": "a", "group": "scene"}], [{"image": "b", "group": "scene"}])


def test_bad_depth_is_counted_and_scale_errors_visible():
    target = torch.ones(2, 2)
    metrics = depth_metrics(target * 2, target, target.bool())
    assert metrics["abs_rel"] == 1
    predicted = torch.full_like(target, float("nan"))
    metrics = depth_metrics(predicted, target, target.bool())
    assert metrics["invalid_fraction"] == 1
    assert metrics["delta1"] == 0


def test_rng_checkpoint_weights_only(tmp_path):
    state = rng_state()
    expected = torch.rand(3), np.random.rand(3)
    atomic_save(state, tmp_path / "rng.pt")
    restore_rng(torch.load(tmp_path / "rng.pt", weights_only=True))
    assert torch.equal(torch.rand(3), expected[0])
    assert np.array_equal(np.random.rand(3), expected[1])
