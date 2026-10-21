from collections import Counter

import pytest
import torch

from radio_moge.adapters import GeometryAdapter, RadioFeatures
from radio_moge.data import MixtureBatches
from radio_moge.metrics import dataset_metrics


def make_adapter(kind):
    return GeometryAdapter(
        kind,
        8,
        12,
        6,
        6,
        indices=[0, 1, 2, 3],
        spatial_width=16,
        attention_heads=4,
        summary_width=8,
        summary_heads=2,
    )


@pytest.mark.parametrize("kind", ["dpt", "attention"])
@pytest.mark.parametrize("shape", [(3, 5), (5, 3), (1, 1)])
def test_spatial_interface_checkpoint_and_gradients(kind, shape):
    torch.manual_seed(12)
    adapter = make_adapter(kind)
    maps = tuple(torch.randn(2, 8, *shape, requires_grad=True) for _ in range(4))
    features = RadioFeatures(maps[-1], torch.randn(2, 12), maps)
    output = adapter(features)
    assert output.spatial.shape == (2, 6, *shape)
    assert output.summary.shape == (2, 6)
    (output.spatial.square().mean() + output.summary.square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())
    assert all(x.grad.abs().sum() > 0 for x in maps)
    loaded = GeometryAdapter(**adapter.config)
    loaded.load_state_dict(adapter.state_dict(), strict=True)
    torch.testing.assert_close(loaded(features).spatial, output.spatial)


@pytest.mark.parametrize("kind", ["dpt", "attention"])
def test_heads_mix_spatial_locations_and_summary_reads_spatial(kind):
    torch.manual_seed(42)
    adapter = make_adapter(kind)
    maps = tuple(torch.randn(1, 8, 5, 7, requires_grad=True) for _ in range(4))
    value = adapter(RadioFeatures(maps[-1], torch.randn(1, 12), maps))
    grad = torch.autograd.grad(value.spatial[0, 0, 2, 3], maps[0], retain_graph=True)[0]
    grad[:, :, 2, 3] = 0
    assert grad.abs().sum() > 0, "Spatial output must use other locations"
    summary_grad = torch.autograd.grad(value.summary.square().sum(), maps[0])[0]
    assert summary_grad.abs().sum() > 0, "Global scale branch must read spatial features"


@pytest.mark.parametrize("kind", ["dpt", "attention"])
def test_bfloat16_autocast_spatial_backward(kind):
    adapter = make_adapter(kind)
    maps = tuple(torch.randn(1, 8, 3, 5) for _ in range(4))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = adapter(RadioFeatures(maps[-1], torch.randn(1, 12), maps))
    loss = output.spatial.float().square().mean() + output.summary.float().square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())


def test_mixture_resume_and_scene_weighting():
    # One large video must not dominate a short scene of the same source.
    records = (
        [dict(dataset="a", group="long")] * 100
        + [dict(dataset="a", group="short")]
        + [dict(dataset="b", group="b")]
    )
    args = (records, {"a": 0.6, "b": 0.4}, 16, [[32, 48], [48, 32]], 42)
    batches = list(MixtureBatches(*args, 0, 1000))
    assert list(MixtureBatches(*args, 17, 30)) == batches[17:47]
    counts = Counter(records[key[0]]["group"] for batch in batches for key in batch)
    assert counts["long"] / counts["short"] == pytest.approx(1, abs=0.1)
    assert counts["b"] / 16000 == pytest.approx(0.4, abs=0.02)
    with pytest.raises(ValueError, match="every dataset"):
        MixtureBatches(records, {"a": 1}, 1, [[32, 32]], 42, 0, 1)


def test_selection_gives_equal_weight_to_sources_and_requires_complete_scores():
    rows = [dict(dataset="a", **{"teacher_depth/abs_rel": 1.0})] * 10
    rows += [dict(dataset="b", **{"teacher_depth/abs_rel": 3.0})]
    assert dataset_metrics(rows)["macro/teacher_depth/abs_rel"] == 2
    rows[-1]["teacher_depth/abs_rel"] = None
    assert "macro/teacher_depth/abs_rel" not in dataset_metrics(rows)
