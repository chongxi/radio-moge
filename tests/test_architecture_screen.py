import importlib.util
from pathlib import Path

import pytest
import torch

from radio_moge.adapters import EncoderFeatures, RadioFeatures
from radio_moge.architecture_screen import (
    JointScreenAdapter,
    candidates,
    conditional_predict,
    flow_training,
    rope2d,
)
from radio_moge.losses import FeatureStatistics


def small_spec(**changes):
    return {
        **next(c for c in candidates() if c["name"] == "joint_4x512"),
        "width": 16,
        "heads": 2,
        "blocks": 2,
        **changes,
    }


def inputs():
    return RadioFeatures(
        torch.randn(2, 8, 2, 3),
        torch.randn(2, 12),
        tuple(torch.randn(2, 8, 2, 3) for _ in range(4)),
    )


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"position": "rope"},
        {"gate": True},
        {"registers": 4},
        {"activation": "swiglu"},
        {"local": True},
        {"position": "rope", "gate": True, "registers": 4, "activation": "swiglu"},
    ],
)
def test_architectures_shape_and_gradient(options):
    model = JointScreenAdapter(small_spec(**options), 8, 12, 6, 10)
    f = inputs()
    result = model(f)
    assert result.spatial.shape == (2, 6, 2, 3) and result.summary.shape == (2, 10)
    (result.spatial.square().mean() + result.summary.square().mean()).backward()
    for p in model.parameters():
        if p.numel():
            assert p.grad is not None and torch.isfinite(p.grad).all()
    # Summary participates in the same attention, so spatial depends on it.
    changed = model(RadioFeatures(f.spatial, f.summary + 3, f.intermediates))
    assert not torch.equal(result.spatial, changed.spatial)


def test_rope_preserves_norms_and_global_tokens():
    x = torch.randn(2, 3, 11, 8)
    result = rope2d(x, 2, 3, 5)
    torch.testing.assert_close(x.norm(dim=-1), result.norm(dim=-1))
    torch.testing.assert_close(result[..., 6:, :], x[..., 6:, :], rtol=0, atol=0)
    assert not torch.equal(result[..., 1, :], x[..., 1, :])
    with pytest.raises(ValueError):
        rope2d(x, 2, 4, 5)


def test_conditional_inference_has_no_teacher_and_counts_calls():
    stats = FeatureStatistics(6, 10)
    stats.spatial_scale.fill_(2)
    stats.summary_scale.fill_(3)
    noise = EncoderFeatures(torch.randn(2, 6, 2, 3), torch.randn(2, 10))

    class Constant(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, features, state, t):
            self.calls += 1
            return EncoderFeatures(torch.ones_like(state.spatial), torch.ones_like(state.summary))

    for count in (1, 2, 4):
        model = Constant()
        out = conditional_predict(model, inputs(), stats, noise, count)
        assert model.calls == count
        torch.testing.assert_close(out.spatial, (noise.spatial + 1) * 2)
        torch.testing.assert_close(out.summary, (noise.summary + 1) * 3)


@pytest.mark.parametrize("family", ["conditional_direct", "conditional_flow"])
def test_flow_train_has_finite_gradients(family):
    model = JointScreenAdapter(small_spec(family=family), 8, 12, 6, 10)
    stats = FeatureStatistics(6, 10)
    target = EncoderFeatures(torch.randn(2, 6, 2, 3), torch.randn(2, 10))
    noise = EncoderFeatures(torch.randn_like(target.spatial), torch.randn_like(target.summary))
    t = torch.zeros(2) if family == "conditional_direct" else torch.tensor([0.1, 0.8])
    loss = flow_training(model, inputs(), target, stats, noise, t)
    sum(loss.values()).backward()
    assert set(loss) == {"spatial", "summary", "cosine", "summary_cosine"}
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.numel()
    )


def test_group_split_and_balanced_sampling():
    path = Path(__file__).parents[1] / "scripts/prepare_architecture_screen.py"
    spec = importlib.util.spec_from_file_location("prepare_screen", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    groups = {
        str(g): [dict(image=f"{g}/{i}", group=str(g), sha256=f"h{g}-{i}") for i in range(4)]
        for g in range(8)
    }
    a, b = module.validation_split(groups, 8)
    module.verify_disjoint({"a": a, "b": b})
    assert len(a) == len(b) == 8
    assert len({r["group"] for r in a}) == len({r["group"] for r in b}) == 4
    with pytest.raises(ValueError):
        module.verify_disjoint({"a": a, "b": [a[0]]})


def test_candidate_catalog_has_single_factor_controls():
    specs = {x["name"]: x for x in candidates()}
    assert len(specs) == 16
    base = specs["joint_4x512"]
    for name, field in [
        ("joint_gate", "gate"),
        ("joint_rope", "position"),
        ("joint_register4", "registers"),
        ("joint_swiglu", "activation"),
        ("joint_local", "local"),
    ]:
        changed = {k for k, v in specs[name].items() if v != base[k]}
        assert changed == {"name", field}
