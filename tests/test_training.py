from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from radio_moge.adapters import GeometryAdapter
from radio_moge.config import load_config, ROOT
from radio_moge.system import FrozenSystem
from radio_moge.train import train


class TinyRadio(nn.Module):
    patch_size = min_resolution_step = 16

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 16, stride=16)

    def forward(self, image, feature_fmt):
        x = self.conv(image)
        return SimpleNamespace(features=x, summary=x.mean((2, 3)))

    def forward_intermediates(self, image, indices, **kwargs):
        final = self.forward(image, "NCHW")
        return final, tuple(final.features * (1 + i / 10) for i in indices)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, 1)

    def forward(self, image, h, w, return_class_token):
        x = self.conv(F.interpolate(image, (h, w), mode="bilinear", align_corners=False))
        return x, x.mean((2, 3))


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(6, 4, 1)
        self.scale = nn.Linear(6, 1)

    def forward(self, features, size):
        value = F.interpolate(
            self.conv(features.spatial), size, mode="bilinear", align_corners=False
        )
        points = value[:, :3].permute(0, 2, 3, 1)
        return {
            "points": points,
            "mask": value[:, 3].sigmoid(),
            "metric_scale": self.scale(features.summary).squeeze(1).exp(),
        }


def tiny_system():
    torch.manual_seed(100)
    modules = [TinyRadio(), TinyEncoder(), TinyDecoder()]
    for module in modules:
        module.eval().requires_grad_(False)
    adapter = GeometryAdapter("mlp", 8, 8, 6, 6, hidden_dim=16)
    return FrozenSystem(*modules, model_config={"tiny": True}), adapter


def config_for(tmp_path):
    config = load_config(ROOT / "configs/mlp.yaml")
    config["data"].update(buckets=[[32, 48]], workers=0, augment=True)
    config["train"].update(
        device="cpu",
        precision="no",
        steps=12,
        batch_size=2,
        accumulation=2,
        lr=0.01,
        warmup=2,
        calibration_batches=2,
        log_every=1,
        save_every=4,
        eval_every=100,
        tensorboard=False,
    )
    config["loss"].update(points=0.5, mask=0.1, scale=0.1, gradient=0.05)
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        rows = []
        for i in range(6):
            path = tmp_path / f"{split}-{i}.png"
            Image.fromarray(rng.integers(0, 256, (40, 60, 3), dtype=np.uint8)).save(path)
            rows.append({"image": str(path), "group": split})
        manifest = tmp_path / f"{split}.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        config["data"][f"{split}_manifest"] = str(manifest)
    return config


def test_training_overfits_and_resume_is_identical(tmp_path):
    config = config_for(tmp_path)
    config["output_dir"] = str(tmp_path / "full")
    full_path = train(config, system_adapter=tiny_system())
    initial = torch.load(tmp_path / "full/initial.pt", weights_only=True)
    assert initial["step"] == initial["cursor"] == 0
    _, fresh_adapter = tiny_system()
    for key, value in fresh_adapter.state_dict().items():
        torch.testing.assert_close(value, initial["adapter"][key], rtol=0, atol=0)
    split_config = deepcopy(config)
    split_config["output_dir"] = str(tmp_path / "split")
    partial = train(split_config, stop_after=5, system_adapter=tiny_system())
    resumed_path = train(split_config, resume=partial, system_adapter=tiny_system())
    split_initial = torch.load(tmp_path / "split/initial.pt", weights_only=True)
    for key, value in initial["adapter"].items():
        torch.testing.assert_close(value, split_initial["adapter"][key], rtol=0, atol=0)
    full = torch.load(full_path, weights_only=True)
    resumed = torch.load(resumed_path, weights_only=True)
    assert full["step"] == resumed["step"] == 12
    assert full["cursor"] == resumed["cursor"] == 24
    for key, value in full["adapter"].items():
        torch.testing.assert_close(value, resumed["adapter"][key], rtol=0, atol=0)
    metrics = [
        json.loads(line) for line in (tmp_path / "full/train.jsonl").read_text().splitlines()
    ]
    assert metrics[-1]["total"] < metrics[0]["total"] * 0.7


def test_resume_rejects_changed_data(tmp_path):
    config = config_for(tmp_path)
    config["output_dir"] = str(tmp_path / "run")
    state = train(config, stop_after=1, system_adapter=tiny_system())
    config["train"]["lr"] *= 2
    with pytest.raises(ValueError, match="identical config"):
        train(config, resume=state, system_adapter=tiny_system())


def test_wall_clock_budget_survives_resume_and_saves_complete_update(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("radio_moge.train.time.monotonic", lambda: clock[0])
    original_step = torch.optim.AdamW.step

    def timed_step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        clock[0] += 1.0
        return result

    monkeypatch.setattr(torch.optim.AdamW, "step", timed_step)
    config = config_for(tmp_path)
    config["data"]["val_manifest"] = None
    config["train"]["max_seconds"] = 5.0
    config["output_dir"] = str(tmp_path / "clock-full")
    full = torch.load(train(config, system_adapter=tiny_system()), weights_only=True)
    assert full["step"] == 5 and full["cursor"] == 10

    config["output_dir"] = str(tmp_path / "clock-resumed")
    checkpoint = train(config, stop_after=2, system_adapter=tiny_system())
    assert torch.load(checkpoint, weights_only=True)["wall_clock_elapsed_seconds"] == 2.0
    clock[0] += 1000.0  # Paused time is excluded; the original five-second budget is retained.
    resumed = torch.load(
        train(config, resume=checkpoint, system_adapter=tiny_system()), weights_only=True
    )
    assert resumed["step"] == 5 and resumed["cursor"] == 10
    assert resumed["wall_clock_elapsed_seconds"] == 5.0
    for key in full["adapter"]:
        torch.testing.assert_close(full["adapter"][key], resumed["adapter"][key], rtol=0, atol=0)
    report = json.loads((Path(config["output_dir"]) / "stop-report.json").read_text())
    assert report["reason"] == "wall_clock_budget"
    # A second resume cannot silently receive a fresh five-second budget.
    again = torch.load(
        train(config, resume=checkpoint, system_adapter=tiny_system()), weights_only=True
    )
    assert again["step"] == 5


def test_deadline_evaluates_final_checkpoint_once(tmp_path, monkeypatch):
    clock = [0.0]
    evaluated = []
    monkeypatch.setattr("radio_moge.train.time.monotonic", lambda: clock[0])
    original_step = torch.optim.AdamW.step

    def timed_step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        clock[0] += 1
        return result

    def evaluate(*args, **kwargs):
        evaluated.append(str(args[6]))
        return {"teacher_depth/abs_rel": 0.1, "postprocess_failed": 0.0}

    monkeypatch.setattr(torch.optim.AdamW, "step", timed_step)
    monkeypatch.setattr("radio_moge.train.evaluate", evaluate)
    config = config_for(tmp_path)
    config["train"].update(max_seconds=3, eval_at_start=False, eval_every=100)
    config["output_dir"] = str(tmp_path / "final-eval")
    checkpoint = train(config, system_adapter=tiny_system())
    state = torch.load(checkpoint, weights_only=True)
    assert state["step"] == 3 and state["best"] == 0.1 and len(evaluated) == 1
    train(config, resume=checkpoint, system_adapter=tiny_system())
    assert len(evaluated) == 1


def test_fixed_budget_training_does_not_read_or_evaluate_test_set(tmp_path, monkeypatch):
    config = config_for(tmp_path)
    config["data"]["val_manifest"] = None
    config["data"]["test_manifest"] = str(tmp_path / "must-not-be-opened.jsonl")
    config["train"].update(steps=2, eval_every=1)
    config["output_dir"] = str(tmp_path / "no-validation")

    def forbidden(*args, **kwargs):
        raise AssertionError("Fixed-budget training must not evaluate the held-out test set")

    monkeypatch.setattr("radio_moge.train.evaluate", forbidden)
    state = torch.load(train(config, system_adapter=tiny_system()), weights_only=True)
    assert state["step"] == 2 and state["data_digests"]["val"] is None
    assert state["best"] is None
    assert not (tmp_path / "no-validation/best.pt").exists()


@pytest.mark.parametrize("kind", ["dpt", "attention"])
def test_spatial_head_geometry_training_resume_and_frozen_backbones(tmp_path, kind):
    config = config_for(tmp_path)
    config["train"].update(steps=4, lr=0.001, save_every=2)
    config["data"]["val_manifest"] = None

    def system_adapter():
        system, _ = tiny_system()
        adapter = GeometryAdapter(
            kind,
            8,
            8,
            6,
            6,
            indices=[0, 1, 2, 3],
            spatial_width=16,
            attention_heads=4,
            summary_width=8,
            summary_heads=2,
        )
        return system, adapter

    config["output_dir"] = str(tmp_path / "full")
    system, adapter = system_adapter()
    result = train(config, system_adapter=(system, adapter))
    assert all(
        not p.requires_grad and p.grad is None
        for m in (system.radio, system.encoder, system.decoder)
        for p in m.parameters()
    )
    partial_config = deepcopy(config)
    partial_config["output_dir"] = str(tmp_path / "split")
    partial = train(partial_config, stop_after=2, system_adapter=system_adapter())
    resumed = train(partial_config, resume=partial, system_adapter=system_adapter())
    full, resumed = [torch.load(p, weights_only=True) for p in (result, resumed)]
    for name, tensor in full["adapter"].items():
        torch.testing.assert_close(tensor, resumed["adapter"][name], rtol=0, atol=0)


def test_checkpoint_selection_uses_configured_source_macro(tmp_path, monkeypatch):
    config = config_for(tmp_path)
    config["output_dir"] = str(tmp_path / "macro")
    config["train"].update(steps=2, eval_every=1, eval_at_start=True, save_every=1)
    config["evaluation"]["selection_metric"] = "macro/teacher_depth/abs_rel"
    scores = iter([(1.0, 0.3), (0.8, 0.2), (0.1, 0.25)])

    def evaluate(*args, **kwargs):
        pooled, macro = next(scores)
        return {
            "teacher_depth/abs_rel": pooled,
            "macro/teacher_depth/abs_rel": macro,
            "postprocess_failed": 0.0,
        }

    monkeypatch.setattr("radio_moge.train.evaluate", evaluate)
    train(config, system_adapter=tiny_system())
    best = torch.load(tmp_path / "macro/best.pt", weights_only=True)
    assert best["step"] == 1 and best["best"] == 0.2
