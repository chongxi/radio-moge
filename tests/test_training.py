from copy import deepcopy
import json
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
    split_config = deepcopy(config)
    split_config["output_dir"] = str(tmp_path / "split")
    partial = train(split_config, stop_after=5, system_adapter=tiny_system())
    resumed_path = train(split_config, resume=partial, system_adapter=tiny_system())
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
