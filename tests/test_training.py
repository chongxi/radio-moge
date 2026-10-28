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


def test_plane_constraints_are_opt_in_and_require_anchor(tmp_path):
    from radio_moge.train import validate_config

    config = config_for(tmp_path)
    validate_config(config)
    config['loss']['plane_flatness'] = .1
    with pytest.raises(ValueError, match='plane_anchor'):
        validate_config(config)
    config['loss']['plane_anchor'] = .1
    validate_config(config)


def test_training_probe_is_observational_and_not_used_for_selection(tmp_path, monkeypatch):
    config = config_for(tmp_path)
    config['train'].update(steps=4, eval_every=2, eval_at_start=True)
    calls = []

    def evaluate(system, adapter, stats, records, cfg, destination, limit):
        probe = destination.name.startswith('train-probe-')
        calls.append(destination.name)
        if probe:
            # Even a diagnostic that consumes RNG cannot change the optimization trajectory.
            torch.rand(7)
        return {'feature/spatial': 1., 'teacher_depth/abs_rel': 0. if probe else .2,
                'postprocess_failed': 0}

    monkeypatch.setattr('radio_moge.train.evaluate', evaluate)
    config['output_dir'] = str(tmp_path / 'without-probe')
    baseline = torch.load(train(config, system_adapter=tiny_system()), weights_only=True)
    manifest = tmp_path / 'probe.jsonl'
    manifest.write_text(Path(config['data']['train_manifest']).read_text().splitlines()[0] + '\n')
    config['data']['train_probe_manifest'] = str(manifest)
    config['output_dir'] = str(tmp_path / 'with-probe')
    checked = torch.load(train(config, system_adapter=tiny_system()), weights_only=True)
    assert checked['best'] == baseline['best'] == .2
    for key, value in baseline['adapter'].items():
        torch.testing.assert_close(value, checked['adapter'][key], rtol=0, atol=0)
    torch.testing.assert_close(baseline['rng']['torch'], checked['rng']['torch'], rtol=0, atol=0)
    rows = [json.loads(s) for s in (tmp_path / 'with-probe/train-probe.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows] == [0, 2, 4]
    assert [x for x in calls if x.startswith('train-probe')] == [
        'train-probe-0000000', 'train-probe-0000002', 'train-probe-0000004']


def test_training_probe_rejects_nontraining_images(tmp_path):
    config = config_for(tmp_path)
    config['output_dir'] = str(tmp_path / 'bad-probe')
    config['data']['train_probe_manifest'] = config['data']['val_manifest']
    with pytest.raises(ValueError, match='subset of the training'):
        train(config, system_adapter=tiny_system())


@pytest.mark.parametrize('flatness_mode', ['teacher_axis', 'own_plane'])
def test_plane_loss_reaches_training_logs_and_adapter(tmp_path, monkeypatch, flatness_mode):
    import importlib
    from radio_moge.decoder import camera_points

    trainer = importlib.import_module('radio_moge.train')
    config = config_for(tmp_path)
    config['output_dir'] = str(tmp_path / 'plane-smoke')
    config['train'].update(steps=1, accumulation=1)
    config['loss'].update(plane_flatness=.1, plane_normal=.1, plane_anchor=.1)
    config['plane_flatness_mode'] = flatness_mode

    def projected(raw, teacher=False):
        depth = torch.full_like(raw['mask'], 3) if teacher else 2 + raw['points'][..., 2].sigmoid()
        k = torch.tensor([[1., 0, .5], [0, 1., .5], [0, 0, 1.]])[None].repeat(len(depth), 1, 1)
        return camera_points(depth, k)

    monkeypatch.setattr(trainer, 'postprocess', lambda raw, *args: {
        'points': projected(raw, teacher=True), 'mask': torch.ones_like(raw['mask'], dtype=torch.bool)})
    monkeypatch.setattr(trainer, 'geometry_points_with_grad', lambda raw, *args: projected(raw))
    path = train(config, system_adapter=tiny_system())
    row = json.loads((path.parent / 'train.jsonl').read_text().splitlines()[0])
    assert row['plane_regions'] > 0 and row['plane_anchor'] > 0
    assert all(np.isfinite(row[k]) for k in ('plane_flatness', 'plane_normal', 'plane_anchor'))
    initial = torch.load(path.parent / 'initial.pt', weights_only=True)
    final = torch.load(path, weights_only=True)
    assert any(not torch.equal(v, final['adapter'][k]) for k, v in initial['adapter'].items())


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


def test_lr_fork_preserves_moments_rng_cursor_and_resumes_exactly(tmp_path):
    config = config_for(tmp_path)
    config["data"]["val_manifest"] = None
    config["output_dir"] = str(tmp_path / "parent")
    parent_path = train(config, stop_after=5, system_adapter=tiny_system())
    parent_bytes = parent_path.read_bytes()
    parent = torch.load(parent_path, weights_only=True)
    trial = deepcopy(config)
    trial["train"].update(lr=0.03, lr_schedule="constant", max_seconds=600, update_norm_every=1)
    trial["output_dir"] = str(tmp_path / "trial")
    full_path = train(trial, fork_from=parent_path, system_adapter=tiny_system())
    initial = torch.load(tmp_path / "trial/initial.pt", weights_only=True)
    assert initial["step"] == 5 and initial["cursor"] == 10
    assert initial["optimizer"]["param_groups"][0]["lr"] == 0.03
    assert initial["runtime_limits"] == {"steps": 12, "max_seconds": 600}
    for section in ("adapter", "statistics"):
        for key in parent[section]:
            torch.testing.assert_close(initial[section][key], parent[section][key], rtol=0, atol=0)
    for key, values in parent["optimizer"]["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(initial["optimizer"]["state"][key][name], value, rtol=0, atol=0)
    torch.testing.assert_close(initial["rng"]["torch"], parent["rng"]["torch"], rtol=0, atol=0)
    assert initial["rng"]["python"] == parent["rng"]["python"]
    assert initial["rng"]["numpy"] == parent["rng"]["numpy"]
    metrics = [json.loads(line) for line in (tmp_path / "trial/train.jsonl").read_text().splitlines()]
    assert all(row["lr_used"] == row["lr"] == 0.03 for row in metrics)
    assert all(row["relative_update_norm"] > 0 for row in metrics)
    trial["output_dir"] = str(tmp_path / "split-trial")
    part = train(trial, fork_from=parent_path, stop_after=8, system_adapter=tiny_system())
    resumed = torch.load(train(trial, resume=part, system_adapter=tiny_system()), weights_only=True)
    full = torch.load(full_path, weights_only=True)
    assert full["step"] == resumed["step"] == 12
    assert full["cursor"] == resumed["cursor"] == 24
    assert full["fork_provenance"] == resumed["fork_provenance"]
    for key in full["adapter"]:
        torch.testing.assert_close(full["adapter"][key], resumed["adapter"][key], rtol=0, atol=0)
    assert parent_path.read_bytes() == parent_bytes


@pytest.mark.parametrize("changed", ["data", "loss", "optimizer", "budget", "schedule"])
def test_lr_fork_rejects_uncontrolled_changes(tmp_path, changed):
    from radio_moge.train import check_lr_fork

    original = config_for(tmp_path)
    trial = deepcopy(original)
    trial["train"].update(lr=0.03, lr_schedule="constant", max_seconds=600)
    if changed == "data":
        trial["data"]["augment"] = False
    elif changed == "loss":
        trial["loss"]["cosine"] *= 2
    elif changed == "optimizer":
        trial["train"]["weight_decay"] *= 2
    elif changed == "budget":
        trial["train"]["steps"] = 5
    else:
        trial["train"]["lr_schedule"] = "cosine"
    with pytest.raises(ValueError):
        check_lr_fork(trial, {"config": original, "data_digests": {}, "step": 5}, {})


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


def test_explicit_extension_preserves_optimizer_sampler_schedule_and_remaining_budget(
    tmp_path, monkeypatch
):
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
    config["train"].update(steps=4, max_seconds=3)
    config["output_dir"] = str(tmp_path / "extension")
    base = train(config, system_adapter=tiny_system())
    state = torch.load(base, weights_only=True)
    assert state["step"] == 3
    frozen = tmp_path / "frozen.pt"
    frozen.write_bytes(Path(base).read_bytes())
    extended = train(
        config,
        resume=frozen,
        extension=(3, 10),
        stop_after=5,
        system_adapter=tiny_system(),
        execution_revision="extension-test",
    )
    middle = torch.load(extended, weights_only=True)
    assert middle["step"] == 5 and middle["cursor"] == 10
    assert middle["scheduler"]["last_epoch"] == 5
    assert middle["optimizer"]["param_groups"][0]["lr"] == pytest.approx(0.01 * 0.05)
    assert middle["runtime_limits"]["max_seconds"] == 6
    clock[0] += 1000  # Downtime must not count, nor grant a fresh extension on resume.
    resumed = torch.load(
        train(config, resume=extended, system_adapter=tiny_system()), weights_only=True
    )
    assert resumed["step"] == 6 and resumed["cursor"] == 12
    assert resumed["execution_revision"] == "extension-test"
    again = torch.load(
        train(config, resume=extended, system_adapter=tiny_system()), weights_only=True
    )
    assert again["step"] == 6

    control_config = deepcopy(config)
    control_config["output_dir"] = str(tmp_path / "extension-control")
    control = torch.load(
        train(control_config, resume=frozen, extension=(3, 10), system_adapter=tiny_system()),
        weights_only=True,
    )
    for key in control["adapter"]:
        torch.testing.assert_close(control["adapter"][key], resumed["adapter"][key], rtol=0, atol=0)
    assert torch.load(frozen, weights_only=True)["step"] == 3


def test_extension_cannot_bypass_config_or_positive_budget_guards(tmp_path):
    config = config_for(tmp_path)
    config["output_dir"] = str(tmp_path / "extension-guards")
    with pytest.raises(ValueError, match="requires --resume"):
        train(config, extension=(3, 10), system_adapter=tiny_system())
    checkpoint = train(config, stop_after=1, system_adapter=tiny_system())
    with pytest.raises(ValueError, match="must be positive"):
        train(config, resume=checkpoint, extension=(-1, 10), system_adapter=tiny_system())
    config["train"]["lr"] *= 2
    with pytest.raises(ValueError, match="identical config"):
        train(config, resume=checkpoint, extension=(3, 10), system_adapter=tiny_system())


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
