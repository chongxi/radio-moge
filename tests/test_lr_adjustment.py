from copy import deepcopy
import json
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import pytest
import torch

from radio_moge.lr_adjustment import apply_lr_adjustment, expected_constant_lr
from radio_moge.train import train
from test_performance import feature_config, profile
from test_training import tiny_system


def test_constant_lr_change_keeps_adam_moments_and_scheduler_position():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter], lr=2.5e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    parameter.sum().backward()
    optimizer.step()
    scheduler.step()
    state = dict(
        step=7,
        cursor=56,
        runtime_limits={"steps": 10, "max_seconds": 600},
        fork_provenance={"data_change": {"new_cursor": 0}, "budget_started_at": 1},
    )
    config = {"train": {"lr_schedule": "constant"}, "data": {"coverage_epochs": 1}}
    request = dict(
        authorized=True,
        id="raise-lr",
        expected_step=7,
        expected_cursor=56,
        runtime_limits=state["runtime_limits"],
        budget_started_at=1,
        previous_lr=2.5e-6,
        learning_rate=5e-6,
    )
    before = deepcopy(optimizer.state_dict())
    position = scheduler.last_epoch
    history = apply_lr_adjustment(config, state, optimizer, scheduler, request)
    assert scheduler.last_epoch == position
    for key, values in before["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(
                optimizer.state_dict()["state"][key][name], value, rtol=0, atol=0
            )
    assert optimizer.param_groups[0]["lr"] == 5e-6
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 5e-6
    assert expected_constant_lr(2.5e-6, 7, history) == 2.5e-6
    assert expected_constant_lr(2.5e-6, 8, history) == 5e-6
    with pytest.raises(ValueError, match="already applied"):
        apply_lr_adjustment(
            config, dict(state, lr_adjustments=history), optimizer, scheduler, request
        )
    for key, value in [
        ("authorized", False),
        ("expected_cursor", 0),
        ("expected_step", 8),
        ("budget_started_at", 2),
        ("previous_lr", 1e-6),
        ("learning_rate", float("nan")),
    ]:
        with pytest.raises(ValueError):
            apply_lr_adjustment(config, state, optimizer, scheduler, dict(request, **{key: value}))


def test_lr_resume_preserves_full_coverage_and_original_budget(tmp_path, monkeypatch):
    cfg = feature_config(tmp_path)
    cfg["train"]["lr"] = 1e-5
    cfg["output_dir"] = str(tmp_path / "parent")

    def evaluate(*args):
        return {
            **{f"feature/{k}": 1.0 for k in cfg["loss"]},
            "teacher_depth/abs_rel": 0.2,
            "postprocess_failed": 0,
        }

    monkeypatch.setattr("radio_moge.train.evaluate", evaluate)
    source = train(cfg, stop_after=2, system_adapter=tiny_system())
    trial = deepcopy(cfg)
    trial["output_dir"] = str(tmp_path / "epoch")
    trial["train"].update(lr=2.5e-6, lr_schedule="constant", steps=5, max_seconds=600)
    trial["data"]["coverage_epochs"] = 1
    rows = Path(cfg["data"]["train_manifest"]).read_text()
    for n in range(32):
        image = tmp_path / f"new-{n}.png"
        Image.fromarray(np.full((40, 60, 3), n + 30, np.uint8)).save(image)
        rows += json.dumps(dict(image=str(image), group="new")) + "\n"
    manifest = tmp_path / "expanded.jsonl"
    manifest.write_text(rows)
    trial["data"]["train_manifest"] = str(manifest)
    started = time.time() - 1
    partial = train(
        trial,
        fork_from=source,
        data_fork=True,
        performance=profile(2, 0),
        budget_started_at=started,
        stop_after=3,
        system_adapter=tiny_system(),
    )
    checkpoint = tmp_path / "preserved.pt"
    checkpoint.write_bytes(partial.read_bytes())
    state = torch.load(checkpoint, weights_only=True)
    request = dict(
        authorized=True,
        id="raise-lr",
        expected_step=3,
        expected_cursor=4,
        runtime_limits=state["runtime_limits"],
        budget_started_at=started,
        previous_lr=2.5e-6,
        learning_rate=5e-6,
    )
    path = train(
        trial, resume=checkpoint, lr_adjustment=request, stop_after=4, system_adapter=tiny_system()
    )
    resumed = torch.load(path, weights_only=True)
    proof = json.loads((path.parent / "resume-state.json").read_text())
    assert proof["step"] == 3 and proof["cursor"] == 4 and proof["scheduler_last_epoch"] == 1
    assert resumed["step"] == 4 and resumed["cursor"] == 8
    assert resumed["fork_provenance"] == state["fork_provenance"]
    assert resumed["runtime_limits"] == state["runtime_limits"]
    assert resumed["performance"] == state["performance"]
    assert resumed["scheduler"]["last_epoch"] == 2
    assert all(int(v["step"]) == 4 for v in resumed["optimizer"]["state"].values())
    assert resumed["lr_adjustments"][0]["budget_extended"] is False
    final = train(trial, resume=path, system_adapter=tiny_system())
    finished = torch.load(final, weights_only=True)
    coverage = json.loads((final.parent / "coverage.json").read_text())
    assert finished["step"] == 5 and finished["cursor"] == 12
    assert coverage["unique_indices"] == 38 and coverage["missing_indices"] == 0
    assert coverage["repeated_presentations"] == 10 and coverage["prefetch_excluded"]
    assert finished["optimizer"]["param_groups"][0]["lr"] == 5e-6
    assert finished["lr_adjustments"] == resumed["lr_adjustments"]
    logged = [json.loads(line) for line in (final.parent / "train.jsonl").read_text().splitlines()]
    assert [r["lr_used"] for r in logged] == [2.5e-6, 5e-6, 5e-6]


@pytest.mark.parametrize("expired", [False, True])
def test_resume_controller_keeps_deadline_and_refuses_duplicate(tmp_path, monkeypatch, expired):
    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "lr_controller", scripts / "run_lr_adjustment_resume.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    transition = tmp_path / "lr-adjustment-20261031"
    transition.mkdir()
    source = transition / "source.pt"
    source.write_bytes(b"stopped checkpoint")
    started = time.time() - (700 if expired else 10)
    request = dict(
        authorized=True,
        learning_rate=5e-6,
        expected_step=8,
        expected_cursor=64,
        budget_started_at=started,
        runtime_limits={"steps": 20, "max_seconds": 600},
        id="raise-lr",
    )
    plan = dict(
        authorized=True,
        learning_rate=5e-6,
        source_checkpoint=str(source),
        source_identity=runner.identity(source),
        original_deadline=started + 600,
        outer_deadline=started + 1200,
        previous_pids=[],
        training_lock=str(tmp_path / "lock"),
    )
    values = {
        transition / "plan.json": plan,
        transition / "request.json": request,
        transition / "source-verification.json": dict(
            stage="passed", identity=plan["source_identity"], step=8, cursor=64
        ),
        tmp_path / "config.json": {"train": {"steps": 20, "max_seconds": 600}},
        tmp_path / "status.json": dict(
            pid=123, launch_count=1, last_exit_code=0, stop_report={"reason": "signal"}
        ),
    }
    for path, value in values.items():
        path.write_text(json.dumps(value))
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: "")
    commands = []

    class Child:
        pid = 456
        returncode = 0

        def poll(self):
            return 0

    def launch(command, **kwargs):
        commands.append(command)
        (tmp_path / "train").mkdir()
        (tmp_path / "train/stop-report.json").write_text(
            json.dumps({"reason": "step_limit", "step": 20})
        )
        return Child()

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    if expired:
        with pytest.raises(ValueError, match="exhausted"):
            runner.run(tmp_path, tmp_path, "revision")
        assert not commands
    else:
        runner.run(tmp_path, tmp_path, "revision")
        assert len(commands) == 1 and "--resume" in commands[0] and "--fork-from" not in commands[0]
        assert 600 < int(commands[0][2]) <= 1190
        with pytest.raises(FileExistsError):
            runner.run(tmp_path, tmp_path, "revision")
        assert len(commands) == 1
