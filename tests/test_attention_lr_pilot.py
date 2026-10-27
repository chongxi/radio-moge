import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "lr_pilot", Path(__file__).parents[1] / "scripts/run_attention_lr_pilot.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def setup_plan(root):
    original = root / "attention-24h"
    (original / "train").mkdir(parents=True)
    checkpoint = original / "last.pt"
    checkpoint.write_bytes(b"frozen parent")
    (original / "config.json").write_text(json.dumps({
        "train": {"log_every": 20}, "tracking": {"project": "radio-moge"},
    }))
    (original / "train/stop-report.json").write_text(json.dumps({
        "step": 83494, "final_evaluation": "complete",
    }))
    plan = dict(authorized=True, learning_rate=1e-4, max_seconds=7200, final_step=88000,
                parent_step=83494, parent_cursor=166988, output_root="pilot",
                parent_checkpoint="attention-24h/last.pt", baseline={},
                parent_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    path = root / "plan.json"
    path.write_text(json.dumps(plan))
    return path


def test_pilot_is_a_new_run_and_refuses_duplicate_launch(tmp_path, monkeypatch):
    plan = setup_plan(tmp_path)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: "")
    calls = []

    class Child:
        pid = 123
        returncode = 0

        def poll(self):
            return 0

    def launch(command, **kwargs):
        calls.append(command)
        output = tmp_path / "pilot/train"
        output.mkdir()
        (output / "stop-report.json").write_text(json.dumps({
            "step": 88000, "final_evaluation": "complete",
        }))
        return Child()

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    runner.run(plan, tmp_path, "pilot-revision")
    assert "--fork-from" in calls[0] and "--resume" not in calls[0]
    assert not (tmp_path / "pilot/train/wandb-run.json").exists()
    config = json.loads((tmp_path / "pilot/config.json").read_text())
    assert config["train"]["lr"] == 1e-4 and config["train"]["lr_schedule"] == "constant"
    assert (tmp_path / "attention-24h/last.pt").read_bytes() == b"frozen parent"
    with pytest.raises(FileExistsError):
        runner.run(plan, tmp_path, "pilot-revision")
    assert len(calls) == 1


@pytest.mark.parametrize("problem", ["gpu_active", "checkpoint_changed", "budget", "unauthorized"])
def test_pilot_guards_prevent_launch(tmp_path, monkeypatch, problem):
    path = setup_plan(tmp_path)
    monkeypatch.setattr(runner.subprocess, "check_output",
                        lambda *a, **kw: "456" if problem == "gpu_active" else "")
    plan = json.loads(path.read_text())
    if problem == "budget":
        plan["max_seconds"] = 86400
    elif problem == "unauthorized":
        plan["authorized"] = False
    elif problem == "checkpoint_changed":
        (tmp_path / "attention-24h/last.pt").write_bytes(b"wrong")
    path.write_text(json.dumps(plan))
    with pytest.raises((ValueError, RuntimeError)):
        runner.run(path, tmp_path, "pilot-revision")
    assert not (tmp_path / "pilot").exists()
