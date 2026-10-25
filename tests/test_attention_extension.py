import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "extension_runner", Path(__file__).parents[1] / "scripts/run_attention_extension.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def setup_plan(root):
    original = root / "attention-24h"
    (original / "train").mkdir(parents=True)
    (original / "final-review").mkdir()
    (original / "final-review/last.pt").write_bytes(b"last checkpoint")
    (original / "final-review/selected.pt").write_bytes(b"selected checkpoint")
    (original / "config.json").write_text(json.dumps({"output_dir": "original", "train": {}}))
    (original / "train/stop-report.json").write_text(
        json.dumps({"step": 10, "final_evaluation": "complete"})
    )
    (original / "train/wandb-run.json").write_text(
        json.dumps({"id": "same-id", "mode": "online", "url": "https://example.test/run"})
    )
    (original / "train/validation.jsonl").write_text('{"step": 10, "loss": 0.5}\n')
    plan = {
        "authorized": True,
        "additional_live_seconds": 86400,
        "additional_update_ceiling": 120000,
        "resume_step": 10,
        "resume_cursor": 20,
        "resume_checkpoint": "attention-24h/final-review/last.pt",
        "resume_checkpoint_sha256": hashlib.sha256(b"last checkpoint").hexdigest(),
        "output_root": "continuation",
        "wandb_run_id": "same-id",
    }
    path = root / "plan.json"
    path.write_text(json.dumps(plan))
    return path


def test_wrapper_preserves_original_and_refuses_duplicate_launch(tmp_path, monkeypatch):
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
        (tmp_path / "continuation/train/stop-report.json").write_text(
            json.dumps({"step": 11, "final_evaluation": "complete"})
        )
        return Child()

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    runner.run(plan, tmp_path, "new-revision")
    assert len(calls) == 1 and "--resume" in calls[0] and "--init" not in calls[0]
    assert calls[0][calls[0].index("--extend-seconds") + 1] == "86400"
    assert (tmp_path / "attention-24h/final-review/last.pt").read_bytes() == b"last checkpoint"
    assert (tmp_path / "continuation/train/best.pt").read_bytes() == b"selected checkpoint"
    # Simulate the trainer's atomic save: original selected checkpoint stays intact.
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"new best")
    replacement.replace(tmp_path / "continuation/train/best.pt")
    assert (
        tmp_path / "attention-24h/final-review/selected.pt"
    ).read_bytes() == b"selected checkpoint"
    with pytest.raises(FileExistsError):
        runner.run(plan, tmp_path, "new-revision")
    assert len(calls) == 1


@pytest.mark.parametrize("problem", ["gpu_active", "checkpoint_changed", "unauthorized"])
def test_launch_guards_fail_before_creating_training_output(tmp_path, monkeypatch, problem):
    plan = setup_plan(tmp_path)
    monkeypatch.setattr(
        runner.subprocess, "check_output", lambda *a, **kw: "456" if problem == "gpu_active" else ""
    )
    if problem == "checkpoint_changed":
        (tmp_path / "attention-24h/final-review/last.pt").write_bytes(b"wrong")
    if problem == "unauthorized":
        data = json.loads(plan.read_text())
        data["authorized"] = False
        plan.write_text(json.dumps(data))
    with pytest.raises((ValueError, RuntimeError)):
        runner.run(plan, tmp_path, "new-revision")
    assert not (tmp_path / "continuation").exists()
