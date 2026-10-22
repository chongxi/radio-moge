import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "quick_screen", Path(__file__).resolve().parents[1] / "scripts/run_quick_screen.py"
)
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


def test_rung_preserves_config_and_resumes_existing_optimizer(tmp_path):
    for head in screen.HEADS:
        path = tmp_path / head / "features"
        path.mkdir(parents=True)
        config = dict(
            data={"manifest": "fixed"},
            models={},
            loss={},
            evaluation={},
            train={"steps": 6000, "eval_every": 1000},
            output_dir=str(path / "train"),
        )
        (path / "config.json").write_text(json.dumps(config))
    before = (tmp_path / "dpt/features/config.json").read_bytes()
    screen.validate_matched_configs(tmp_path, 1000)
    cmd = screen.command_for(tmp_path, "dpt", 1000, "python", 0)
    assert cmd[-2:] == ["--stop-after", "1000"] and "--resume" not in cmd
    latest = tmp_path / "dpt/features/train/latest.pt"
    latest.parent.mkdir()
    latest.touch()
    cmd = screen.command_for(tmp_path, "dpt", 3000, "python", 1000)
    assert cmd[-2:] == ["--resume", str(latest)]
    assert (tmp_path / "dpt/features/config.json").read_bytes() == before
    assert screen.command_for(tmp_path, "dpt", 1000, "python", 1000) is None
    config["data"] = {"manifest": "different"}
    (tmp_path / "attention/features/config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Unmatched"):
        screen.validate_matched_configs(tmp_path, 1000)


def test_feedback_never_ranks_unequal_budgets_or_failed_validation(tmp_path):
    for head, step, score in [("fusion", 1000, 0.2), ("dpt", 1000, 0.3), ("attention", 500, 0.1)]:
        path = tmp_path / head / "features/train/validation.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {"step": step, "macro/teacher_depth/abs_rel": score, "postprocess_failed": 0}
            )
            + "\n"
        )
    assert screen.feedback(tmp_path, 1000)["descriptive_order"] is None
    path = tmp_path / "attention/features/train/validation.jsonl"
    path.write_text(
        json.dumps({"step": 1000, "macro/teacher_depth/abs_rel": 0.1, "postprocess_failed": 1})
        + "\n"
    )
    assert screen.feedback(tmp_path, 1000)["descriptive_order"] is None
    path.write_text(path.read_text().replace('"postprocess_failed": 1', '"postprocess_failed": 0'))
    assert screen.feedback(tmp_path, 1000)["descriptive_order"] == ["attention", "fusion", "dpt"]
