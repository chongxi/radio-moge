"""Summarize architecture trials using the preregistered validation criterion."""

import argparse
import json
import math
from pathlib import Path
from statistics import median

from radio_moge.checkpoint import atomic_json

CRITERION = "macro/teacher_depth/abs_rel"
METRICS = [
    CRITERION,
    "teacher_depth/abs_rel",
    "teacher_depth/rmse",
    "scale_log_error",
    "raw/points",
    "raw/normal",
    "postprocess_failed",
    "branch_ms",
]


def read_rows(path):
    if not path.exists():
        return []
    result = []
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return result


def metrics(row):
    return {k: v for k, v in row.items() if k in METRICS or k.startswith("dataset/")}


def stage_summary(directory):
    validation = read_rows(directory / "validation.jsonl")
    training = read_rows(directory / "train.jsonl")
    if not validation:
        return {"status": "not_evaluated"}
    eligible = [
        r
        for r in validation
        if isinstance(r.get(CRITERION), (int, float))
        and math.isfinite(r[CRITERION])
        and r.get("postprocess_failed") == 0
    ]
    best = min(eligible, key=lambda r: r[CRITERION]) if eligible else None
    config = json.loads((directory / "config.json").read_text())
    finished = bool(training and training[-1]["step"] == config["train"]["steps"])
    result = {
        "status": "finished" if finished else "in_progress",
        "target_steps": config["train"]["steps"],
        "last_train_step": training[-1]["step"] if training else 0,
        "initial": {"step": validation[0]["step"], **metrics(validation[0])},
        "last": {"step": validation[-1]["step"], **metrics(validation[-1])},
        "selected": {"step": best["step"], **metrics(best)} if best else None,
        "selection_metric": CRITERION,
        "validation_curve": [{"step": r["step"], **metrics(r)} for r in validation],
        "median_logged_step_seconds": median([r["seconds"] for r in training])
        if training
        else None,
        "peak_training_memory_gib": max((r.get("peak_memory_gib", 0) for r in training), default=0),
        "config": str(directory / "config.json"),
    }
    if best and CRITERION in validation[0]:
        result["selected_minus_initial_macro_abs_rel"] = best[CRITERION] - validation[0][CRITERION]
    if (directory / "wandb-run.json").exists():
        result["wandb"] = json.loads((directory / "wandb-run.json").read_text())
    return result


def summarize(root):
    heads = {
        head: {
            stage: stage_summary(root / head / stage / "train")
            for stage in ["features", "geometry"]
        }
        for head in ["fusion", "dpt", "attention"]
    }
    # Do not rank partial runs against completed runs or choose checkpoints using test metrics.
    rankings = {}
    for stage in ["features", "geometry"]:
        if all(h[stage]["status"] == "finished" and h[stage]["selected"] for h in heads.values()):
            rankings[stage] = sorted(
                heads, key=lambda name: heads[name][stage]["selected"][CRITERION]
            )
    return {
        "selection_metric": CRITERION,
        "heads": heads,
        "completed_stage_rankings": rankings,
        "limits": "One seed, matched optimizer updates and examples, not equal FLOPs or convergence. "
        "Internal teacher-agreement validation; not ground-truth accuracy. "
        "RMSE is reported without replacing the checkpoint-selection criterion. "
        "No test scores are read by this report.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    atomic_json(summarize(args.root), args.output)
