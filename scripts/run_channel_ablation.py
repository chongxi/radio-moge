"""Run one fresh, budget-matched channel-cleaning comparison on the existing GPU."""

import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import time

import torch

from radio_moge.checkpoint import atomic_json
from radio_moge.config import load_config


def verify_control(config, baseline):
    for key in ("models", "data", "train", "loss", "evaluation"):
        if config[key] != baseline[key]:
            raise ValueError(f"Channel ablation changed baseline {key}")
    adapter = dict(config["adapter"])
    if adapter.pop("channel_cleaning", None) != "zero-channel":
        raise ValueError("Expected zero-channel cleaning")
    if adapter.pop("channel_threshold", None) != 8.0:
        raise ValueError("Expected toromi's fixed 8x threshold")
    if adapter != baseline["adapter"]:
        raise ValueError("Channel ablation changed adapter architecture")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline", required=True, help="Original fusion initial.pt")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "plan.json").exists():
            raise FileExistsError("Run already planned; inspect status before any resume")
        config = load_config(args.config)
        baseline = torch.load(args.baseline, weights_only=True, map_location="cpu", mmap=True)
        verify_control(config, baseline["config"])
        if baseline["step"] != 0:
            raise ValueError("Baseline must be its saved initial.pt")
        del baseline
        config["output_dir"] = str(root / "train")
        config["code_revision"] = args.revision
        atomic_json(config, root / "config.json")
        atomic_json(
            dict(
                revision=args.revision,
                baseline_initial=str(Path(args.baseline).resolve()),
                steps=config["train"]["steps"],
                initialization="Fresh seed-matched adapter; no weights copied from baseline",
                selection="validation/teacher_depth/abs_rel",
                change="Per-image, per-layer spatial mean-absolute channel magnitude > 8x median",
                summary="Unchanged; spatial maps only",
                test_policy="Outer test is not read during training or model selection",
                gpu_policy="Keep GPU running after this experiment finishes",
            ),
            root / "plan.json",
        )
        status = dict(stage="starting", started_at=time.time(), revision=args.revision)
        atomic_json(status, root / "status.json")
        with (root / "console.log").open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "radio_moge.train",
                    "--config",
                    str(root / "config.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            status.update(stage="training", training_pid=process.pid)
            atomic_json(status, root / "status.json")
            result = process.wait()
        status.update(
            stage="failed" if result else "complete", returncode=result, updated_at=time.time()
        )
        if not result:
            state = torch.load(
                root / "train/latest.pt", weights_only=True, map_location="cpu", mmap=True
            )
            status["completed_steps"] = state["step"]
            if state["step"] != config["train"]["steps"]:
                status["stage"] = "interrupted"
        atomic_json(status, root / "status.json")
        raise SystemExit(result)


if __name__ == "__main__":
    main()
