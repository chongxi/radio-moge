"""Run one bounded constant-LR fork with fresh artifacts and W&B identity."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from radio_moge.checkpoint import atomic_json


PILOT_LR_TAGS = {1e-4: "lr-1e-4", 1e-5: "lr-1e-5"}


def stable_identity(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def run(plan_path, repo, revision):
    plan = json.loads(plan_path.read_text())
    if plan.get("authorized") is not True or plan["learning_rate"] not in PILOT_LR_TAGS:
        raise ValueError("This pilot requires explicit authorization for a supported learning rate")
    if not 0 < plan["max_seconds"] <= 7200:
        raise ValueError("Pilot must have a wall-clock budget of at most two hours")
    if not 0 < plan["final_step"] - plan["parent_step"] <= 5000:
        raise ValueError("Pilot must have at most 5000 optimizer updates")
    base = plan_path.parent
    original = base / "attention-24h"
    root = base / plan["output_root"]
    checkpoint = base / plan["parent_checkpoint"]
    with (original / ".training.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip()
        if active:
            raise RuntimeError("GPU already has compute processes; inspect without replacing them")
        identity = stable_identity(checkpoint)
        with checkpoint.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != plan["parent_sha256"] or stable_identity(checkpoint) != identity:
            raise ValueError("Parent checkpoint hash or identity changed")
        stop = json.loads((original / "train/stop-report.json").read_text())
        if stop["step"] != plan["parent_step"] or stop["final_evaluation"] != "complete":
            raise ValueError("Parent completion report does not match")
        config = json.loads((original / "config.json").read_text())
        config["output_dir"] = str(root / "train")
        config["train"].update(
            lr=plan["learning_rate"], lr_schedule="constant", steps=plan["final_step"],
            max_seconds=plan["max_seconds"], update_norm_every=config["train"]["log_every"],
        )
        config["tracking"].update(
            name=plan["output_root"],
            tags=["attention", "lr-pilot", PILOT_LR_TAGS[plan["learning_rate"]],
                  "batch-32", "fork-83494"],
        )
        root.mkdir(exist_ok=False)
        atomic_json(config, root / "config.json")
        atomic_json(plan, root / "plan.json")
        atomic_json(plan["baseline"], root / "baseline.json")
        command = [
            sys.executable, "-u", "-m", "radio_moge.train", "--config",
            str(root / "config.json"), "--fork-from", str(checkpoint),
            "--execution-revision", revision,
        ]
        state = dict(
            stage="starting", pid=os.getpid(), started_at=time.time(), revision=revision,
            parent_sha256=digest, command=command, parent_step=plan["parent_step"],
            parent_cursor=plan["parent_cursor"], final_step=plan["final_step"],
            learning_rate=plan["learning_rate"], max_seconds=plan["max_seconds"],
        )

        def record(**changes):
            state.update(changes, updated_at=time.time())
            atomic_json(state, root / "status.json")

        record()
        environment = dict(os.environ, PYTHONPATH=str(repo / "src"), OMP_NUM_THREADS="1",
                           MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with (root / "console.log").open("w") as log:
            child = subprocess.Popen(command, cwd=repo, env=environment,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            record(stage="training", training_pid=child.pid)
            while child.poll() is None:
                record()
                time.sleep(10)
        if child.returncode:
            record(stage="needs_inspection", exit_code=child.returncode)
            raise RuntimeError("Pilot failed; inspect before any retry")
        stop = json.loads((root / "train/stop-report.json").read_text())
        if stable_identity(checkpoint) != identity:
            record(stage="needs_inspection", error="Parent identity changed")
            raise RuntimeError("Parent checkpoint identity changed")
        record(stage="training_complete" if stop["final_evaluation"] == "complete" else "interrupted",
               exit_code=0, stop_report=stop, finished_at=time.time())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--training-repo", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    run(args.plan.resolve(), args.training_repo.resolve(), args.revision)
