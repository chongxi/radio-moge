"""Launch one explicitly authorized continuation; preserve the first-day artifacts."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def atomic_json(path, data):
    temporary = path.with_name(path.name + f".new-{os.getpid()}")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, path)


def run(plan_path, repo, revision):
    plan = json.loads(plan_path.read_text())
    if plan.get("authorized") is not True or plan["additional_live_seconds"] != 86400:
        raise ValueError("Explicit additional24h authorization is required")
    base = plan_path.parent
    original = base / "attention-24h"
    checkpoint = base / plan["resume_checkpoint"]
    root = base / plan["output_root"]
    # Use the original training lock as well, so both wrappers share a guard.
    lock = (original / ".training.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if active:
        raise RuntimeError("GPU compute processes already exist; inspect without replacing them")
    with checkpoint.open("rb") as handle:
        sha = hashlib.file_digest(handle, "sha256").hexdigest()
    if sha != plan["resume_checkpoint_sha256"]:
        raise ValueError("Frozen resume checkpoint SHA256 mismatch")
    stop = json.loads((original / "train/stop-report.json").read_text())
    if stop["step"] != plan["resume_step"] or stop["final_evaluation"] != "complete":
        raise ValueError("First-day training is not complete at the declared checkpoint")
    root.mkdir(exist_ok=False)
    output = root / "train"
    output.mkdir()
    config = json.loads((original / "config.json").read_text())
    config["output_dir"] = str(output)
    atomic_json(root / "config.json", config)
    identity = json.loads((original / "train/wandb-run.json").read_text())
    if identity["id"] != plan["wandb_run_id"] or identity["mode"] != "online":
        raise ValueError("Unexpected W&B run identity")
    atomic_json(output / "wandb-run.json", identity)
    # Atomic checkpoint writes in the trainer replace these links, never modify them.
    os.link(checkpoint, output / "initial.pt")
    os.link(original / "final-review/selected.pt", output / "best.pt")
    old_validation = [
        json.loads(line)
        for line in (original / "train/validation.jsonl").read_text().splitlines()
        if line.strip()
    ]
    baseline = [row for row in old_validation if row["step"] == plan["resume_step"]]
    if len(baseline) != 1:
        raise ValueError("Expected one verified baseline evaluation")
    (output / "validation.jsonl").write_text(json.dumps(baseline[0]) + "\n")
    command = [
        sys.executable,
        "-u",
        "-m",
        "radio_moge.train",
        "--config",
        str(root / "config.json"),
        "--resume",
        str(checkpoint),
        "--extend-seconds",
        str(plan["additional_live_seconds"]),
        "--extend-updates",
        str(plan["additional_update_ceiling"]),
        "--execution-revision",
        revision,
    ]
    state = {
        "stage": "starting",
        "pid": os.getpid(),
        "started_at": time.time(),
        "revision": revision,
        "checkpoint_sha256": sha,
        "command": command,
        "additional_seconds": plan["additional_live_seconds"],
        "start_step": plan["resume_step"],
        "start_cursor": plan["resume_cursor"],
        "wandb_url": identity["url"],
        "gpu_policy": "Never stop or replace GPU",
    }

    def record(**changes):
        state.update(changes, updated_at=time.time())
        atomic_json(root / "status.json", state)

    record()
    environment = dict(
        os.environ,
        PYTHONPATH=str(repo / "src"),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    with (root / "console.log").open("w") as log:
        child = subprocess.Popen(
            command,
            cwd=repo,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        record(stage="training", training_pid=child.pid)
        while child.poll() is None:
            record()
            time.sleep(10)
    if child.returncode:
        record(stage="needs_inspection", exit_code=child.returncode)
        raise RuntimeError(f"Continuation exited {child.returncode}; inspect before any retry")
    stop = json.loads((output / "stop-report.json").read_text())
    record(
        stage="training_complete" if stop["final_evaluation"] == "complete" else "interrupted",
        exit_code=0,
        stop_report=stop,
        finished_at=time.time(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--training-repo", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    run(args.plan.resolve(), args.training_repo.resolve(), args.revision)
