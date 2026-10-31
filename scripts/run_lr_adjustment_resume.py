"""One reviewed LR-only resume; retain the existing run, coverage and absolute deadline."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from run_data_continuation import atomic, identity


def run(root, repo, revision):
    transition = root / "lr-adjustment-20261031"
    plan = json.loads((transition / "plan.json").read_text())
    request = json.loads((transition / "request.json").read_text())
    source = Path(plan["source_checkpoint"])
    proof = json.loads((transition / "source-verification.json").read_text())
    config = json.loads((root / "config.json").read_text())
    if (
        plan.get("authorized") is not True
        or request.get("authorized") is not True
        or plan["learning_rate"] != request["learning_rate"]
        or plan["original_deadline"]
        != request["budget_started_at"] + config["train"]["max_seconds"]
        or request["runtime_limits"]
        != {"steps": config["train"]["steps"], "max_seconds": config["train"]["max_seconds"]}
        or proof["stage"] != "passed"
        or proof["identity"] != plan["source_identity"]
        or proof["step"] != request["expected_step"]
        or proof["cursor"] != request["expected_cursor"]
    ):
        raise ValueError("Resume plan, source proof or original budget differs")
    with Path(plan["training_lock"]).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (transition / "status.json").exists():
            raise FileExistsError("Adjustment already launched; no automatic retry")
        if any(Path(f"/proc/{pid}").exists() for pid in plan["previous_pids"]):
            raise RuntimeError("Previous training processes must have exited")
        if subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip():
            raise RuntimeError("GPU compute is not empty")
        if identity(source) != plan["source_identity"]:
            raise ValueError("Stopped source identity changed")
        remaining = math.ceil(plan["outer_deadline"] - time.time())
        if remaining <= 600 or time.time() >= plan["original_deadline"]:
            raise ValueError("Original training budget is exhausted")
        old = json.loads((root / "status.json").read_text())
        if old.get("last_exit_code") != 0 or old.get("stop_report", {}).get("reason") != "signal":
            raise ValueError("Expected a clean, intentionally stopped previous invocation")
        command = [
            "timeout",
            "--kill-after=120",
            str(remaining),
            sys.executable,
            "-u",
            "-m",
            "radio_moge.train",
            "--config",
            str(root / "config.json"),
            "--resume",
            str(source),
            "--lr-adjustment",
            str(transition / "request.json"),
            "--execution-revision",
            revision,
        ]
        state = dict(
            old,
            stage="lr_adjustment_resume_starting",
            pid=os.getpid(),
            child_pid=None,
            previous_controller_pid=old["pid"],
            resumed_at=time.time(),
            command=command,
            execution_revision=revision,
            lr_adjustment_id=request["id"],
            current_learning_rate=request["learning_rate"],
            resume_count=1,
        )
        for key in ["last_exit_code", "finished_at", "stop_report"]:
            state.pop(key, None)

        def record(**changes):
            state.update(changes, updated_at=time.time())
            atomic(transition / "status.json", state)
            atomic(root / "status.json", state)

        record()
        env = dict(
            os.environ,
            PYTHONPATH=str(repo / "src"),
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            WANDB_BASE_URL="https://api.forge.coreweave.com",
        )
        with (transition / "resume-console.log").open("x") as log:
            child = subprocess.Popen(
                command,
                cwd=repo,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            record(stage="training", child_pid=child.pid)
            while child.poll() is None:
                record()
                time.sleep(5)
        record(
            child_pid=None,
            last_exit_code=child.returncode,
            finished_at=time.time(),
            resume_source_identity_unchanged=identity(source) == plan["source_identity"],
        )
        if child.returncode:
            record(stage="needs_inspection")
            return
        stop = json.loads((root / "train/stop-report.json").read_text())
        record(stage="training_complete_review_pending", stop_report=stop)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("repo", type=Path)
    parser.add_argument("revision")
    args = parser.parse_args()
    run(args.root, args.repo, args.revision)
