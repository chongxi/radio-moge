"""Run the two frozen LR arms sequentially, with no retries or deployment."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from radio_moge.checkpoint import atomic_json
from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import publish_json
from run_room_lr_arm import compare_arm_configs, LEARNING_RATES, UPDATES


def run(path):
    plan = checked_json(file_ref(path))
    root = path.parent
    if checked_path(plan["controller"]) != Path(__file__).resolve():
        raise ValueError("Controller code changed")
    arms = [checked_json(ref) for ref in plan["arms"]]
    if [a["arm"] for a in arms] != list(LEARNING_RATES) or any(
        a["additional_updates"] != UPDATES for a in arms
    ):
        raise ValueError("Exactly the two declared bounded arms are required")
    configs = [checked_json(a["config"]) for a in arms]
    compare_arm_configs(*configs)
    for key in (
        "source_checkpoint",
        "source_step",
        "source_parent_cursor",
        "source_cursor",
        "source_results",
        "execution_revision",
        "runner",
        "trainer",
        "training_lock",
    ):
        if arms[0][key] != arms[1][key]:
            raise ValueError("Paired source or implementation mismatch: " + key)
    status = dict(
        stage="starting",
        started_at=time.time(),
        controller_pid=os.getpid(),
        completed_arms=[],
        automatic_retry=False,
    )
    publish_json(root / "controller-start.json", status)
    try:
        for ref, arm in zip(plan["arms"], arms, strict=True):
            arm_root = Path(ref["path"]).parent
            if (arm_root / "status.json").exists() or (arm_root / "training").exists():
                raise FileExistsError("Arm already launched; inspect rather than retry")
            env = dict(os.environ)
            env.pop("WANDB_SERVICE", None)
            with (arm_root / "console.log").open("x") as log:
                child = subprocess.Popen(
                    [
                        sys.executable,
                        str(checked_path(arm["runner"])),
                        "--plan",
                        str(checked_path(ref)),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                status.update(stage="arm_running", arm=arm["arm"], trainer_pid=child.pid)
                atomic_json(status, root / "status.json")
                try:
                    code = child.wait(timeout=5700)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    raise TimeoutError("Arm outer wall budget reached; no retry") from None
            if code:
                raise RuntimeError(f"Arm {arm['arm']} exited {code}; no retry")
            completed = checked_json(file_ref(arm_root / "completed.json"))
            if not completed["budget_complete"] or completed["new_updates"] != UPDATES:
                raise ValueError("Incomplete arm; do not start the next comparison arm")
            status["completed_arms"].append(arm["arm"])
        reports = [
            checked_json(file_ref(Path(r["path"]).parent / "training/actual-rgb-input-audit.json"))
            for r in plan["arms"]
        ]
        if reports[0]["records"]["sha256"] != reports[1]["records"]["sha256"]:
            raise ValueError("Full actual input bytes differ between arms")
        status.update(
            stage="training_complete_review_pending",
            actual_input_sha256=reports[0]["records"]["sha256"],
            finished_at=time.time(),
            automatic_next_training=False,
        )
        publish_json(root / "report.json", status)
        atomic_json(status, root / "status.json")
    except Exception as error:
        status.update(
            stage="failed_no_retry",
            error=type(error).__name__,
            message=str(error),
            finished_at=time.time(),
        )
        atomic_json(status, root / "status.json")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    run(parser.parse_args().plan.resolve())
