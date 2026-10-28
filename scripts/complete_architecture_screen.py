"""Seal a completed, audited architecture screen after the serial backup passes."""

import argparse
import hashlib
import json
from pathlib import Path
import time


def read(path):
    return json.loads(path.read_text())


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".incoming")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def complete(root, capacity=False):
    plan = read(root / "plan.json")
    count = len(plan["candidates"])
    candidate_count, point_count, confirmation_count = (
        (count, 2 * len(plan["evaluation_steps"]) * count, count) if capacity else (16, 180, 4)
    )
    proofs = {
        name: read(root / name)
        for name in (
            "status.json",
            "queue-status.json",
            "curve-verification.json",
            "confirmation-status.json",
            "confirmation-verification.json",
            "wandb-final-verification.json",
            "visualization-verification.json",
            "browser-final-verification.json",
            "final-review/backup-verification.json",
        )
    }
    training = proofs["status.json"]
    assert training["stage"] == "complete" and len(training["completed"]) == candidate_count
    assert all(
        row["step"] == 1000 and row["cursor"] == 16000 and row["reason"] == "step_limit"
        for row in training["completed"]
    )
    queue = proofs["queue-status.json"]
    assert queue["child_pid"] is None and queue["child_exit"] == 0
    assert proofs["curve-verification.json"]["total_proofs"] == point_count
    confirmation = proofs["confirmation-status.json"]
    assert confirmation["stage"] == "complete" and confirmation["optimizer_updates"] == 0
    assert len(confirmation["completed"]) == confirmation_count
    for name in (
        "confirmation-verification.json",
        "wandb-final-verification.json",
        "visualization-verification.json",
        "browser-final-verification.json",
        "final-review/backup-verification.json",
    ):
        assert proofs[name]["stage"] == "passed", name
    runs = proofs["wandb-final-verification.json"]["runs"]
    assert len(runs) == candidate_count and all(
        r["state"] == "finished" and r["step"] == 1000 for r in runs.values()
    )
    backup = proofs["final-review/backup-verification.json"]
    expected_files = (
        len(read(root / "final-review/plan.json")["checkpoints_first"]) if capacity else 907
    )
    assert backup["file_count"] == expected_files and len(backup["checkpoints"]) == candidate_count
    assert backup["postcopy_source_check"]["all_source_identities_unchanged"]
    assert backup["all_plan_files_serially_copied_and_sha256_readback_verified"]
    now = time.time()
    report = read(root / "result-report.json")
    report.update(
        stage="complete",
        completed_at=now,
        backup_verification=str(root / "final-review/backup-verification.json"),
    )
    atomic(root / "result-report.json", report)
    result = dict(
        stage="passed",
        checked_at=now,
        candidate_count=candidate_count,
        updates_per_candidate=1000,
        all_stop_reasons="step_limit",
        optimizer_cursor=16000,
        audited_points=point_count,
        confirmation_candidates=confirmation_count,
        confirmation_optimizer_updates=0,
        confirmation_feature_images=224,
        confirmation_geometry_images=56,
        gt_images=32,
        all_wandb_finished=True,
        backup_files=backup["file_count"],
        backup_bytes=backup["bytes"],
        browser_url=proofs["browser-final-verification.json"]["url"],
        automatic_extension=False,
        input_sha256={
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in proofs
        },
        result_report=str(root / "result-report.json"),
        remaining=[],
    )
    atomic(root / "completion-verification.json", result)
    continuation = read(root / "continuation.json")
    continuation.update(
        stage="complete",
        updated_at=now,
        pending=[],
        completion_verification=str(root / "completion-verification.json"),
    )
    continuation["commands"]["sync"] = (
        "Final metadata synchronization only; all computation and backups complete."
    )
    atomic(root / "continuation.json", continuation)
    handoff_path = root.parent / "handoff.json"
    handoff = read(handoff_path)
    key = (
        "active_adapter_wide4096_screen"
        if plan.get("screen_profile") == "wide4096"
        else "active_adapter_wide2048_screen"
        if plan.get("screen_profile") == "wide2048"
        else "active_adapter_capacity_screen"
    )
    active = handoff[key if capacity else "active_adapter_architecture_screen"]
    assert active["name"] == root.name
    active.update(
        stage="complete",
        completed_at=now,
        remaining=[],
        completion_verification=str(root / "completion-verification.json"),
        result_report=str(root / "result-report.json"),
        backup_verification=str(root / "final-review/backup-verification.json"),
        note=f"All {candidate_count} scratch candidates, predeclared {confirmation_count} endpoint confirmations, audits, W&B, browser and serial backups complete. No automatic follow-on training.",
    )
    atomic(handoff_path, handoff)
    atomic(root / "handoff-completion-snapshot.json", active)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--capacity", action="store_true")
    args = parser.parse_args()
    complete(args.root, args.capacity)
