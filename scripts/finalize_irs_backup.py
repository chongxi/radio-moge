"""Verify IRS exclusions and receipts after its completed retained RGB audit."""

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import time

import finalize_structured3d_backup as common
from verify_attention_dataset_backup import Reader, atomic_json, declared_exclusions, require

DATASET = "irs-rgb-ready-v1-20261022"
PLAN_SHA = "5429a555e6653dc9b452f85a5a2781f6ebd96c39080d729ed963fefe063016a6"
VERIFIER_SHA = "78c6a70f6ae3884cba395efd095e69260fbda7889a30c0aee73c6f6320b74aed"
COMMON_SHA = "448c881e6ad89297672cfa3c8256d4ea93a64f17591c0f8846da5f738c996271"
RETAINED_SHA = "58929d6746022801088bd09f913b44a666d59f673f1a8007caa7d3a79e26122e"
COUNTS = {"train": 133879, "val": 25646, "test": 17580, "test-official": 29524}


def extended_pins(pins, info):
    """The pinned dataset_info includes the additional official holdout count."""
    require(info["counts"] == pins["counts"], "Primary count mismatch")
    require(info["extra_holdout_manifests"] == ["test-official.jsonl"],
            "Unexpected official holdout manifest")
    official = info["official_test_rgb"]
    require(type(official) is int and official > 0, "Invalid official count")
    result = copy.deepcopy(pins)
    require("test-official" not in result["counts"], "Official count already in primary pins")
    result["counts"]["test-official"] = official
    manifests = {n for n in pins["metadata_files"] if n.endswith(".jsonl")}
    require(manifests == {n + ".jsonl" for n in result["counts"]},
            "Pinned manifest set differs from split counts")
    return result


def source_audit(root, pins, reader, report):
    report(stage="source_metadata")
    identities = common.metadata_check(root, pins, reader)
    info = json.loads((root / "dataset_info.json").read_text())
    full_pins = extended_pins(pins, info)
    excluded = declared_exclusions(root, pins["metadata_files"])
    receipts = {f"receipts/{name}.json" for name in info["provenance"]["archive_sha256"]}
    actual = {p.relative_to(root).as_posix() for p in (root / "receipts").iterdir()}
    require(actual == receipts, "Unexpected source receipts inventory")
    files = {}
    for name in sorted(receipts | excluded):
        report(stage="source_excluded_and_receipts", current=name, verified_files=len(files))
        sha, st = reader.digest(common.safe_file(root, name))
        files[name] = {"sha256": sha, "bytes": st.st_size}
        identities[name] = common.identity(st)
    common.check_stability(root, identities)
    return {"stage": "passed", "root": str(root), "metadata": pins["metadata_files"],
            "counts": full_pins["counts"],
            "excluded": {n: files[n] for n in sorted(excluded)},
            "receipts": {n: files[n] for n in sorted(receipts)}}


def validate_retained_proof(proof, launch, pins, root):
    require(proof["stage"] == "passed" and proof["failures"] == 0,
            "Retained audit did not pass")
    require(proof["dataset"] == launch["dataset"] == DATASET and
            proof["pid"] == launch["pid"] and proof["root"] == str(root),
            "Retained audit identity mismatch")
    require(proof["plan_sha256"] == launch["plan_sha256"] == PLAN_SHA and
            proof["code_sha256"] == launch["code_sha256"] == VERIFIER_SHA,
            "Retained audit provenance mismatch")
    require(proof["metadata"] == pins["metadata_files"], "Retained metadata mismatch")
    require(proof["counts"] == COUNTS and proof["verified_images"] == sum(COUNTS.values()),
            "Retained count mismatch including official holdout")
    require(proof["inventory_count"] == 206632 and proof["excluded_images"] == 3,
            "Retained inventory mismatch")


def finalize(root, pins, source, retained, reader, report):
    # Check pinned bytes before deriving the additional official split count.
    info_bytes = common.stable_bytes(root / "dataset_info.json")
    require(common.digest_bytes(info_bytes) == pins["metadata_files"]["dataset_info.json"]["sha256"],
            "Pinned dataset info mismatch")
    info = json.loads(info_bytes)
    full_pins = extended_pins(pins, info)
    require(source["counts"] == full_pins["counts"] == retained["counts"],
            "Source/retained split count mismatch")
    result = common.finalize(root, full_pins, source, reader, report)
    require(result["retained_images"] == retained["verified_images"] and
            result["retained_rgb_bytes"] == retained["verified_rgb_bytes"],
            "Retained count or byte total differs from completed audit")
    result["retained_hash_evidence"] = "Completed, pinned IRS retained RGB audit"
    result["checks"][0] = "completed retained RGB SHA readback including official holdout"
    result["root_identity"] = {"device": root.stat().st_dev, "inode": root.stat().st_ino}
    result["metadata_identities"] = {
        n: common.identity(common.safe_file(root, n).lstat()) for n in pins["metadata_files"]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["source", "finalize"])
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--source-proof", type=Path)
    parser.add_argument("--source-proof-sha256")
    args = parser.parse_args()
    plan_bytes = common.stable_bytes(args.base / "prepared-dataset-backup-plan.json")
    require(common.digest_bytes(plan_bytes) == PLAN_SHA, "Plan SHA mismatch")
    plan = json.loads(plan_bytes)
    pins = plan["pinned_source_inventory"][DATASET]
    root = Path(plan["source_root"] if args.mode == "source" else plan["staging"]) / DATASET
    locks = []
    for path in [args.report.with_suffix(".lock")] + (
        [args.base / "irs-backup-verification.lock"] if args.mode == "finalize" else []
    ):
        lock = path.open("a+")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    require(not args.report.exists(), "Completed report exists; inspect rather than rerun")
    code_sha = common.digest_bytes(common.stable_bytes(Path(__file__)))
    require(common.digest_bytes(common.stable_bytes(Path(common.__file__))) == COMMON_SHA,
            "Shared finalizer code mismatch")
    require(common.digest_bytes(common.stable_bytes(Path(__file__).with_name(
        "verify_attention_dataset_backup.py"))) == VERIFIER_SHA, "Verifier code mismatch")
    state = {"pid": os.getpid(), "started_at": time.time(), "mode": args.mode,
             "code_sha256": code_sha, "common_sha256": COMMON_SHA,
             "verifier_sha256": VERIFIER_SHA, "plan_sha256": PLAN_SHA,
             "dataset": DATASET, "mib_per_second": 20, "published": False}
    def report(**changes):
        state.update(changes, updated_at=time.time())
        atomic_json(args.status, state)
    def copied():
        controller = json.loads((args.base / "prepared-dataset-backup-status.json").read_text())
        require(controller["plan_sha256"] == PLAN_SHA and
                controller["stage"] == "all_copies_finished_pending_verification" and
                controller.get("child_pid") is None and controller.get("dataset") is None and
                DATASET in controller["copied_pending_verification"], "Copies not finished")
    try:
        reader = Reader(20)
        if args.mode == "source":
            result = source_audit(root, pins, reader, report)
            require(result["counts"] == COUNTS and len(result["excluded"]) == 3 and
                    len(result["receipts"]) == 6, "Unexpected source inventory")
        else:
            copied()
            evidence_path = args.base / "irs-backup-verification.json"
            evidence = common.stable_bytes(evidence_path)
            require(common.digest_bytes(evidence) == RETAINED_SHA, "Retained report SHA mismatch")
            retained = json.loads(evidence)
            launch = json.loads(common.stable_bytes(args.base / "irs-backup-verification-launch.json"))
            validate_retained_proof(retained, launch, pins, root)
            process = Path(f"/proc/{retained['pid']}/stat")
            require(not process.exists() or process.read_text().split(') ')[1].split()[0] == 'Z',
                    "Retained reader is still active")
            source_bytes = common.stable_bytes(args.source_proof)
            require(common.digest_bytes(source_bytes) == args.source_proof_sha256,
                    "Source proof SHA mismatch")
            source = json.loads(source_bytes)
            require(source["plan_sha256"] == PLAN_SHA and source["dataset"] == DATASET and
                    source["code_sha256"] == code_sha and source["common_sha256"] == COMMON_SHA and
                    source["verifier_sha256"] == VERIFIER_SHA and
                    source["root"] == str(Path(plan["source_root"]) / DATASET),
                    "Source proof provenance mismatch")
            result = finalize(root, pins, source, retained, reader, report)
            inventory = result["inventory"]
            require(result["counts"] == COUNTS and result["excluded_images"] == 3 and
                    result["receipt_files_verified"] == 6, "Unexpected final counts")
            require(inventory["files"] == launch["copy_evidence"]["regular_files"] == 206647 and
                    inventory["bytes"] == launch["copy_evidence"]["total_regular_file_bytes"] ==
                    178912437590, "Copy inventory total mismatch")
            require(common.stable_bytes(evidence_path) == evidence, "Retained proof changed")
            copied()
            result.update(source_proof_sha256=args.source_proof_sha256,
                          retained_proof_sha256=RETAINED_SHA,
                          retained_finished_at=retained["updated_at"])
        report(**result, finished_at=time.time())
        atomic_json(args.report, state)
    except Exception as error:
        report(stage="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
