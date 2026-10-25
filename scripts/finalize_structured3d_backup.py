"""Complete the pinned legacy Structured3D audit without rehashing retained RGB."""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import time

from verify_attention_dataset_backup import (
    Reader, atomic_json, declared_exclusions, identity, relative_path, require,
)

DATASET = "structured3d-perspective-ready-v1-20261022"
LEGACY_SHA = "c9676902fb254185b9759a6badf0d9e7665cd75d2e806cd3465c7b02b43317b3"
PLAN_SHA = "5429a555e6653dc9b452f85a5a2781f6ebd96c39080d729ed963fefe063016a6"
EXPECTED_ERROR = "Unreferenced RGB files: 289"


def stable_bytes(path):
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), f"Nonregular evidence: {path}")
    value = path.read_bytes()
    require(identity(before) == identity(path.lstat()), f"Evidence changed: {path}")
    return value


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def safe_file(root, name):
    relative = relative_path(name)
    require(root.is_dir() and not root.is_symlink(), "Unsafe dataset root")
    for parent in (root / relative).parents:
        if parent == root:
            break
        require(parent.is_dir() and not parent.is_symlink(), "Unsafe parent directory")
    return root / relative


def metadata_check(root, pins, reader):
    identities = {}
    for name, expected in pins["metadata_files"].items():
        require(Path(name).name == name, "Metadata must be at dataset root")
        sha, info = reader.digest(safe_file(root, name))
        require(
            sha == expected["sha256"] and info.st_size == expected["bytes"],
            f"Pinned metadata mismatch: {name}",
        )
        identities[name] = identity(info)
    return identities


def check_stability(root, identities):
    for name, expected in identities.items():
        require(identity(safe_file(root, name).lstat()) == expected, f"Changed: {name}")


def source_audit(root, pins, reader, report):
    report(stage="source_metadata")
    identities = metadata_check(root, pins, reader)
    excluded = declared_exclusions(root, pins["metadata_files"])
    require(len(excluded) == 289, "Unexpected exclusion count")
    info = json.loads((root / "dataset_info.json").read_text())
    receipts = {f"receipts/{name}.json" for name in info["provenance"]["archive_sha256"]}
    actual = {p.relative_to(root).as_posix() for p in (root / "receipts").iterdir()}
    require(actual == receipts, "Unexpected source receipts inventory")
    files = {}
    for name in sorted(receipts | excluded):
        report(stage="source_excluded_and_receipts", current=name, verified_files=len(files))
        sha, st = reader.digest(safe_file(root, name))
        files[name] = {"sha256": sha, "bytes": st.st_size}
        identities[name] = identity(st)
    check_stability(root, identities)
    return {"stage": "passed", "root": str(root), "metadata": pins["metadata_files"],
            "excluded": {n: files[n] for n in sorted(excluded)},
            "receipts": {n: files[n] for n in sorted(receipts)}}


def validate_terminal_evidence(status, launch, log, code, plan_sha):
    require(digest_bytes(code) == LEGACY_SHA, "Legacy code SHA mismatch")
    require(status["stage"] == "failed" and status.get("error") == EXPECTED_ERROR,
            "Different legacy terminal error")
    require(status["code_sha256"] == launch["code_sha256"] == LEGACY_SHA,
            "Legacy launch/status code mismatch")
    require(status["plan_sha256"] == launch["plan_sha256"] == plan_sha == PLAN_SHA,
            "Legacy plan mismatch")
    require(status["pid"] == launch["pid"] and status["dataset"] == DATASET,
            "Legacy identity mismatch")
    require(log.rstrip().endswith("ValueError: " + EXPECTED_ERROR), "Unexpected traceback")
    require('line 127, in verify\n' in log and
            'require(not remaining, f"Unreferenced RGB files: {len(remaining)}")' in log,
            "Failure did not reach the post-manifest check")


def retained_rows(root, pins, excluded):
    rows = {}
    counts = Counter()
    for name in sorted(n for n in pins["metadata_files"] if n.endswith(".jsonl")):
        split = name.removesuffix(".jsonl")
        with (root / name).open() as handle:
            for line in handle:
                row = json.loads(line)
                name = row["image"]
                require(relative_path(name).parts[0] == "rgb", "Retained path outside RGB")
                require(name not in rows and name not in excluded, "Overlapping image path")
                require(row["sha256_kind"] == "encoded_file_bytes", "Missing encoded SHA")
                require(isinstance(row["bytes"], int) and row["bytes"] > 0, "Invalid RGB size")
                rows[name] = {"sha256": row["sha256"], "bytes": row["bytes"]}
                counts[split] += 1
    require(dict(counts) == pins["counts"], "Pinned split count mismatch")
    return rows, dict(counts)


def exact_tree(root, expected, report):
    """Read names and stat only; retained RGB bytes were read by the legacy job."""
    remaining = set(expected)
    total_bytes = directories_count = files_count = 0
    last = time.monotonic()
    require(root.is_dir() and not root.is_symlink(), "Unsafe dataset root")
    def fail(error):
        raise error
    for folder, directories, files in os.walk(root, followlinks=False, onerror=fail):
        directories_count += 1
        for name in directories:
            require(not (Path(folder) / name).is_symlink(), "Symlink directory in dataset")
        for name in files:
            path = Path(folder) / name
            relative = path.relative_to(root).as_posix()
            require(relative in remaining, f"Unexpected file: {relative}")
            st = path.lstat()
            require(stat.S_ISREG(st.st_mode), f"Nonregular file: {relative}")
            require(st.st_size == expected[relative]["bytes"], f"File size mismatch: {relative}")
            total_bytes += st.st_size
            files_count += 1
            remaining.remove(relative)
        if time.monotonic() - last >= 30:
            report(stage="checking_exact_tree", directories=directories_count,
                   files=files_count, bytes=total_bytes)
            last = time.monotonic()
    require(not remaining, f"Missing files: {len(remaining)}")
    return {"files": files_count, "directories": directories_count, "bytes": total_bytes}


def finalize(root, pins, source, reader, report):
    require(source["stage"] == "passed" and source["metadata"] == pins["metadata_files"],
            "Source proof metadata mismatch")
    report(stage="checking_metadata_stability")
    identities = metadata_check(root, pins, reader)
    excluded = declared_exclusions(root, pins["metadata_files"])
    require(set(source["excluded"]) == excluded, "Source exclusions mismatch")
    rows, counts = retained_rows(root, pins, excluded)
    retained_hashes = {row["sha256"] for row in rows.values()}
    info = json.loads((root / "dataset_info.json").read_text())
    expected_receipts = {f"receipts/{n}.json" for n in info["provenance"]["archive_sha256"]}
    require(set(source["receipts"]) == expected_receipts, "Source receipt inventory mismatch")
    supplemental = {**source["excluded"], **source["receipts"]}
    for name, expected in supplemental.items():
        report(stage="checking_excluded_bytes_and_receipts", current=name)
        sha, st = reader.digest(safe_file(root, name))
        require(sha == expected["sha256"] and st.st_size == expected["bytes"],
                f"Supplemental SHA/size mismatch: {name}")
        if name in excluded:
            require(sha in retained_hashes, "Excluded bytes do not match any retained image")
        identities[name] = identity(st)
    expected = {**rows, **supplemental, **pins["metadata_files"]}
    require(len(expected) == len(rows) + len(supplemental) + len(pins["metadata_files"]),
            "Overlapping metadata path")
    report(stage="checking_exact_tree", directories=0, files=0)
    inventory = exact_tree(root, expected, report)
    check_stability(root, identities)
    return {"stage": "passed", "published": False, "counts": counts,
            "retained_images": len(rows), "retained_rgb_bytes": sum(r["bytes"] for r in rows.values()),
            "retained_hash_evidence": "Pinned legacy code reached its post-manifest check",
            "retained_rehashed": False, "excluded_images": len(excluded),
            "excluded_bytes_verified": True,
            "excluded_rgb_bytes": sum(v["bytes"] for v in source["excluded"].values()),
            "receipt_files_verified": len(source["receipts"]), "inventory": inventory,
            "metadata": pins["metadata_files"], "failures": 0,
            "checks": ["legacy full retained RGB SHA readback and per-manifest identity checks",
                       "pinned split counts and stable metadata SHA/size",
                       "excluded RGB bytes and all receipts match fresh source SHA/size",
                       "excluded bytes match retained SHA values",
                       "exact full-tree file inventory and sizes, no symlinks or special files"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["source", "finalize"])
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--source-proof", type=Path)
    parser.add_argument("--source-proof-sha256")
    args = parser.parse_args()
    plan_bytes = stable_bytes(args.base / "prepared-dataset-backup-plan.json")
    require(digest_bytes(plan_bytes) == PLAN_SHA, "Plan SHA mismatch")
    plan = json.loads(plan_bytes)
    pins = plan["pinned_source_inventory"][DATASET]
    locks = []
    for path in [args.report.with_suffix(".lock")] + (
        [args.base / "structured3d-backup-verification.lock"] if args.mode == "finalize" else []
    ):
        lock = path.open("a+")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    require(not args.report.exists(), "Completed report exists; inspect rather than rerun")
    state = {"pid": os.getpid(), "started_at": time.time(), "mode": args.mode,
             "code_sha256": digest_bytes(Path(__file__).read_bytes()),
             "verifier_sha256": digest_bytes(Path(__file__).with_name(
                 "verify_attention_dataset_backup.py").read_bytes()),
             "plan_sha256": PLAN_SHA, "dataset": DATASET, "mib_per_second": 20,
             "published": False}
    def report(**changes):
        state.update(changes, updated_at=time.time())
        atomic_json(args.status, state)
    def copied():
        controller = json.loads((args.base / "prepared-dataset-backup-status.json").read_text())
        require(controller["plan_sha256"] == PLAN_SHA, "Controller plan mismatch")
        require(DATASET in controller["copied_pending_verification"], "Copy incomplete")
        require(controller.get("dataset") != DATASET, "Dataset has an active copy")
    try:
        reader = Reader(20)
        if args.mode == "source":
            result = source_audit(Path(plan["source_root"]) / DATASET, pins, reader, report)
        else:
            copied()
            evidence_names = ["structured3d-backup-verification-launch.json",
                              "structured3d-backup-verification-status.json",
                              "structured3d-backup-verification.log"]
            evidence = {n: stable_bytes(args.base / n) for n in evidence_names}
            launch, status = (json.loads(evidence[n]) for n in evidence_names[:2])
            code = stable_bytes(args.base / "code/dee3f89/verify_attention_dataset_backup.py")
            validate_terminal_evidence(status, launch, evidence[evidence_names[2]].decode(),
                                       code, PLAN_SHA)
            process = Path(f"/proc/{status['pid']}/stat")
            require(not process.exists() or process.read_text().split(') ')[1].split()[0] == 'Z',
                    "Legacy reader is still active")
            evidence["legacy-verifier.py"] = code
            archive = args.base / "structured3d-legacy-terminal-evidence"
            archive.mkdir(exist_ok=True)
            for name, value in evidence.items():
                path = archive / name
                if path.exists():
                    require(stable_bytes(path) == value, "Preserved evidence differs")
                else:
                    temp = path.with_name(path.name + f".new-{os.getpid()}")
                    temp.write_bytes(value)
                    os.replace(temp, path)
            report(legacy_evidence={n: digest_bytes(v) for n, v in evidence.items()})
            source_bytes = stable_bytes(args.source_proof)
            require(digest_bytes(source_bytes) == args.source_proof_sha256,
                    "Source proof SHA mismatch")
            source = json.loads(source_bytes)
            require(source["plan_sha256"] == PLAN_SHA and source["dataset"] == DATASET,
                    "Source proof identity mismatch")
            require(source["code_sha256"] == state["code_sha256"] and
                    source["verifier_sha256"] == state["verifier_sha256"],
                    "Source proof code mismatch")
            root = Path(plan["staging"]) / DATASET
            require(status["root"] == str(root), "Legacy root mismatch")
            result = finalize(root, pins, source, reader, report)
            require(result["retained_images"] == 161010 and result["excluded_images"] == 289,
                    "Unexpected Structured3D counts")
            require(result["inventory"]["bytes"] == launch["copy_evidence"][
                "total_regular_file_bytes"], "Copy byte total mismatch")
            require(result["inventory"]["files"] == launch["copy_evidence"]["regular_files"],
                    "Copy file count mismatch")
            copied()
            result["source_proof_sha256"] = args.source_proof_sha256
            result["source_proof"] = str(args.source_proof)
            result["legacy_terminal_at"] = status["updated_at"]
        report(**result, finished_at=time.time())
        atomic_json(args.report, state)
    except Exception as error:
        report(stage="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
