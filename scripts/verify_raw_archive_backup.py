"""Verify completed raw-archive copies against pinned source evidence, without publishing."""

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time

from verify_attention_dataset_backup import atomic_json, identity, relative_path, require


ARCHIVE_EVIDENCE = "completed_download_receipt_and_prepared_provenance"
SMALL_EVIDENCE = "fresh_small_file_readback"
MAX_METADATA_BYTES = 16 * 1024**2


def small_bytes(path):
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), f"Not a regular metadata file: {path}")
    require(before.st_size <= MAX_METADATA_BYTES, f"Metadata too large: {path}")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as handle:
        require(identity(before) == identity(os.fstat(handle.fileno())), f"Replaced: {path}")
        value = handle.read(MAX_METADATA_BYTES + 1)
        require(identity(before) == identity(os.fstat(handle.fileno())), f"Changed: {path}")
    require(len(value) == before.st_size, f"Metadata size changed: {path}")
    require(identity(before) == identity(path.lstat()), f"Replaced: {path}")
    return value


def pinned_json(path, expected_sha):
    value = small_bytes(path)
    require(hashlib.sha256(value).hexdigest() == expected_sha, f"SHA256 mismatch: {path}")
    return json.loads(value)


def validate_pins(pins):
    files = pins["files"]
    archives = {}
    for name, entry in files.items():
        path = relative_path(name)
        require(not any(p.startswith(".") for p in path.parts), "Hidden path in source pins")
        require(type(entry["bytes"]) is int and entry["bytes"] >= 0, "Invalid pinned size")
        sha = entry["sha256"]
        require(isinstance(sha, str) and len(sha) == 64
                and all(c in "0123456789abcdef" for c in sha), "Invalid SHA256 pin")
        is_archive = len(path.parts) == 2 and path.parts[0] == "archives" and (
            name.endswith(".zip") or name.endswith(".tar.gz"))
        require(entry["sha256_evidence"] == (ARCHIVE_EVIDENCE if is_archive else SMALL_EVIDENCE),
                f"Unexpected SHA256 evidence: {name}")
        if is_archive:
            archives[path.name] = entry
            require(f"receipts/{path.name}.json" in files, f"Receipt missing: {name}")
            require(name + ".part.json" in files, f"Download sidecar missing: {name}")
    require(len(files) == pins["file_count"], "Pinned file count mismatch")
    require(sum(f["bytes"] for f in files.values()) == pins["total_bytes"],
            "Pinned total size mismatch")
    require(len(archives) == pins["archive_count"] > 0, "Pinned archive count mismatch")
    require(sum(f["bytes"] for f in archives.values()) == pins["archive_bytes"],
            "Pinned archive size mismatch")
    require({n for n in files if n.startswith("receipts/")} ==
            {f"receipts/{n}.json" for n in archives}, "Unexpected receipt set")
    require({n for n in files if n.startswith("archives/")} ==
            {f"archives/{n}{suffix}" for n in archives for suffix in ("", ".part.json")},
            "Unexpected archive or partial file")
    return archives


def tree_snapshot(root, files):
    """Stat the small archive tree, rejecting even empty unexpected directories."""
    expected_dirs = {"."}
    for name in files:
        expected_dirs.update(p.as_posix() for p in relative_path(name).parents)
    found_files, found_dirs = {}, {}

    def visit(folder):
        info = folder.lstat()
        require(stat.S_ISDIR(info.st_mode), f"Directory missing or symlinked: {folder}")
        name = folder.relative_to(root).as_posix()
        require(name in expected_dirs, f"Unexpected directory: {name}")
        found_dirs[name] = identity(info)
        with os.scandir(folder) as entries:
            for entry in entries:
                path = Path(entry.path)
                info = entry.stat(follow_symlinks=False)
                relative = path.relative_to(root).as_posix()
                if stat.S_ISDIR(info.st_mode):
                    visit(path)
                else:
                    require(stat.S_ISREG(info.st_mode), f"Symlink or special file: {relative}")
                    require(relative in files, f"Unexpected file: {relative}")
                    require(info.st_size == files[relative]["bytes"], f"Size mismatch: {relative}")
                    found_files[relative] = identity(info)

    visit(root)
    require(set(found_files) == set(files), "Missing destination files")
    require(set(found_dirs) == expected_dirs, "Missing destination directories")
    return {"files": found_files, "directories": found_dirs}


class ArchiveReader:
    """One bounded reader, with progress during each large file, not just between files."""

    def __init__(self, mib_per_second, report, heartbeat_seconds=30, block_bytes=1024**2):
        require(math.isfinite(mib_per_second) and 0 < mib_per_second <= 20,
                "Read rate must be positive and at most 20 MiB/s")
        self.rate = mib_per_second * 1024**2
        self.report = report
        self.heartbeat_seconds = heartbeat_seconds
        self.block_bytes = block_bytes
        self.bytes_read = 0
        self.started = time.monotonic()

    def digest(self, path, expected_identity, dataset, name):
        digest = hashlib.sha256()
        file_bytes = 0
        last_report = time.monotonic()
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            require(stat.S_ISREG(before.st_mode), f"Not a regular file: {path}")
            require(identity(before) == expected_identity, f"Changed before hashing: {path}")
            while block := handle.read(self.block_bytes):
                digest.update(block)
                file_bytes += len(block)
                self.bytes_read += len(block)
                delay = self.bytes_read / self.rate - (time.monotonic() - self.started)
                if delay > 0:
                    time.sleep(delay)
                if time.monotonic() - last_report >= self.heartbeat_seconds:
                    self.report(stage="checking_file_hashes", dataset=dataset, current_file=name,
                                current_file_bytes_read=file_bytes, bytes_read=self.bytes_read)
                    last_report = time.monotonic()
            require(identity(before) == identity(os.fstat(handle.fileno())), f"Changed: {path}")
        require(identity(before) == identity(path.lstat()), f"Replaced: {path}")
        require(file_bytes == before.st_size, f"Read size mismatch: {path}")
        return digest.hexdigest()


def receipt_relationships(root, pins, archives):
    for name, pin in archives.items():
        receipt = json.loads(small_bytes(root / "receipts" / (name + ".json")))
        sidecar = json.loads(small_bytes(root / "archives" / (name + ".part.json")))
        require(receipt["sha256"] == pin["sha256"] and receipt["bytes"] == pin["bytes"],
                f"Archive/receipt mismatch: {name}")
        require(sidecar["bytes"] == pin["bytes"], f"Archive/sidecar size mismatch: {name}")
        for item in (receipt, sidecar):
            require(item.get("name", name) == name, f"Archive name mismatch: {name}")
    status = json.loads(small_bytes(root / "status.json"))
    require(status["stage"] == pins["source_stage"] and status["stage"] in
            {"archives_complete", "archives_complete_with_missing_assets"},
            "Source download status is not complete")
    require(status.get("missing", []) == pins["missing_archives"], "Missing-archive scope changed")
    require(not set(pins["missing_archives"]) & archives.keys(), "Missing archive also included")
    completed = status["completed"]
    require(len(completed) == len(archives), "Download status archive count mismatch")
    if all(isinstance(x, dict) for x in completed):
        require({x["name"]: x["sha256"] for x in completed} ==
                {n: x["sha256"] for n, x in archives.items()}, "Download status SHA mismatch")
    else:
        require(all(isinstance(x, str) for x in completed) and set(completed) == set(archives),
                "Download status archive names mismatch")


def verify_dataset(root, pins, reader, report):
    archives = validate_pins(pins)
    report(stage="checking_initial_inventory", dataset=root.name)
    initial = tree_snapshot(root, pins["files"])
    checked, verified_bytes = {}, 0
    for name, pin in sorted(pins["files"].items()):
        report(stage="checking_file_hashes", dataset=root.name, current_file=name,
               current_file_bytes_read=0, verified_files=len(checked), verified_bytes=verified_bytes)
        sha = reader.digest(root / name, initial["files"][name], root.name, name)
        require(sha == pin["sha256"], f"File SHA256 mismatch: {name}")
        checked[name] = {"sha256": sha, "bytes": pin["bytes"],
                         "identity": initial["files"][name]}
        verified_bytes += pin["bytes"]
    report(stage="checking_receipt_relationships", verified_files=len(checked),
           verified_bytes=verified_bytes)
    receipt_relationships(root, pins, archives)
    report(stage="checking_final_inventory")
    require(tree_snapshot(root, pins["files"]) == initial, "Destination tree changed during audit")
    return {"stage": "passed", "published": False, "failures": 0, "files": checked,
            "root": str(root), "root_identity": initial["directories"]["."],
            "directory_identities": initial["directories"],
            "inventory": {"files": len(checked), "directories": len(initial["directories"]),
                          "bytes": verified_bytes},
            "archive_count": len(archives), "archive_bytes": pins["archive_bytes"],
            "missing_archives": pins["missing_archives"],
            "every_destination_file_sha256_verified": True,
            "archive_receipt_relationships_verified": True,
            "fresh_source_archive_hashes_claimed": False,
            "archive_decompression_checked": False}


def process_active(pid):
    require(type(pid) is int and pid > 0, "Invalid controller PID")
    try:
        value = (Path("/proc") / str(pid) / "stat").read_text()
    except FileNotFoundError:
        return False
    return value.rsplit(")", 1)[1].split()[0] != "Z"


def require_copies_finished(base, plan, plan_sha):
    controller = json.loads(small_bytes(base / "prepared-dataset-backup-status.json"))
    require(controller["plan_sha256"] == plan_sha, "Controller plan mismatch")
    require(controller["stage"] == "all_copies_finished_pending_verification"
            and controller.get("child_pid") is None and controller.get("dataset") is None,
            "All bulk copies must finish before raw archive verification")
    require(len(controller["copied_pending_verification"]) == len(plan["datasets"])
            and set(controller["copied_pending_verification"]) == set(plan["datasets"]),
            "Copy completion dataset set mismatch")
    require(not process_active(controller["pid"]), "Copy controller is still active")
    require(not (base / "prepared-dataset-backup.pause").exists(), "Checkpoint transfer pause is set")
    staging_arg = (str(Path(plan["staging"])) + "/").encode()
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = (process / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        require(not (args and Path(os.fsdecode(args[0])).name == "rsync" and
                     staging_arg in args), "An orphan rsync for this staging tree is still active")


def load_plan(path, plan_sha):
    plan = pinned_json(path, plan_sha)
    names = plan["datasets"]
    require(len(names) == len(set(names)) > 0, "Invalid dataset list")
    for name in names:
        require(len(relative_path(name).parts) == 1, "Invalid dataset name")
    source_name = relative_path(plan["source_inventory"])
    require(len(source_name.parts) == 1, "Source inventory must be at plan root")
    source = pinned_json(path.parent / source_name, plan["source_inventory_sha256"])
    require(source["datasets"] == plan["pinned_source_inventory"]
            and set(source["datasets"]) == set(names)
            and source["source_root"] == plan["source_root"], "Source inventory disagrees with plan")
    require(Path(plan["staging"]) == path.parent / "staging", "Unexpected staging path")
    for pins in source["datasets"].values():
        validate_pins(pins)
    for name, sha in plan["publication_prerequisites"].items():
        require(len(relative_path(name).parts) == 1, "Invalid publication proof path")
        proof = pinned_json(path.parent.parent / name, sha)
        require(proof["stage"] == "published" and proof["passed"], "Prepared backup not published")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--mib-per-second", type=float, default=20)
    args = parser.parse_args()
    require(args.plan.is_absolute(), "Use an absolute plan path")
    base = args.plan.parent
    report_path = base / "raw-archive-backup-verification.json"
    status_path = base / "raw-archive-backup-verification-status.json"
    with ExitStack() as stack:
        for name in [".raw-archive-verification.lock", ".prepared-dataset-backup.lock"]:
            lock = stack.enter_context((base / name).open("a+"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not report_path.exists() and not status_path.exists(),
                "Audit evidence already exists; preserve and inspect it before a new attempt")
        plan = load_plan(args.plan, args.plan_sha256)
        require_copies_finished(base, plan, args.plan_sha256)
        state = {"pid": os.getpid(), "started_at": time.time(), "plan_sha256": args.plan_sha256,
                 "source_inventory_sha256": plan["source_inventory_sha256"],
                 "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "dependency_sha256": hashlib.sha256(
                     Path(__file__).with_name("verify_attention_dataset_backup.py").read_bytes()
                 ).hexdigest(), "mib_per_second": args.mib_per_second, "published": False}

        def report(**changes):
            state.update(changes, updated_at=time.time())
            atomic_json(status_path, state)

        reader = ArchiveReader(args.mib_per_second, report)
        try:
            results = {}
            for name in plan["datasets"]:
                results[name] = verify_dataset(Path(plan["staging"]) / name,
                                               plan["pinned_source_inventory"][name], reader, report)
                report(completed_datasets=list(results))
            require_copies_finished(base, plan, args.plan_sha256)
            require(load_plan(args.plan, args.plan_sha256) == plan, "Plan changed during audit")
            for name, result in results.items():
                snapshot = tree_snapshot(Path(plan["staging"]) / name,
                                         plan["pinned_source_inventory"][name]["files"])
                require(snapshot["files"] == {n: x["identity"] for n, x in result["files"].items()}
                        and snapshot["directories"] == result["directory_identities"],
                        f"Previously verified dataset changed: {name}")
            report(stage="passed", datasets=results, failures=0, dataset=None, current_file=None,
                   verified_files=sum(x["inventory"]["files"] for x in results.values()),
                   verified_bytes=sum(x["inventory"]["bytes"] for x in results.values()),
                   bytes_read=reader.bytes_read,
                   archive_count=sum(x["archive_count"] for x in results.values()))
            atomic_json(report_path, state)
        except Exception as error:
            report(stage="failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
