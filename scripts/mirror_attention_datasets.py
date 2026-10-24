"""Serial, resumable copies into a private staging directory.

Copy completion deliberately remains pending verification. A separate audit must
check the saved manifests and destination image hashes before publishing a dataset.
No source file or existing published dataset is modified.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time


def atomic_json(path, value):
    temporary = path.with_name(path.name + f".new-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def copy_checkpoint(item, plan, report):
    """Copy a frozen source, resume our partial, then read back the destination."""
    destination = Path(item["destination"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert destination.stat().st_size == item["bytes"]
        assert digest(destination) == item["sha256"]
        return
    partial = destination.with_name("." + destination.name + ".incoming")
    offset = partial.stat().st_size if partial.exists() else 0
    assert offset <= item["bytes"]
    source_command = "tail -c +" + str(offset + 1) + " -- " + shlex.quote(item["source"])
    command = shlex.split(plan["ssh_command"]) + [plan["ssh_host"], source_command]
    report(stage="copying_checkpoint", checkpoint=item, copied_bytes=offset, command=command, child_pid=None)
    started = last_report = time.monotonic()
    copied = offset
    # stderr goes to the controller log, never to an unread pipe.
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
    report(child_pid=child.pid)
    try:
        with partial.open("ab", buffering=8 * 1024**2) as output:
            while block := child.stdout.read(8 * 1024**2):
                copied += len(block)
                assert copied <= item["bytes"]
                output.write(block)
                delay = (copied - offset) / (plan["bwlimit_kib"] * 1024) - (time.monotonic() - started)
                if delay > 0:
                    time.sleep(delay)
                if time.monotonic() - last_report >= 30:
                    report(copied_bytes=copied)
                    last_report = time.monotonic()
            output.flush()
            os.fsync(output.fileno())
        assert child.wait() == 0 and copied == item["bytes"]
    finally:
        child.stdout.close()
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)
    report(stage="verifying_checkpoint", child_pid=None, copied_bytes=copied)
    assert digest(partial) == item["sha256"]
    os.replace(partial, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    plan_hash = hashlib.sha256(plan_bytes).hexdigest()
    root = args.plan.resolve().parent
    status_path = root / "prepared-dataset-backup-status.json"
    staging = Path(plan["staging"])
    assert staging.is_absolute() and staging.is_dir()
    assert plan["bwlimit_kib"] > 0
    assert len(plan["datasets"]) == len(set(plan["datasets"]))
    assert all(Path(name).name == name and name not in (".", "..") for name in plan["datasets"])
    lock = (root / ".prepared-dataset-backup.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner_path = staging / ".backup-owner.json"
    if owner_path.exists():
        assert json.loads(owner_path.read_text())["plan_sha256"] == plan_hash
    else:
        atomic_json(owner_path, {"plan_sha256": plan_hash, "purpose": plan["purpose"]})
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    if previous:
        assert previous["plan_sha256"] == plan_hash
        # A controller can disappear while its rsync child survives. A released
        # controller lock alone therefore does not authorize another transfer.
        old_child = Path("/proc") / str(previous.get("child_pid", "absent"))
        try:
            old_args = (old_child / "cmdline").read_bytes().split(b"\0")
        except OSError:
            old_args = []
        expected_args = [x.encode() for x in previous.get("command", [])]
        if (old_args and old_args[:-1] == expected_args) or (
            b"rsync" in old_args and (str(staging) + "/").encode() in old_args
        ):
            raise RuntimeError("Previous transfer child is still active; do not duplicate it")
    completed = previous.get("copied_pending_verification", [])
    state = {
        "pid": os.getpid(), "plan_sha256": plan_hash,
        "started_at": time.time(), "copied_pending_verification": completed,
        "verified_datasets": [], "published_datasets": [],
        "verified_checkpoints": previous.get("verified_checkpoints", []),
        "staging": str(staging), "policy": "One rsync process; no deletion; staged copies only",
    }

    def report(**changes):
        state.update(changes, updated_at=time.time())
        atomic_json(status_path, state)

    try:
        for item in plan.get("checkpoints_first", []):
            copy_checkpoint(item, plan, report)
            if item not in state["verified_checkpoints"]:
                state["verified_checkpoints"].append(item)
            report(stage="checkpoint_verified", child_pid=None)
        for name in plan["datasets"]:
            if name in completed:
                assert (staging / name).is_dir()
                continue
            # Exclude transient lock/partial files, preserving all ordinary data
            # and provenance files. --partial permits resuming failed transfers.
            command = [
                "rsync", "-rlt", "--partial", "--stats",
                "--bwlimit=" + str(plan["bwlimit_kib"]), "--exclude=.*",
                "-e", plan["ssh_command"],
                plan["ssh_host"] + ":" + plan["source_root"].rstrip("/") + "/" + name,
                str(staging) + "/",
            ]
            log_path = root / ("backup-" + name + ".log")
            report(stage="starting_copy", dataset=name, command=command, log=str(log_path), child_pid=None)
            with log_path.open("ab", buffering=0) as log:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                report(stage="copying", child_pid=child.pid)
                next_report = time.monotonic() + 30
                paused = False
                while child.poll() is None:
                    time.sleep(5)
                    # Pause only this owned transfer group when a higher-priority
                    # checkpoint copy needs the single bulk-transfer slot.
                    requested = (root / "prepared-dataset-backup.pause").exists()
                    if requested != paused and child.poll() is None:
                        os.killpg(child.pid, signal.SIGSTOP if requested else signal.SIGCONT)
                        paused = requested
                        report(stage="paused_for_checkpoint" if paused else "copying")
                    if time.monotonic() >= next_report:
                        try:
                            io = (Path("/proc") / str(child.pid) / "io").read_text()
                        except OSError:
                            io = None
                        report(child_io=io, log_bytes=log_path.stat().st_size)
                        next_report = time.monotonic() + 30
                if child.returncode != 0:
                    raise RuntimeError(f"rsync failed for {name}: exit {child.returncode}; see {log_path}")
            completed.append(name)
            report(stage="copy_finished_pending_verification", child_pid=None, child_io=None)
        report(stage="all_copies_finished_pending_verification", dataset=None)
    except Exception as error:
        report(stage="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
