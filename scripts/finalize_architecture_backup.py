"""Verify completed serial backup receipts, immutable sources and optional metadata bundle."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

from backup_architecture_screen import atomic


def extract_bundle(archive, destination, expected):
    """Accept only the exact regular files whose source hashes were recorded."""
    with tarfile.open(archive, "r:") as tar:
        members = tar.getmembers()
        assert len(members) == len(expected) and {m.name for m in members} == set(expected)
        for member in members:
            relative = Path(member.name)
            assert member.isfile() and not relative.is_absolute() and ".." not in relative.parts
            item = expected[member.name]
            assert member.size == item["bytes"]
            handle = tar.extractfile(member)
            data = handle.read()
            assert hashlib.sha256(data).hexdigest() == item["sha256"]
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(output.suffix + ".incoming")
            temporary.write_bytes(data)
            assert hashlib.sha256(temporary.read_bytes()).hexdigest() == item["sha256"]
            temporary.replace(output)
    return dict(files=len(expected), every_extracted_file_sha256_verified=True)


def finish(root):
    review = root / "final-review"

    def read(path):
        return json.loads(path.read_text())

    if (review / "backup-verification.json").exists():
        print('{"stage":"already_verified"}')
        return
    state, plan, source = [
        read(review / n)
        for n in ("prepared-dataset-backup-status.json", "plan.json", "source-verification.json")
    ]
    if state["stage"] != "all_copies_finished_pending_verification":
        print(
            json.dumps(
                dict(
                    stage=state["stage"],
                    verified=len(state["verified_checkpoints"]),
                    total=len(plan["checkpoints_first"]),
                )
            )
        )
        return
    assert state["child_pid"] is None and state["verified_checkpoints"] == plan["checkpoints_first"]
    for item in plan["checkpoints_first"]:
        assert Path(item["destination"]).stat().st_size == item["bytes"]
    pid = read(review / "launch.json")["pid"]
    process = Path("/proc") / str(pid) / "stat"
    controller = process.read_text().split(") ")[1].split()[0] if process.exists() else "absent"
    assert controller in ("absent", "Z")
    with (review / ".prepared-dataset-backup.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    code = """import sys,json
from pathlib import Path
rows=json.load(sys.stdin)
for row in rows:
 s=Path(row['source']).stat()
 assert [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]==row['identity'],row['relative']
print(json.dumps(dict(checked=len(rows),all_source_identities_unchanged=True)))
"""
    response = subprocess.run(
        [
            "ssh",
            "-F",
            "/home/hub/.ssh/radio-moge-config",
            "radio-moge-gpu",
            shlex.join(["/opt/conda/bin/python3", "-c", code]),
        ],
        input=json.dumps(source["files"]),
        text=True,
        capture_output=True,
        check=True,
    )
    bundled = set(source.get("bundled_metadata", []))
    extracted = (
        extract_bundle(
            review / "backup-metadata.tar",
            review / "artifacts",
            {i["relative"]: i for i in source["files"] if i["relative"] in bundled},
        )
        if bundled
        else None
    )
    report = dict(
        stage="passed",
        checked_at=time.time(),
        file_count=len(plan["checkpoints_first"]),
        source_artifact_count=len(source["files"]),
        bytes=sum(i["bytes"] for i in plan["checkpoints_first"]),
        controller_pid=pid,
        controller_state=controller,
        child_pid=None,
        lock_free=True,
        all_plan_files_serially_copied_and_sha256_readback_verified=True,
        all_destination_sizes_match=True,
        postcopy_source_check=json.loads(response.stdout),
        metadata_extraction=extracted,
        checkpoints={i["relative"]: i["checkpoint"] for i in source["files"] if "checkpoint" in i},
        source_proof=str(review / "source-verification.json"),
        destination_sha256_receipts=str(review / "prepared-dataset-backup-status.json"),
    )
    atomic(review / "backup-verification.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    finish(parser.parse_args().root)
