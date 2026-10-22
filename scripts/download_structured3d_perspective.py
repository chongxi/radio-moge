"""Serial, resumable download of the user-provided Structured3D perspective release.

Panorama archives are excluded. Missing official assets remain explicit. This
downloads and CRC-verifies archives; RGB decoding and training preparation are
separate steps. No training or GPU lifecycle operations are performed.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile
import time
import zipfile


BASE = "https://zju-kjl-jointlab-azure.kujiale.com/Structured3D/"


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def inventory(path):
    rows = json.loads(path.read_text())["archives"]
    expected = {
        f"Structured3D_perspective_{kind}_{i:02d}.zip"
        for kind in ("full", "empty")
        for i in range(18)
    } | {"Structured3D_annotation_3d.zip", "Structured3D_bbox.zip"}
    if len(rows) != len(expected) or {r["name"] for r in rows} != expected:
        raise ValueError(
            "Inventory must contain exactly the approved perspective and annotation files"
        )
    for row in rows:
        if row["url"] != BASE + row["name"] or row["status"] not in (200, 404):
            raise ValueError("Unexpected archive URL or availability status")
        if row["status"] == 200 and (not row.get("etag") or row.get("bytes", 0) <= 0):
            raise ValueError("Available archive needs positive size and immutable ETag")
    return rows


def verify_archive(path, expected_bytes):
    if path.stat().st_size != expected_bytes:
        raise ValueError("Downloaded archive size differs from inventory")
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        for member in members:
            name = PurePosixPath(member.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in member.filename:
                raise ValueError("Unsafe ZIP member path")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Unexpected ZIP symbolic link")
        bad = archive.testzip()  # Bounded decompression checks every member's CRC.
        if bad:
            raise ValueError(f"ZIP CRC failed: {bad}")
    return dict(
        sha256=sha,
        bytes=expected_bytes,
        members=len(members),
        uncompressed_bytes=sum(m.file_size for m in members),
        scene_ids=sorted(
            {
                component
                for member in members
                for component in PurePosixPath(member.filename).parts
                if re.fullmatch(r"scene_\d{5}", component)
            }
        ),
        verification="Full archive SHA256 and ZIP CRC for every member; RGB not yet decoded",
    )


def download_one(row, output, record):
    directory = output / "archives"
    directory.mkdir(exist_ok=True)
    destination = directory / row["name"]
    partial = destination.with_suffix(".zip.part")
    partial_identity = destination.with_suffix(".zip.part.json")
    receipt = output / "receipts" / (row["name"] + ".json")
    if destination.exists():
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            with destination.open("rb") as stream:
                sha = hashlib.file_digest(stream, "sha256").hexdigest()
            if previous["etag"] != row["etag"] or sha != previous["sha256"]:
                raise ValueError("Previously verified archive changed")
            return previous
        candidate = destination
    else:
        identity = {key: row[key] for key in ("url", "bytes", "etag")}
        if partial_identity.exists():
            if json.loads(partial_identity.read_text()) != identity:
                raise ValueError("Partial download belongs to a different archive version")
        elif partial.exists():
            raise ValueError("Partial download has no source identity; inspect before resuming")
        else:
            atomic_json(partial_identity, identity)
        size = partial.stat().st_size if partial.exists() else 0
        if size > row["bytes"]:
            raise ValueError("Partial download exceeds declared archive size")
        if size < row["bytes"]:
            command = [
                "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--connect-timeout",
                "20",
                "--speed-limit",
                "1024",
                "--speed-time",
                "120",
                "--retry",
                "5",
                "--retry-delay",
                "10",
                "--continue-at",
                "-",
                "--header",
                "If-Match: " + row["etag"],
                "--output",
                str(partial),
                "--url",
                row["url"],
            ]
            with (output / "transfer.log").open("ab") as log:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
                while child.poll() is None:
                    record(
                        stage="downloading",
                        archive=row["name"],
                        transfer_pid=child.pid,
                        downloaded_bytes=partial.stat().st_size if partial.exists() else 0,
                        archive_bytes=row["bytes"],
                    )
                    time.sleep(5)
                if child.returncode:
                    raise RuntimeError(f"curl exited {child.returncode}; partial file preserved")
        candidate = partial
    record(stage="verifying_archive", archive=row["name"], transfer_pid=None)
    result = dict(
        name=row["name"],
        url=row["url"],
        etag=row["etag"],
        **verify_archive(candidate, row["bytes"]),
    )
    if candidate != destination:
        candidate.replace(destination)
    atomic_json(receipt, result)
    return result


def run(args):
    rows = inventory(args.inventory)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / ".download.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(
        stage="waiting",
        pid=os.getpid(),
        started=time.time(),
        completed=[],
        missing=[r["name"] for r in rows if r["status"] == 404],
        scope="Perspective full and empty, structure and bbox; no panorama archives",
    )

    def record(**changes):
        state.update(changes, updated=time.time())
        atomic_json(args.output / "status.json", state)

    record()
    try:
        while args.wait_for_dataset:
            source = args.wait_for_dataset
            previous = json.loads((source / "status.json").read_text())
            if previous["stage"] == "failed":
                raise RuntimeError(
                    "Preceding dataset failed; inspect before starting another transfer"
                )
            if previous["stage"] == "complete":
                with (source / ".download.lock").open("a") as prior_lock:
                    try:
                        fcntl.flock(prior_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        pass
                    else:
                        break
            record(
                stage="waiting_for_preceding_dataset",
                predecessor=str(source),
                predecessor_stage=previous["stage"],
            )
            time.sleep(30)
        for row in rows:
            if row["status"] == 404:
                continue
            result = download_one(row, args.output, record)
            state["completed"].append(dict(name=row["name"], sha256=result["sha256"]))
            record(stage="between_archives", transfer_pid=None)
        record(
            stage="archives_complete_with_missing_assets"
            if state["missing"]
            else "archives_complete",
            finished=time.time(),
            training_ready=False,
        )
    except BaseException as error:
        record(stage="failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-for-dataset", type=Path)
    run(parser.parse_args())
