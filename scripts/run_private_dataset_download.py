"""Run an approved private ScanNet++ script after training starts, with safe I/O.

The licensed script and token are supplied by the user outside git. This wrapper
contains neither. It leaves the official split/asset selection logic intact.
"""

import argparse
import builtins
import fcntl
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import sys
import tempfile
import time
from types import SimpleNamespace
import urllib.error
import urllib.request
from urllib.parse import urlsplit
import zipfile


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class RedactedWriter(io.TextIOBase):
    """Buffer whole lines so secrets split across writes cannot escape."""

    def __init__(self, sink, secret):
        self.sink, self.secret, self.pending = sink, secret, ""

    def clean(self, text):
        text = text.replace(self.secret, "[REDACTED]") if self.secret else text
        return re.sub(r'https?://[^\s<>"\']+', "[URL REDACTED]", text)

    def write(self, text):
        self.pending += text
        while re.search(r"[\r\n]", self.pending):
            at = re.search(r"[\r\n]", self.pending).end()
            self.sink.write(self.clean(self.pending[:at]))
            self.pending = self.pending[at:]
        self.sink.flush()
        return len(text)

    def flush(self):
        self.sink.flush()  # Do not flush a possible partial credential.

    def finish(self):
        if self.pending:
            self.sink.write(self.clean(self.pending))
            self.pending = ""
        self.sink.flush()

    def isatty(self):
        return False


def has_updates(training):
    path = training / "train/train.jsonl"
    if not path.exists():
        return False
    with path.open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("step", 0) >= 1:
                assert row["lr_used"] > 0 and row["gradient_norm"] > 0
                assert all(math.isfinite(v) for v in row.values() if type(v) in (int, float))
                return True
    return False


class AtomicRetriever:
    def __init__(
        self, data_root, control, record, bytes_per_second=20 * 1024**2, before_asset=None
    ):
        self.root, self.control, self.record = data_root.resolve(), control, record
        self.rate = bytes_per_second
        self.completed = 0
        self.verified_bytes = 0
        self.before_asset = before_asset

    def __call__(self, url, filename, reporthook=None, data=None):
        path = Path(filename).resolve()
        relative = str(path.relative_to(self.root))
        if self.before_asset and Path(relative).parts[0] == "data":
            self.before_asset()
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname != "scannetpp.mlsg.cit.tum.de":
            raise ValueError("Unexpected dataset download origin")
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        side = part.with_name(part.name + ".identity.json")
        previous = json.loads(side.read_text()) if side.exists() else {}
        offset = part.stat().st_size if part.exists() else 0
        # Resume only with a server validator. Otherwise safely start over.
        validator = previous.get("etag") or previous.get("last_modified")
        if offset and not validator:
            offset = 0
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers.update(Range=f"bytes={offset}-", **{"If-Range": validator})
        request = urllib.request.Request(url, data=data, headers=headers)
        self.record(
            stage="downloading",
            current=relative,
            current_bytes=offset,
            completed_files=self.completed,
            verified_download_bytes=self.verified_bytes,
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            status = response.status
            length = response.headers.get("Content-Length")
            if length is None:
                raise ValueError("Missing Content-Length; refusing unverified completion")
            length = int(length)
            current = dict(
                etag=response.headers.get("ETag"),
                last_modified=response.headers.get("Last-Modified"),
            )
            if offset and any(
                previous.get(k) and previous[k] != current.get(k) for k in ("etag", "last_modified")
            ):
                raise ValueError("Source identity changed; partial preserved for inspection")
            if status == 206:
                match = re.fullmatch(
                    r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
                )
                if not match or int(match[1]) != offset or int(match[2]) - offset + 1 != length:
                    raise ValueError("Incorrect Content-Range; refusing partial append")
                total = int(match[3])
            elif status == 200:
                offset, total = 0, length  # A server ignoring Range cannot cause concatenation.
            else:
                raise ValueError(f"Unexpected transfer status {status}")
            if previous.get("total") is not None and offset and previous["total"] != total:
                raise ValueError("Source length changed; partial preserved")
            current["total"] = total
            atomic_json(side, current)
            sha = hashlib.sha256()
            if offset:
                with part.open("rb") as stream:
                    while chunk := stream.read(1024**2):
                        sha.update(chunk)
            received = offset
            started, last = time.monotonic(), time.monotonic()
            with part.open("ab" if offset else "wb") as stream:
                while chunk := response.read(1024**2):
                    stream.write(chunk)
                    sha.update(chunk)
                    received += len(chunk)
                    if received > total:
                        raise ValueError("Transfer exceeded declared size")
                    delay = (received - offset) / self.rate - (time.monotonic() - started)
                    if delay > 0:
                        time.sleep(delay)
                    if time.monotonic() - last > 15:
                        stream.flush()
                        self.record(current=relative, current_bytes=received, current_total=total)
                        last = time.monotonic()
                stream.flush()
                os.fsync(stream.fileno())
            if received != total:
                raise urllib.error.ContentTooShortError(
                    "Incomplete transfer; partial preserved", None
                )
            part.replace(path)
            key = hashlib.sha256(relative.encode()).hexdigest()
            atomic_json(
                self.control / "receipts" / f"{key}.json",
                dict(
                    path=relative,
                    bytes=total,
                    sha256=sha.hexdigest(),
                    identity=current,
                    completed_at=time.time(),
                    checksum_kind="local streamed SHA256; no publisher hash supplied",
                ),
            )
            side.unlink(missing_ok=True)
            self.completed += 1
            self.verified_bytes += total
            self.record(
                completed_files=self.completed,
                verified_download_bytes=self.verified_bytes,
                current_bytes=total,
                current_total=total,
            )
            return str(path), response.headers


def verified_extract(archive, path, record, control):
    """One CRC-checked extraction into staging, then publish; never extract traversal."""
    destination = Path(path).resolve()
    stage = Path(tempfile.mkdtemp(prefix=".scannetpp-extract-", dir=destination))
    members = archive.infolist()
    seen = set()
    for member in members:
        name = Path(member.filename)
        if name.is_absolute() or ".." in name.parts or stat.S_ISLNK(member.external_attr >> 16):
            raise ValueError("Unsafe archive member")
        if str(name) in seen:
            raise ValueError("Duplicate archive member")
        seen.add(str(name))
        if member.flag_bits & 1:
            raise ValueError("Encrypted archive is unsupported")
    record(
        stage="extracting",
        archive=Path(archive.filename).name,
        staging=str(stage),
        extraction_members=len(members),
    )
    # ZipFile.open validates CRC at EOF; no second full read or extra worker pool.
    for member in members:
        target = stage / member.filename
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024**2)
        if target.stat().st_size != member.file_size:
            raise ValueError("Extracted size mismatch")

    def publish(source, target):
        if not target.exists():
            source.replace(target)
        elif source.is_dir() and target.is_dir():
            for item in source.iterdir():
                publish(item, target / item.name)
            source.rmdir()
        else:
            raise FileExistsError("Existing extracted asset must be inspected before replacement")

    for source in stage.iterdir():
        publish(source, destination / source.name)
    stage.rmdir()
    key = hashlib.sha256(str(archive.filename).encode()).hexdigest()
    atomic_json(
        control / "extractions" / f"{key}.json",
        dict(
            stage="passed",
            archive=Path(archive.filename).name,
            crc_checked=True,
            files=sum(not m.is_dir() for m in members),
            expanded_bytes=sum(m.file_size for m in members),
            completed_at=time.time(),
        ),
    )
    record(stage="downloading", staging=None)


def inventory(module, cfg, root):
    split_lists = {s: module.read_txt_list(root / "splits" / f"{s}.txt") for s in cfg["splits"]}
    scenes = list(dict.fromkeys(s for split in cfg["download_splits"] for s in split_lists[split]))
    expected = []
    for scene in scenes:
        split = next(s for s in cfg["splits"] if scene in split_lists[s])
        item = module.ScannetppScene_Release(scene, data_root=root / "data")
        for asset in cfg["default_assets"]:
            if asset not in cfg["exclude_assets"].get(split, []):
                path = Path(getattr(item, asset))
                expected.append(
                    dict(scene=scene, split=split, asset=asset, path=str(path.relative_to(root)))
                )
    return dict(
        scenes=len(scenes),
        splits={s: len(v) for s, v in split_lists.items()},
        assets=expected,
        metadata_sha256={
            n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in cfg["meta_files"]
        },
    )


def run(args):
    import yaml

    os.umask(0o077)
    os.nice(15)
    socket.setdefaulttimeout(60)
    control = args.control.resolve()
    control.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = (control / ".download.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (control / "status.json").exists():
        raise FileExistsError("Existing controller state: inspect before any relaunch")
    cfg = yaml.safe_load(args.config.read_text())
    secret = cfg["token"]
    assert secret and "TOKEN_HERE" not in secret
    root = Path(cfg["data_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    assert not any(root.iterdir()), "Initial launch requires new dataset destination"
    state = dict(
        stage="waiting_for_training_updates",
        pid=os.getpid(),
        started_at=time.time(),
        data_root=str(root),
        training_root=str(args.training),
        rate_limit_mib_s=20,
        processes=1,
        completed_files=0,
        verified_download_bytes=0,
    )

    def record(**changes):
        state.update(changes, updated_at=time.time())
        atomic_json(control / "status.json", state)

    record()
    with (control / "console.log").open("x") as sink:
        sanitized = RedactedWriter(sink, secret)
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = sanitized
        try:
            while not has_updates(args.training):
                status_path = args.training / "status.json"
                status = json.loads(status_path.read_text()) if status_path.exists() else {}
                if status.get("stage") in (
                    "needs_inspection",
                    "failed",
                    "training_complete_review_pending",
                ):
                    raise RuntimeError(
                        "Training did not start successfully; download gate remains closed"
                    )
                record(stage="waiting_for_training_updates")
                time.sleep(20)
            if shutil.disk_usage(root).free < 4 * 1024**4:
                raise RuntimeError("Less than4TiB filesystem headroom; inspect before download")
            record(
                stage="starting_download",
                training_updates_observed=True,
                download_started_at=time.time(),
            )
            sys.path.insert(0, str(args.script.parent))
            spec = importlib.util.spec_from_file_location("private_official_download", args.script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            def freeze_inventory():
                if not (control / "inventory.json").exists():
                    frozen = inventory(module, cfg, root)
                    atomic_json(control / "inventory.json", frozen)
                    record(
                        scenes=frozen["scenes"],
                        expected_assets=len(frozen["assets"]),
                        inventory_frozen=True,
                    )

            module.urlretrieve = AtomicRetriever(
                root, control, record, before_asset=freeze_inventory
            )
            original_zip = zipfile.ZipFile

            class CheckedZip(original_zip):
                def extractall(self, path=None, members=None, pwd=None):
                    if members is not None or pwd is not None:
                        raise ValueError("Unexpected partial/encrypted extraction")
                    verified_extract(self, path, record, control)

            # Private module shares zipfile; this process has no other archive users.
            module.zipfile.ZipFile = CheckedZip
            original_input = builtins.input

            def approved_proceed(prompt=""):
                if prompt:
                    raise ValueError("Unexpected interactive credential/location prompt")
                return "y"

            builtins.input = approved_proceed
            try:
                module.main(SimpleNamespace(config_file=str(args.config)))
            finally:
                builtins.input = original_input
            freeze_inventory()
            frozen = json.loads((control / "inventory.json").read_text())
            assert inventory(module, cfg, root) == frozen, (
                "Official metadata changed during download"
            )
            missing = [item for item in frozen["assets"] if not (root / item["path"]).exists()]
            atomic_json(
                control / "inventory-verification.json",
                dict(
                    stage="passed" if not missing else "missing_assets",
                    scenes=frozen["scenes"],
                    splits=frozen["splits"],
                    expected_assets=len(frozen["assets"]),
                    missing=missing,
                    metadata_sha256=frozen["metadata_sha256"],
                ),
            )
            if missing:
                raise RuntimeError(f"Official script returned with {len(missing)} assets missing")
            record(
                stage="complete",
                finished_at=time.time(),
                scenes=frozen["scenes"],
                expected_assets=len(frozen["assets"]),
            )
        except BaseException as error:
            record(
                stage="needs_inspection", error=sanitized.clean(f"{type(error).__name__}: {error}")
            )
            raise RuntimeError("Private download stopped; inspect sanitized status/log") from None
        finally:
            sanitized.finish()
            sys.stdout, sys.stderr = old_out, old_err


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--script", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--training", type=Path, required=True)
    run(p.parse_args())
