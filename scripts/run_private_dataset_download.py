"""Run an approved private ScanNet++ script after training starts, with safe I/O.

The licensed script and token are supplied by the user outside git. This wrapper
contains neither. It leaves the official split/asset selection logic intact.
"""

import argparse
import builtins
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
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
import traceback
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


def transfer_chunks(
    response, url, data, offset, total, identity, workers, record, chunk_bytes=16 * 1024**2
):
    """Bounded HTTP ranges, assembled in order before the existing streamed SHA.

    The first range proves support before parallel requests start. Every range is
    checked against the original length and validator. A server ignoring Range
    falls back to the still-unconsumed original response, never concatenation.
    """
    validator = identity.get("etag") or identity.get("last_modified")
    parallel = workers > 1 and validator and data is None and total - offset >= 2 * chunk_bytes
    if parallel:

        def fetch(start, end):
            request = urllib.request.Request(
                url,
                headers={
                    "Accept-Encoding": "identity",
                    "Range": f"bytes={start}-{end}",
                    "If-Range": validator,
                },
            )
            with urllib.request.urlopen(request, timeout=60) as part:
                for key, header in [("etag", "ETag"), ("last_modified", "Last-Modified")]:
                    if identity.get(key) and part.headers.get(header) != identity[key]:
                        raise ValueError("Source identity changed during parallel transfer")
                if part.status == 200:
                    return None
                expected = end - start + 1
                if (
                    part.status != 206
                    or part.headers.get("Content-Range") != f"bytes {start}-{end}/{total}"
                    or int(part.headers.get("Content-Length", "-1")) != expected
                ):
                    raise ValueError("Incorrect parallel Content-Range or length")
                # Never let an incorrect server response allocate unbounded memory.
                body = part.read(expected + 1)
                if len(body) != expected:
                    raise ValueError("Truncated or oversized parallel range")
                return body

        # Prove Range/validator support with a small prefix; the previous full
        # 16 MiB probe delayed every parallel worker behind one serial transfer.
        # All remaining chunks still undergo the same independent validation.
        probe_bytes = min(chunk_bytes, 64 * 1024)
        end = min(total - 1, offset + probe_bytes - 1)
        probe_started = time.monotonic()
        first = fetch(offset, end)
        probe_seconds = time.monotonic() - probe_started
        if first is not None:
            response.close()
            record(
                active_range_workers=workers,
                range_chunk_bytes=chunk_bytes,
                parallel_range_verified=True,
                range_probe_bytes=end - offset + 1,
                range_probe_seconds=probe_seconds,
            )
            position = end + 1
            pending = deque()
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="dataset-range"
            ) as pool:

                def fill():
                    nonlocal position
                    while position < total and len(pending) < workers:
                        last = min(total - 1, position + chunk_bytes - 1)
                        pending.append(pool.submit(fetch, position, last))
                        position = last + 1

                fill()
                try:
                    yield first
                    del first
                    while pending:
                        body = pending.popleft().result()
                        if body is None:
                            raise ValueError("Server stopped honoring verified ranges")
                        fill()
                        yield body
                finally:
                    for future in pending:
                        future.cancel()
            return
    record(active_range_workers=1, parallel_range_verified=False)
    while chunk := response.read(1024**2):
        yield chunk


class AtomicRetriever:
    def __init__(
        self,
        data_root,
        control,
        record,
        bytes_per_second=20 * 1024**2,
        before_asset=None,
        resume=False,
        range_workers=1,
        performance_file=None,
    ):
        self.root, self.control, self.record = data_root.resolve(), control, record
        if not math.isfinite(bytes_per_second) or bytes_per_second < 0:
            raise ValueError("Transfer limit must be finite and nonnegative; zero means uncapped")
        self.rate = bytes_per_second
        self.completed = 0
        self.verified_bytes = 0
        self.before_asset = before_asset
        self.resume = resume
        self.range_workers = range_workers
        self.performance_file = performance_file

    def selected_range_workers(self):
        workers = self.range_workers
        if self.performance_file and Path(self.performance_file).exists():
            workers = json.loads(Path(self.performance_file).read_text())["range_workers"]
        if type(workers) is not int or not 1 <= workers <= 8:
            raise ValueError("Range worker count must be an integer from1to8")
        return workers

    def __call__(self, url, filename, reporthook=None, data=None):
        path = Path(filename).resolve()
        relative = str(path.relative_to(self.root))
        if self.before_asset and Path(relative).parts[0] == "data":
            self.before_asset()
        receipt_path = (
            self.control / "receipts" / f"{hashlib.sha256(relative.encode()).hexdigest()}.json"
        )
        if self.resume and receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["path"] != relative or not path.is_file():
                raise ValueError("Previously completed transfer requested again without its file")
            if path.stat().st_size != receipt["bytes"]:
                raise ValueError("Previously completed file size changed")
            # Frozen metadata and retained raw files are never overwritten or counted twice.
            return str(path), {"Content-Length": str(receipt["bytes"])}
        if self.resume and path.exists():
            raise FileExistsError("Existing file without receipt; refusing replacement")
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
            if self.resume:
                raise ValueError("Partial lacks source validator; preserve for inspection")
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
            throttle_seconds = 0.0
            workers = self.selected_range_workers()
            transfer_info = {}

            def transfer_record(**changes):
                transfer_info.update(changes)
                self.record(**changes)

            chunks = transfer_chunks(
                response,
                url,
                data,
                offset,
                total,
                current,
                workers=workers if not self.rate else 1,
                record=transfer_record,
            )
            with closing(chunks), part.open("ab" if offset else "wb") as stream:
                for chunk in chunks:
                    stream.write(chunk)
                    sha.update(chunk)
                    received += len(chunk)
                    if received > total:
                        raise ValueError("Transfer exceeded declared size")
                    delay = (
                        (received - offset) / self.rate - (time.monotonic() - started)
                        if self.rate
                        else 0.0
                    )
                    if delay > 0:
                        time.sleep(delay)
                        throttle_seconds += delay
                    if time.monotonic() - last > 15:
                        stream.flush()
                        elapsed = time.monotonic() - started
                        self.record(
                            current=relative,
                            current_bytes=received,
                            current_total=total,
                            current_transfer_seconds=elapsed,
                            current_transfer_mib_s=(received - offset) / elapsed / 1024**2,
                            current_throttle_seconds=throttle_seconds,
                        )
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
                    resumed_from_bytes=offset,
                    transferred_this_attempt_bytes=received - offset,
                    transfer_seconds=time.monotonic() - started,
                    rate_limit_mib_s=self.rate / 1024**2,
                    throttle_seconds=throttle_seconds,
                    requested_range_workers=workers,
                    active_range_workers=transfer_info.get("active_range_workers", 1),
                    range_probe_bytes=transfer_info.get("range_probe_bytes"),
                    range_probe_seconds=transfer_info.get("range_probe_seconds"),
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
                last_transfer_seconds=time.monotonic() - started,
                last_transfer_mib_s=(received - offset) / (time.monotonic() - started) / 1024**2,
                last_throttle_seconds=throttle_seconds,
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


def verify_resume_state(root, control, previous, record=lambda **kw: None):
    """Validate retained receipts and published paths without rehashing the dataset.

    ZIP bytes are deliberately removed by the official downloader after extraction.
    Their retained transfer SHA and CRC/extraction receipt establish prior completion;
    this is a recovery check, not a new full-content audit of extracted data.
    """
    root, control = Path(root).resolve(), Path(control)
    frozen = json.loads((control / "inventory.json").read_text())
    receipts = {}
    last = time.monotonic()
    for receipt_path in (control / "receipts").glob("*.json"):
        value = json.loads(receipt_path.read_text())
        relative = value["path"]
        target = (root / relative).resolve()
        if str(target.relative_to(root)) != relative:
            raise ValueError("Invalid receipt destination")
        key = hashlib.sha256(relative.encode()).hexdigest()
        if receipt_path.stem != key or relative in receipts:
            raise ValueError("Invalid or duplicate receipt identity")
        if not re.fullmatch(r"[0-9a-f]{64}", value["sha256"]):
            raise ValueError("Missing transfer checksum")
        if type(value["bytes"]) is not int or value["bytes"] < 0:
            raise ValueError("Invalid receipt length")
        receipts[relative] = value
        if time.monotonic() - last > 10:
            record(recovery_receipts_checked=len(receipts))
            last = time.monotonic()
    if (
        len(receipts) != previous["completed_files"]
        or sum(r["bytes"] for r in receipts.values()) != previous["verified_download_bytes"]
    ):
        raise ValueError("Receipt totals differ from stopped status")
    metadata = set(frozen["metadata_sha256"])
    for name, digest in frozen["metadata_sha256"].items():
        path = root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen metadata changed")
        if name not in receipts or receipts[name]["sha256"] != digest:
            raise ValueError("Metadata transfer receipt absent or inconsistent")
    seen, used, completed, extractions = set(), set(metadata), 0, 0
    for i, item in enumerate(frozen["assets"]):
        relative = item["path"]
        path = (root / relative).resolve()
        if str(path.relative_to(root)) != relative or relative in seen:
            raise ValueError("Invalid or duplicate inventory destination")
        seen.add(relative)
        archive_relative = str(Path(relative).with_suffix(".zip"))
        direct, archive = receipts.get(relative), receipts.get(archive_relative)
        if direct and archive and relative != archive_relative:
            raise ValueError("Ambiguous transfer receipts for asset")
        if not direct and not archive:
            if path.exists():
                raise ValueError("Published asset without transfer receipt")
            continue
        if not path.exists():
            raise ValueError("Completed transfer has no published asset")
        if archive and not direct:
            key = hashlib.sha256(str(root / archive_relative).encode()).hexdigest()
            extraction = json.loads((control / "extractions" / f"{key}.json").read_text())
            if extraction.get("stage") != "passed" or extraction.get("crc_checked") is not True:
                raise ValueError("ZIP extraction proof absent or failed")
            if extraction["archive"] != Path(archive_relative).name:
                raise ValueError("ZIP extraction proof names another archive")
            used.add(archive_relative)
            extractions += 1
        else:
            if not path.is_file() or path.stat().st_size != direct["bytes"]:
                raise ValueError("Published raw asset size differs from receipt")
            used.add(relative)
        completed += 1
        if time.monotonic() - last > 10:
            record(recovery_assets_checked=i + 1, recovery_completed_assets=completed)
            last = time.monotonic()
    if set(receipts) != used:
        raise ValueError("Receipts outside frozen inventory")
    current = previous.get("current")
    partial_bytes = 0
    if current and current not in receipts:
        path = (root / current).resolve()
        path.relative_to(root)
        partial = path.with_name(path.name + ".part")
        if partial.exists():
            identity = json.loads(partial.with_name(partial.name + ".identity.json").read_text())
            partial_bytes = partial.stat().st_size
            if not (identity.get("etag") or identity.get("last_modified")):
                raise ValueError("Partial lacks a validator")
            if not 0 <= partial_bytes <= identity["total"]:
                raise ValueError("Partial exceeds source length")
    return dict(
        stage="passed",
        checked_at=time.time(),
        completed_files=len(receipts),
        verified_download_bytes=sum(r["bytes"] for r in receipts.values()),
        completed_assets=completed,
        expected_assets=len(frozen["assets"]),
        metadata_receipts=len(metadata),
        crc_extraction_receipts=extractions,
        partial_bytes=partial_bytes,
        retained_files_rehashed=False,
    )


def run(args):
    import yaml

    os.umask(0o077)
    nice = getattr(args, "nice", 15)
    rate_limit = getattr(args, "rate_limit_mib_s", 20)
    if not 0 <= nice <= 19 or not math.isfinite(rate_limit) or rate_limit < 0:
        raise ValueError("Invalid download priority or rate limit")
    os.nice(nice)
    socket.setdefaulttimeout(60)
    control = args.control.resolve()
    control.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = (control / ".download.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    resume = getattr(args, "resume", False)
    status_path = control / "status.json"
    previous = json.loads(status_path.read_text()) if status_path.exists() else None
    if resume:
        if not previous or previous.get("stage") != "needs_inspection":
            raise ValueError("Resume requires an inspected, stopped download")
        # The lock, not a potentially recycled PID from a previous machine boot, excludes peers.
        resume_name = getattr(args, "resume_id", None)
        if not resume_name or not re.fullmatch(r"[a-zA-Z0-9_-]+", resume_name):
            raise ValueError("A unique resume ID is required")
        history = control / "recoveries" / resume_name
        history.mkdir(parents=True, exist_ok=False)
        atomic_json(history / "prior-status.json", previous)
    elif previous:
        raise FileExistsError("Existing controller state: inspect before any relaunch")
    cfg = yaml.safe_load(args.config.read_text())
    secret = cfg["token"]
    assert secret and "TOKEN_HERE" not in secret
    root = Path(cfg["data_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not resume:
        assert not any(root.iterdir()), "Initial launch requires new dataset destination"
    elif str(root) != previous["data_root"]:
        raise ValueError("Resume dataset destination differs from the stopped download")
    state = dict(
        stage="waiting_for_training_updates",
        pid=os.getpid(),
        started_at=time.time(),
        data_root=str(root),
        training_root=str(args.training),
        rate_limit_mib_s=rate_limit,
        bandwidth_uncapped=rate_limit == 0,
        nice=nice,
        requested_range_workers=getattr(args, "range_workers", 1),
        processes=1,
        completed_files=0,
        verified_download_bytes=0,
    )

    def record(**changes):
        state.update(changes, updated_at=time.time())
        atomic_json(control / "status.json", state)

    if resume:
        state.update(
            started_at=previous["started_at"],
            resumed_at=time.time(),
            resume_id=resume_name,
            previous_pid=previous["pid"],
            stage="checking_resume_receipts",
            completed_files=previous["completed_files"],
            verified_download_bytes=previous["verified_download_bytes"],
            training_updates_observed=True,
        )
    record()
    log_path = history / "console.log" if resume else control / "console.log"
    with log_path.open("x") as sink:
        sanitized = RedactedWriter(sink, secret)
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = sanitized
        try:
            while not resume and not has_updates(args.training):
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

            if resume:
                proof = verify_resume_state(root, control, previous, record)
                frozen = json.loads((control / "inventory.json").read_text())
                if inventory(module, cfg, root) != frozen:
                    raise ValueError("Official selection or frozen metadata changed")
                atomic_json(history / "preflight.json", proof)
                record(
                    stage="resuming_download",
                    resume_preflight_passed=True,
                    completed_assets=proof["completed_assets"],
                    expected_assets=proof["expected_assets"],
                    scenes=frozen["scenes"],
                    inventory_frozen=True,
                )
            module.urlretrieve = AtomicRetriever(
                root,
                control,
                record,
                before_asset=freeze_inventory,
                resume=resume,
                bytes_per_second=rate_limit * 1024**2,
                range_workers=getattr(args, "range_workers", 1),
                performance_file=getattr(args, "performance_file", None),
            )
            if resume:
                module.urlretrieve.completed = previous["completed_files"]
                module.urlretrieve.verified_bytes = previous["verified_download_bytes"]
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
                module.zipfile.ZipFile = original_zip
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
            proof = verify_resume_state(root, control, state, record)
            if proof["completed_assets"] != proof["expected_assets"]:
                raise RuntimeError("Asset receipt coverage incomplete")
            atomic_json(control / "completion-receipts.json", proof)
            record(
                stage="complete",
                finished_at=time.time(),
                scenes=frozen["scenes"],
                expected_assets=len(frozen["assets"]),
            )
        except BaseException as error:
            record(
                stage="needs_inspection",
                error=sanitized.clean(f"{type(error).__name__}: {error}"),
                error_frames=[
                    dict(file=Path(f.filename).name, line=f.lineno, function=f.name)
                    for f in traceback.extract_tb(error.__traceback__)
                ],
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
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-id")
    p.add_argument(
        "--rate-limit-mib-s",
        type=float,
        default=20,
        help="Transfer cap in MiB/s; zero disables all bandwidth sleeps",
    )
    p.add_argument("--nice", type=int, default=15, choices=range(20))
    p.add_argument("--range-workers", type=int, default=1, choices=range(1, 9))
    p.add_argument(
        "--performance-file",
        type=Path,
        help="Optional public JSON with range_workers; reread once per transfer",
    )
    run(p.parse_args())
