import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.error
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_private_dataset_download as download


def test_redaction_holds_fragmented_secret_and_url():
    sink = io.StringIO()
    writer = download.RedactedWriter(sink, "test-secret")
    writer.write("token=test-")
    writer.flush()
    assert not sink.getvalue()
    writer.write("secret https://example.test/private?token=test-")
    writer.write("secret\nremaining")
    writer.finish()
    assert sink.getvalue() == "token=[REDACTED] [URL REDACTED]\nremaining"


def test_gate_requires_finite_actual_optimizer_row(tmp_path):
    folder = tmp_path / "train"
    folder.mkdir()
    log = folder / "train.jsonl"
    log.write_text('{"step":0}\npartial')
    assert not download.has_updates(tmp_path)
    row = dict(step=20, lr_used=6e-6, gradient_norm=1.1, parameter_update_l2=0.01)
    log.write_text(json.dumps(row) + "\n")
    assert download.has_updates(tmp_path)
    row["loss"] = float("nan")
    log.write_text(json.dumps(row) + "\n")
    with pytest.raises(AssertionError):
        download.has_updates(tmp_path)


class Response(io.BytesIO):
    def __init__(self, body, status, headers):
        super().__init__(body)
        self.status, self.headers = status, headers


@pytest.mark.parametrize("resume", [False, True])
def test_transfer_retries_truncated_file_safely(tmp_path, monkeypatch, resume):
    root = tmp_path / "data"
    root.mkdir()
    state = tmp_path / "state"
    retriever = download.AtomicRetriever(root, state, lambda **kw: None, bytes_per_second=10**9)
    destination = root / "asset.zip"
    responses = [Response(b"abc", 200, {"Content-Length": "6", "ETag": "stable"})]
    if resume:
        responses.append(
            Response(
                b"def",
                206,
                {"Content-Length": "3", "ETag": "stable", "Content-Range": "bytes 3-5/6"},
            )
        )
    else:
        responses.append(Response(b"abcdef", 200, {"Content-Length": "6", "ETag": "stable"}))
    requests = []

    def open_mock(request, timeout):
        requests.append(request)
        return responses.pop(0)

    monkeypatch.setattr(download.urllib.request, "urlopen", open_mock)
    with pytest.raises(urllib.error.ContentTooShortError):
        retriever("https://scannetpp.mlsg.cit.tum.de/fixture", destination)
    assert not destination.exists()
    assert destination.with_suffix(".zip.part").read_bytes() == b"abc"
    retriever("https://scannetpp.mlsg.cit.tum.de/fixture", destination)
    assert requests[1].get_header("Range") == "bytes=3-"
    assert destination.read_bytes() == b"abcdef"
    receipt = json.loads(next((state / "receipts").glob("*.json")).read_text())
    assert receipt["sha256"] == hashlib.sha256(b"abcdef").hexdigest()
    assert receipt["bytes"] == 6


def test_changed_source_cannot_append(tmp_path, monkeypatch):
    target = tmp_path / "asset"
    target.with_name("asset.part").write_bytes(b"old")
    download.atomic_json(target.with_name("asset.part.identity.json"), dict(etag="old", total=6))
    response = Response(
        b"new", 206, {"Content-Length": "3", "ETag": "new", "Content-Range": "bytes 3-5/6"}
    )
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **kw: response)
    retriever = download.AtomicRetriever(tmp_path, tmp_path / "state", lambda **kw: None)
    with pytest.raises(ValueError, match="identity changed"):
        retriever("https://scannetpp.mlsg.cit.tum.de/fixture", target)
    assert not target.exists()
    assert target.with_name("asset.part").read_bytes() == b"old"


@pytest.mark.parametrize("name", ["../escape", "/absolute"])
def test_extract_rejects_traversal_before_publish(tmp_path, name):
    archive = tmp_path / "asset.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(name, "unsafe")
    with zipfile.ZipFile(archive) as z, pytest.raises(ValueError, match="Unsafe"):
        download.verified_extract(z, tmp_path, lambda **kw: None, tmp_path / "state")
    assert not (tmp_path.parent / "escape").exists()


def test_extract_crc_and_publish_with_receipt(tmp_path):
    archive = tmp_path / "asset.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("rgb/image.txt", "fixture")
    with zipfile.ZipFile(archive) as z:
        download.verified_extract(z, tmp_path, lambda **kw: None, tmp_path / "state")
    assert (tmp_path / "rgb/image.txt").read_text() == "fixture"
    receipt = json.loads(next((tmp_path / "state/extractions").glob("*.json")).read_text())
    assert receipt["crc_checked"] and receipt["expanded_bytes"] == 7


def test_corrupt_archive_never_publishes(tmp_path):
    archive = tmp_path / "asset.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("rgb/image.txt", "fixture")
    archive.write_bytes(archive.read_bytes().replace(b"fixture", b"corrupt"))
    with zipfile.ZipFile(archive) as z, pytest.raises(zipfile.BadZipFile):
        download.verified_extract(z, tmp_path, lambda **kw: None, tmp_path / "state")
    assert not (tmp_path / "rgb").exists()


def recovery_fixture(tmp_path):
    root, state = tmp_path / "dataset", tmp_path / "state"
    root.mkdir()
    meta = root / "splits/train.txt"
    meta.parent.mkdir()
    meta.write_text("fixture\n")
    raw = root / "data/scene/rgb.mkv"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"raw-video")
    extracted = raw.parent / "depth.bin"
    extracted.write_bytes(b"expanded-depth")
    records = [(str(p.relative_to(root)), p.read_bytes()) for p in [meta, raw]]
    records.append(("data/scene/depth.zip", b"prior-zip"))
    for name, body in records:
        key = hashlib.sha256(name.encode()).hexdigest()
        download.atomic_json(
            state / "receipts" / f"{key}.json",
            dict(path=name, bytes=len(body), sha256=hashlib.sha256(body).hexdigest()),
        )
    key = hashlib.sha256(str(root / "data/scene/depth.zip").encode()).hexdigest()
    download.atomic_json(
        state / "extractions" / f"{key}.json",
        dict(stage="passed", archive="depth.zip", crc_checked=True, files=1, expanded_bytes=14),
    )
    download.atomic_json(
        state / "inventory.json",
        dict(
            metadata_sha256={"splits/train.txt": hashlib.sha256(meta.read_bytes()).hexdigest()},
            assets=[
                dict(path="data/scene/" + name) for name in ["rgb.mkv", "depth.bin", "pending.bin"]
            ],
        ),
    )
    previous = dict(completed_files=3, verified_download_bytes=sum(len(b) for _, b in records))
    return root, state, previous


def test_resume_accepts_deleted_zip_with_crc_receipt_and_keeps_all_bytes(tmp_path):
    root, state, previous = recovery_fixture(tmp_path)
    files = [p for folder in [root, state] for p in folder.rglob("*") if p.is_file()]
    before = {str(p): (p.stat().st_ino, p.stat().st_mtime_ns, p.read_bytes()) for p in files}
    proof = download.verify_resume_state(root, state, previous)
    assert proof["completed_assets"] == 2 and proof["expected_assets"] == 3
    assert proof["crc_extraction_receipts"] == 1
    assert before == {
        str(p): (p.stat().st_ino, p.stat().st_mtime_ns, p.read_bytes()) for p in files
    }


@pytest.mark.parametrize("damage", ["metadata", "raw_size", "crc", "unreceipted", "missing"])
def test_resume_rejects_inconsistent_retained_state(tmp_path, damage):
    root, state, previous = recovery_fixture(tmp_path)
    if damage == "metadata":
        (root / "splits/train.txt").write_text("changed")
    elif damage == "raw_size":
        (root / "data/scene/rgb.mkv").write_bytes(b"x")
    elif damage == "crc":
        p = next((state / "extractions").glob("*.json"))
        v = json.loads(p.read_text())
        v["crc_checked"] = False
        download.atomic_json(p, v)
    elif damage == "unreceipted":
        (root / "data/scene/pending.bin").write_bytes(b"unknown")
    else:
        (root / "data/scene/depth.bin").unlink()
    with pytest.raises(ValueError):
        download.verify_resume_state(root, state, previous)


def test_resume_partial_validator_and_counter_preservation(tmp_path, monkeypatch):
    root, state, previous = recovery_fixture(tmp_path)
    target = root / "data/scene/pending.zip"
    target.with_name("pending.zip.part").write_bytes(b"abc")
    download.atomic_json(
        target.with_name("pending.zip.part.identity.json"), dict(etag="stable", total=6)
    )
    previous["current"] = "data/scene/pending.zip"
    assert download.verify_resume_state(root, state, previous)["partial_bytes"] == 3
    requests = []

    def open_mock(request, timeout):
        requests.append(request)
        return Response(
            b"def", 206, {"Content-Length": "3", "ETag": "stable", "Content-Range": "bytes 3-5/6"}
        )

    monkeypatch.setattr(download.urllib.request, "urlopen", open_mock)
    retriever = download.AtomicRetriever(
        root, state, lambda **kw: None, bytes_per_second=10**9, resume=True
    )
    retriever.completed = previous["completed_files"]
    retriever.verified_bytes = previous["verified_download_bytes"]
    metadata = root / "splits/train.txt"
    before = metadata.stat().st_mtime_ns
    retriever("https://scannetpp.mlsg.cit.tum.de/metadata", metadata)
    assert not requests and metadata.stat().st_mtime_ns == before
    retriever("https://scannetpp.mlsg.cit.tum.de/pending", target)
    assert requests[0].get_header("Range") == "bytes=3-"
    assert target.read_bytes() == b"abcdef"
    assert retriever.completed == previous["completed_files"] + 1
    assert retriever.verified_bytes == previous["verified_download_bytes"] + 6


def test_resume_never_truncates_unvalidated_partial(tmp_path):
    target = tmp_path / "asset"
    part = tmp_path / "asset.part"
    part.write_bytes(b"preserved")
    retriever = download.AtomicRetriever(
        tmp_path, tmp_path / "state", lambda **kw: None, resume=True
    )
    with pytest.raises(ValueError, match="validator"):
        retriever("https://scannetpp.mlsg.cit.tum.de/fixture", target)
    assert part.read_bytes() == b"preserved"


def test_full_resume_preserves_history_and_completes_only_missing_asset(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import yaml

    root, state, previous = recovery_fixture(tmp_path)
    previous.update(
        stage="needs_inspection",
        data_root=str(root),
        pid=987654,
        started_at=123.0,
        training_updates_observed=True,
    )
    download.atomic_json(state / "status.json", previous)
    old_log = state / "console.log"
    old_log.write_text("previous interruption\n")
    cfg = tmp_path / "private.yml"
    cfg.write_text(yaml.safe_dump(dict(token="fixture-only-token", data_root=str(root))))
    official = tmp_path / "fixture.py"
    official.write_text("""import zipfile, yaml
from pathlib import Path
from urllib.request import urlretrieve
def main(args):
    root = Path(yaml.safe_load(Path(args.config_file).read_text())["data_root"])
    urlretrieve("https://scannetpp.mlsg.cit.tum.de/meta", root / "splits/train.txt")
    for name in ["rgb.mkv", "depth.bin", "pending.bin"]:
        path = root / "data/scene" / name
        if not path.exists():
            urlretrieve("https://scannetpp.mlsg.cit.tum.de/new", path)
""")
    frozen = json.loads((state / "inventory.json").read_text())
    frozen.update(scenes=1, splits={"fixture": 1})
    download.atomic_json(state / "inventory.json", frozen)
    monkeypatch.setattr(download, "inventory", lambda *a: frozen)
    monkeypatch.setattr(download.os, "nice", lambda *a: None)
    monkeypatch.setattr(download.os, "umask", lambda *a: None)
    monkeypatch.setattr(download.shutil, "disk_usage", lambda *a: SimpleNamespace(free=5 * 1024**4))
    calls = []

    def response(request, timeout):
        calls.append(request)
        return Response(b"new-data", 200, {"Content-Length": "8", "ETag": "stable"})

    monkeypatch.setattr(download.urllib.request, "urlopen", response)
    args = SimpleNamespace(
        control=state,
        config=cfg,
        script=official,
        training=tmp_path / "training",
        resume=True,
        resume_id="fixture-recovery",
    )
    download.run(args)
    result = json.loads((state / "status.json").read_text())
    assert result["stage"] == "complete" and result["completed_files"] == 4
    assert result["verified_download_bytes"] == previous["verified_download_bytes"] + 8
    assert len(calls) == 1
    assert old_log.read_text() == "previous interruption\n"
    assert (
        json.loads((state / "recoveries/fixture-recovery/prior-status.json").read_text())
        == previous
    )
    proof = json.loads((state / "completion-receipts.json").read_text())
    assert proof["completed_assets"] == proof["expected_assets"] == 3
    with pytest.raises(ValueError, match="inspected, stopped"):
        download.run(args)


def test_uncapped_transfer_never_sleeps_and_records_measured_bytes(tmp_path, monkeypatch):
    body = b"payload" * 200000
    monkeypatch.setattr(
        download.urllib.request,
        "urlopen",
        lambda *a, **kw: Response(body, 200, {"Content-Length": str(len(body)), "ETag": "stable"}),
    )

    def forbidden_sleep(seconds):
        pytest.fail("Uncapped transfer attempted to sleep")

    monkeypatch.setattr(download.time, "sleep", forbidden_sleep)
    states = []
    retriever = download.AtomicRetriever(
        tmp_path, tmp_path / "state", lambda **kw: states.append(kw), bytes_per_second=0
    )
    retriever("https://scannetpp.mlsg.cit.tum.de/fixture", tmp_path / "asset")
    receipt = json.loads(next((tmp_path / "state/receipts").glob("*.json")).read_text())
    assert (tmp_path / "asset").read_bytes() == body
    assert receipt["rate_limit_mib_s"] == receipt["throttle_seconds"] == 0
    assert receipt["transferred_this_attempt_bytes"] == len(body)
    assert receipt["resumed_from_bytes"] == 0 and receipt["transfer_seconds"] > 0
    assert next(s["last_transfer_mib_s"] for s in states if "last_transfer_mib_s" in s) > 0
    assert states[-1]["network_active"] is False


@pytest.mark.parametrize("rate", [-1, float("nan"), float("inf")])
def test_invalid_transfer_rate_rejected(tmp_path, rate):
    with pytest.raises(ValueError, match="limit"):
        download.AtomicRetriever(
            tmp_path, tmp_path / "state", lambda **kw: None, bytes_per_second=rate
        )


def test_parallel_ranges_are_ordered_bounded_and_validate_resumed_offsets(monkeypatch):
    import threading
    import time

    body = bytes(range(256)) * 512
    offset, chunk = 111, 4096
    active = maximum = 0
    lock = threading.Lock()
    calls = []

    def response(request, timeout):
        nonlocal active, maximum
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        with lock:
            active += 1
            maximum = max(maximum, active)
            calls.append((start, end))
        # Complete ranges out of order to check assembly ordering.
        time.sleep(0.002 if (start // chunk) % 2 else 0.006)
        with lock:
            active -= 1
        assert request.get_header("If-range") == "stable"
        return Response(
            body[start : end + 1],
            206,
            {
                "Content-Length": str(end - start + 1),
                "Content-Range": f"bytes {start}-{end}/{len(body)}",
                "ETag": "stable",
            },
        )

    monkeypatch.setattr(download.urllib.request, "urlopen", response)
    original = Response(body[offset:], 206, {})
    records = []
    result = b"".join(
        download.transfer_chunks(
            original,
            "https://example.test/asset",
            None,
            offset,
            len(body),
            {"etag": "stable"},
            4,
            lambda **kw: records.append(kw),
            chunk,
        )
    )
    assert result == body[offset:]
    assert 1 < maximum <= 4
    assert original.closed
    assert calls[0][0] == offset and sum(b - a + 1 for a, b in calls) == len(body) - offset
    assert records[-1]["active_range_workers"] == 4


@pytest.mark.parametrize("offset", [0, 12345])
def test_small_probe_starts_parallel_workers_without_serial_full_chunk(monkeypatch, offset):
    import threading

    body = bytes(range(256)) * (5 * 1024 * 1024 // 256)
    chunk = 2 * 1024 * 1024
    calls, records = [], []
    lock = threading.Lock()

    def response(request, timeout):
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        with lock:
            calls.append((start, end))
        return Response(
            body[start : end + 1],
            206,
            {
                "Content-Length": str(end - start + 1),
                "Content-Range": f"bytes {start}-{end}/{len(body)}",
                "ETag": "stable",
            },
        )

    monkeypatch.setattr(download.urllib.request, "urlopen", response)
    result = b"".join(
        download.transfer_chunks(
            Response(body[offset:], 200, {}),
            "https://example.test/asset",
            None,
            offset,
            len(body),
            {"etag": "stable"},
            4,
            lambda **kw: records.append(kw),
            chunk,
        )
    )
    assert calls[0] == (offset, offset + 64 * 1024 - 1)
    ordered = sorted(calls)
    assert ordered[1] == (offset + 64 * 1024, offset + 64 * 1024 + chunk - 1)
    assert all(left[1] + 1 == right[0] for left, right in zip(ordered, ordered[1:]))
    assert ordered[-1][1] == len(body) - 1
    assert result == body[offset:]
    assert records[-1]["range_probe_bytes"] == 64 * 1024
    assert records[-1]["range_probe_seconds"] >= 0


@pytest.mark.parametrize("bad", ["identity", "range", "truncated"])
def test_small_probe_does_not_relax_later_chunk_checks(monkeypatch, bad):
    body = b"x" * (5 * 1024 * 1024)

    def response(request, timeout):
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        headers = {
            "ETag": "stable",
            "Content-Length": str(end - start + 1),
            "Content-Range": f"bytes {start}-{end}/{len(body)}",
        }
        part = body[start : end + 1]
        if start == 64 * 1024:
            if bad == "identity":
                headers["ETag"] = "changed"
            elif bad == "range":
                headers["Content-Range"] = "bytes 0-1/2"
            else:
                part = part[:-1]
        return Response(part, 206, headers)

    monkeypatch.setattr(download.urllib.request, "urlopen", response)
    with pytest.raises(ValueError):
        b"".join(
            download.transfer_chunks(
                Response(body, 200, {}), "https://example.test/asset", None, 0, len(body),
                {"etag": "stable"}, 4, lambda **kw: None, 2 * 1024 * 1024,
            )
        )


@pytest.mark.parametrize("bad", ["identity", "range", "length", "truncated", "oversized"])
def test_parallel_range_rejects_mismatched_or_truncated_source(monkeypatch, bad):
    body = b"x" * 32

    def response(request, timeout):
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        headers = {
            "ETag": "stable",
            "Content-Length": str(end - start + 1),
            "Content-Range": f"bytes {start}-{end}/32",
        }
        part = body[start : end + 1]
        if bad == "identity":
            headers["ETag"] = "changed"
        if bad == "range":
            headers["Content-Range"] = "bytes 1-8/32"
        if bad == "length":
            headers["Content-Length"] = "999"
        if bad == "truncated":
            part = part[:-1]
        if bad == "oversized":
            part += b"extra"
        return Response(part, 206, headers)

    monkeypatch.setattr(download.urllib.request, "urlopen", response)
    with pytest.raises(ValueError):
        b"".join(
            download.transfer_chunks(
                Response(body, 200, {}),
                "https://example.test/a",
                None,
                0,
                len(body),
                {"etag": "stable"},
                4,
                lambda **kw: None,
                8,
            )
        )


def test_ignored_parallel_range_falls_back_without_duplicate_data(monkeypatch):
    body = b"abcdefghijklmnopqrstuvwxyz"
    monkeypatch.setattr(
        download.urllib.request,
        "urlopen",
        lambda *a, **kw: Response(body, 200, {"ETag": "stable", "Content-Length": str(len(body))}),
    )
    records = []
    result = b"".join(
        download.transfer_chunks(
            Response(body, 200, {}),
            "https://example.test/a",
            None,
            0,
            len(body),
            {"etag": "stable"},
            4,
            lambda **kw: records.append(kw),
            8,
        )
    )
    assert result == body
    assert records[-1]["active_range_workers"] == 1


def test_invalid_live_range_worker_setting_is_rejected(tmp_path):
    control = tmp_path / "performance.json"
    control.write_text('{"range_workers": 4}')
    retriever = download.AtomicRetriever(
        tmp_path,
        tmp_path / "state",
        lambda **kw: None,
        bytes_per_second=0,
        performance_file=control,
    )
    assert retriever.selected_range_workers() == 4
    control.write_text('{"range_workers": 8}')
    assert retriever.selected_range_workers() == 8
    control.write_text('{"range_workers": 100}')
    with pytest.raises(ValueError):
        retriever.selected_range_workers()


def test_parallel_retriever_resume_keeps_streamed_full_file_checksum(tmp_path, monkeypatch):
    body = bytes(range(256)) * 16
    target = tmp_path / "asset.bin"
    (tmp_path / "asset.bin.part").write_bytes(body[:123])
    download.atomic_json(
        tmp_path / "asset.bin.part.identity.json", dict(etag="stable", total=len(body))
    )
    original_chunks = download.transfer_chunks
    monkeypatch.setattr(
        download, "transfer_chunks", lambda *a, **kw: original_chunks(*a, **kw, chunk_bytes=512)
    )

    def response(request, timeout):
        raw = request.get_header("Range").removeprefix("bytes=").split("-")
        start = int(raw[0])
        end = int(raw[1]) if raw[1] else len(body) - 1
        return Response(
            body[start : end + 1],
            206,
            {
                "ETag": "stable",
                "Content-Length": str(end - start + 1),
                "Content-Range": f"bytes {start}-{end}/{len(body)}",
            },
        )

    monkeypatch.setattr(download.urllib.request, "urlopen", response)
    retriever = download.AtomicRetriever(
        tmp_path,
        tmp_path / "state",
        lambda **kw: None,
        bytes_per_second=0,
        resume=True,
        range_workers=4,
    )
    retriever("https://scannetpp.mlsg.cit.tum.de/fixture", target)
    receipt = json.loads(next((tmp_path / "state/receipts").glob("*.json")).read_text())
    assert target.read_bytes() == body
    assert receipt["sha256"] == hashlib.sha256(body).hexdigest()
    assert receipt["resumed_from_bytes"] == 123
    assert receipt["transferred_this_attempt_bytes"] == len(body) - 123
    assert receipt["active_range_workers"] == 4 and receipt["throttle_seconds"] == 0


def pipeline_archive(root, control, name, payload=b'fixture'):
    path = root / 'data/scene' / (name + '.zip')
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr(name + '/frame.bin', payload)
    relative = str(path.relative_to(root))
    download.atomic_json(
        control / 'receipts' / (hashlib.sha256(relative.encode()).hexdigest() + '.json'),
        dict(path=relative, bytes=path.stat().st_size,
             sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
    )
    return path


def test_pipeline_overlaps_next_download_and_survives_official_zip_unlink(tmp_path, monkeypatch):
    import threading
    root, control = tmp_path / 'dataset', tmp_path / 'state'
    source = pipeline_archive(root, control, 'first')
    started, release = threading.Event(), threading.Event()
    original_extract = download.verified_extract

    def blocked(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(download, 'verified_extract', blocked)
    pipeline = download.ExtractionPipeline(root, control, lambda **kw: None, zipfile.ZipFile)
    try:
        pipeline.submit(source, source.parent)
        assert started.wait(5)
        source.unlink()  # Exactly what the unchanged official caller does.
        overlap = []
        monkeypatch.setattr(download.urllib.request, 'urlopen', lambda *a, **k: Response(
            b'next-network-data', 200, {'Content-Length': '17', 'ETag': 'stable'}))
        retriever = download.AtomicRetriever(
            root, control, lambda **kw: None, bytes_per_second=0,
            check=pipeline.check, on_chunk=overlap.append,
        )
        retriever('https://scannetpp.mlsg.cit.tum.de/fixture', source.parent / 'next.bin')
        assert sum(overlap) == 17
        assert not (source.parent / 'first').exists()  # Download finished before extraction.
        retained = next((source.parent / '.scannetpp-pipeline').glob('*.zip'))
        assert retained.exists()
    finally:
        release.set()
        pipeline.close()
    assert (source.parent / 'first/frame.bin').read_bytes() == b'fixture'
    assert not retained.exists()
    receipt = json.loads(next((control / 'extractions').glob('*.json')).read_text())
    assert receipt['archive'] == source.name and receipt['crc_checked']
    assert receipt['completed_at'] >= receipt['started_at']
    download.ExtractionPipeline.require_clean(control)


@pytest.mark.parametrize('bound', ['jobs', 'bytes', 'oversized'])
def test_pipeline_backpressure_bounds_unfinished_archives(tmp_path, monkeypatch, bound):
    import threading
    root, control = tmp_path / 'dataset', tmp_path / 'state'
    sources = [pipeline_archive(root, control, f'asset{i}', bytes(128)) for i in range(3)]
    started, release, pressure, returned = (threading.Event() for _ in range(4))
    original_extract = download.verified_extract

    def blocked(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(download, 'verified_extract', blocked)
    size = sources[0].stat().st_size
    limit = 10**9 if bound == 'jobs' else (size if bound == 'bytes' else 1)
    pipeline = download.ExtractionPipeline(
        root, control, lambda **kw: pressure.set() if kw.get('extraction_backpressure') else None,
        zipfile.ZipFile, max_bytes=limit,
    )
    pipeline.submit(sources[0], sources[0].parent)
    assert started.wait(5)
    next_index = 1
    if bound == 'jobs':
        pipeline.submit(sources[1], sources[1].parent)
        next_index = 2
    errors = []

    def submit_next():
        try:
            pipeline.submit(sources[next_index], sources[next_index].parent)
            returned.set()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=submit_next)
    thread.start()
    try:
        assert pressure.wait(5)
        assert not returned.is_set()
        assert len(list((control / 'extraction-pipeline').glob('*.json'))) == next_index
    finally:
        release.set()
        thread.join(5)
        pipeline.close()
    assert not thread.is_alive() and not errors and returned.is_set()


def test_pipeline_crc_failure_propagates_and_retains_archive(tmp_path):
    root, control = tmp_path / 'dataset', tmp_path / 'state'
    source = pipeline_archive(root, control, 'broken')
    source.write_bytes(source.read_bytes().replace(b'fixture', b'corrupt'))
    pipeline = download.ExtractionPipeline(root, control, lambda **kw: None, zipfile.ZipFile)
    pipeline.submit(source, source.parent)
    source.unlink()
    with pytest.raises(RuntimeError, match='Background extraction failed'):
        pipeline.close()
    retained = next((source.parent / '.scannetpp-pipeline').glob('*.zip'))
    assert retained.exists() and not (source.parent / 'broken').exists()
    assert not list((control / 'extractions').glob('*.json'))
    with pytest.raises(ValueError, match='requires inspection'):
        download.ExtractionPipeline.require_clean(control)
    with pytest.raises(RuntimeError, match='Background extraction failed'):
        pipeline.check()


def test_pipeline_cannot_complete_over_preexisting_extracted_file(tmp_path):
    root, control = tmp_path / 'dataset', tmp_path / 'state'
    source = pipeline_archive(root, control, 'existing')
    target = source.parent / 'existing/frame.bin'
    target.parent.mkdir()
    target.write_bytes(b'original')
    pipeline = download.ExtractionPipeline(root, control, lambda **kw: None, zipfile.ZipFile)
    pipeline.submit(source, source.parent)
    with pytest.raises(RuntimeError, match='Background extraction failed'):
        pipeline.close()
    assert target.read_bytes() == b'original'
    assert next((source.parent / '.scannetpp-pipeline').glob('*.zip')).exists()


@pytest.mark.parametrize('failure', [None, 'network', 'interrupt', 'crc'])
def test_run_pipeline_drains_before_completion_or_network_error(tmp_path, monkeypatch, failure):
    import threading
    from types import SimpleNamespace
    import yaml
    root, state, previous = recovery_fixture(tmp_path)
    original_metadata = (root / 'splits/train.txt').read_bytes()
    frozen = json.loads((state / 'inventory.json').read_text())
    frozen.update(scenes=1, splits={'fixture': 1})
    frozen['assets'] = frozen['assets'][:2] + [dict(path='data/scene/newdir'), dict(path='data/scene/next.bin')]
    download.atomic_json(state / 'inventory.json', frozen)
    previous.update(stage='needs_inspection', data_root=str(root), pid=987654, started_at=123.0)
    download.atomic_json(state / 'status.json', previous)
    cfg = tmp_path / 'private.yml'
    cfg.write_text(yaml.safe_dump(dict(token='fixture-only-token', data_root=str(root))))
    official = tmp_path / 'fixture.py'
    official.write_text('''import zipfile,yaml
from pathlib import Path
from urllib.request import urlretrieve
def main(args):
 root=Path(yaml.safe_load(Path(args.config_file).read_text())["data_root"])
 p=root/"data/scene/newdir.zip"
 urlretrieve("https://scannetpp.mlsg.cit.tum.de/zip", p)
 with zipfile.ZipFile(p) as z: z.extractall(p.parent)
 p.unlink()
 urlretrieve("https://scannetpp.mlsg.cit.tum.de/next", root/"data/scene/next.bin")
''')
    body = io.BytesIO()
    with zipfile.ZipFile(body, 'w') as z:
        z.writestr('newdir/image.bin', b'fixture')
    payload = body.getvalue()
    if failure == 'crc':
        payload = payload.replace(b'fixture', b'corrupt')
    extracting, release = threading.Event(), threading.Event()
    original_extract = download.verified_extract

    def blocked(*args, **kwargs):
        extracting.set()
        assert release.wait(5)
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(download, 'verified_extract', blocked)

    def response(request, timeout):
        if request.full_url.endswith('/zip'):
            return Response(payload, 200, {'Content-Length': str(len(payload)), 'ETag': 'stable'})
        assert extracting.wait(5)
        # The actual next transfer is entered while the ZIP worker is blocked.
        assert not (root / 'data/scene/newdir').exists()
        release.set()
        if failure == 'interrupt':
            raise KeyboardInterrupt()
        return Response(b'new', 200, {'Content-Length': '10' if failure == 'network' else '3', 'ETag': 'stable'})

    monkeypatch.setattr(download.urllib.request, 'urlopen', response)
    monkeypatch.setattr(download, 'inventory', lambda *a: frozen)
    monkeypatch.setattr(download.os, 'nice', lambda *a: None)
    monkeypatch.setattr(download.os, 'umask', lambda *a: None)
    monkeypatch.setattr(download.shutil, 'disk_usage', lambda *a: SimpleNamespace(free=5 * 1024**4))
    args = SimpleNamespace(control=state, config=cfg, script=official, training=tmp_path/'training',
                           resume=True, resume_id='overlap-fixture', overlap_extraction=True,
                           rate_limit_mib_s=0)
    if failure:
        with pytest.raises(RuntimeError, match='Private download stopped'):
            download.run(args)
    else:
        download.run(args)
    result = json.loads((state / 'status.json').read_text())
    assert result['stage'] == ('needs_inspection' if failure else 'complete')
    assert not result['extraction_active']
    assert (root / 'splits/train.txt').read_bytes() == original_metadata
    if failure != 'crc':
        assert result['extraction_pending_jobs'] == 0
        proof = download.verify_resume_state(root, state, result)
        assert proof['completed_assets'] == (3 if failure in ('network', 'interrupt') else 4)
        assert not list((root / 'data/scene/.scannetpp-pipeline').glob('*.zip'))
    else:
        assert list((root / 'data/scene/.scannetpp-pipeline').glob('*.zip'))
        assert not (state / 'completion-receipts.json').exists()
