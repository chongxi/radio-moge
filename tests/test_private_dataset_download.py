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
