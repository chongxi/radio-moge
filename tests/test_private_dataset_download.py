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
