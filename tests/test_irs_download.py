import gzip
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


spec = importlib.util.spec_from_file_location(
    "irs_download", Path(__file__).parents[1] / "scripts/download_irs.py"
)
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


@pytest.fixture
def source():
    data = gzip.compress(os.urandom(32000), mtime=0)
    state = {"mode": "valid", "ranges": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/expired":
                self.send_response(403)
                self.end_headers()
                return
            start, end = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
            state["ranges"].append((start, end))
            self.send_response(200 if state["mode"] == "ignore_range" else 206)
            offset = start + (1 if state["mode"] == "wrong_offset" else 0)
            self.send_header("Content-Range", f"bytes {offset}-{end}/{len(data)}")
            self.send_header("ETag", '"changed"' if state["mode"] == "changed" else '"version1"')
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            try:
                self.wfile.write(data[start : end + 1])
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    item = dict(name="Store.tar.gz", bytes=len(data), id="public-item-id", etag='"version1"')

    class Client:
        calls = 0

        def resolve(self, expected):
            assert expected == item
            self.calls += 1
            return f"http://127.0.0.1:{server.server_port}/" + (
                "expired" if state["mode"] == "expired_once" and self.calls == 1 else "file"
            )

    try:
        yield Client(), item, data, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def make_partial(output, item, data):
    directory = output / "archives"
    directory.mkdir()
    partial = directory / (item["name"] + ".part")
    partial.write_bytes(data)
    download.atomic_json(directory / (item["name"] + ".part.json"), item)
    return partial


def test_resume_and_reuse_verified_archive(source, tmp_path):
    client, item, data, state = source
    partial = make_partial(tmp_path, item, data[:113])
    result = download.download_one(client, item, tmp_path, lambda **kw: None, retries=0)
    assert state["ranges"] == [(113, len(data) - 1)]
    assert not partial.exists()
    assert (tmp_path / "archives" / item["name"]).read_bytes() == data
    assert result["sha256"] == hashlib.sha256(data).hexdigest()
    assert result["uncompressed_bytes"] == 32000
    assert download.download_one(client, item, tmp_path, lambda **kw: None) == result
    assert client.calls == 1
    final = tmp_path / "archives" / item["name"]
    final.write_bytes(b"changed")
    with pytest.raises(download.SourceChanged):
        download.download_one(client, item, tmp_path, lambda **kw: None)


@pytest.mark.parametrize("mode", ["ignore_range", "wrong_offset", "changed"])
def test_unsafe_range_or_version_never_appends(source, tmp_path, mode):
    client, item, data, state = source
    state["mode"] = mode
    partial = make_partial(tmp_path, item, data[:113])
    with pytest.raises(download.DownloadError):
        download.download_one(client, item, tmp_path, lambda **kw: None, retries=0)
    assert partial.read_bytes() == data[:113]
    assert not (tmp_path / "archives" / item["name"]).exists()


def test_expired_download_url_is_refreshed(source, tmp_path, monkeypatch):
    client, item, data, state = source
    state["mode"] = "expired_once"
    monkeypatch.setattr(download.time, "sleep", lambda seconds: None)
    result = download.download_one(client, item, tmp_path, lambda **kw: None, retries=1)
    assert client.calls == 2
    assert result["sha256"] == hashlib.sha256(data).hexdigest()


def test_partial_identity_and_gzip_crc_fail_closed(source, tmp_path):
    client, item, data, state = source
    partial = make_partial(tmp_path, item, data[:113])
    with pytest.raises(download.SourceChanged):
        download.download_one(client, dict(item, etag='"new-version"'), tmp_path, lambda **kw: None)
    assert client.calls == 0
    corrupt = bytearray(data)
    corrupt[-8] ^= 1
    partial.write_bytes(corrupt)
    with pytest.raises(gzip.BadGzipFile):
        download.download_one(client, item, tmp_path, lambda **kw: None)
    assert not (tmp_path / "archives" / item["name"]).exists()


def test_default_selection_is_full_release_without_duplicate_small_subset():
    names = list(download.FULL) + ["IRS_small.tar.gz"]
    items = {
        name: dict(name=name, bytes=42, id=name, etag='"v1"', item_url="private") for name in names
    }
    selected = download.select_items(items)
    assert [x["name"] for x in selected] == list(download.FULL)
    assert all("item_url" not in x for x in selected)
    with pytest.raises(download.DownloadError, match="Missing"):
        download.select_items({})


def test_queue_waits_for_verified_completion_and_released_lock(tmp_path, monkeypatch):
    source = tmp_path / "predecessor"
    source.mkdir()
    status = source / "status.json"
    download.atomic_json(status, {"stage": "downloading"})
    held = (source / ".download.lock").open("a")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    events = []

    def advance(_):
        if len(events) == 1:
            download.atomic_json(status, {"stage": "archives_complete_with_missing_assets"})
        elif len(events) == 2:
            held.close()
        else:
            pytest.fail("Queue did not advance after predecessor lock release")

    monkeypatch.setattr(download.time, "sleep", advance)
    download.wait_for_dataset(source, lambda **event: events.append(event))
    assert [e["predecessor_stage"] for e in events] == [
        "downloading",
        "archives_complete_with_missing_assets",
    ]


@pytest.mark.parametrize("stage", ["failed", "interrupted"])
def test_failed_predecessor_cannot_start_irs(tmp_path, stage):
    download.atomic_json(tmp_path / "status.json", {"stage": stage})
    with pytest.raises(download.DownloadError, match="Preceding dataset"):
        download.wait_for_dataset(tmp_path, lambda **kw: None)


def test_requests_only_web_flow_and_source_identity(monkeypatch):
    item = dict(name="Store.tar.gz", bytes=100, id="public-id", etag='"v1"')
    row = dict(
        FileLeafRef=item["name"],
        File_x0020_Size="100",
        UniqueId=item["id"],
        FSObjType="0",
        **{".etag": item["etag"], ".spItemUrl": "https://onedrive.live.com/_api/item"},
    )

    class Response:
        status_code = 200
        url = "https://onedrive.live.com/?spopath=/personal/abc/Documents/IRS&listurl=/personal/abc/Documents"

        def __init__(self, body=None):
            self.body = body

        def json(self):
            return self.body

    class Session:
        def close(self):
            pass

        def get(self, url, **kw):
            if url.endswith("/_api/item"):
                assert kw["headers"]["Authorization"] == "Bearer temporary-token"
                return Response(
                    {
                        "size": 100,
                        "eTag": '"v1"',
                        "@content.downloadUrl": "https://example.com/temporary-signed-link",
                    }
                )
            return Response()

        def post(self, url, **kw):
            if url.endswith("contextinfo"):
                return Response(
                    {"d": {"GetContextWebInformation": {"FormDigestValue": "temporary-digest"}}}
                )
            assert kw["headers"]["X-RequestDigest"] == "temporary-digest"
            assert kw["params"]["RootFolder"] == "/personal/abc/Documents/IRS"
            return Response(
                {
                    "ListData": {"Row": [row]},
                    "ListSchema": {".driveAccessToken": "access_token=temporary-token"},
                }
            )

    monkeypatch.setattr(download.requests, "Session", Session)
    client = download.IRSClient()
    assert client.resolve(item) == "https://example.com/temporary-signed-link"
    assert "temporary-token" not in json.dumps(download.public_item(client.items[item["name"]]))
    with pytest.raises(download.SourceChanged):
        client.resolve(dict(item, bytes=101))
