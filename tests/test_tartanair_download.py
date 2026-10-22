import importlib.util
import gzip
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import zlib

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location(
    "tartanair_download", Path(__file__).parents[1] / "scripts/download_tartanair_variety.py"
)
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


def test_family_isolation_and_balanced_temporal_sampling():
    index = []
    for environment in [
        "hospital",
        "gascola",
        "office",
        "office2",
        "seasonsforest",
        "seasonsforest_winter",
    ]:
        for trajectory, count in [("P000", 100), ("P001", 200)]:
            for frame in range(count):
                index.append(
                    dict(
                        archive=f"{environment}/Easy/image_left.zip",
                        member=f"{environment}/Easy/{trajectory}/image_left/{frame:06}_left.png",
                        family=download.family(environment),
                        environment=environment,
                        trajectory=f"{environment}/Easy/{trajectory}",
                        split=download.split_for(environment),
                    )
                )
    targets = dict(train=24, val=16, test=16)
    selected = download.select(index, targets, seed=42, stride=5)
    assert selected == download.select(list(reversed(index)), targets, seed=42, stride=5)
    groups = {s: {r["family"] for r in selected if r["split"] == s} for s in targets}
    assert not groups["train"] & (groups["val"] | groups["test"])
    assert not groups["val"] & groups["test"]
    assert all(sum(r["split"] == s for r in selected) == n for s, n in targets.items())
    assert sum(r["environment"] == "hospital" for r in selected) == 12
    for trajectory in {r["trajectory"] for r in selected}:
        frames = sorted(
            int(Path(r["member"]).name[:6]) for r in selected if r["trajectory"] == trajectory
        )
        assert all(b - a >= 5 for a, b in zip(frames, frames[1:]))
        assert frames[-1] > 70  # Sampling spans the track, not just its beginning.


def png(compress_level):
    stream = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(stream, format="PNG", compress_level=compress_level)
    return stream.getvalue()


def test_crc_full_decode_and_decoded_duplicate_prefers_heldout():
    records = []
    for split, compression in [("train", 0), ("test", 9)]:
        data = png(compression)
        info = download.validate_rgb(data, dict(bytes=len(data), crc32=zlib.crc32(data)))
        records.append(dict(image=split + ".png", split=split, **info))
    assert records[0]["sha256"] != records[1]["sha256"]
    kept, rejected = download.deduplicate(records)
    assert [r["split"] for r in kept] == ["test"]
    assert [r["split"] for r in rejected] == ["train"]
    with pytest.raises(ValueError, match="CRC"):
        download.validate_rgb(png(9), dict(bytes=len(png(9)), crc32=0))


def test_corrupted_resume_fails_without_network(tmp_path):
    path = tmp_path / "rgb" / "sample.png"
    path.parent.mkdir()
    path.write_bytes(b"corrupt")
    data = png(9)
    with pytest.raises(ValueError, match="CRC"):
        download.fetch(dict(member="sample.png", bytes=len(data), crc32=zlib.crc32(data)), tmp_path)


def test_small_group_capacity_is_redistributed_without_oversampling():
    assert download.quotas({"tiny": 1, "large": 9}, 6) == {"large": 5, "tiny": 1}
    assert download.quotas({"tiny": 1, "large": 9}, 30) == {"large": 9, "tiny": 1}


def test_gzip_http_catalog_is_cached_as_utf8():
    expected = b"hospital/Easy/image_left.zip 7.192729599\nhospital/Easy/depth_left.zip 2.1\n"
    compressed = gzip.compress(expected)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(compressed)))
            self.end_headers()
            self.wfile.write(compressed)

        def log_message(self, *_):
            pass

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(target=server.handle_request)
        worker.start()
        try:
            data = download.fetch_catalog(f"http://127.0.0.1:{server.server_port}/")
            assert data == expected
            assert download.rgb_archives(data.decode()) == ["hospital/Easy/image_left.zip"]
        finally:
            worker.join(timeout=5)


def test_full_inventory_keeps_every_frame_without_crossing_family_splits():
    index = []
    for env in ("hospital", "office", "office2", "seasonsforest_winter"):
        for frame in range(7):
            index.append(
                dict(
                    archive=f"{env}/Easy/image_left.zip",
                    member=f"{env}/Easy/P000/image_left/{frame:06}_left.png",
                    environment=env,
                    family=download.family(env),
                    split=download.split_for(env),
                )
            )
    selected = download.select_all(list(reversed(index)))
    assert len(selected) == 28
    assert {r["member"] for r in selected} == {r["member"] for r in index}
    assert all(r["split"] == "val" for r in selected if r["environment"] == "office2")
    with pytest.raises(ValueError, match="Repeated"):
        download.select_all(index + index[:1])
    bad = dict(index[0], environment="office2")
    with pytest.raises(ValueError, match="split policy"):
        download.select_all([bad])


def test_full_download_reuses_valid_rgb_and_refuses_corrupt_source(tmp_path):
    task = dict(
        member="hospital/Easy/P000/image_left/000000_left.png",
        archive="hospital/Easy/image_left.zip",
        environment="hospital",
        family="hospital",
        trajectory="hospital/Easy/P000",
        split="train",
    )
    data = png(9)
    task.update(bytes=len(data), crc32=zlib.crc32(data))
    source = tmp_path / "old"
    path = source / "rgb" / task["member"]
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    target = tmp_path / "full"
    result = download.fetch(task, target, source)
    assert result["sha256"] == download.validate_rgb(data, task)["sha256"]
    assert (target / result["image"]).stat().st_ino == path.stat().st_ino
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="CRC"):
        download.fetch(task, tmp_path / "another", source)
    assert not (tmp_path / "another" / "rgb" / task["member"]).exists()


def test_new_selection_cannot_overwrite_existing_download_state(tmp_path):
    import json

    status = b'{"stage":"complete","done":28}'
    (tmp_path / "status.json").write_bytes(status)
    (tmp_path / "plan.json").write_text(json.dumps({"settings": {"selection": "subset"}}))
    args = SimpleNamespace(output=tmp_path, all_frames=True, reuse_root=None, seed=42)
    with pytest.raises(ValueError, match="new directory"):
        download.run(args)
    assert (tmp_path / "status.json").read_bytes() == status


def test_reuse_wait_never_downloads_while_source_active(tmp_path, monkeypatch):
    import fcntl
    import json

    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    p = source / "status.json"
    p.write_text(json.dumps(dict(stage="downloading", done=4, total=8)))
    sleeps = []

    def finish_source(delay):
        sleeps.append(delay)
        assert (
            json.loads((output / "status.json").read_text())["stage"] == "waiting_for_reuse_source"
        )
        p.write_text(json.dumps(dict(stage="complete", done=8, total=8)))

    monkeypatch.setattr(download.time, "sleep", finish_source)
    download.wait_for_reuse_source(source, output, {}, True)
    assert sleeps == [30]
    with (source / ".download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="finish downloading"):
            download.wait_for_reuse_source(source, output, {}, False)
    p.write_text(json.dumps(dict(stage="failed")))
    with pytest.raises(ValueError, match="failed"):
        download.wait_for_reuse_source(source, output, {}, True)
