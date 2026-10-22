import importlib.util
import gzip
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
from pathlib import Path
import sys
import threading
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
