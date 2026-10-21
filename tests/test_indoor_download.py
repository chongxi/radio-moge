import csv
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import pytest


spec = importlib.util.spec_from_file_location(
    "indoor_download", Path(__file__).parents[1] / "scripts" / "indoor_download.py"
)
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


class Response:
    def __init__(self, data, status, headers):
        self.raw = io.BytesIO(data)
        self.status_code, self.headers = status, headers

    def raise_for_status(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class Server:
    def __init__(self, data, ignore=False):
        self.data, self.ignore = data, ignore

    def head(self, *_args, **_kwargs):
        return Response(b"", 200, {"Content-Length": str(len(self.data))})

    def get(self, _url, headers, **_kwargs):
        start, end = map(int, headers["Range"].split("=")[1].split("-"))
        if self.ignore:
            return Response(self.data, 200, {})
        return Response(
            self.data[start : end + 1],
            206,
            {"Content-Range": f"bytes {start}-{end}/{len(self.data)}"},
        )


def test_remote_zip_reads_members_and_rejects_ignored_range(monkeypatch):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("sample.png", b"pixels" * 10000)
        archive.writestr("other.txt", b"camera")
    remote = download.RemoteZip("https://test/archive.zip", Server(stream.getvalue()))
    assert remote.read("sample.png") == b"pixels" * 10000
    assert remote.read("other.txt") == b"camera"
    monkeypatch.setattr(download.time, "sleep", lambda *_: None)
    with pytest.raises(ValueError, match="honor range"):
        download.RemoteZip("https://test/archive.zip", Server(stream.getvalue(), ignore=True))
    broken = bytearray(stream.getvalue())
    broken[50] ^= 255
    with pytest.raises(Exception):
        download.RemoteZip("https://test/archive.zip", Server(broken)).read("sample.png")


def write_csv(path, rows):
    with path.open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_plan_excludes_cross_fold_visits_and_unreleased_hypersim(tmp_path):
    write_csv(
        tmp_path / "arkitscenes-raw-metadata.csv",
        [
            dict(video_id="1", visit_id="a", fold="Training"),
            dict(video_id="2", visit_id="b", fold="Training"),
            dict(video_id="3", visit_id="b", fold="Validation"),
            dict(video_id="4", visit_id="NA", fold="Training"),
            dict(video_id="5", visit_id="c", fold="Training"),
        ],
    )
    write_csv(
        tmp_path / "hypersim-scene-splits.csv",
        [
            dict(scene_name="one", included_in_public_release="True", split_partition_name="train"),
            dict(
                scene_name="two", included_in_public_release="False", split_partition_name="train"
            ),
            dict(
                scene_name="three", included_in_public_release="True", split_partition_name="test"
            ),
        ],
    )
    tasks = download.make_tasks(tmp_path, 101, 42, 24)
    train = [t for t in tasks if t["split"] == "train" and t["dataset"] == "arkitscenes"]
    assert {t["id"] for t in train} == {"a", "c"}
    assert sum(t["target"] for t in train) == 101
    assert [t["id"] for t in tasks if t["dataset"] == "hypersim"] == ["one"]
    assert set(download.farthest_order(33)) == set(range(33))


def test_resume_detects_corruption_and_finalize_detects_leakage(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"old")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"records": [{"image": "image.jpg", "sha256": "bad"}]}))
    with pytest.raises(ValueError, match="hash mismatch"):
        download.verify_existing(tmp_path, receipt)
    tasks = [dict(dataset="a", split=split, id="x") for split in ["train", "val"]]
    for task in tasks:
        download.write_json(
            tmp_path / "receipts" / (download.task_key(task) + ".json"),
            {
                "records": [
                    {
                        "image": "image.jpg",
                        "sha256": "same",
                        "group": "shared",
                        "split": task["split"],
                    }
                ]
            },
        )
    with pytest.raises(ValueError, match="Cross-split"):
        download.finalize(tmp_path, tasks)
