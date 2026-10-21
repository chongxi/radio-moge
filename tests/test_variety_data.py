import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import zlib

from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_coco_fetch_validates_crc_and_reuses_existing_bytes(tmp_path):
    download = module("download_coco_variety")
    buffer = io.BytesIO()
    Image.new("RGB", (20, 30), (20, 40, 60)).save(buffer, format="JPEG")
    data = buffer.getvalue()
    path = tmp_path / "rgb/train2017/000001.jpg"
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    task = dict(
        member="train2017/000001.jpg",
        crc32=zlib.crc32(data),
        bytes=len(data),
        split="train",
        official_split="train2017",
    )
    row = download.fetch(task, tmp_path)
    assert row["sha256"] == hashlib.sha256(data).hexdigest()
    assert (row["width"], row["height"]) == (20, 30)


def test_mixture_preserves_nyu_test_and_checks_scene_leakage(tmp_path):
    prep = module("prepare_variety_mix")
    indoor = tmp_path / "indoor-mix-ready-v1-20261021"
    nyu = tmp_path / "nyuv2-internal-val-s43"
    labeled = tmp_path / "nyuv2-labeled-s42"
    for p in (indoor, nyu, labeled):
        p.mkdir()

    def write(directory, split, name, group, dataset=None):
        image = tmp_path / f"{name}.png"
        Image.new("RGB", (20, 20), (len(name) * 10, len(group) * 5, 1)).save(image)
        row = dict(
            image=str(image), group=group, sha256=hashlib.sha256(image.read_bytes()).hexdigest()
        )
        if dataset:
            row["dataset"] = dataset
        path = directory / f"{split}.jsonl"
        path.write_text(json.dumps(row) + "\n")
        return row

    write(indoor, "train", "indoor_train", "room_one", "arkitscenes")
    row = write(indoor, "val", "indoor_validation", "room_two", "arkitscenes")
    (indoor / "val-full.jsonl").write_text(json.dumps(row) + "\n")
    write(nyu, "train", "nyu_train", "nyu_room_one")
    write(nyu, "val", "nyu_val", "nyu_room_two")
    write(labeled, "test", "nyu_test", "nyu_room_three")
    original = (labeled / "test.jsonl").read_bytes()
    info = prep.prepare(tmp_path, tmp_path / "mixed")
    assert info["counts"]["train"]["nyuv2"] == 1
    assert (labeled / "test.jsonl").read_bytes() == original
    test = json.loads((tmp_path / "mixed/test-nyu.jsonl").read_text())
    assert test["group"] == "nyuv2:nyu_room_three"
    # A held-out image from another view of a training scene must be rejected.
    bad = json.loads(original)
    bad["group"] = "nyu_room_one"
    (labeled / "test.jsonl").write_text(json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match="leakage"):
        prep.prepare(tmp_path, tmp_path / "invalid")
