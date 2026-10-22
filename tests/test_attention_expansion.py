import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "expansion", ROOT / "scripts/prepare_attention_expansion.py"
)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def setup_sources(root):
    base, new = root / "base", root / "tartan"
    base.mkdir()
    new.mkdir()
    counter = 0

    def row(dataset, group, environment=None, split=None):
        nonlocal counter
        counter += 1
        image = root / f"rgb-{counter}.png"
        Image.new("RGB", (20, 20), (counter, 20, 40)).save(image)
        result = dict(
            image=str(image),
            dataset=dataset,
            group=group,
            sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
            sha256_kind="encoded_file_bytes",
        )
        if environment:
            result.update(environment=environment, split=split)
        return result

    def save(folder, split, rows):
        portable = [dict(r, image=os.path.relpath(r["image"], folder)) for r in rows]
        (folder / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in portable))

    training = [row(d, d + ":train") for d in prep.WEIGHTS if d != "tartanair"]
    val = row("nyuv2", "nyuv2:val")
    save(base, "train", training)
    save(base, "val", [val])
    save(base, "val-full", [val])
    save(base, "test-nyu", [row("nyuv2", "nyuv2:test")])
    save(base, "test-coco", [row("coco2017", "coco2017:test")])
    for split, environment in [("train", "hospital"), ("val", "office"), ("test", "amusement")]:
        save(
            new,
            split,
            [row("tartanair", "tartanair:environment:" + environment, environment, split)],
        )
    (base / "dataset_info.json").write_text(json.dumps(dict(stage="ready")))
    (new / "dataset_info.json").write_text(
        json.dumps(dict(stage="ready", counts=dict(train=1, val=1, test=1)))
    )
    (new / "status.json").write_text(json.dumps(dict(stage="complete")))
    return base, new, training, val


def test_existing_holdouts_unchanged_new_training_duplicates_excluded(tmp_path):
    base, new, _, val = setup_sources(tmp_path)
    original = {s: (base / f"{s}.jsonl").read_bytes() for s in prep.HELDOUT}
    leaked = dict(
        val,
        dataset="tartanair",
        group="tartanair:environment:hospital",
        environment="hospital",
        split="train",
    )
    with (new / "train.jsonl").open("a") as file:
        file.write(json.dumps(leaked) + "\n")
    info = json.loads((new / "dataset_info.json").read_text())
    info["counts"]["train"] = 2
    (new / "dataset_info.json").write_text(json.dumps(info))
    out = tmp_path / "expanded"
    result = prep.prepare(base, new, out)
    assert result["exact_duplicate_exclusions"] == 1
    assert result["train_images"] == 5
    assert result["tartanair_counts"] == dict(train=1, val=1, test=1)
    assert all((out / f"{s}.jsonl").read_bytes() == content for s, content in original.items())
    assert all((base / f"{s}.jsonl").read_bytes() == content for s, content in original.items())
    assert len(prep.read_manifest(out / "train.jsonl")) == 5


def test_environment_variant_cannot_leak_into_training(tmp_path):
    base, new, _, _ = setup_sources(tmp_path)
    path = new / "train.jsonl"
    bad = json.loads(path.read_text())
    bad.update(environment="office2", group="tartanair:environment:office")
    path.write_text(json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match="environment-family"):
        prep.prepare(base, new, tmp_path / "expanded")
    assert not (tmp_path / "expanded").exists()


def test_unfinished_download_and_corrupted_image_fail_closed(tmp_path):
    base, new, _, _ = setup_sources(tmp_path)
    (new / "status.json").write_text(json.dumps(dict(stage="downloading")))
    with pytest.raises(ValueError, match="finish first"):
        prep.prepare(base, new, tmp_path / "expanded")
    (new / "status.json").write_text(json.dumps(dict(stage="complete")))
    path = new / json.loads((new / "train.jsonl").read_text())["image"]
    path.write_bytes(b"corrupted image")
    with pytest.raises(ValueError, match="SHA256"):
        prep.prepare(base, new, tmp_path / "expanded")
    assert not (tmp_path / "expanded").exists()


def test_refuse_byte_copy_when_relative_path_would_change_meaning(tmp_path):
    base, new, _, _ = setup_sources(tmp_path)
    p = base / "val.jsonl"
    row = json.loads(p.read_text())
    source = (base / row["image"]).resolve()
    (base / "local.png").write_bytes(source.read_bytes())
    row["image"] = "local.png"
    p.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="change path meaning"):
        prep.prepare(base, new, tmp_path / "expanded")
