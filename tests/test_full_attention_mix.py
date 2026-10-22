import importlib.util
import json
from pathlib import Path

import pytest

from test_attention_expansion import setup_sources


spec = importlib.util.spec_from_file_location(
    "full_mix", Path(__file__).resolve().parents[1] / "scripts/prepare_full_attention_mix.py"
)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def sources(root):
    base, tartan, _, _ = setup_sources(root)
    tartan_info = json.loads((tartan / "dataset_info.json").read_text())
    tartan_info["selection_mode"] = "all_left_rgb_frames"
    (tartan / "dataset_info.json").write_text(json.dumps(tartan_info))
    result = {"tartanair": tartan}
    from PIL import Image

    for index, name in enumerate(("structured3d", "irs")):
        folder = root / name
        folder.mkdir()
        for j, split in enumerate(("train", "val", "test")):
            path = folder / f"{split}.png"
            Image.new("RGB", (20, 20), (70 + index * 10 + j, 11, 20)).save(path)
            row = dict(
                image=path.name,
                dataset=name,
                split=split,
                group=f"{name}:{split}",
                sha256=prep.digest(path),
                sha256_kind="encoded_file_bytes",
            )
            (folder / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
        (folder / "dataset_info.json").write_text(
            json.dumps(
                dict(
                    stage="ready",
                    scope="all_eligible_rgb",
                    counts=dict(train=1, val=1, test=1),
                    verification=dict(rgb_decode_all=True),
                )
            )
        )
        result[name] = folder
    return base, result


def test_complete_mixture_preserves_holdouts_and_excludes_cross_source_leak(tmp_path):
    base, new = sources(tmp_path)
    original = {s: (base / f"{s}.jsonl").read_bytes() for s in prep.HELDOUT}
    # Add an IRS training image identical to held-out Structured3D pixels.
    irs = new["irs"]
    val = json.loads((new["structured3d"] / "val.jsonl").read_text())
    leak = dict(
        val,
        image=str(new["structured3d"] / val["image"]),
        dataset="irs",
        group="irs:another-training-scene",
        split="train",
    )
    with (irs / "train.jsonl").open("a") as f:
        f.write(json.dumps(leak) + "\n")
    info = json.loads((irs / "dataset_info.json").read_text())
    info["counts"]["train"] = 2
    (irs / "dataset_info.json").write_text(json.dumps(info))
    output = tmp_path / "merged"
    result = prep.prepare(base, new, output, validation_per_source=1)
    assert result["train_images"] == 7 and result["selection_images"] == 4
    assert result["exact_duplicate_exclusions"] == 1
    assert len(prep.read_manifest(output / "train.jsonl")) == 7
    assert all((output / f"{s}.jsonl").read_bytes() == data for s, data in original.items())
    assert set(result["train_sources"]) == set(prep.WEIGHTS)
    assert abs(sum(result["mixture_weights"].values()) - 1) < 1e-12


def test_archive_completion_or_absent_source_cannot_start_mixture(tmp_path):
    base, new = sources(tmp_path)
    with pytest.raises(ValueError, match="All three"):
        prep.prepare(base, {"tartanair": new["tartanair"]}, tmp_path / "out", 1)
    path = new["irs"] / "dataset_info.json"
    info = json.loads(path.read_text())
    info.update(stage="archives_complete")
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="ready"):
        prep.prepare(base, new, tmp_path / "out", 1)
    info.update(stage="ready", verification={})
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="decoding"):
        prep.prepare(base, new, tmp_path / "out", 1)
    assert not (tmp_path / "out").exists()


def test_scene_leak_and_rgb_corruption_fail_closed(tmp_path):
    base, new = sources(tmp_path)
    p = new["irs"] / "train.jsonl"
    row = json.loads(p.read_text())
    row["group"] = "irs:val"
    p.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="leakage"):
        prep.prepare(base, new, tmp_path / "out", 1)
    row["group"] = "irs:train"
    p.write_text(json.dumps(row) + "\n")
    (new["irs"] / "train.png").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        prep.prepare(base, new, tmp_path / "out", 1)
