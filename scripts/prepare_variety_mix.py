"""Combine audited indoor RGB with NYU training scenes; keep existing test intact."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from radio_moge.checkpoint import atomic_json
from radio_moge.data import check_split, read_manifest


def load(path, dataset=None):
    rows = read_manifest(path)
    for row in rows:
        if dataset:
            row["dataset"] = dataset
            row["group"] = f"{dataset}:{row['group']}"
    return rows


def prepare(base, output, coco=None):
    if output.exists():
        raise FileExistsError(f"Use a new dataset directory: {output}")
    indoor = base / "indoor-mix-ready-v1-20261021"
    nyu = base / "nyuv2-internal-val-s43"
    inputs = {
        "indoor_train": indoor / "train.jsonl",
        "indoor_val": indoor / "val.jsonl",
        "indoor_val_full": indoor / "val-full.jsonl",
        "nyu_train": nyu / "train.jsonl",
        "nyu_val": nyu / "val.jsonl",
        "nyu_test": base / "nyuv2-labeled-s42/test.jsonl",
    }
    train = load(inputs["indoor_train"]) + load(inputs["nyu_train"], "nyuv2")
    validation = load(inputs["indoor_val"]) + load(inputs["nyu_val"], "nyuv2")
    full_validation = load(inputs["indoor_val_full"]) + load(inputs["nyu_val"], "nyuv2")
    test = load(inputs["nyu_test"], "nyuv2")
    extra_test = []
    if coco:
        if json.loads((coco / "dataset_info.json").read_text())["stage"] != "ready":
            raise ValueError("COCO subset is not ready")
        for split in ["train", "val", "test"]:
            inputs[f"coco_{split}"] = coco / f"{split}.jsonl"
        train += load(inputs["coco_train"])
        coco_val = load(inputs["coco_val"])
        validation += coco_val
        full_validation += coco_val
        extra_test = load(inputs["coco_test"])
    # NYU sha256 is decoded RGB, indoor sha256 is encoded file bytes. Preserve
    # both provenance and a common encoded hash for cross-source duplicate checks.
    encoded = {}
    for rows in (train, validation, full_validation, test, extra_test):
        for row in rows:
            if row["dataset"] == "nyuv2":
                row["decoded_rgb_sha256"] = row["sha256"]
                if row["image"] not in encoded:
                    with open(row["image"], "rb") as file:
                        encoded[row["image"]] = hashlib.file_digest(file, "sha256").hexdigest()
                row["sha256"] = encoded[row["image"]]
            row["sha256_kind"] = "encoded_file_bytes"
    check_split(train, full_validation)
    check_split(train, test)
    check_split(full_validation, test)
    check_split(train, extra_test)
    check_split(full_validation, extra_test)
    check_split(test, extra_test)
    assert {r["image"] for r in validation} <= {r["image"] for r in full_validation}
    temporary = output.with_name(output.name + ".incoming")
    temporary.mkdir(parents=True, exist_ok=False)
    splits = {"train": train, "val": validation, "val-full": full_validation, "test-nyu": test}
    if extra_test:
        splits["test-coco"] = extra_test
    for split, rows in splits.items():
        with (temporary / f"{split}.jsonl").open("w") as file:
            for row in rows:
                row = dict(row, split=split)
                for key in ("image", "depth"):
                    if key in row:
                        row[key] = os.path.relpath(row[key], output)
                file.write(json.dumps(row) + "\n")
    info = {
        "stage": "ready",
        "counts": {s: dict(Counter(r["dataset"] for r in rows)) for s, rows in splits.items()},
        "scene_groups": {s: len({r["group"] for r in rows}) for s, rows in splits.items()},
        "source_manifest_sha256": {
            k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in inputs.items()
        },
        "mixture_weights": (
            {"arkitscenes": 0.35, "hypersim": 0.3, "nyuv2": 0.15, "coco2017": 0.2}
            if coco
            else {"arkitscenes": 0.4, "hypersim": 0.4, "nyuv2": 0.2}
        ),
        "sampling": "Source weights, uniform scenes within source, uniform frames within scene.",
        "selection": "Equal-source mean teacher-depth AbsRel on fixed internal validation.",
        "test_policy": "Existing 290-image NYU test unchanged; 20-scene visualization remains fixed.",
        "verification": "Source scene/path/hash split checks; NYU encoded hashes computed. "
        "Indoor RGB uses prior complete integrity audit; not rehashed here.",
    }
    atomic_json(info, temporary / "dataset_info.json")
    temporary.replace(output)
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coco", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.base.resolve(), args.output.resolve(), args.coco)))
