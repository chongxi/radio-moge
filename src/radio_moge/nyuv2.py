"""Prepare the original NYU Depth V2 labeled MAT file without loading it into RAM."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import tempfile
import urllib.request

import numpy as np
from PIL import Image, ImageDraw


SOURCE_URL = (
    "https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
)
SOURCE_BYTES = 2972037809
DATASET_PAGE = "https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while block := file.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def download_source(path):
    """Publish only complete downloads. Existing sources are checked, never replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.stat().st_size != SOURCE_BYTES:
            raise ValueError(f"Unexpected source size: {path}; expected {SOURCE_BYTES} bytes")
        return
    # Unique temporary file prevents two writers from corrupting a partial download.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name, suffix=".part") as file:
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as response:
            size, next_report = 0, 256 * 1024**2
            while block := response.read(8 * 1024**2):
                file.write(block)
                size += len(block)
                if size >= next_report:
                    print(f"Downloaded {size / 1024**3:.2f} GiB", flush=True)
                    next_report += 256 * 1024**2
        if size != SOURCE_BYTES:
            raise ValueError(f"Incomplete source: {size} bytes, expected {SOURCE_BYTES}")
        file.flush()
        os.fsync(file.fileno())
        # Link is atomic and refuses to replace an existing destination.
        os.link(file.name, path)


def scene_group(scene):
    # Conservatively group repeated captures such as office_0001a / office_0001b.
    return re.sub(r"(?<=\d)[a-z]+$", "", scene)


def split_scenes(records, test_fraction=0.2, seed=42):
    """Choose whole scene groups near the requested image ratio, without using labels."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    counts = Counter(row["group"] for row in records)
    if len(counts) < 2:
        raise ValueError("At least two scene groups are required")
    target = round(len(records) * test_fraction)
    rng = random.Random(seed)
    best_error, test_groups = float("inf"), None
    # Each trial shuffles complete groups, then finds the closest prefix. Selecting
    # a prefix avoids favoring scenes with only one image, as a subset-sum would.
    for _ in range(1000):
        order = sorted(counts)
        rng.shuffle(order)
        total = 0
        for k, group in enumerate(order[:-1], 1):
            total += counts[group]
            error = abs(total - target)
            if error < best_error:
                best_error, test_groups = error, set(order[:k])
            if total >= target:
                break
        if best_error == 0:
            break
    train = [row for row in records if row["group"] not in test_groups]
    test = [row for row in records if row["group"] in test_groups]
    if {r["sha256"] for r in train} & {r["sha256"] for r in test}:
        raise ValueError("Duplicate RGB content crosses the scene split")
    return train, test


def matlab_string(file, reference):
    return "".join(chr(int(x)) for x in file[reference][...].ravel())


def write_manifest(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def preview(root, records, destination):
    """Dataset audit only: training RGB and aligned metric depth, no model output."""
    rows = []
    selected, seen = [], set()
    for row in records:
        if row["group"] not in seen:
            selected.append(row)
            seen.add(row["group"])
        if len(selected) == 8:
            break
    for row in selected:
        with Image.open(root / row["image"]) as source:
            rgb = source.resize((320, 240))
        depth = np.load(root / row["depth"], allow_pickle=False)
        # Fixed 0–10m display; black indicates missing sensor measurements.
        gray = (np.nan_to_num(depth, nan=0).clip(0, 10) * 25.5).astype(np.uint8)
        panel = Image.new("RGB", (640, 264), "white")
        panel.paste(rgb, (0, 24))
        panel.paste(Image.fromarray(gray).resize((320, 240), Image.Resampling.NEAREST), (320, 24))
        ImageDraw.Draw(panel).text((5, 5), f'{row["scene"]}: RGB | depth 0-10m', fill="black")
        rows.append(panel)
    sheet = Image.new("RGB", (640 * 2, 264 * ((len(rows) + 1) // 2)), "white")
    for i, panel in enumerate(rows):
        sheet.paste(panel, ((i % 2) * 640, (i // 2) * 264))
    sheet.save(destination)


def prepare(source, output, test_fraction=0.2, seed=42, depth_key="rawDepths"):
    import h5py

    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Dataset already exists: {output}; choose a new output name")
    if depth_key not in {"rawDepths", "depths"}:
        raise ValueError("Use rawDepths (measured) or depths (inpainted)")
    output.parent.mkdir(parents=True, exist_ok=True)
    # The entire completed dataset appears atomically under its final name.
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=f".{output.name}-") as temp:
        staging = Path(temp)
        (staging / "rgb").mkdir()
        (staging / "depth").mkdir()
        records, duplicates, fingerprints = [], [], set()
        with h5py.File(source, "r") as file:
            images, depths = file["images"], file[depth_key]
            # MATLAB H,W,C,N is exposed by HDF5 as N,C,W,H.
            n, channels, width, height = images.shape
            if channels != 3 or depths.shape != (n, width, height):
                raise ValueError("Unexpected NYUv2 array dimensions")
            scenes = file["scenes"][...].ravel()
            types = file["sceneTypes"][...].ravel()
            if len(scenes) != n or len(types) != n:
                raise ValueError("Scene metadata does not match the image count")
            for i in range(n):
                rgb = np.ascontiguousarray(images[i].transpose(2, 1, 0))
                if rgb.dtype != np.uint8:
                    raise ValueError("Expected uint8 RGB")
                fingerprint = hashlib.sha256(rgb.tobytes()).hexdigest()
                if fingerprint in fingerprints:
                    duplicates.append(i + 1)
                    continue
                fingerprints.add(fingerprint)
                depth = np.ascontiguousarray(depths[i].T, dtype=np.float32)
                depth[~np.isfinite(depth) | (depth <= 0)] = 0
                if not (depth > 0).any():
                    raise ValueError(f"Image {i + 1} has no valid depth pixels")
                scene = matlab_string(file, scenes[i])
                row = {
                    "id": f"nyuv2_{i + 1:04d}",
                    "image": f"rgb/{i + 1:04d}.png",
                    "depth": f"depth/{i + 1:04d}.npy",
                    "depth_scale": 1.0,
                    "scene": scene,
                    "group": scene_group(scene),
                    "scene_type": matlab_string(file, types[i]),
                    "sha256": fingerprint,
                }
                Image.fromarray(rgb).save(staging / row["image"], compress_level=3)
                np.save(staging / row["depth"], depth, allow_pickle=False)
                records.append(row)
                if (i + 1) % 100 == 0:
                    print(f"Extracted {i + 1}/{n} images", flush=True)
        train, test = split_scenes(records, test_fraction, seed)
        write_manifest(staging / "train.jsonl", train)
        write_manifest(staging / "test.jsonl", test)
        subset = random.Random(seed).sample(train, min(32, len(train)))
        write_manifest(staging / "overfit.jsonl", subset)
        def summary(rows):
            return {
                "images": len(rows),
                "scene_groups": len({r["group"] for r in rows}),
                "scene_types": dict(sorted(Counter(r["scene_type"] for r in rows).items())),
            }
        info = {
            "dataset": "NYU Depth V2 labeled release",
            "source_url": SOURCE_URL,
            "dataset_page": DATASET_PAGE,
            "citation": "Silberman et al., Indoor Segmentation and Support Inference from RGBD Images, ECCV 2012",
            "source_sha256": sha256_file(source),
            "source_bytes": source.stat().st_size,
            "original_images": n,
            "duplicate_indices_removed": duplicates,
            "image_size": [height, width],
            "sha256_field": "SHA-256 of decoded contiguous RGB uint8 pixels",
            "depth_key": depth_key,
            "depth_units": "meters",
            "depth_protocol": "Full frame, finite positive depth pixels; no benchmark crop or scale alignment",
            "seed": seed,
            "test_fraction_requested": test_fraction,
            "test_fraction_actual": len(test) / len(records),
            "split_algorithm": "Seeded scene permutation closest-prefix v1 (up to 1000 trials)",
            "grouping": "Scene names with trailing letter suffixes merged conservatively",
            "train": summary(train),
            "test": summary(test),
            "notes": "Custom sanity split, not the official NYUv2 benchmark. RGB-only training; depth for final evaluation. Teacher pretraining may contain these scenes.",
        }
        (staging / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")
        preview(staging, train, staging / "training-preview.jpg")
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", default="/shared/radio-moge/datasets/downloads/nyu_depth_v2_labeled.mat"
    )
    parser.add_argument("--output", default="/shared/radio-moge/datasets/nyuv2-labeled-s42")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth-key", choices=["rawDepths", "depths"], default="rawDepths")
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Output already exists; choose a new name to preserve existing artifacts")
    download_source(args.source)
    print(json.dumps(prepare(
        args.source, args.output, args.test_fraction, args.seed, args.depth_key
    ), indent=2))


if __name__ == "__main__":
    main()
