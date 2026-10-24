"""Replay consumed mixture microbatches without opening or decoding any images.

Use an immutable checkpoint and the manifest SHA recorded before training. Run
under the training Python version with data.py identical to the pinned version.
This counts optimizer inputs, excluding calibration, validation and prefetch.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import sys

import torch

from . import data
from .checkpoint import atomic_json


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sampling_records(manifest, expected_sha256):
    """Preserve manifest row order and sampler keys; discard unused metadata."""
    records, digest = [], hashlib.sha256()
    with Path(manifest).open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            if "image" not in row or any(
                not isinstance(row.get(key), str) or not row[key] for key in ("dataset", "group")
            ):
                raise ValueError("Every training row needs image, dataset and group")
            records.append({key: sys.intern(row[key]) for key in ("dataset", "group")})
    if digest.hexdigest() != expected_sha256:
        raise ValueError("Training manifest SHA256 mismatch")
    if not records:
        raise ValueError("Training manifest is empty")
    return records


def replay(records, config, step, cursor):
    settings, source = config["train"], config["data"]
    for name, value in (("step", step), ("cursor", cursor)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    accumulation = settings["accumulation"]
    if type(accumulation) is not int or accumulation < 1 or cursor != step * accumulation:
        raise ValueError("Checkpoint cursor must equal step * accumulation")
    sampler = data.MixtureBatches(
        records,
        source["mixture_weights"],
        settings["batch_size"],
        source["buckets"],
        settings["seed"],
        0,
        cursor,
    )
    available, groups = Counter(), defaultdict(set)
    for row in records:
        available[row["dataset"]] += 1
        groups[row["dataset"]].add(row["group"])
    draws, unique, visited_groups = Counter(), Counter(), defaultdict(set)
    seen = bytearray(len(records))
    for batch in sampler:
        for index, _, _, _ in batch:
            row = records[index]
            name = row["dataset"]
            draws[name] += 1
            if not seen[index]:
                seen[index] = 1
                unique[name] += 1
                visited_groups[name].add(row["group"])
    total = sum(draws.values())
    if total != cursor * settings["batch_size"]:
        raise ValueError("Sampler draw count differs from consumed microbatch count")
    return {
        "method": "Exact MixtureBatches replay from absolute microbatch 0 to cursor (exclusive)",
        "scope": "Optimizer inputs only; excludes calibration, validation and prefetched batches",
        "step": step,
        "cursor": cursor,
        "total_draws": total,
        "available_images": len(records),
        "unique_images": sum(unique.values()),
        "seed": settings["seed"],
        "batch_size": settings["batch_size"],
        "accumulation": accumulation,
        "buckets": source["buckets"],
        "mixture_weights": source["mixture_weights"],
        "per_source": {
            name: {
                "draws": draws[name],
                "available_images": available[name],
                "unique_images": unique[name],
                "image_coverage_fraction": unique[name] / available[name],
                "available_groups": len(groups[name]),
                "unique_groups": len(visited_groups[name]),
                "group_coverage_fraction": len(visited_groups[name]) / len(groups[name]),
            }
            for name in sorted(available)
        },
        "limits": "Unique image rows, not crop uniqueness. Manifest identity relies on the "
        "pre-training audit and pinned file SHA; image bytes are not re-audited here.",
    }


def checkpoint_coverage(checkpoint, manifest_sha256, sampler_sha256, manifest=None):
    sampler_path = Path(data.__file__).resolve()
    actual_sampler_sha = file_sha256(sampler_path)
    if actual_sampler_sha != sampler_sha256:
        raise ValueError("Sampler file SHA256 differs from pinned training code")
    checkpoint = Path(checkpoint)
    # Hash and deserialize the same open inode, even if a caller supplies a
    # replaceable pathname. Frozen selected/last checkpoint paths are preferred.
    with checkpoint.open("rb") as stream:
        before = stream_stat(stream)
        checkpoint_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=True)
        if stream_stat(stream) != before:
            raise ValueError("Checkpoint changed while being read")
    config, step, cursor = state["config"], state["step"], state["cursor"]
    metadata = {key: state.get(key) for key in ("data_digests", "wall_clock_elapsed_seconds")}
    del state  # Release adapter and optimizer tensors before reading the full manifest.
    manifest = Path(manifest or config["data"]["train_manifest"])
    records = sampling_records(manifest, manifest_sha256)
    report = replay(records, config, step, cursor)
    report.update(
        checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_sha,
        checkpoint_metadata=metadata,
        manifest=str(manifest),
        manifest_sha256=manifest_sha256,
        sampler_file=str(sampler_path),
        sampler_sha256=actual_sampler_sha,
        python_version=platform.python_version(),
        training_code_revision=config.get("code_revision"),
    )
    return report


def stream_stat(stream):
    import os

    stat = os.fstat(stream.fileno())
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, help="Optional relocated copy with identical bytes"
    )
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--sampler-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new report path to preserve prior evidence")
    torch.set_num_threads(1)
    report = checkpoint_coverage(
        args.checkpoint,
        args.manifest_sha256,
        args.sampler_sha256,
        args.manifest,
    )
    atomic_json(report, args.output)
    print(json.dumps({key: report[key] for key in ("step", "total_draws", "unique_images")}))


if __name__ == "__main__":
    main()
