"""Add audited TartanAir RGB while preserving all existing held-out manifests."""

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import random

from radio_moge.checkpoint import atomic_json
from radio_moge.data import check_split, read_manifest

from download_tartanair_variety import family, split_for


WEIGHTS = dict(arkitscenes=0.28, hypersim=0.24, nyuv2=0.12, coco2017=0.16, tartanair=0.20)
HELDOUT = ("val", "val-full", "test-nyu", "test-coco")


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def prepare(base, tartanair, output):
    base, tartanair, output = (Path(p).resolve() for p in (base, tartanair, output))
    if output.exists() or output.with_name(output.name + ".incoming").exists():
        raise FileExistsError("Use a new expansion dataset directory")
    # Copying the original manifest bytes must retain their relative-path meanings.
    if output.parent != base.parent:
        raise ValueError("Expansion must be a sibling of the base dataset")
    source_info = json.loads((tartanair / "dataset_info.json").read_text())
    source_status = json.loads((tartanair / "status.json").read_text())
    if source_info.get("stage") != "ready" or source_status.get("stage") != "complete":
        raise ValueError("TartanAir download and verification must finish first")
    if json.loads((base / "dataset_info.json").read_text()).get("stage") != "ready":
        raise ValueError("Base dataset must be ready")
    paths = {f"base/{s}": base / f"{s}.jsonl" for s in ("train", *HELDOUT)}
    paths.update({f"tartanair/{s}": tartanair / f"{s}.jsonl" for s in ("train", "val", "test")})
    before = {k: digest(p) for k, p in paths.items()}
    for split in HELDOUT:
        for line in paths[f"base/{split}"].read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("image", "depth"):
                if key in row and (base / row[key]).resolve() != (output / row[key]).resolve():
                    raise ValueError("Copying held-out manifest bytes would change path meaning")
    old = {s: read_manifest(paths[f"base/{s}"]) for s in ("train", *HELDOUT)}
    new = {s: read_manifest(paths[f"tartanair/{s}"]) for s in ("train", "val", "test")}
    if set(r["dataset"] for r in old["train"]) != set(WEIGHTS) - {"tartanair"}:
        raise ValueError("Unexpected base training sources")
    for split, rows in new.items():
        if len(rows) != source_info["counts"][split]:
            raise ValueError("TartanAir manifest count differs from verified receipt")
        for row in rows:
            if (
                row["dataset"] != "tartanair"
                or row["split"] != split
                or split_for(row["environment"]) != split
                or row["group"] != "tartanair:environment:" + family(row["environment"])
            ):
                raise ValueError("TartanAir environment-family split policy violated")
    for rows in (*old.values(), *new.values()):
        if any(r.get("sha256_kind") != "encoded_file_bytes" or not r.get("sha256") for r in rows):
            raise ValueError("Cross-source audit requires comparable encoded-file SHA256 values")
    if not {r["image"] for r in old["val"]} <= {r["image"] for r in old["val-full"]}:
        raise ValueError("Fixed validation must remain a subset of full validation")
    for left, right in combinations(
        (old["train"], old["val-full"], old["test-nyu"], old["test-coco"]), 2
    ):
        check_split(left, right)
    for left, right in combinations(new.values(), 2):
        check_split(left, right)
    # Keep the established base partitions immutable. Exclude any new image that
    # already occurs anywhere in them, including held-out and training partitions.
    old_hashes = {r["sha256"] for rows in old.values() for r in rows}
    excluded = []
    for split, rows in new.items():
        excluded.extend(
            dict(image=r["image"], split=split, reason="encoded_duplicate_of_base")
            for r in rows
            if r["sha256"] in old_hashes
        )
        new[split] = [r for r in rows if r["sha256"] not in old_hashes]
        if not new[split]:
            raise ValueError(f"No TartanAir {split} remains after cross-source exclusion")
    combined = old["train"] + new["train"]
    holdouts = [old["val-full"], old["test-nyu"], old["test-coco"], new["val"], new["test"]]
    for left, right in combinations([combined, *holdouts], 2):
        check_split(left, right)
    # Download already CRC/decode/hash-checked every new RGB. Recheck a fixed
    # sample from every environment here; do not claim a second full byte audit.
    groups = defaultdict(list)
    for rows in new.values():
        for row in rows:
            groups[(row["split"], row["environment"])].append(row)
    samples = []
    for key, rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda r: r["image"])
        for row in random.Random(f"42:{key}").sample(ordered, min(4, len(ordered))):
            if digest(row["image"]) != row["sha256"]:
                raise ValueError("New RGB sample failed SHA256 verification")
            samples.append(
                dict(image=row["image"], split=row["split"], environment=row["environment"])
            )
    if {k: digest(p) for k, p in paths.items()} != before:
        raise ValueError("Source manifests changed during preparation")
    temporary = output.with_name(output.name + ".incoming")
    temporary.mkdir(parents=True, exist_ok=False)
    for split in HELDOUT:
        (temporary / f"{split}.jsonl").write_bytes((base / f"{split}.jsonl").read_bytes())
    for split, rows in {
        "train": combined,
        "val-tartanair": new["val"],
        "test-tartanair": new["test"],
    }.items():
        with (temporary / f"{split}.jsonl").open("w") as file:
            for row in rows:
                row = dict(row)
                for key in ("image", "depth"):
                    if key in row:
                        row[key] = os.path.relpath(row[key], output)
                file.write(json.dumps(row) + "\n")
    if any(digest(temporary / f"{s}.jsonl") != before[f"base/{s}"] for s in HELDOUT):
        raise ValueError("Existing held-out manifests were changed")
    info = dict(
        stage="ready",
        train_images=len(combined),
        train_sources=dict(Counter(r["dataset"] for r in combined)),
        fixed_validation_images=len(old["val"]),
        tartanair_counts={s: len(v) for s, v in new.items()},
        mixture_weights=WEIGHTS,
        source_manifest_sha256=before,
        preserved_heldout_sha256={s: before[f"base/{s}"] for s in HELDOUT},
        exact_duplicate_exclusions=len(excluded),
        verified_rgb_samples=samples,
        checks="Every manifest path exists; all split pairs checked for path/group/encoded SHA overlap; "
        "new source samples rehashed; original held-out manifest bytes preserved.",
        limits="Cross-source duplicate check uses encoded hashes, not a full cross-source pixel/perceptual "
        "audit. Shared synthetic assets are not guaranteed disjoint. TartanAir split is internal.",
        supervision="Frozen MoGe-v2 feature targets from RGB; no TartanAir native depth used.",
        selection="Original fixed884 validation only. New environment-held-out validation reported "
        "separately; no test data in optimization or checkpoint selection.",
    )
    atomic_json(excluded, temporary / "excluded-new-images.json")
    atomic_json(info, temporary / "dataset_info.json")
    temporary.replace(output)
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tartanair", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.base, args.tartanair, args.output)))
