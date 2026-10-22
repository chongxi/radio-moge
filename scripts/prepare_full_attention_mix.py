"""Combine three verified new RGB sources with the established four-source mixture.

This writes manifests referencing existing RGB; it does not duplicate the images.
Structured3D/IRS archive completion is deliberately insufficient for readiness.
"""

import argparse
from collections import Counter, defaultdict, deque
import hashlib
from itertools import combinations, zip_longest
import json
import os
from pathlib import Path

from radio_moge.checkpoint import atomic_json
from radio_moge.data import check_split, read_manifest


SOURCES = ("tartanair", "structured3d", "irs")
HELDOUT = ("val", "val-full", "test-nyu", "test-coco")
WEIGHTS = dict(
    arkitscenes=0.175,
    hypersim=0.15,
    nyuv2=0.075,
    coco2017=0.1,
    tartanair=0.2,
    structured3d=0.15,
    irs=0.15,
)


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def scene_sample(rows, count):
    """Fixed seeded ordering, round-robin across scenes, no image duplication."""
    if len(rows) < count:
        raise ValueError(f"Need {count} validation images, only {len(rows)} available")

    def order(value):
        return hashlib.sha256(("42:" + value).encode()).hexdigest()

    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    queues = deque(
        deque(sorted(groups[g], key=lambda r: order(r["image"]))) for g in sorted(groups, key=order)
    )
    result = []
    while len(result) < count:
        group = queues.popleft()
        result.append(group.popleft())
        if group:
            queues.append(group)
    return result


def source_rows(name, folder):
    info = json.loads((folder / "dataset_info.json").read_text())
    if info.get("stage") != "ready" or info.get("pilot"):
        raise ValueError(f"{name}: extracted and audited RGB manifests must be ready")
    if name == "tartanair":
        if json.loads((folder / "status.json").read_text()).get("stage") != "complete":
            raise ValueError("TartanAir download must finish")
        if info.get("selection_mode") != "all_left_rgb_frames":
            raise ValueError("This round requires full TartanAir, not the earlier subset")
    elif info.get("verification", {}).get("rgb_decode_all") is not True:
        raise ValueError(f"{name}: all RGB must pass decoding, not only archive CRC")
    elif info.get("scope") != "all_eligible_rgb":
        raise ValueError(f"{name}: complete eligible RGB scope is required")
    rows = {s: read_manifest(folder / f"{s}.jsonl") for s in ("train", "val", "test")}
    for split, values in rows.items():
        if len(values) != info["counts"][split]:
            raise ValueError(f"{name}/{split}: manifest/receipt count mismatch")
        if any(
            r.get("dataset") != name or r.get("split") != split or not r.get("group")
            for r in values
        ):
            raise ValueError(f"{name}: invalid source/split/scene identity")
    for a, b in combinations(rows.values(), 2):
        check_split(a, b)
    return rows


def prepare(base, sources, output, validation_per_source=128):
    base, output = Path(base).resolve(), Path(output).resolve()
    sources = {name: Path(path).resolve() for name, path in sources.items()}
    if set(sources) != set(SOURCES):
        raise ValueError("All three queued sources are required")
    temporary = output.with_name(output.name + ".incoming")
    if output.exists() or temporary.exists():
        raise FileExistsError("Use a new output directory; inspect interrupted preparation")
    if output.parent != base.parent:
        raise ValueError("Output must be a sibling of the base to preserve held-out paths")
    if json.loads((base / "dataset_info.json").read_text()).get("stage") != "ready":
        raise ValueError("Base data is not ready")
    paths = {f"base/{s}": base / f"{s}.jsonl" for s in ("train", *HELDOUT)}
    for name, folder in sources.items():
        paths.update({f"{name}/{s}": folder / f"{s}.jsonl" for s in ("train", "val", "test")})
        paths[f"{name}/info"] = folder / "dataset_info.json"
    paths["irs/official-test"] = sources["irs"] / "test-official.jsonl"
    before = {key: digest(path) for key, path in paths.items()}
    old = {s: read_manifest(base / f"{s}.jsonl") for s in ("train", *HELDOUT)}
    new = {name: source_rows(name, folder) for name, folder in sources.items()}
    irs_info = json.loads((sources["irs"] / "dataset_info.json").read_text())
    official_test = read_manifest(paths["irs/official-test"])
    if (
        irs_info.get("extra_holdout_manifests") != ["test-official.jsonl"]
        or irs_info.get("official_test_rgb") != len(official_test)
        or any(
            r.get("dataset") != "irs"
            or r.get("split") != "official-test"
            or r.get("official_split") != "test"
            for r in official_test
        )
    ):
        raise ValueError("Original IRS test audit must remain separate and complete")
    if set(r["dataset"] for r in old["train"]) != set(WEIGHTS) - set(SOURCES):
        raise ValueError("Unexpected base source mixture")
    for split in HELDOUT:
        for line in (base / f"{split}.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            for key, value in json.loads(line).items():
                if (
                    key in ("image", "depth")
                    and (base / value).resolve() != (output / value).resolve()
                ):
                    raise ValueError("Copying held-out bytes would change relative path meaning")
    if not {r["image"] for r in old["val"]} <= {r["image"] for r in old["val-full"]}:
        raise ValueError("Original fixed validation is not a full-validation subset")
    all_rows = (
        list(old.values())
        + [rows for splits in new.values() for rows in splits.values()]
        + [official_test]
    )
    if any(
        r.get("sha256_kind") != "encoded_file_bytes" or not r.get("sha256")
        for rows in all_rows
        for r in rows
    ):
        raise ValueError("Comparable encoded SHA256 is required for every image")
    # Immutable established partitions win. Among new sources, held-out data
    # wins over training. Decode hashes catch byte-different duplicates where supplied.
    encoded = {r["sha256"] for rows in old.values() for r in rows}
    decoded = {
        r["decoded_rgb_sha256"]
        for rows in old.values()
        for r in rows
        if r.get("decoded_rgb_sha256")
    }
    official_encoded = {r["sha256"] for r in official_test}
    official_decoded = {
        r["decoded_rgb_sha256"] for r in official_test if r.get("decoded_rgb_sha256")
    }
    if {r["sha256"] for r in old["train"] + old["val"]} & official_encoded or {
        r["decoded_rgb_sha256"] for r in old["train"] + old["val"] if r.get("decoded_rgb_sha256")
    } & official_decoded:
        raise ValueError(
            "IRS official test overlaps established training/selection; inspect before mixing"
        )
    encoded.update(official_encoded)
    decoded.update(official_decoded)
    excluded = []
    for split in ("test", "val", "train"):
        for name in SOURCES:
            kept = []
            for row in new[name][split]:
                pixel_hash = row.get("decoded_rgb_sha256")
                if row["sha256"] in encoded or (pixel_hash and pixel_hash in decoded):
                    excluded.append(
                        dict(
                            dataset=name,
                            split=split,
                            image=row["image"],
                            reason="duplicate_of_preserved_partition",
                        )
                    )
                    continue
                encoded.add(row["sha256"])
                if pixel_hash:
                    decoded.add(pixel_hash)
                kept.append(row)
            if not kept:
                raise ValueError(f"No {name}/{split} remains after duplicate exclusion")
            new[name][split] = kept
    training = old["train"] + [r for name in SOURCES for r in new[name]["train"]]
    partitions = [training, old["val-full"], old["test-nyu"], old["test-coco"]]
    partitions += [new[n][s] for n in SOURCES for s in ("val", "test")]
    for a, b in combinations(partitions, 2):
        check_split(a, b)
        da = {r["decoded_rgb_sha256"] for r in a if r.get("decoded_rgb_sha256")}
        if da & {r["decoded_rgb_sha256"] for r in b if r.get("decoded_rgb_sha256")}:
            raise ValueError("Decoded RGB overlap across partitions")
    fixed = {name: scene_sample(new[name]["val"], validation_per_source) for name in SOURCES}
    selection_by_source = defaultdict(list)
    for row in old["val"] + [r for values in fixed.values() for r in values]:
        selection_by_source[row["dataset"]].append(row)
    selection = [r for batch in zip_longest(*selection_by_source.values()) for r in batch if r]
    # The source extraction pass checks all new RGB. Rehash a bounded, reproducible
    # sample here; do not claim this is a second full-file audit.
    samples = []
    for name in SOURCES:
        for split in ("train", "val", "test"):
            for row in scene_sample(new[name][split], min(16, len(new[name][split]))):
                if digest(row["image"]) != row["sha256"]:
                    raise ValueError(f"{name}: RGB sample hash mismatch")
                samples.append(dict(dataset=name, split=split, image=row["image"]))
    if {key: digest(path) for key, path in paths.items()} != before:
        raise ValueError("Source manifests changed during preparation")
    temporary.mkdir()
    for split in HELDOUT:
        (temporary / f"{split}.jsonl").write_bytes((base / f"{split}.jsonl").read_bytes())
    generated = {"train": training, "val-selection": selection, "test-irs-official": official_test}
    generated.update({f"{s}-{n}": new[n][s] for n in SOURCES for s in ("val", "test")})
    for split, rows in generated.items():
        with (temporary / f"{split}.jsonl").open("w") as file:
            for row in rows:
                portable = dict(row)
                for key in ("image", "depth"):
                    if key in portable:
                        portable[key] = os.path.relpath(portable[key], output)
                file.write(json.dumps(portable) + "\n")
    info = dict(
        stage="ready",
        train_images=len(training),
        train_sources=dict(Counter(r["dataset"] for r in training)),
        selection_images=len(selection),
        selection_sources=dict(Counter(r["dataset"] for r in selection)),
        mixture_weights=WEIGHTS,
        source_manifest_sha256=before,
        preserved_heldout_sha256={s: before[f"base/{s}"] for s in HELDOUT},
        manifest_sha256={p.name: digest(p) for p in temporary.glob("*.jsonl")},
        new_counts={n: {s: len(rows) for s, rows in new[n].items()} for n in SOURCES},
        verified_samples=samples,
        exact_duplicate_exclusions=len(excluded),
        audit_only_manifests=["test-irs-official.jsonl"],
        verification=dict(
            all_paths_exist=True,
            split_groups_paths_hashes_disjoint=True,
            old_holdouts_byte_identical=True,
        ),
        selection="Macro teacher AbsRel over unchanged old fixed validation plus fixed scene-balanced "
        "new-domain validation. No test selection; report each domain and NYU sensor separately.",
        supervision="RGB-only frozen MoGe-v2 targets; no native depth targets added",
        limits="Pixel duplicate audit only where decode hashes are supplied; no full perceptual/asset "
        "overlap guarantee. Weighted sampling does not guarantee every pool image is visited.",
    )
    if any(digest(temporary / f"{s}.jsonl") != before[f"base/{s}"] for s in HELDOUT):
        raise ValueError("Existing hold-out bytes changed")
    atomic_json(excluded, temporary / "excluded-new-images.json")
    atomic_json(info, temporary / "dataset_info.json")
    temporary.replace(output)
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for name in SOURCES:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.base, {n: getattr(args, n) for n in SOURCES}, args.output)))
