"""Select diverse disjoint scenes, then cache frozen paired features once on the GPU."""

import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import heapq
import json
import os
from pathlib import Path
import random
import time


SOURCES = ["arkitscenes", "hypersim", "nyuv2", "coco2017", "tartanair", "structured3d", "irs"]


def atomic(path, value):
    p = path.with_suffix(path.suffix + ".incoming")
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    p.replace(path)


def key(text):
    return int(hashlib.sha256(("scratch-screen-20261028:" + text).encode()).hexdigest(), 16)


def pools(path):
    """Keep at most 256 uniformly hash-ranked frames per group, streaming the large file."""
    groups = defaultdict(lambda: defaultdict(list))
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for line in file:
            digest.update(line)
            row = json.loads(line)
            for field in ("image", "depth"):
                if field in row:
                    row[field] = os.path.abspath(path.parent / row[field])
            score = key(row["image"])
            group = groups[row["dataset"]][row["group"]]
            item = (-score, row["image"], row)
            if len(group) < 256:
                heapq.heappush(group, item)
            elif score < -group[0][0]:
                heapq.heapreplace(group, item)
    return {
        source: {g: [item[2] for item in sorted(v, reverse=True)] for g, v in source_groups.items()}
        for source, source_groups in groups.items()
    }, digest.hexdigest()


def balanced_select(groups, count):
    order = sorted(groups, key=key)
    selected = []
    for index in range(max(map(len, groups.values()))):
        for group in order:
            if index < len(groups[group]):
                selected.append(groups[group][index])
                if len(selected) == count:
                    return selected
    raise ValueError(f"Insufficient images: {len(selected)} < {count}")


def validation_split(groups, count):
    # Alternate groups; for low-scene domains this explicitly leaves one environment
    # per partition. It does not invent scene independence from adjacent video frames.
    ordered = sorted(groups, key=key)
    a, b = ({g: groups[g] for g in ordered[::2]}, {g: groups[g] for g in ordered[1::2]})
    return balanced_select(a, count), balanced_select(b, count)


def verify_disjoint(partitions):
    for i, (name, rows) in enumerate(partitions.items()):
        assert len({r["image"] for r in rows}) == len(rows)
        for other, values in list(partitions.items())[i + 1 :]:
            for field in ("image", "group", "sha256", "decoded_rgb_sha256"):
                left = {r[field] for r in rows if field in r}
                right = {r[field] for r in values if field in r}
                if left & right:
                    raise ValueError(f"{name}/{other} overlap in {field}")


def prepare(root, dataset, base, revision):
    from radio_moge.architecture_screen import candidates

    root.mkdir(exist_ok=False)
    train, train_sha = pools(dataset / "train.jsonl")
    val, val_sha = pools(dataset / "val-selection.jsonl")
    partitions = {name: [] for name in ("train", "validation", "confirmation")}
    for source in SOURCES:
        partitions["train"] += balanced_select(train[source], 256)
        a, b = validation_split(val[source], 32)
        partitions["validation"] += a
        partitions["confirmation"] += b
    verify_disjoint(partitions)
    hashes, sizes, scene_counts = {}, {}, {}
    for name, rows in partitions.items():
        for row in rows:
            row["screen_id"] = hashlib.sha256(row["image"].encode()).hexdigest()[:20]
        payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        p = root / f"{name}.jsonl"
        t = p.with_suffix(".incoming")
        t.write_text(payload)
        t.replace(p)
        hashes[name] = hashlib.sha256(payload.encode()).hexdigest()
        sizes[name] = dict(Counter(r["dataset"] for r in rows))
        scene_counts[name] = {
            s: len({r["group"] for r in rows if r["dataset"] == s}) for s in SOURCES
        }
    rng = random.Random(47)
    probe = []
    for source in SOURCES:
        ids = [i for i, r in enumerate(partitions["train"]) if r["dataset"] == source]
        probe += rng.sample(ids, 16)
    models = json.loads((base / "attention-24h/config.json").read_text())["models"]
    plan = dict(
        name=root.name,
        stage="prepared",
        execution_revision=revision,
        created_at=time.time(),
        source_manifest_sha256={"train": train_sha, "validation": val_sha},
        manifest_sha256=hashes,
        source_counts=sizes,
        scene_counts=scene_counts,
        candidates=candidates(),
        models=models,
        train_probe_indices=probe,
        train_images=1792,
        validation_images=224,
        confirmation_images=224,
        image_shape=[480, 640],
        indices=[7, 15, 23, 31],
        cache_input_dtype="bfloat16",
        cache_target_dtype="float32",
        calibration="training cache only, all 1792 views",
        loss_weights=dict(spatial=1.0, summary=1.0, cosine=1.0, summary_cosine=1.0),
        seed=42,
        data_seed=47,
        effective_batch=16,
        microbatch=4,
        lr=0.0003,
        warmup_updates=100,
        lr_schedule="warmup then constant",
        weight_decay=0.01,
        clip_grad=1.0,
        updates=1000,
        evaluation_steps=[0, 100, 250, 500, 1000],
        max_seconds_per_candidate=600,
        total_screen_seconds=7200,
        candidate_order=[x["name"] for x in candidates()],
        selection="Pareto of separate source-macro spatial/summary cosine versus updates and GPU time; min(two macro cosines) as transparent shortlist score",
        confirmation_policy="No confirmation metrics consulted for candidate selection; evaluate selected finalists only after shortlist is frozen.",
        next_rung="Review rank stability at 500 and 1000 before deciding any longer comparison; no unbounded sweep.",
        initialization="Random adapter and fresh Adam; no trained adapter weights or parent statistics.",
        frozen=["RADIO", "MoGe encoder", "MoGe decoder"],
        flow="Conditional feature-space linear flow, Gaussian normalized initial state, 1/2/4 Euler calls; matched-parameter t=0 direct control.",
        limitations=[
            "Fixed cached RGB views: fast architecture screen, not augmented full-data convergence.",
            "Different sizes compare useful speed/capacity tradeoffs, not parameter-controlled causal effects.",
            "COCO image IDs do not prove physical-scene independence.",
            "TartanAir and IRS have few disjoint validation environments/families.",
            "Confirmation pool is unused in this screen selection, not necessarily never viewed in older experiments.",
            "Feature cosine does not substitute for decoded geometry; finalists require depth checks.",
        ],
    )
    atomic(root / "plan.json", plan)
    atomic(
        root / "data-verification.json",
        dict(
            stage="passed",
            checked_at=time.time(),
            exact_paths_groups_hashes_disjoint=True,
            source_counts=sizes,
            scene_counts=scene_counts,
            source_manifest_sha256=plan["source_manifest_sha256"],
            manifest_sha256=hashes,
        ),
    )
    print(json.dumps(dict(stage="prepared", root=str(root), counts=sizes, scenes=scene_counts)))


def cache(root):
    import torch
    from radio_moge.checkpoint import atomic_save
    from radio_moge.data import TrainingImages
    from radio_moge.system import build_system

    torch.set_num_threads(2)
    lock = (root / ".cache.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "cache-index.json").exists():
        raise FileExistsError("Cache already complete; do not recompute")
    plan = json.loads((root / "plan.json").read_text())
    config = dict(models=plan["models"], adapter={"kind": "linear"}, data={"buckets": [[480, 640]]})
    start = time.time()
    status = dict(stage="loading_frozen_models", pid=os.getpid(), started_at=start)
    atomic(root / "cache-status.json", status)
    system, unused = build_system(config, "cuda")
    del unused
    directory = root / "cache"
    directory.mkdir(exist_ok=True)
    index = dict(
        stage="passed",
        started_at=start,
        shards=[],
        partitions={},
        input_dtype="bfloat16",
        target_dtype="float32",
        views="fixed seeded crop for train, center crop for validation/confirmation",
        source_images_verified=0,
        quantization_max_absolute={},
    )
    for partition in ("train", "validation", "confirmation"):
        records = [json.loads(x) for x in (root / f"{partition}.jsonl").read_text().splitlines()]
        assert (
            hashlib.sha256((root / f"{partition}.jsonl").read_bytes()).hexdigest()
            == plan["manifest_sha256"][partition]
        )
        data = TrainingImages(records, augment=partition == "train")
        index["partitions"][partition] = len(records)
        for offset in range(0, len(records), 8):
            if time.time() - start > 1800:
                raise TimeoutError(
                    "Cache preparation budget exhausted; inspect partial cache, no automatic retry"
                )
            path = directory / f"{partition}-{offset:05d}.pt"
            if path.exists():
                raise FileExistsError("Partial cache needs explicit audit before reuse")
            storage = defaultdict(list)
            identities = []
            for i in range(offset, min(offset + 8, len(records))):
                source = Path(records[i]["image"])
                before = source.stat()
                with source.open("rb") as file:
                    digest = hashlib.file_digest(file, "sha256").hexdigest()
                assert digest == records[i]["sha256"], str(source)
                image = data[(i, plan["data_seed"] + i * 104729, 480, 640)].unsqueeze(0).cuda()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    features, target = system.encode(image, plan["indices"])
                values = {
                    "spatial": features.spatial,
                    "summary": features.summary,
                    **{f"tap{j}": value for j, value in enumerate(features.intermediates)},
                    "target_spatial": target.spatial,
                    "target_summary": target.summary,
                }
                for name, value in values.items():
                    assert torch.isfinite(value).all()
                    converted = (
                        value.float() if name.startswith("target") else value.to(torch.bfloat16)
                    )
                    if partition == "train" and i < 2:
                        error = float((converted.float() - value.float()).abs().max())
                        index["quantization_max_absolute"][name] = max(
                            error, index["quantization_max_absolute"].get(name, 0)
                        )
                    storage[name].append(converted.cpu())
                after = source.stat()
                assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
                identities.append(
                    dict(
                        image=str(source),
                        sha256=digest,
                        bytes=after.st_size,
                        mtime_ns=after.st_mtime_ns,
                    )
                )
                index["source_images_verified"] += 1
            tensors = {name: torch.cat(values) for name, values in storage.items()}
            atomic_save(tensors, path)
            with path.open("rb") as file:
                digest = hashlib.file_digest(file, "sha256").hexdigest()
            index["shards"].append(
                dict(
                    path=str(path),
                    partition=partition,
                    offset=offset,
                    count=len(identities),
                    bytes=path.stat().st_size,
                    sha256=digest,
                    shapes={k: list(v.shape) for k, v in tensors.items()},
                    images=identities,
                )
            )
            atomic(root / "cache-progress.json", index)
            status.update(
                stage="encoding",
                partition=partition,
                images=index["source_images_verified"],
                updated_at=time.time(),
                last_shard=str(path),
            )
            atomic(root / "cache-status.json", status)
    index.update(finished_at=time.time(), elapsed_seconds=time.time() - start)
    atomic(root / "cache-index.json", index)
    status.update(stage="complete", updated_at=time.time(), images=index["source_images_verified"])
    atomic(root / "cache-status.json", status)
    print(
        json.dumps(
            dict(
                stage="complete",
                images=index["source_images_verified"],
                seconds=index["elapsed_seconds"],
            )
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--base", type=Path)
    p.add_argument("--revision")
    p.add_argument("--cache", action="store_true")
    a = p.parse_args()
    if a.cache:
        try:
            cache(a.root)
        except Exception as error:
            atomic(
                a.root / "cache-error.json",
                dict(
                    stage="failed",
                    type=type(error).__name__,
                    message=str(error),
                    checked_at=time.time(),
                ),
            )
            raise
    else:
        prepare(a.root, a.dataset, a.base, a.revision)
