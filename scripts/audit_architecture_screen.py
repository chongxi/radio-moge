"""Independently audit small scratch-screen records; never load checkpoints or run inference."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time


METRICS = tuple(
    f"{part}/{metric}"
    for part in ("spatial", "summary")
    for metric in ("cosine", "norm_ratio", "abs_log_norm_ratio", "normalized_mse")
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return (
        [json.loads(s) for s in path.read_text().splitlines(keepends=True) if s.endswith("\n")]
        if path.exists()
        else []
    )


def atomic(path, value):
    p = path.with_suffix(path.suffix + ".incoming")
    p.write_text(json.dumps(value, indent=2, allow_nan=False))
    p.replace(path)


def aggregate(rows):
    sources = sorted({r["dataset"] for r in rows})
    result = {key: math.fsum(r[key] for r in rows) / len(rows) for key in METRICS}
    for source in sources:
        group = [r for r in rows if r["dataset"] == source]
        for key in METRICS:
            result[f"dataset/{source}/{key}"] = math.fsum(r[key] for r in group) / len(group)
    for key in METRICS:
        result[f"macro/{key}"] = math.fsum(result[f"dataset/{s}/{key}"] for s in sources) / len(
            sources
        )
    return result


def audit_point(directory, records, indices, source_counts, step, nfe, logged):
    rows, metrics, integrity = (
        read(directory / name) for name in ("per_image.json", "metrics.json", "integrity.json")
    )
    assert digest(directory / "per_image.json") == integrity["per_image_sha256"]
    assert len(rows) == integrity["image_count"] == len(indices) == metrics["images"]
    assert Counter(r["dataset"] for r in rows) == source_counts
    for row, index in zip(rows, indices, strict=True):
        reference = records[index]
        assert row["cache_index"] == index
        for key in ("image", "group", "dataset", "screen_id"):
            assert row[key] == reference[key], (directory, index, key)
        assert all(math.isfinite(row[k]) for k in METRICS)
        for part in ("spatial", "summary"):
            assert -1.000001 <= row[f"{part}/cosine"] <= 1.000001
            assert row[f"{part}/norm_ratio"] > 0
            assert row[f"{part}/abs_log_norm_ratio"] >= 0
            assert row[f"{part}/normalized_mse"] >= 0
    expected = aggregate(rows)
    errors = {key: abs(value - metrics[key]) for key, value in expected.items()}
    assert max(errors.values()) < 1e-8, (directory, errors)
    assert metrics["step"] == step and metrics["nfe"] == nfe
    assert metrics == logged, directory
    assert all(math.isfinite(x) for x in metrics.values() if isinstance(x, (int, float)))
    return dict(
        stage="passed",
        checked_at=time.time(),
        step=step,
        nfe=nfe,
        images=len(rows),
        source_counts=dict(source_counts),
        exact_identity_order=True,
        finite=True,
        max_independent_mean_error=max(errors.values()),
        input_sha256={
            name: digest(directory / name)
            for name in ("per_image.json", "metrics.json", "integrity.json")
        },
        spatial_cosine=metrics["macro/spatial/cosine"],
        summary_cosine=metrics["macro/summary/cosine"],
        note="Audit of stored values and aggregation, not a second inference.",
    )


def audit(root):
    plan = read(root / "plan.json")
    manifests = {
        part: lines(root / f"{part}.jsonl") for part in ("train", "validation", "confirmation")
    }
    for part in manifests:
        assert digest(root / f"{part}.jsonl") == plan["manifest_sha256"][part]
    proofs, stability, new = {}, {}, 0
    proofdir = root / "curve-audits"
    proofdir.mkdir(exist_ok=True)
    for spec in plan["candidates"]:
        name, arm = spec["name"], root / spec["name"]
        config_path = arm / "config.json"
        if config_path.exists():
            config = read(config_path)
            assert config["architecture"] == spec and config["fresh_initialization"]
            assert config["seed"] == plan["seed"] and config["loss"] == plan["loss_weights"]
            assert all(v == plan[k] for k, v in config["train"].items())
        train = lines(arm / "train.jsonl")
        for row in train:
            assert all(math.isfinite(v) for v in row.values())
            expected_lr = plan["lr"] * min(1.0, row["step"] / plan["warmup_updates"])
            assert math.isclose(row["lr_used"], expected_lr, rel_tol=1e-12, abs_tol=1e-15)
            expected_total = sum(row[k] * weight for k, weight in plan["loss_weights"].items())
            assert math.isclose(row["total"], expected_total, rel_tol=2e-6, abs_tol=2e-6)
            assert row["gradient_norm"] > 0 and row["seconds"] > 0
        assert all(a["step"] < b["step"] for a, b in zip(train, train[1:]))
        stability[name] = dict(
            logged_rows=len(train),
            last_logged_step=train[-1]["step"] if train else None,
            lr_schedule_checked=True,
            finite=True,
            max_logged_gradient=max((r["gradient_norm"] for r in train), default=None),
        )
        stop_path = arm / "stop-report.json"
        if stop_path.exists():
            stop = read(stop_path)
            assert 0 <= stop["step"] <= plan["updates"]
            assert stop["cursor"] == stop["step"] * plan["effective_batch"]
            stability[name]["stop"] = stop
        for nfe in [1, 2, 4] if spec["family"] == "conditional_flow" else [1]:
            points = {}
            for partition in ("validation", "train_probe"):
                records = manifests["validation" if partition == "validation" else "train"]
                indices = (
                    list(range(len(records)))
                    if partition == "validation"
                    else plan["train_probe_indices"]
                )
                counts = Counter(records[i]["dataset"] for i in indices)
                for logged in lines(arm / f"{partition}-nfe{nfe}.jsonl"):
                    step = logged["step"]
                    directory = arm / f"{partition}-{step:07d}-nfe{nfe}"
                    if not all(
                        (directory / n).exists()
                        for n in ("per_image.json", "metrics.json", "integrity.json")
                    ):
                        continue
                    proofpath = proofdir / f"{name}-{partition}-{step:07d}-nfe{nfe}.json"
                    if proofpath.exists():
                        proof = read(proofpath)
                        assert all(
                            digest(directory / n) == sha for n, sha in proof["input_sha256"].items()
                        )
                        assert read(directory / "metrics.json") == logged
                    else:
                        proof = audit_point(directory, records, indices, counts, step, nfe, logged)
                        atomic(proofpath, proof)
                        new += 1
                    points.setdefault(step, {})[partition] = proof
            proofs[f"{name}/nfe{nfe}"] = points
    rungs = {}
    for step in plan["evaluation_steps"]:
        rows = []
        for name, points in proofs.items():
            if step in points and set(points[step]) == {"validation", "train_probe"}:
                p = points[step]["validation"]
                rows.append(
                    dict(
                        candidate=name,
                        spatial=p["spatial_cosine"],
                        summary=p["summary_cosine"],
                        min_cosine=min(p["spatial_cosine"], p["summary_cosine"]),
                    )
                )
        rungs[str(step)] = sorted(rows, key=lambda r: r["min_cosine"], reverse=True)
    report = dict(
        stage="passed_for_available_completed_points",
        checked_at=time.time(),
        new_proofs=new,
        total_proofs=sum(len(parts) for points in proofs.values() for parts in points.values()),
        stability=stability,
        matched_step_rankings=rungs,
        confirmation_evaluated=False,
        limitations="Single seed/common LR/fixed views; rankings include completed paired points only; no convergence claim.",
    )
    atomic(root / "curve-verification.json", report)
    print(
        json.dumps(
            dict(
                stage=report["stage"],
                new_proofs=new,
                total_proofs=report["total_proofs"],
                current_candidates=[n for n, s in stability.items() if s["logged_rows"]],
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    audit(parser.parse_args().root)
