"""Independently audit completed full-data evaluations, without model inference."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import posixpath
import time

from run_private_dataset_download import atomic_json


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    reference = root.parent / "attention-lr1e5-pilot-20261027"
    reference_proof = load(reference / "eval-0088000-verification.json")
    reference_rows_path = reference / "train/eval-0088000/per_image.json"
    assert reference_proof["stage"] == "passed"
    assert sha(reference_rows_path) == reference_proof["sha256"]["per_image.json"]
    before = load(reference_rows_path)
    config = load(root / "config.json")
    remote_manifest = config["data"]["val_manifest"]
    local_manifest = Path(remote_manifest.replace("/home/buildhub/shared/", "/shared/", 1))
    manifest = [json.loads(line) for line in local_manifest.read_text().splitlines()]
    expected = [
        (
            posixpath.normpath(posixpath.join(posixpath.dirname(remote_manifest), row["image"])),
            row["dataset"],
        )
        for row in manifest
    ]
    assert len(expected) == 1268 and len(set(expected)) == 1268
    assert expected == [(row["image"], row["dataset"]) for row in before]
    logs = [json.loads(line) for line in (root / "train/validation.jsonl").read_text().splitlines()]
    new_proofs = []
    for log in logs:
        step = log["step"]
        folder = root / f"train/eval-{step:07d}"
        proof_path = root / f"eval-{step:07d}-verification.json"
        hashes = {name: sha(folder / name) for name in ("metrics.json", "per_image.json")}
        if proof_path.exists():
            proof = load(proof_path)
            assert proof["stage"] == "passed" and proof["sha256"] == hashes
            continue
        rows, metrics = load(folder / "per_image.json"), load(folder / "metrics.json")
        assert [(row["image"], row["dataset"]) for row in rows] == expected
        counts = Counter(row["dataset"] for row in rows)
        assert dict(counts) == reference_proof["source_counts"]
        for row in rows:
            assert all(math.isfinite(v) for v in row.values() if type(v) in (float, int))
        errors = []

        def check(key, actual):
            assert math.isfinite(metrics[key]) and math.isclose(
                metrics[key], actual, abs_tol=1e-10, rel_tol=1e-12
            ), key
            assert metrics[key] == log[key], key
            errors.append(abs(metrics[key] - actual))

        for key in metrics:
            if key == "images":
                actual = len(rows)
            elif key.endswith("/count"):
                actual = sum(key[:-6] in row for row in rows)
            elif key.startswith("dataset/"):
                _, source, field = key.split("/", 2)
                selected = [row for row in rows if row["dataset"] == source]
                actual = (
                    len(selected)
                    if field == "images"
                    else math.fsum(row[field] for row in selected if field in row)
                    / sum(field in row for row in selected)
                )
            elif key.startswith("macro/"):
                field = key[6:]
                means = []
                for source in counts:
                    values = [
                        row[field] for row in rows if row["dataset"] == source and field in row
                    ]
                    if values:
                        means.append(math.fsum(values) / len(values))
                actual = math.fsum(means) / len(means)
            else:
                values = [row[key] for row in rows if key in row]
                actual = math.fsum(values) / len(values)
            check(key, actual)
        total = math.fsum(
            metrics["feature/" + key] * weight for key, weight in config["loss"].items()
        )
        assert math.isclose(total, log["feature/total"], abs_tol=1e-12)
        gt_rows = [row for row in rows if "gt_teacher/abs_rel" in row]
        assert len(gt_rows) == 116 and all(row["dataset"] == "nyuv2" for row in gt_rows)
        teacher_differences = [
            abs(a[key] - b[key])
            for a, b in zip(rows, before, strict=True)
            for key in b
            if key.startswith("gt_teacher/")
        ]
        assert max(teacher_differences) < 1e-8, "Frozen teacher GT changed"
        proof = dict(
            stage="passed",
            checked_at=time.time(),
            step=step,
            images=len(rows),
            source_counts=dict(counts),
            nyu_sensor_rows=len(gt_rows),
            exact_identity_order_source_match=True,
            all_numeric_per_image_values_finite=True,
            all_aggregates_independently_recomputed=True,
            max_aggregate_difference=max(errors),
            max_gt_teacher_difference=max(teacher_differences),
            postprocess_failures=sum(row.get("postprocess_failed", 0) for row in rows),
            feature_weighted_total=total,
            sha256=hashes,
            validation_manifest_sha256=sha(local_manifest),
            metrics={
                key: value
                for key, value in metrics.items()
                if key.startswith(("macro/", "gt_student/"))
            },
            scope="Full fixed1268 validation; teacher agreement and sensor accuracy remain distinct. No inference.",
        )
        atomic_json(proof_path, proof)
        new_proofs.append(
            dict(
                step=step, max_error=max(errors), postprocess_failures=proof["postprocess_failures"]
            )
        )
    print(json.dumps(dict(stage="passed", evaluations=len(logs), new_proofs=new_proofs)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    audit(parser.parse_args().root)
