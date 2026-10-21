"""Paired evaluation differences, bootstrapping scenes within each source."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from radio_moge.checkpoint import atomic_json
from radio_moge.data import read_manifest


DEFAULT_METRICS = [
    "teacher_depth/abs_rel",
    "teacher_depth/rmse",
    "scale_log_error",
    "raw/points",
    "raw/normal",
    "gt_student/abs_rel",
    "gt_student/rmse",
]


def paired_report(before, after, records, metrics=DEFAULT_METRICS, repeats=4000, seed=42):
    def indexed(rows):
        result = {row["image"]: row for row in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate image in paired comparison")
        return result

    left, right, manifest = indexed(before), indexed(after), indexed(records)
    if left.keys() != right.keys() or left.keys() != manifest.keys():
        raise ValueError("Both predictions must cover exactly the same manifest images")
    if repeats < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    groups = defaultdict(lambda: defaultdict(list))
    for image, record in manifest.items():
        groups[record.get("dataset", "unspecified")][record["group"]].append(image)
    random = np.random.default_rng(seed)
    resamples = {
        source: random.integers(len(scenes), size=(repeats, len(scenes)))
        for source, scenes in sorted(groups.items())
    }
    result = {
        "images": len(records),
        "scenes": {source: len(scenes) for source, scenes in groups.items()},
        "bootstrap": {
            "repeats": repeats,
            "seed": seed,
            "unit": "scene within source",
            "interval": "percentile95",
        },
        "sign": "candidate minus baseline; negative is better for these error metrics",
        "postprocess_failures": {
            "baseline": sum(row.get("postprocess_failed", 1) != 0 for row in before),
            "candidate": sum(row.get("postprocess_failed", 1) != 0 for row in after),
        },
        "metrics": {},
        "limits": "Conditional on these checkpoints and this validation sample. Does not include training-seed uncertainty or correct for repeated checkpoint selection. Not an independent test or causal loss comparison unless the inputs are matched experimental controls. Independent-normal-head loss is not point-derived surface quality.",
    }

    def complete(value):
        return isinstance(value, (int, float)) and math.isfinite(value)

    for metric in metrics:
        sources, draws, pooled_sums, pooled_counts = {}, [], [], []
        for source, scenes in sorted(groups.items()):
            images = [image for members in scenes.values() for image in members]
            if not all(
                complete(left[image].get(metric)) and complete(right[image].get(metric))
                for image in images
            ):
                sources[source] = {"status": "incomplete_metric", "images": len(images)}
                continue
            values = np.array(
                [
                    [
                        sum(right[image][metric] - left[image][metric] for image in members),
                        len(members),
                    ]
                    for members in scenes.values()
                ]
            )
            sampled = values[resamples[source]].sum(axis=1)
            distribution = sampled[:, 0] / sampled[:, 1]
            a = float(np.mean([left[image][metric] for image in images]))
            b = float(np.mean([right[image][metric] for image in images]))
            sources[source] = {
                "status": "complete",
                "baseline": a,
                "candidate": b,
                "difference": b - a,
                "difference_ci95": np.quantile(distribution, [0.025, 0.975]).tolist(),
                "scenes_improved": int((values[:, 0] < 0).sum()),
                "scene_count": len(scenes),
                "per_scene_difference": {
                    name: float(value[0] / value[1]) for name, value in zip(scenes, values)
                },
            }
            draws.append(distribution)
            pooled_sums.append(sampled[:, 0])
            pooled_counts.append(sampled[:, 1])
        entry = {"sources": sources}
        if len(draws) == len(groups):
            macro = np.mean(draws, axis=0)
            pooled = np.sum(pooled_sums, axis=0) / np.sum(pooled_counts, axis=0)
            entry["macro_difference"] = float(np.mean([s["difference"] for s in sources.values()]))
            entry["macro_difference_ci95"] = np.quantile(macro, [0.025, 0.975]).tolist()
            entry["image_mean_difference"] = float(
                np.mean([right[i][metric] - left[i][metric] for i in manifest])
            )
            entry["image_mean_difference_ci95"] = np.quantile(pooled, [0.025, 0.975]).tolist()
        result["metrics"][metric] = entry
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--image-prefix", nargs=2, metavar=("OLD", "NEW"))
    parser.add_argument("--repeats", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    predictions = []
    for path in (args.baseline, args.candidate):
        rows = json.loads(path.read_text())
        if args.image_prefix:
            old, new = args.image_prefix
            for row in rows:
                row["image"] = str(Path(new) / Path(row["image"]).relative_to(old))
        predictions.append(rows)
    result = paired_report(
        *predictions, read_manifest(args.manifest), repeats=args.repeats, seed=args.seed
    )
    result["inputs"] = {
        name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in (
            ("baseline", args.baseline),
            ("candidate", args.candidate),
            ("manifest", args.manifest),
        )
    }
    result["image_prefix_translation"] = args.image_prefix
    atomic_json(result, args.output)
