"""Frozen feature diagnostics and teacher-feature replacement at the decoder boundary."""

import argparse
from collections import defaultdict
import copy
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F

from .adapters import EncoderFeatures, GeometryAdapter
from .checkpoint import atomic_json
from .data import evaluation_image, read_manifest
from .decoder import postprocess
from .losses import FeatureStatistics
from .metrics import aggregate, depth_metrics
from .system import build_system


def feature_metrics(predicted, target, statistics):
    result = {f"feature/{k}": v.item() for k, v in statistics.loss(predicted, target).items()}
    for name in ("spatial", "summary"):
        p, t = getattr(predicted, name).float(), getattr(target, name).float()
        cosine = F.cosine_similarity(p, t, dim=1).flatten()
        ratio = p.norm(dim=1).clamp_min(1e-8) / t.norm(dim=1).clamp_min(1e-8)
        result.update({
            f"{name}/cosine": cosine.mean().item(),
            f"{name}/norm_ratio": ratio.mean().item(),
            f"{name}/abs_log_norm_ratio": ratio.log().abs().mean().item(),
            f"{name}/relative_l2": ((p - t).norm(dim=1)
                                     / t.norm(dim=1).clamp_min(1e-8)).mean().item(),
        })
        if name == "spatial":
            for q in (0.1, 0.5, 0.9):
                result[f"spatial/cosine_p{int(q * 100)}"] = cosine.quantile(q).item()
    return result


def replace_scale(raw, scale_source):
    """Exact for MoGeDecoder: summary feeds only scale_head; keep inputs immutable."""
    return {**raw, "metric_scale": scale_source["metric_scale"]}


def factorized_features(predicted, target, component):
    """Replace only vector norm or only direction, along the channel dimension."""
    branch, part = component.split("_")
    if branch not in ("spatial", "summary") or part not in ("norm", "direction"):
        raise ValueError(f"Unknown feature intervention: {component}")
    p, t = getattr(predicted, branch).float(), getattr(target, branch).float()
    pnorm, tnorm = p.norm(dim=1, keepdim=True), t.norm(dim=1, keepdim=True)
    if (pnorm <= 1e-8).any() or (tnorm <= 1e-8).any():
        raise ValueError("Cannot separate norm and direction for zero vectors")
    mixed = p / pnorm * tnorm if part == "norm" else t / tnorm * pnorm
    return EncoderFeatures(mixed if branch == "spatial" else predicted.spatial,
                           mixed if branch == "summary" else predicted.summary)


def scene_difference(rows, candidate, baseline, repeats=2000):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row[candidate] - row[baseline])
    values = np.array([[sum(v), len(v)] for v in groups.values()])
    draws = np.random.default_rng(42).integers(len(values), size=(repeats, len(values)))
    totals = values[draws].sum(axis=1)
    ci = np.quantile(totals[:, 0] / totals[:, 1], [0.025, 0.975])
    return {
        "difference": sum(v[0] for v in values) / sum(v[1] for v in values),
        "scene_bootstrap_ci95": ci.tolist(), "scenes": len(groups),
        "images_improved": sum(row[candidate] < row[baseline] for row in rows),
        "repeats": repeats, "seed": 42,
    }


def summarize(rows):
    by_checkpoint = defaultdict(list)
    for row in rows:
        by_checkpoint[row["checkpoint"]].append(row)
    result = {}
    predictors = ["spatial/cosine", "summary/cosine", "feature/spatial", "feature/summary",
                  "spatial/abs_log_norm_ratio", "summary/abs_log_norm_ratio"]
    outcomes = ["student/gt/abs_rel", "student/teacher/abs_rel", "scale_log_error"]
    for name, subset in by_checkpoint.items():
        correlations = {}
        for x in predictors:
            for y in outcomes:
                xs, ys = [r[x] for r in subset], [r[y] for r in subset]
                correlations[f"{x} -> {y}"] = (
                    float(spearmanr(xs, ys).statistic)
                    if len(set(xs)) > 1 and len(set(ys)) > 1 else None
                )
        interventions = {}
        variants = [k.removesuffix("/gt/abs_rel") for k in subset[0]
                    if k.endswith("/gt/abs_rel") and not k.startswith("student/")]
        for intervention in variants:
            interventions[intervention] = {
                target: scene_difference(subset, f"{intervention}/{target}/abs_rel",
                                         f"student/{target}/abs_rel")
                for target in ("gt", "teacher")
            }
        result[name] = {"means": aggregate(subset), "spearman": correlations,
                        "paired_interventions": interventions}
    return result


def identity(path):
    s = Path(path).stat()
    return {"inode": s.st_ino, "bytes": s.st_size, "mtime_ns": s.st_mtime_ns}


@torch.no_grad()
def run(plan, output, report):
    torch.set_num_threads(2)
    started = time.monotonic()
    records = read_manifest(plan["manifest"])
    records = [r for r in records if r.get("dataset") == "nyuv2" and "depth" in r]
    assert len(records) == plan["expected_images"] == 116
    atomic_json(records, output / "panel.json")
    models, references, sources = {}, {}, []
    system = config = None
    for spec in plan["checkpoints"]:
        report(stage="loading_checkpoint", checkpoint=spec["name"])
        before = identity(spec["path"])
        state = torch.load(spec["path"], map_location="cpu", weights_only=True, mmap=True)
        assert state["step"] == spec["step"]
        if system is None:
            config = copy.deepcopy(state["config"])
            system, adapter = build_system(config, "cuda")
        else:
            for key in ("models", "adapter", "evaluation"):
                assert state["config"][key] == config[key], key
            adapter = GeometryAdapter(**state["adapter_config"]).cuda()
        adapter.load_state_dict(state["adapter"], strict=True)
        adapter.eval().requires_grad_(False)
        statistics = FeatureStatistics(adapter.config["target_dim"],
                                       adapter.config["target_summary_dim"]).cuda()
        statistics.load_state_dict(state["statistics"], strict=True)
        models[spec["name"]] = (adapter, statistics)
        reference_bytes = Path(spec["reference"]).read_bytes()
        ref = {r["image"]: r for r in json.loads(reference_bytes) if r.get("dataset") == "nyuv2"}
        assert set(ref) == {r["image"] for r in records}
        references[spec["name"]] = ref
        assert identity(spec["path"]) == before
        sources.append({**spec, "identity": before,
                        "reference_sha256": hashlib.sha256(reference_bytes).hexdigest()})
        del state
        gc.collect()
    assert all(not p.requires_grad for adapter, _ in models.values() for p in adapter.parameters())
    atomic_json({"sources": sources, "config": config, "plan": plan}, output / "provenance.json")
    rows, reproduction_differences, replacement_checks = [], defaultdict(float), []
    for index, record in enumerate(records):
        if time.monotonic() - started > plan["max_seconds"]:
            raise TimeoutError("Diagnostic inference wall budget reached; no automatic restart")
        image, gt = evaluation_image(record, config["evaluation"]["max_side"])
        image, gt = image.cuda(), gt.cuda().unsqueeze(0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features, target_features = system.encode(
                image, adapter.indices, num_tokens=config["evaluation"]["num_tokens"])
        target_raw = system.decoder(target_features, image.shape[-2:])
        teacher = postprocess(target_raw, image, config["models"].get("moge_source"))
        teacher_valid = teacher["mask"] & torch.isfinite(teacher["depth"]) & (teacher["depth"] > 0)
        for name, (adapter, statistics) in models.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted_features = adapter(features)
            raw = system.decoder(predicted_features, image.shape[-2:])
            row = {"image": record["image"], "group": record["group"], "dataset": "nyuv2",
                   "checkpoint": name, **feature_metrics(predicted_features, target_features,
                                                         statistics)}
            row["scale_log_error"] = (raw["metric_scale"].log()
                                      - target_raw["metric_scale"].log()).abs().mean().item()
            variants = {"student": raw, "oracle_summary": replace_scale(raw, target_raw),
                        "oracle_spatial": replace_scale(target_raw, raw)}
            for component in plan.get("factorized_interventions", []):
                mixed = factorized_features(predicted_features, target_features, component)
                variants["oracle_" + component] = system.decoder(mixed, image.shape[-2:])
            # Check the shortcut against real mixed-feature decoder calls once per checkpoint.
            if index == 0:
                for tag, mixed in (
                    ("oracle_summary", EncoderFeatures(predicted_features.spatial,
                                                       target_features.summary)),
                    ("oracle_spatial", EncoderFeatures(target_features.spatial,
                                                       predicted_features.summary)),
                ):
                    direct = system.decoder(mixed, image.shape[-2:])
                    for key in direct:
                        torch.testing.assert_close(direct[key], variants[tag][key], rtol=0, atol=0)
                    replacement_checks.append({"checkpoint": name, "variant": tag,
                                               "exact_raw_decoder_match": True})
            predictions = {tag: postprocess(value, image, config["models"].get("moge_source"))
                           for tag, value in variants.items()}
            predictions["teacher"] = teacher
            for tag, prediction in predictions.items():
                row.update({f"{tag}/gt/{k}": v for k, v in depth_metrics(
                    prediction["depth"], gt, gt > 0).items()})
                row.update({f"{tag}/teacher/{k}": v for k, v in depth_metrics(
                    prediction["depth"], teacher["depth"], teacher_valid).items()})
            ref = references[name][record["image"]]
            for here, there in [("feature/cosine", "feature/cosine"),
                                ("feature/spatial", "feature/spatial"),
                                ("feature/summary", "feature/summary"),
                                ("scale_log_error", "scale_log_error"),
                                ("student/gt/abs_rel", "gt_student/abs_rel"),
                                ("student/teacher/abs_rel", "teacher_depth/abs_rel"),
                                ("teacher/gt/abs_rel", "gt_teacher/abs_rel")]:
                diff = abs(row[here] - ref[there])
                reproduction_differences[f"{name}/{here}"] = max(
                    reproduction_differences[f"{name}/{here}"], diff)
                if diff > plan["reproduction_tolerance"]:
                    raise ValueError(f"Cached evaluation mismatch: {name} {here} {diff}")
            assert all(np.isfinite(v) for v in row.values() if isinstance(v, (int, float)))
            rows.append(row)
        if (index + 1) % 10 == 0 or index + 1 == len(records):
            atomic_json(rows, output / "per_image.json")
            report(stage="evaluating", completed_images=index + 1, rows=len(rows),
                   elapsed_seconds=time.monotonic() - started)
    assert len(rows) == len(records) * len(models)
    assert all(identity(s["path"]) == s["identity"] for s in sources)
    summary = {"stage": "passed", "images": len(records), "rows": len(rows),
               "elapsed_seconds": time.monotonic() - started, "checkpoints": summarize(rows),
               "reproduction_max_differences": dict(reproduction_differences),
               "replacement_checks": replacement_checks, "source_identities_unchanged": True,
               "limits": "NYU116 validation, paired teacher-feature interventions with frozen models. "
               "Teacher features are diagnostic oracles, not deployable inputs. Correlations are "
               "descriptive, not causal. Scene-bootstrap intervals do not include training-seed "
               "uncertainty or repeated validation selection. No optimizer or training updates."}
    atomic_json(summary, output / "summary.json")
    report(stage="complete", completed_images=len(records), rows=len(rows),
           elapsed_seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".diagnostic.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "status.json").exists():
            raise RuntimeError("Existing diagnostic attempt; inspect it instead of restarting")

        def report(**changes):
            atomic_json({"pid": os.getpid(), "updated_at": time.time(), **changes},
                        output / "status.json")

        try:
            report(stage="initializing")
            run(plan, output, report)
        except Exception as error:
            report(stage="failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
