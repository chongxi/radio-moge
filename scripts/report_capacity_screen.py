"""Report paired initialization-seed capacity comparisons without choosing new trials."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from audit_architecture_screen import atomic, lines, read


FEATURES = ("spatial/cosine", "summary/cosine", "spatial/norm_ratio", "summary/norm_ratio")
GEOMETRY = (
    "gt_student/abs_rel",
    "gt_student/rmse",
    "gt_student/delta1",
    "planes/student/normalized_rms",
    "depth_edges/log_gradient_mae",
    "depth_edges/retention_ratio",
    "outside_plane/teacher_abs_rel",
    "macro/teacher_depth/abs_rel",
)


def paired(rows, specs, seeds):
    result = {}
    shapes = sorted({(s["blocks"], s["width"]) for s in specs})
    for blocks, width in shapes:
        group = [rows[f"combined_{blocks}x{width}_s{seed}"] for seed in seeds]
        bases = [rows[f"combined_4x512_s{seed}"] for seed in seeds]
        keys = set(group[0]) & set(bases[0])
        keys = sorted(k for k in keys if all(type(r[k]) in (int, float) for r in group + bases))
        result[f"{blocks}x{width}"] = {
            "mean": {k: math.fsum(r[k] for r in group) / len(group) for k in keys},
            "range": {k: [min(r[k] for r in group), max(r[k] for r in group)] for k in keys},
            "paired_difference_vs_4x512": {
                str(seed): {k: r[k] - b[k] for k in keys}
                for seed, r, b in zip(seeds, group, bases, strict=True)
            },
            "both_cosines_higher_in_each_seed": all(
                r[k] > b[k]
                for r, b in zip(group, bases, strict=True)
                for k in ("spatial/cosine", "summary/cosine")
            ),
        }
    return result


def run(root):
    plan = read(root / "plan.json")
    specs, seeds = plan["candidates"], plan["initialization_seeds"]
    status, queue = read(root / "status.json"), read(root / "queue-status.json")
    audit = read(root / "curve-verification.json")
    assert status["stage"] == "complete" and len(status["completed"]) == len(specs) == 12
    assert queue["child_pid"] is None and queue["child_exit"] == 0
    assert audit["total_proofs"] == 2 * len(plan["evaluation_steps"]) * len(specs)
    stops = {r["name"]: r for r in status["completed"]}
    assert all(
        r["step"] == 1000 and r["cursor"] == 16000 and r["reason"] == "step_limit"
        for r in stops.values()
    )
    rows, crossings, curves = {}, {}, {}
    for spec in specs:
        name = spec["name"]
        config = read(root / name / "config.json")
        assert config["seed"] == spec["seed"]
        val = lines(root / name / "validation-nfe1.jsonl")
        probe = lines(root / name / "train_probe-nfe1.jsonl")
        assert [r["step"] for r in val] == [r["step"] for r in probe] == plan["evaluation_steps"]
        curves[name] = val
        rows[name] = {k: val[-1]["macro/" + k] for k in FEATURES}
        rows[name].update({"train_probe/" + k: probe[-1]["macro/" + k] for k in FEATURES})
        rows[name].update(
            parameters=stops[name]["parameters"],
            train_seconds=stops[name]["optimizer_seconds"],
            elapsed_seconds=stops[name]["elapsed_seconds"],
        )
        crossings[name] = {}
        for spatial, summary in ((0.85, 0.75), (0.90, 0.80), (0.95, 0.95), (0.98, 0.98)):
            hits = [
                r
                for r in val
                if r["macro/spatial/cosine"] >= spatial and r["macro/summary/cosine"] >= summary
            ]
            crossings[name][f"{spatial}/{summary}"] = (
                {k: hits[0][k] for k in ("step", "optimizer_seconds", "elapsed_seconds")}
                if hits
                else None
            )
    ranks = {}
    for seed in seeds:
        ranks[str(seed)] = {}
        names = [s["name"] for s in specs if s["seed"] == seed]
        for metric in ("spatial/cosine", "summary/cosine"):
            ranks[str(seed)][metric] = {
                str(step): sorted(
                    names,
                    key=lambda n: next(r for r in curves[n] if r["step"] == step)[
                        "macro/" + metric
                    ],
                    reverse=True,
                )
                for step in (500, 1000)
            }
    screen = dict(
        stage="screen_complete_confirmation_pending",
        checked_at=time.time(),
        matched_steps=1000,
        all_candidates_finished=len(specs),
        independent_audit_points=audit["total_proofs"],
        initialization_seeds=seeds,
        endpoints=rows,
        paired=paired(rows, specs, seeds),
        first_observed_crossings=crossings,
        ranks_500_1000=ranks,
        limitations=plan["limitations"],
    )
    atomic(root / "screen-report.json", screen)
    confirmation_path = root / "confirmation-verification.json"
    if not confirmation_path.exists() or read(confirmation_path)["stage"] != "passed":
        print(json.dumps(dict(stage=screen["stage"], candidates=len(rows))))
        return
    verification = read(confirmation_path)
    assert set(verification["results"]) == set(rows)
    confirmation = {}
    for name, checked in verification["results"].items():
        assert checked["gt_count"] == 32
        assert all(
            checked[k] == 0
            for k in (
                "teacher_failures",
                "postprocess_failures",
                "plane_failures",
                "preservation_failures",
            )
        )
        f = read(root / "confirmation" / name / "features-nfe1/metrics.json")
        g = read(root / "confirmation" / name / "geometry-metrics.json")
        confirmation[name] = {k: f["macro/" + k] for k in FEATURES}
        confirmation[name].update({k: g[k] for k in GEOMETRY})
        confirmation[name].update(
            {k: rows[name][k] for k in ("parameters", "train_seconds", "elapsed_seconds")}
        )
        confirmation[name]["gt_count"] = checked["gt_count"]
    grouped = paired(confirmation, specs, seeds)
    better = [
        shape for shape, values in grouped.items() if values["both_cosines_higher_in_each_seed"]
    ]
    conclusion = (
        "相同 1,000 步下，"
        + (
            "、".join(better) + " 在两个初始化 seed 中都比 4×512 的两项 cosine 更高。"
            if better
            else "没有更大规模在两个初始化 seed 中同时提高两项 cosine。"
        )
        + "深度误差、平面残差和耗时需分别比较，参数最多不等于综合最优。"
    )
    report = dict(
        stage="evaluated_pending_backup",
        checked_at=time.time(),
        conclusion_zh=conclusion,
        confirmation=confirmation,
        paired=grouped,
        target_both_098_reached=any(
            all(r[k] > 0.98 for k in ("spatial/cosine", "summary/cosine"))
            for r in confirmation.values()
        ),
        first_observed_crossings=crossings,
        comparison_panel_previously_viewed=True,
        geometry_images=56,
        gt_images=32,
        optimizer_updates_in_confirmation=0,
        limitations=plan["limitations"],
        screen_report=str(root / "screen-report.json"),
        proof_sha256={
            n: hashlib.sha256((root / n).read_bytes()).hexdigest()
            for n in ("curve-verification.json", "confirmation-verification.json")
        },
    )
    atomic(root / "capacity-report.json", report)
    atomic(root / "result-report.json", report)
    print(
        json.dumps(
            dict(stage=report["stage"], conclusion=conclusion, paired=grouped), ensure_ascii=False
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root)
