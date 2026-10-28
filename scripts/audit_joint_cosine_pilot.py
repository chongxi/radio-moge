"""Final paired cosine/geometry audit with independently recomputed saved evaluations."""

import argparse
import json
import math
from pathlib import Path
import statistics
import time

from audit_plane_subset_pilot import audit_eval, fixed_teacher, KEYS, numeric


def load(p):
    return json.loads(p.read_text())


def rows(p):
    return [json.loads(s) for s in p.read_text().splitlines()]


def atomic(p, d):
    t = p.with_suffix(".incoming")
    t.write_text(json.dumps(d, indent=2, allow_nan=False))
    t.replace(p)


def interpolate(points, x):
    if x <= points[0][0]:
        return points[0][1]
    for (a, va), (b, vb) in zip(points, points[1:]):
        if x <= b:
            return va + (vb - va) * (x - a) / (b - a)
    return points[-1][1]


def speed_report(root):
    result = {}
    plan = load(root / "plan.json")
    target = plan["speed_target_absolute_cosine_gain"]
    for partition, filename in [
        ("validation", "validation.jsonl"),
        ("train-probe", "train-probe.jsonl"),
    ]:
        result[partition] = {}
        values = {a: rows(root / a / filename) for a in plan["arms"]}
        for metric in ("spatial/cosine", "summary/cosine"):
            initial = values["control"][0][metric]
            assert initial == values["joint"][0][metric]
            entry = {
                "initial": initial,
                "absolute_gain_target": target,
                "first_observed_crossing": {},
            }
            for arm, points in values.items():
                first = next((r for r in points if r[metric] >= initial + target), None)
                entry["first_observed_crossing"][arm] = (
                    {k: first[k] for k in ("step", "optimizer_seconds", "elapsed_seconds", metric)}
                    if first
                    else None
                )
            entry["common_interval_average_cosine_gain"] = {}
            for axis in ("step", "optimizer_seconds", "elapsed_seconds"):
                curves = {a: [(r[axis], r[metric] - initial) for r in v] for a, v in values.items()}
                for curve in curves.values():
                    assert all(b[0] > a[0] for a, b in zip(curve, curve[1:]))
                start = max(c[0][0] for c in curves.values())
                end = min(c[-1][0] for c in curves.values())
                knots = sorted(
                    {start, end} | {x for c in curves.values() for x, _ in c if start < x < end}
                )
                averages = {}
                for arm, curve in curves.items():
                    area = sum(
                        (b - a) * (interpolate(curve, a) + interpolate(curve, b)) / 2
                        for a, b in zip(knots, knots[1:])
                    )
                    averages[arm] = area / (end - start)
                entry["common_interval_average_cosine_gain"][axis] = dict(
                    start=start,
                    end=end,
                    **averages,
                    joint_minus_control=averages["joint"] - averages["control"],
                )
            result[partition][metric] = entry
    result["method"] = (
        "First completed evaluation crossing only; no exact crossing-time claim. AUC uses piecewise-linear interpolation over a shared interval; descriptive, not significance."
    )
    return result


def run(root):
    status, plan = load(root / "status.json"), load(root / "plan.json")
    assert (
        status["stage"] == "complete" and status["child_pid"] is None and status["child_exit"] == 0
    )
    assert status["parent_unchanged"] and plan["arms"] == ["control", "joint"]
    steps = status["matched_steps"]
    assert 0 < steps <= 600
    curve = load(root / "curve-verification.json")
    assert curve["stage"] == "passed" and curve["matched_steps"] == plan["evaluation_steps"]
    initial = load(root / "initial-adapter-verification.json")
    assert initial["stage"] == "passed" and not initial["full_architectures_identical"]
    assert len(set(initial["shared_base_fingerprints"].values())) == 1
    preflight = load(root / "preflight-verification.json")
    assert all(
        v == preflight["base_fingerprint"] for v in initial["shared_base_fingerprints"].values()
    )
    arms = {}
    for arm in plan["arms"]:
        stop = load(root / arm / "stop-report.json")
        logs = rows(root / arm / "train.jsonl")
        assert stop["step"] == steps and stop["final_evaluation"] == "complete"
        assert logs[-1]["step"] == steps and all(r["lr"] == r["lr_used"] == 1e-5 for r in logs)
        assert all(math.isfinite(v) for r in logs for v in r.values() if numeric(v))
        assert all(r["update_norm"] > 0 for r in logs if "update_norm" in r)
        arms[arm] = dict(
            stop=stop,
            logged_rows=len(logs),
            max_gradient=max(r["gradient_norm"] for r in logs),
            optimizer_seconds=logs[-1]["optimizer_seconds"],
            parameters=preflight["arms"][arm]["parameters"],
            run=load(root / arm / "wandb-run.json"),
        )
    results, receipts = {}, {}
    for part in ("validation", "train-probe", "screenshot"):
        images = [r["image"] for r in rows(root / f"{part}.jsonl")]
        paths = (
            {
                "initial": root / "control/eval-0000000",
                **{a: root / a / f"eval-{steps:07d}" for a in plan["arms"]},
            }
            if part == "validation"
            else {a: root / "assessment" / a / part for a in ("initial", *plan["arms"])}
        )
        data, metrics, checks = {}, {}, {}
        for arm, path in paths.items():
            data[arm], metrics[arm], checks[arm] = audit_eval(path, images)
            raw = load(path / "metrics.json")
            metrics[arm]["branch_ms"] = raw["branch_ms"]
        fixed_teacher(list(data.values()))
        if part == "validation":
            other, _, check = audit_eval(root / "joint/eval-0000000", images)
            fixed_teacher([data["initial"], other])
            assert all(
                a[k] == b[k] for a, b in zip(data["initial"], other, strict=True) for k in KEYS
            )
            checks["joint_initial"] = check
        changes = {
            k: dict(
                initial=metrics["initial"][k],
                control=metrics["control"][k],
                joint=metrics["joint"][k],
                joint_minus_control=metrics["joint"][k] - metrics["control"][k],
                joint_vs_control_percent=100 * (metrics["joint"][k] / metrics["control"][k] - 1),
                joint_vs_initial_percent=100 * (metrics["joint"][k] / metrics["initial"][k] - 1),
            )
            for k in [*KEYS, "branch_ms"]
        }
        key = "planes/student/normalized_rms"
        wins = sum(
            j[key] < c[key] for j, c in zip(data["joint"], data["control"], strict=True) if key in j
        )
        results[part] = dict(
            images=len(images),
            metrics=changes,
            flatness_better_than_control=wins,
            images_with_planes=sum(key in r for r in data["joint"]),
            paired_cosine_differences={
                k: dict(
                    mean=statistics.mean(
                        j[k] - c[k] for j, c in zip(data["joint"], data["control"], strict=True)
                    ),
                    images_improved=sum(
                        j[k] > c[k] for j, c in zip(data["joint"], data["control"], strict=True)
                    ),
                )
                for k in ("spatial/cosine", "summary/cosine")
            },
        )
        receipts[part] = checks
    report = dict(
        stage="passed",
        checked_at=time.time(),
        steps=steps,
        arms=arms,
        results=results,
        aggregates=receipts,
        initial_outputs_identical=True,
        shared_base_fingerprints=initial["shared_base_fingerprints"],
        full_architectures_identical=False,
        fixed_teacher_masks=True,
        frozen_teacher_gt_unchanged=True,
        speed=speed_report(root),
        limitations=plan["limitations"],
    )
    atomic(root / "evaluation-verification.json", report)
    print(
        json.dumps(
            dict(
                stage="passed",
                steps=steps,
                validation=results["validation"],
                speed=report["speed"]["validation"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root)
