"""Freeze a small shortlist using screening validation only, before opening confirmation."""

import argparse
import hashlib
import json
from pathlib import Path
import time


POLICY = {
    "baseline": "legacy_2x1280",
    "balanced_slots": 2,
    "spatial_slots": 1,
    "efficiency_slots": 1,
    "maximum_architectures": 5,
    "efficiency_cosine_floor": [0.85, 0.75],
    "flow_nfe": "Pick each architecture's best minimum-of-two cosine NFE using screening validation only.",
    "criteria": "Baseline, top two balanced, best spatial, fastest nondominated qualifying endpoint; deduplicate architectures.",
}


def pareto(rows):
    return [
        a
        for a in rows
        if not any(
            b["spatial"] >= a["spatial"]
            and b["summary"] >= a["summary"]
            and b["optimizer_seconds"] <= a["optimizer_seconds"]
            and (
                b["spatial"] > a["spatial"]
                or b["summary"] > a["summary"]
                or b["optimizer_seconds"] < a["optimizer_seconds"]
            )
            for b in rows
        )
    ]


def select(rows):
    best = {}
    for row in rows:
        score = (min(row["spatial"], row["summary"]), -row["nfe"])
        if row["name"] not in best or score > best[row["name"]][0]:
            best[row["name"]] = (score, row)
    unique = [v[1] for v in best.values()]
    chosen = {}

    def add(row, reason):
        chosen.setdefault(row["name"], dict(row, reasons=[]))["reasons"].append(reason)

    add(best[POLICY["baseline"]][1], "original architecture baseline")
    for row in sorted(unique, key=lambda r: min(r["spatial"], r["summary"]), reverse=True)[:2]:
        add(row, "top two by minimum of spatial and summary cosine")
    add(max(unique, key=lambda r: r["spatial"]), "highest spatial cosine")
    frontier = pareto(unique)
    qualifying = [r for r in frontier if r["spatial"] >= 0.85 and r["summary"] >= 0.75]
    add(
        min(qualifying or frontier, key=lambda r: r["optimizer_seconds"]),
        "fastest qualifying Pareto endpoint",
    )
    assert len(chosen) <= POLICY["maximum_architectures"]
    return list(chosen.values()), frontier


def run(root):
    if (root / "shortlist.json").exists():
        raise FileExistsError("Shortlist already frozen; do not select again using confirmation.")

    def read(p):
        return json.loads(p.read_text())

    plan, status, queue, audit = (
        read(root / p)
        for p in ("plan.json", "status.json", "queue-status.json", "curve-verification.json")
    )
    assert status["stage"] == "complete" and queue["child_pid"] is None and queue["child_exit"] == 0
    assert not (root / "confirmation-status.json").exists()
    assert len(status["completed"]) == len(plan["candidates"]) == 16
    rows, hashes = [], {}
    for spec in plan["candidates"]:
        stop = read(root / spec["name"] / "stop-report.json")
        # A time-cut screen needs manual matched-budget review, not silent selection.
        assert stop["step"] == plan["updates"] == 1000
        for nfe in [1, 2, 4] if spec["family"] == "conditional_flow" else [1]:
            p = root / spec["name"] / f"validation-0001000-nfe{nfe}/metrics.json"
            metric = read(p)
            proof = read(root / "curve-audits" / f"{spec['name']}-validation-0001000-nfe{nfe}.json")
            assert proof["stage"] == "passed"
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            assert proof["input_sha256"]["metrics.json"] == digest
            hashes[str(p.relative_to(root))] = digest
            rows.append(
                dict(
                    name=spec["name"],
                    nfe=nfe,
                    spatial=metric["macro/spatial/cosine"],
                    summary=metric["macro/summary/cosine"],
                    optimizer_seconds=metric["optimizer_seconds"],
                    parameters=stop["parameters"],
                    step=metric["step"],
                )
            )
    selected, frontier = select(rows)
    report = dict(
        stage="frozen_before_confirmation",
        frozen_at=time.time(),
        policy=POLICY,
        selected=selected,
        pareto_frontier=frontier,
        all_screen_endpoints=rows,
        screen_metric_sha256=hashes,
        input_audit_checked_at=audit["checked_at"],
        confirmation_consulted=False,
        geometry_policy="Fixed first four confirmation images per source plus all 32 NYU GT images: 56 images. No metric-based image selection.",
    )
    p = root / "shortlist.json"
    t = p.with_suffix(".incoming")
    t.write_text(json.dumps(report, indent=2, allow_nan=False))
    t.replace(p)
    print(json.dumps(report))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    run(p.parse_args().root)
