"""Verify actual queued variant configurations and finite optimizer logs, without GPU work."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from backup_architecture_screen import atomic


def read(path):
    return json.loads(path.read_text())


def check(root):
    plan = read(root / "plan.json")
    report = root / "variant-training-verification.json"
    previous = read(report).get("variants", {}) if report.exists() else {}
    verified = dict(previous)
    for spec in plan["candidates"]:
        name = spec["name"]
        arm = root / name
        path = arm / "train.jsonl"
        if not path.exists():
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if previous.get(name, {}).get("log_sha256") == digest:
            continue
        config = read(arm / "config.json")
        assert config["architecture"] == spec and config["seed"] == spec["seed"]
        assert config["fresh_initialization"] and config["loss"] == plan["loss_weights"]
        assert config["train"] == {k: plan[k] for k in config["train"]}
        rows = [json.loads(x) for x in data.splitlines(keepends=True) if x.endswith(b'\n')]
        if not rows:
            continue
        errors = []
        for row in rows:
            assert all(math.isfinite(v) for v in row.values() if isinstance(v, (int, float)))
            assert row["gradient_norm"] > 0
            assert math.isclose(row["lr_used"], plan["lr"] * min(row["step"] / plan["warmup_updates"], 1), rel_tol=1e-12)
            errors.append(abs(row["total"] - sum(row[k] * w for k, w in plan["loss_weights"].items())))
        assert max(errors) < 1e-6
        verified[name] = dict(stage="passed", checked_at=time.time(), seed=spec["seed"],
                              heads=spec["heads"], qk_norm=spec.get("qk_norm", False),
                              summary_readout=spec.get("summary_readout", False),
                              logged_rows=len(rows), last_step=rows[-1]["step"],
                              weighted_total_max_error=max(errors), log_sha256=digest,
                              wandb_id=read(arm / "wandb-run.json")["id"])
    assert len({v["wandb_id"] for v in verified.values()}) == len(verified)
    result = dict(stage="passed" if len(verified) == len(plan["candidates"]) else "partial",
                  checked_at=time.time(), variants=verified,
                  pending=[s["name"] for s in plan["candidates"] if s["name"] not in verified])
    atomic(report, result)
    print(json.dumps(dict(stage=result["stage"], verified=len(verified), pending=result["pending"])))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    check(p.parse_args().root)
