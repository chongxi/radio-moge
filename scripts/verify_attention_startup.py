"""Check the ten-model GPU preflight and the real baseline's first logged updates."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from backup_architecture_screen import atomic


def read(p):
    return json.loads(p.read_text())


def run(root):
    proof_path = root / "startup-verification.json"
    if proof_path.exists():
        print('{"stage":"already_verified"}')
        return
    arm = root / "attention_baseline_s42"
    if not (arm / "train.jsonl").exists():
        print('{"stage":"pending_actual_updates"}')
        return
    plan = read(root / "plan.json")
    old = root.parent / "adapter-combined-wide4096-screen-20261028"
    prior = old / "combined_4x2048_s42"
    assert read(old / "completion-verification.json")["stage"] == "passed"
    pre = read(root / "preflight-verification.json")
    params = read(root / "planned-parameters.json")["candidates"]
    assert pre["stage"] == "passed" and pre["optimizer_updates"] == 0
    assert set(pre["candidates"]) == set(params) and len(params) == 10
    for name, item in pre["candidates"].items():
        assert item["parameters"] == params[name] and item["finite_nonzero_gradients"]
        assert item["optimizer_updates"] == 0
        assert math.isfinite(item["finite_loss"]) and math.isfinite(item["gradient_norm"])
        assert item["gradient_norm"] > 0
    cache = read(root / "cache-load-verification.json")
    pcache = read(old / "cache-load-verification.json")
    for key in ("dimensions", "training_images", "validation_images", "calibration_images",
                "sample_order_sha256", "data_seed", "calibration_uses_validation", "cache_gpu_bytes"):
        assert cache[key] == pcache[key], key
    assert cache["training_images"] == cache["calibration_images"] == 1792
    assert cache["calibration_uses_validation"] is False
    assert cache["cache_index_sha256"] == hashlib.sha256((root / "cache-index.json").read_bytes()).hexdigest()
    config, previous = read(arm / "config.json"), read(prior / "config.json")
    for key in ("dimensions", "fresh_initialization", "seed", "train", "loss"):
        assert config[key] == previous[key], key
    assert {k: v for k, v in config["architecture"].items() if k not in ("name", "variant")} == {
        k: v for k, v in previous["architecture"].items() if k != "name"}
    rows = [json.loads(s) for s in (arm / "train.jsonl").read_text().splitlines()]
    assert rows
    errors = []
    for row in rows:
        assert all(math.isfinite(v) for v in row.values() if isinstance(v, (int, float)))
        assert row["gradient_norm"] > 0
        assert math.isclose(row["lr_used"], plan["lr"] * min(row["step"] / plan["warmup_updates"], 1), rel_tol=1e-12)
        errors.append(abs(row["total"] - sum(row[k] * w for k, w in plan["loss_weights"].items())))
    assert max(errors) < 1e-6
    name = "validation-0000000-nfe1/per_image.json"
    current, reference = read(arm / name), read(prior / name)
    assert len(current) == len(reference) == 224
    differences = []
    for a, b in zip(current, reference, strict=True):
        assert set(a) == set(b)
        for key, value in a.items():
            if isinstance(value, (int, float)):
                differences.append(abs(value - b[key]))
            else:
                assert value == b[key]
    assert max(differences) == 0
    tracker = read(arm / "wandb-run.json")["id"]
    assert tracker not in {read(p)["id"] for p in old.glob("combined_*/wandb-run.json")}
    proof = dict(stage="passed", checked_at=time.time(), gpu_preflight_models=10,
                 gpu_preflight_optimizer_updates=0, actual_parameters=params,
                 all_preflight_finite_nonzero_gradients=True, calibration_train_only=1792,
                 data_order_identical_to_prior=True, first_logged_rows=len(rows),
                 weighted_total_max_error=max(errors), baseline_initial_per_image_max_difference=0,
                 baseline_reference_sha256=hashlib.sha256((prior / name).read_bytes()).hexdigest(),
                 fresh_wandb_id=tracker, execution_revision=plan["execution_revision"],
                 note="Preflight has zero updates; inspect each variant's actual training as its queue turn begins.")
    atomic(proof_path, proof)
    print(json.dumps(proof))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    run(p.parse_args().root)
