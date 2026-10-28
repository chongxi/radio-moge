"""Freeze a bounded replicated capacity experiment referencing an immutable feature cache."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

from radio_moge.capacity_screen import capacity_candidates
from radio_moge.attention_tricks import attention_candidates


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic(path, data):
    p = path.with_suffix(path.suffix + ".incoming")
    p.write_bytes(data)
    p.replace(path)


def prepare(root, source, revision, wide2048=False, wide4096=False, attention=False):
    assert sum((wide2048, wide4096, attention)) <= 1
    assert json.loads((source / "completion-verification.json").read_text())["stage"] == "passed"
    plan = copy.deepcopy(json.loads((source / "plan.json").read_text()))
    cache = json.loads((source / "cache-index.json").read_text())
    old_index_hash = digest((source / "cache-index.json").read_bytes())
    root.mkdir(exist_ok=False)
    for name in ("train.jsonl", "validation.jsonl", "confirmation.jsonl", "data-verification.json"):
        data = (source / name).read_bytes()
        if name.endswith(".jsonl"):
            assert digest(data) == plan["manifest_sha256"][Path(name).stem]
        atomic(root / name, data)
    cache["reused_cache_setup_seconds"] = cache["elapsed_seconds"]
    cache["elapsed_seconds"] = 0.0
    payload = json.dumps(cache, indent=2).encode()
    atomic(root / "cache-index.json", payload)
    shapes = (
        ((4, 2048), (4, 4096))
        if wide4096
        else (
            ((4, 1024), (4, 1536), (4, 2048))
            if wide2048
            else ((4, 512), (4, 768), (4, 1024), (8, 512), (8, 768), (8, 1024))
        )
    )
    candidates = attention_candidates() if attention else capacity_candidates(shapes)
    count = len(candidates)
    plan.update(
        name=root.name,
        execution_revision=revision,
        created_at=time.time(),
        candidates=candidates,
        candidate_order=[c["name"] for c in candidates],
        screen_profile="wide4096" if wide4096 else "wide2048" if wide2048 else "capacity",
        comparison_baseline_shape=[4, 2048] if wide4096 else [4, 1024] if wide2048 else [4, 512],
        experiment_shapes=[list(s) for s in shapes],
        total_screen_seconds=3600,
        max_seconds_per_candidate=600,
        selection=f"All {len(shapes)} shapes and both seeds prespecified; no adaptive pruning or winner selection.",
        confirmation_policy=f"Evaluate all {count} endpoints on the same224 feature/56 geometry comparison panel. This panel was viewed previously; it is not a new untouched holdout.",
        next_rung="No automatic extension after1000 updates per candidate.",
        initialization_seeds=[42, 43],
        data_seed=47,
        cache_reuse={
            "source_root": str(Path(cache["shards"][0]["path"]).parent.parent),
            "local_source_metadata": str(source),
            "source_index_sha256": old_index_hash,
            "index_sha256": digest(payload),
            "source_complete_proof_sha256": digest(
                (source / "completion-verification.json").read_bytes()
            ),
            "new_encoder_inference": False,
        },
        audit_limitations="Two initialization seeds with identical fixed data order; common LR, cached fixed views, previously inspected comparison panel; no convergence or broad statistical significance claim.",
        limitations=[
            "Same train/selection/confirmation images as prior screen; confirmation panel is reused, not a fresh untouched test.",
            "Two seeds vary initialization only. Shared fixed data order and LR are controlled, not optimized per size.",
            "Cached fixed-view learning curves do not establish full-data convergence.",
            "Geometry56 images includes only32NYUGT; COCO scenes unknown and Tartan/IRS few environments.",
        ],
    )
    if attention:
        plan.update(
            screen_profile="attention_tricks",
            comparison_baseline_variant="baseline",
            comparison_baseline_shape=[4, 2048],
            experiment_shapes=[[4, 2048]],
            experiment_variants=list(dict.fromkeys(c["variant"] for c in candidates)),
            selection="All five variants and both seeds fixed before training; no adaptive pruning.",
            variant_details={
                "baseline": "Unmodified 4x2048,32heads,headwidth64.",
                "heads16": "16heads/headwidth128; same total width. Per-head gates change parameter count slightly; axial RoPE frequency allocation follows head dimension.",
                "heads64": "64heads/headwidth32; same caveats as heads16.",
                "qknorm": "Per-head LayerNorm on projected Q,K before RoPE; standard sqrt(head_dim) softmax scaling retained. Output magnitudes are not normalized.",
                "summary_readout": "One extra gated attention-only residual summary update, all spatial/summary/register tokens as memory; no extra FFN or spatial update. Shared initialization unchanged. Extra parameters reported; not parameter-matched.",
            },
        )
    atomic(root / "plan.json", json.dumps(plan, indent=2).encode())
    atomic(
        root / "cache-status.json",
        json.dumps(
            dict(
                stage="complete_reused",
                images=2240,
                original_elapsed_seconds=cache["reused_cache_setup_seconds"],
                new_encoder_inference=False,
            )
        ).encode(),
    )
    frozen = dict(
        stage="frozen_before_confirmation",
        frozen_at=time.time(),
        policy=plan["confirmation_policy"],
        selected=[dict(name=c["name"], nfe=1) for c in plan["candidates"]],
        screen_metric_sha256={},
        confirmation_consulted=False,
        previous_panel_viewed=True,
        all_candidates_predeclared=True,
    )
    atomic(root / "shortlist.json", json.dumps(frozen, indent=2).encode())
    atomic(root / "confirmation-policy.json", json.dumps(frozen, indent=2).encode())
    fixed = [
        "manifest_sha256",
        "train_probe_indices",
        "models",
        "loss_weights",
        "effective_batch",
        "microbatch",
        "lr",
        "warmup_updates",
        "weight_decay",
        "clip_grad",
        "updates",
        "evaluation_steps",
        "image_shape",
    ]
    original = json.loads((source / "plan.json").read_text())
    assert all(plan[k] == original[k] for k in fixed)
    atomic(
        root / "configuration-verification.json",
        json.dumps(
            dict(
                stage="passed",
                checked_at=time.time(),
                unchanged=fixed,
                candidates=plan["candidates"],
                reused_shard_count=len(cache["shards"]),
                no_old_adapter_weights=True,
            ),
            indent=2,
        ).encode(),
    )
    print(
        json.dumps(
            dict(
                stage="prepared",
                name=root.name,
                candidates=len(plan["candidates"]),
                cache_reused=True,
            )
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--revision", required=True)
    widths = p.add_mutually_exclusive_group()
    widths.add_argument("--wide2048", action="store_true")
    widths.add_argument("--wide4096", action="store_true")
    widths.add_argument("--attention", action="store_true")
    a = p.parse_args()
    prepare(a.root, a.source, a.revision, a.wide2048, a.wide4096, a.attention)
