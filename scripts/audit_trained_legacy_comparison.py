"""Audit existing per-image results on one identical cached comparison panel."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import time

from radio_moge.checkpoint import atomic_json


def audit(root):
    base = root.parent
    screen = base / "adapter-attention-tricks2048-screen-20261028"
    scratch = base / "adapter-scratch-diverse-screen-20261028"
    paths = {
        "trained_legacy_156000": root / "evaluation/features",
        "scratch_legacy_s42": scratch / "confirmation/legacy_2x1280/features-nfe1",
        "qknorm_s42": screen / "confirmation/attention_qknorm_s42/features-nfe1",
        "qknorm_s43": screen / "confirmation/attention_qknorm_s43/features-nfe1",
    }
    plan = json.loads((screen / "plan.json").read_text())
    scratch_plan = json.loads((scratch / "plan.json").read_text())
    for key in ["loss_weights", "lr", "warmup_updates", "effective_batch", "microbatch",
                "weight_decay", "clip_grad", "updates", "data_seed"]:
        assert plan[key] == scratch_plan[key], key
    provenance = json.loads((root / "evaluation/provenance.json").read_text())
    assert provenance["optimizer_updates"] == 0 and provenance["step"] == 156000
    for name, key in [("cache-index.json", "cache_index_sha256"),
                      ("confirmation.jsonl", "manifest_sha256")]:
        assert hashlib.sha256((screen / name).read_bytes()).hexdigest() == provenance[key]
    assert (screen / "confirmation.jsonl").read_bytes() == (scratch / "confirmation.jsonl").read_bytes()
    metrics = ["spatial/cosine", "summary/cosine"]
    identities, models, hashes, errors = None, {}, {}, []
    for name, path in paths.items():
        data = (path / "per_image.json").read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        assert digest == json.loads((path / "integrity.json").read_text())["per_image_sha256"]
        rows = json.loads(data)
        keys = [(r["image"], r["dataset"], r["group"], r["cache_index"]) for r in rows]
        assert len(rows) == len(set(keys)) == 224
        if identities is None:
            identities = keys
        assert keys == identities
        sources = sorted({r["dataset"] for r in rows})
        assert len(sources) == 7
        assert all(sum(r["dataset"] == s for r in rows) == 32 for s in sources)
        by_source = {s: {k: mean(r[k] for r in rows if r["dataset"] == s)
                         for k in metrics} for s in sources}
        macro = {k: mean(by_source[s][k] for s in sources) for k in metrics}
        stored = json.loads((path / "metrics.json").read_text())
        for k in metrics:
            assert all(math.isfinite(r[k]) and -1 <= r[k] <= 1 for r in rows)
            errors.append(abs(macro[k] - stored["macro/" + k]))
        models[name] = dict(mean=macro, by_source=by_source)
        hashes[name] = dict(path=str(path / "per_image.json"), sha256=digest)
    assert max(errors) < 1e-12
    qk = {k: mean(models[f"qknorm_s{s}"]["mean"][k] for s in [42, 43]) for k in metrics}
    report = dict(
        stage="passed", checked_at=time.time(), images=224, sources=7, images_per_source=32,
        exact_image_order_identity=True, identical_cached_inputs_and_targets=True,
        macro_recomputation_max_error=max(errors), optimizer_updates=0,
        new_encoder_inference=False, models=models, qknorm_two_seed_mean=qk,
        delta_vs_trained={k: qk[k] - models["trained_legacy_156000"]["mean"][k] for k in metrics},
        matched_1000_seed42_delta={k: models["qknorm_s42"]["mean"][k] -
                                 models["scratch_legacy_s42"]["mean"][k] for k in metrics},
        new_training=dict(unique_images=plan["train_images"], updates=plan["updates"],
                          effective_batch=plan["effective_batch"], sample_presentations=16000,
                          approximate_passes=16000/1792),
        old_training=dict(pool_images=796478, selected_step=156000,
                          note="Selected checkpoint after approximately two days; initialized from an earlier pilot"),
        per_image_sources=hashes,
        limits=["Previously viewed panel; not an untouched test",
                "Old full-training versus new short-screen compares unequal training budgets",
                "Seed42 short-screen comparison holds objective/data/steps/seed fixed but changes the combined architecture; not a QK-only effect",
                "No convergence or statistical significance claim"],
    )
    atomic_json(report, root / "comparison-report.json")
    print(json.dumps({k: report[k] for k in ["stage", "models", "qknorm_two_seed_mean", "delta_vs_trained", "matched_1000_seed42_delta"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    audit(parser.parse_args().root)
