"""Freeze the authorized longer run from the completed trained Muon arm."""

import argparse
from copy import deepcopy
from pathlib import Path
import json

from radio_moge.corpus_admission import checked_json, file_ref
from radio_moge.geometry_cache import publish_json
from run_room_feature_epoch import real_cursor
from run_room_muon_continuation import (
    BATCH,
    EPOCHS,
    UPDATES_PER_EPOCH,
    UPDATES,
    SECONDS,
    CADENCE,
    IMAGES,
    COVERAGE_GOAL,
    BUCKET_POLICY,
    check_contract,
)


def prepare(source, output, revision):
    code = Path(__file__).resolve().parents[1]
    old = checked_json(file_ref(source / "plan.json"))
    config = checked_json(old["config"])
    finish = checked_json(file_ref(source / "finish-verification-01/report.json"))
    online = checked_json(file_ref(source / "online-final-01/report.json"))
    results = checked_json(file_ref(source / f"validation-{finish['step']}-review-01/report.json"))
    provenance = checked_json(results["provenance"])
    if (
        old["arm"] != "core-muon"
        or finish["step"] != 263121
        or finish["new_updates"] != 1000
        or online["state"] != "finished"
        or not online["all_committed_values_numerically_exact"]
    ):
        raise ValueError("Expected the closed trained Muon arm")
    parent = dict(
        config=deepcopy(config),
        step=finish["step"],
        cursor=finish["cursor"],
        performance=provenance["performance"],
        data_digests=provenance["data_digests"],
    )
    output.mkdir(exist_ok=False)
    config["output_dir"] = str(output / "training")
    config["train"].update(
        steps=finish["step"] + UPDATES,
        max_seconds=SECONDS,
        eval_every=CADENCE,
        save_every=CADENCE,
        eval_at_start=False,
    )
    config["data"]["rgb_replacement"]["real_start_cursor"] = real_cursor(parent)
    config["tracking"].update(
        group=output.name,
        name="actual128-muon-three-passes-from263121",
        tags=[
            "feature-only",
            "geometry-off",
            "actual-batch128",
            "continued-muon-momentum",
            "preserved-aux-adam",
            "three-synthetic-passes",
            "lr1e-5",
        ],
    )
    for spec in config["evaluation"]["checkpoint_metrics"].values():
        metric, view = spec["metric"], "original"
        if metric.startswith("resolution/native-640x400/"):
            view, metric = "native-640x400", metric.removeprefix("resolution/native-640x400/")
        spec["baseline"] = results["views"][view]["metrics"][metric]["final"]
    publish_json(output / "config.json", config)
    implementation = [
        "scripts/run_room_muon_continuation.py",
        "scripts/run_room_optimizer_arm.py",
        "scripts/prepare_room_muon_continuation.py",
        "src/radio_moge/train.py",
        "src/radio_moge/optimizer_trial.py",
    ]
    plan = dict(
        authorized=True,
        user_direction="Continue with Muon for longer training",
        arm="core-muon",
        learning_rate=1e-5,
        actual_batch=BATCH,
        epochs=EPOCHS,
        updates_per_epoch=UPDATES_PER_EPOCH,
        additional_updates=UPDATES,
        synthetic_images=IMAGES,
        synthetic_presentations=EPOCHS * IMAGES,
        original_presentations=UPDATES * BATCH - EPOCHS * IMAGES,
        coverage_goal=COVERAGE_GOAL,
        bucket_policy=BUCKET_POLICY,
        maximum_training_seconds=SECONDS,
        source_step=finish["step"],
        source_parent_cursor=finish["cursor"],
        source_cursor=real_cursor(parent),
        source_optimizer_counts=finish["optimizer_counts"],
        source_checkpoint={k: finish["checkpoint"][k] for k in ("path", "sha256")},
        source_finish=file_ref(source / "finish-verification-01/report.json"),
        source_results=file_ref(source / f"validation-{finish['step']}-review-01/report.json"),
        source_metadata=file_ref(source / "metadata-report.json"),
        source_model_gate=file_ref(source / "mixed-model-zero-update.json"),
        source_execution=str(Path(old["runner"]["path"]).parents[1]),
        batch_probe=old["batch_probe"],
        dependency=old["dependency"],
        training_lock=old["training_lock"],
        config=file_ref(output / "config.json"),
        runner=file_ref(code / "scripts/run_room_muon_continuation.py"),
        trainer=file_ref(code / "src/radio_moge/train.py"),
        execution_revision=revision,
        implementation_files=[file_ref(code / name) for name in implementation],
        automatic_retry=False,
        automatic_next_training=False,
        automatic_deployment=False,
        sampler_transition="Fresh three synthetic permutations seeds43/44/45; continue original keys and crop seeds",
        optimizer_transition="All16 Muon momentum and84 AdamW states preserved; no optimizer reset",
        **{
            k: old[k]
            for k in (
                "real_manifest",
                "synthetic_manifest",
                "validation_manifest",
                "structural_validation_manifest",
            )
        },
    )
    check_contract(config, parent, plan)
    publish_json(output / "plan.json", plan)
    print(json.dumps(file_ref(output / "plan.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    prepare(args.source.resolve(), args.output.resolve(), args.revision)
