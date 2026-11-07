"""Freeze the authorized core-reset AdamW/Muon comparison and its closed AdamW reference."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from radio_moge.corpus_admission import checked_json, file_ref
from radio_moge.geometry_cache import publish_json
from run_room_optimizer_arm import (
    BATCH,
    UPDATES,
    SECONDS,
    SYNTHETIC_PRESENTATIONS,
    COVERAGE_GOAL,
    MODES,
    policy,
    BUCKET_POLICY,
    check_contract,
    compare_arm_configs,
)


def prepare(source, baseline, dependency, trainer_patch, output, revision):
    code = Path(__file__).resolve().parents[1]
    previous = checked_json(file_ref(source / "plan.json"))
    config = checked_json(file_ref(source / "config.json"))
    finish = checked_json(file_ref(source / "finish-verification-01/report.json"))
    results = checked_json(file_ref(source / f"validation-{finish['step']}-review-01/report.json"))
    provenance = checked_json(file_ref(source / "training/provenance.json"))
    if (
        finish["step"] != 262121
        or finish["new_updates"] != 3175
        or finish["actual_rgb_presentations"] != 406400
        or not finish["every_synthetic_image_exactly_once"]
    ):
        raise ValueError("Expected the completed, verified actual128 source epoch")
    parent = dict(
        config=config,
        step=finish["step"],
        cursor=finish["cursor"],
        performance=provenance["performance"],
        data_digests=provenance["data_digests"],
    )
    source_cursor = config["data"]["rgb_replacement"]["real_start_cursor"] + finish["cursor"]
    output.mkdir(exist_ok=False)
    baseline_plan_ref = file_ref(baseline / "plan.json")
    baseline_plan = checked_json(baseline_plan_ref)
    baseline_finish = checked_json(file_ref(baseline / "finish-verification-01/report.json"))
    baseline_online = checked_json(file_ref(baseline / "online-final-01/report.json"))
    if (
        baseline_finish["step"] != 263121
        or baseline_finish["new_updates"] != UPDATES
        or not baseline_online["all_committed_values_numerically_exact"]
        or baseline_online["state"] != "finished"
        or baseline_plan["learning_rate"] != 1e-5
        or baseline_plan["source_step"] != finish["step"]
    ):
        raise ValueError("The reused continued-AdamW reference is not fully closed")
    configs, refs = [], []
    for arm in MODES:
        lr = 1e-5
        root = output / arm
        root.mkdir()
        current = deepcopy(config)
        current["output_dir"] = str(root / "training")
        current["optimizer_trial"] = policy(arm)
        current["train"].update(
            lr=lr,
            steps=finish["step"] + UPDATES,
            max_seconds=SECONDS,
            eval_every=250,
            save_every=250,
            eval_at_start=False,
        )
        current["data"]["rgb_replacement"]["real_start_cursor"] = source_cursor
        current["tracking"].update(
            group=output.name,
            name=f"actual128-{arm}-from262121",
            tags=[
                "feature-only",
                "feature-weight1",
                "geometry-off",
                "actual-batch128",
                "one-forward-per-update",
                "preserved-aux-adam-rng-rms",
                "fresh-core-optimizer-state",
                "matched-optimizer-comparison",
                "fixed1000-updates",
                "same-inputs-crops-buckets",
                arm,
            ],
        )
        for spec in current["evaluation"]["checkpoint_metrics"].values():
            metric = spec["metric"]
            view = "original"
            if metric.startswith("resolution/native-640x400/"):
                view, metric = "native-640x400", metric.removeprefix("resolution/native-640x400/")
            spec["baseline"] = results["views"][view]["metrics"][metric]["final"]
        publish_json(root / "config.json", current)
        plan = dict(
            authorized=True,
            user_direction="Authorized Muon/core-reset AdamW comparison against saved continued AdamW",
            arm=arm,
            learning_rate=lr,
            actual_batch=BATCH,
            additional_updates=UPDATES,
            synthetic_images=50787,
            synthetic_presentations=SYNTHETIC_PRESENTATIONS,
            original_presentations=UPDATES * BATCH - SYNTHETIC_PRESENTATIONS,
            coverage_goal=COVERAGE_GOAL,
            bucket_policy=BUCKET_POLICY,
            maximum_training_seconds=SECONDS,
            source_step=finish["step"],
            source_parent_cursor=finish["cursor"],
            source_cursor=source_cursor,
            source_checkpoint={k: finish["checkpoint"][k] for k in ("path", "sha256")},
            source_finish=file_ref(source / "finish-verification-01/report.json"),
            source_results=file_ref(source / f"validation-{finish['step']}-review-01/report.json"),
            source_metadata=file_ref(source / "metadata-report.json"),
            source_model_gate=file_ref(source / "mixed-model-zero-update.json"),
            source_execution=str(Path(previous["runner"]["path"]).parents[1]),
            batch_probe=previous["batch_probe"],
            prior_probe_is_historical_not_current_checkpoint_gate=True,
            reference_arm=None,
            baseline_arm=str(baseline),
            baseline_plan=baseline_plan_ref,
            baseline_finish=file_ref(baseline / "finish-verification-01/report.json"),
            baseline_online=file_ref(baseline / "online-final-01/report.json"),
            baseline_input_sha256="b4fd4a309ae3750f36fe97818bf490df2fc805f2a98de817e0382f69627adcab",
            dependency=file_ref(dependency),
            trainer_patch=file_ref(trainer_patch),
            optimizer_implementation=file_ref(code / "src/radio_moge/optimizer_trial.py"),
            training_lock=previous["training_lock"],
            config=file_ref(root / "config.json"),
            runner=file_ref(code / "scripts/run_room_optimizer_arm.py"),
            trainer=file_ref(code / "src/radio_moge/train.py"),
            execution_revision=revision,
            automatic_retry=False,
            automatic_next_training=False,
            automatic_deployment=False,
            predefined_result="fixed1000 allsixviews; bothmacrocosupandLiDARdown; allmetricsreported",
            **{
                k: previous[k]
                for k in (
                    "real_manifest",
                    "synthetic_manifest",
                    "validation_manifest",
                    "structural_validation_manifest",
                )
            },
        )
        check_contract(current, parent, plan)
        publish_json(root / "plan.json", plan)
        configs.append(current)
        refs.append(file_ref(root / "plan.json"))
    compare_arm_configs(*configs)
    publish_json(
        output / "plan.json",
        dict(
            arms=refs,
            controller=file_ref(code / "scripts/run_room_optimizer_pair.py"),
            source_step=finish["step"],
            fixed_updates_per_arm=UPDATES,
            actual_batch=BATCH,
            total_presentations_per_arm=UPDATES * BATCH,
            new_optimizer_state=True,
            changed_optimizer_policy="16 fresh core tensors; 84 preserved native AdamW tensors",
            baseline_arm=str(baseline),
            baseline_finish=file_ref(baseline / "finish-verification-01/report.json"),
            baseline_online=file_ref(baseline / "online-final-01/report.json"),
            comparison="fixed1000 continued AdamW vs core-reset AdamW vs core Muon; no best selection",
            no_baseline_inference=True,
            automatic_deployment=False,
        ),
    )
    print(json.dumps(file_ref(output / "plan.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--dependency", type=Path, required=True)
    parser.add_argument("--trainer-patch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    prepare(
        args.source.resolve(),
        args.baseline.resolve(),
        args.dependency.resolve(),
        args.trainer_patch.resolve(),
        args.output.resolve(),
        args.revision,
    )
