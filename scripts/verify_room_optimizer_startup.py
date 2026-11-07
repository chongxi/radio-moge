"""Verify actual same-shape large-batch startup and unchanged image/crop-seed stream."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from radio_moge.corpus_admission import checked_json, file_ref
from radio_moge.data import DeterministicBatches
from radio_moge.geometry_cache import publish_json
from radio_moge.optimizer_trial import check_counts, policy
import hashlib


def run(root):
    plan = checked_json(file_ref(root / "plan.json"))
    config = checked_json(plan["config"])
    batch = plan["actual_batch"]
    accumulation = batch // 4
    presentations = batch * 20
    metadata = checked_json(file_ref(root / "metadata-report.json"))
    initial = checked_json(file_ref(root / "initial-verification.json"))
    model = checked_json(file_ref(root / "mixed-model-zero-update.json"))
    check_counts(initial["optimizer_counts"], plan["arm"], plan["source_step"], 0)
    if (
        initial["step"] != plan["source_step"]
        or initial["cursor"] != 0
        or initial["real_stream_start_cursor"] != plan["source_cursor"]
        or initial["parameter_count"] != 100
        or config["optimizer_trial"] != policy(plan["arm"])
        or initial["core_optimizer"] != config["optimizer_trial"]
        or model["temporary_optimizer_updates"] != 2
        or model["temporary_updates_discarded_and_state_restored"] is not True
        or model["optimizer_updates"] != 0
        or model["state_before"] != model["state_after"]
        or model["rng_restored_bitwise"] is not True
        or model["geometry_branch_executed"] is not False
        or config["loss"] != dict(spatial=1.0, summary=1.0, cosine=1.0, summary_cosine=1.0)
    ):
        raise ValueError("Missing actual initial/model proof or changed feature coefficients")
    training = root / "training"
    log = []
    with (training / "train.jsonl").open() as file:
        for line in file:
            log.append(json.loads(line))
            if len(log) == 20:
                break
    if len(log) != 20:
        raise ValueError("Wait for twenty committed optimizer updates")
    for i, row in enumerate(log):
        if (
            row["step"] != plan["source_step"] + i + 1
            or row["logical_data_cursor"] != (i + 1) * accumulation
            or row["effective_batch"] != batch
            or row.get("actual_forward_batch") != batch
            or row.get("model_forwards_per_update") != 1
            or row["lr_used"] != plan["learning_rate"]
            or row["gradient_norm"] <= 0
            or any(not math.isfinite(v) for v in row.values() if isinstance(v, (float, int)))
            or not math.isclose(
                row["total"], sum(row[k] for k in config["loss"]), rel_tol=1e-5, abs_tol=2e-6
            )
        ):
            raise ValueError(
                "Training loss/batch/step/cursor is not the declared pure-feature update"
            )
    update_norms = [r["update_norm"] for r in log if "update_norm" in r]
    if not update_norms or any(v <= 0 for v in update_norms):
        raise ValueError("No measured nonzero adapter update")
    original = [
        list(k)
        for batch in DeterministicBatches(
            metadata["real_images"],
            4,
            config["data"]["buckets"],
            config["train"]["seed"],
            plan["source_cursor"],
            20 * accumulation,
        )
        for k in batch
    ]
    planned = checked_json(file_ref(root / "planned-keys.json"))[:presentations]
    counts, houses, observed = Counter(), set(), []
    with (training / "actual-rgb-inputs.jsonl").open() as file:
        for i, line in enumerate(file):
            if i == 20:
                break
            row = json.loads(line)
            if row["update"] != log[i]["step"] or len(row["logical_images"]) != batch:
                raise ValueError("Input audit and successful optimizer updates differ")
            if (
                len(row["compute_groups"]) != 1
                or row["compute_groups"][0]["shape"][0] != batch
                or row["compute_groups"][0]["logical_image_slots"] != list(range(batch))
            ):
                raise ValueError("Actual forward batch split or reordered")
            for item in row["logical_images"]:
                slot = len(observed)
                if item["key"] != planned[slot] or "ordinary_geometry" in item:
                    raise ValueError("Planned RGB key or geometry-free input contract differs")
                replaced = (slot + 1) % 8 == 0
                if replaced != (item["dataset"] in ("procthor_empty", "structural_rooms")):
                    raise ValueError("Synthetic source share changed")
                if item["key"][1] != original[slot][1] or (
                    not replaced and item["key"][:2] != original[slot][:2]
                ):
                    raise ValueError("Original real identity or crop seed changed")
                if replaced:
                    houses.add(item["group"])
                counts[item["dataset"]] += 1
                observed.append(item)
    if (
        len(observed) != presentations
        or sum(counts[k] for k in ("procthor_empty", "structural_rooms")) != presentations // 8
        or counts.get("coco2017", 0)
    ):
        raise ValueError("Committed startup RGB population differs")
    digests = []
    for directory in [root, Path(plan["baseline_arm"])]:
        digest = hashlib.sha256()
        with (directory / "training/actual-rgb-inputs.jsonl").open("rb") as stream:
            for _ in range(20):
                raw = stream.readline()
                if not raw:
                    raise ValueError("Missing actual startup input line")
                digest.update(raw)
        digests.append(digest.hexdigest())
    if len(set(digests)) != 1:
        raise ValueError("Actual startup pixels/crops/buckets differ from the AdamW reference")
    report = dict(
        stage="actual_matched_optimizer_startup_verified",
        optimizer_policy=config["optimizer_trial"],
        actual_input_prefix_sha256=digests[0],
        historical_reference_actual_inputs_identical=True,
        learning_rate=plan["learning_rate"],
        updates=20,
        initial=file_ref(root / "initial-verification.json"),
        model=file_ref(root / "mixed-model-zero-update.json"),
        plan=file_ref(root / "plan.json"),
        actual_rgb_inputs=file_ref(training / "actual-rgb-inputs.jsonl"),
        input_file_is_growing=True,
        verified_prefix_updates=20,
        presentations=presentations,
        source_counts=dict(counts),
        synthetic_houses=len(houses),
        original_unreplaced_identities_and_crop_seeds_exact=True,
        planned_uniform_bucket_per_forward_verified=True,
        actual_forward_batch=batch,
        all_feature_weights_one=True,
        geometry_loss=False,
        finite_loss_and_gradients=True,
        nonzero_measured_update_norms=update_norms,
        config=plan["config"],
        verifier=file_ref(Path(__file__)),
        optimizer_updates_by_verifier=0,
    )
    # Bind the exact inspected prefix separately; the trainer's full audit keeps growing.
    publish_json(root / "startup-input-prefix.json", observed)
    report["fixed_prefix"] = file_ref(root / "startup-input-prefix.json")
    publish_json(root / "startup-verification-01.json", report)
    print(
        json.dumps(
            dict(
                report=file_ref(root / "startup-verification-01.json"),
                updates=20,
                source_counts=dict(counts),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root.resolve())
