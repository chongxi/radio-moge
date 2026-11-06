"""Review saved six-view results and committed image coverage of the room epoch."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import publish_json
from review_corpus_controls import joint_result
from review_room_feature_mix import compare_saved
from run_room_feature_epoch import IMAGES, SOURCES
from run_room_lr_arm import UPDATES, BATCH, SYNTHETIC_PRESENTATIONS, COVERAGE_GOAL
from review_room_feature_epoch import committed_prefix

VIEWS = {
    "original",
    "native-640x400",
    "dataset-scannetpp-dslr",
    "dataset-scannetpp-lidar",
    "dataset-procthor-empty",
    "dataset-structural-rooms",
}


def epoch_shape(plan):
    if (
        plan["actual_batch"] != BATCH
        or plan["additional_updates"] != UPDATES
        or plan["coverage_goal"] != COVERAGE_GOAL
        or plan["synthetic_presentations"] != SYNTHETIC_PRESENTATIONS
    ):
        raise ValueError("Wrong matched-LR budget")
    return BATCH, UPDATES, 250


def endpoints(source_step, batch=BATCH):
    if batch != BATCH:
        raise ValueError("Actual128 required")
    return [source_step + n for n in range(250, UPDATES + 1, 250)]


def check_identity(plan, config, parent, old, provenance, initial, metadata, baseline):
    _, updates, _ = epoch_shape(plan)
    if (
        plan["synthetic_images"] != IMAGES
        or plan["additional_updates"] != updates
        or plan["coverage_goal"] != COVERAGE_GOAL
        or plan["source_step"] != parent["step"]
        or set(parent["views"]) != VIEWS
        or provenance["revisions"] != old["revisions"]
        or provenance["adapter"] != old["adapter"]
        or initial["step"] != parent["step"]
        or initial["optimizer_states"] != 100
        or initial["cursor"] != 0
        or initial["real_stream_start_cursor"] != plan["source_cursor"]
        or initial["parent_mixed_cursor"] != plan["source_parent_cursor"]
        or metadata["synthetic_images"] != IMAGES
        or metadata["planned_synthetic_unique"] != SYNTHETIC_PRESENTATIONS
        or metadata["planned_synthetic_repeats"] != 0
        or metadata["geometry_loss"]
        or baseline["source"] != plan["source_results"]
        or baseline["step"] != parent["step"]
        or baseline["model_inference"] is not False
        or baseline["views"] != {k: v["files"] for k, v in parent["views"].items()}
        or config["loss"] != dict(spatial=1.0, summary=1.0, cosine=1.0, summary_cosine=1.0)
        or any(k in config for k in ("geometry_supervision", "ordinary_geometry", "feature_replay"))
    ):
        raise ValueError("Parent, saved baseline, exact-epoch or feature-only contract changed")
    if provenance["data_digests"] != dict(old["data_digests"], train=metadata["mixed_digest"]):
        raise ValueError(
            "Only the declared training population may change; all six heldouts stay fixed"
        )
    if initial["data_digests"] != provenance["data_digests"]:
        raise ValueError("Actual initial and evaluated populations differ")


def run(root, step, output):
    plan = checked_json(file_ref(root / "plan.json"))
    batch, total_updates, _ = epoch_shape(plan)
    parent = checked_json(plan["source_results"])
    config = checked_json(plan["config"])
    provenance_ref = file_ref(root / "training/provenance.json")
    provenance = checked_json(provenance_ref)
    old = checked_json(parent["provenance"])
    initial = checked_json(file_ref(root / "initial-verification.json"))
    metadata = checked_json(file_ref(root / "metadata-report.json"))
    baseline = checked_json(file_ref(root / "baseline-reuse.json"))
    check_identity(plan, config, parent, old, provenance, initial, metadata, baseline)
    updates = step - parent["step"]
    if step not in endpoints(parent["step"], batch):
        raise ValueError("Only a predefined committed validation endpoint may be reviewed")
    chosen = [
        r
        for line in (root / "training/validation.jsonl").read_text().splitlines()
        if (r := json.loads(line))["step"] == step
    ]
    if len(chosen) != 1:
        raise ValueError("A single completed validation record is required")
    planned = checked_json(metadata["planned_keys"])
    with checked_path(plan["synthetic_manifest"]).open() as file:
        synthetic = [json.loads(line) for line in file]
    if len(synthetic) != IMAGES or Counter(r["dataset"] for r in synthetic) != SOURCES:
        raise ValueError("Frozen source totals changed")
    prefix = committed_prefix(
        root / "training/actual-rgb-inputs.jsonl",
        planned,
        synthetic,
        metadata["real_images"],
        parent["step"],
        updates,
        batch,
    )
    if updates == total_updates and (
        prefix["synthetic_coverage"]["unique"] != SYNTHETIC_PRESENTATIONS
        or prefix["synthetic_coverage"]["repeated"]
    ):
        raise ValueError("The final endpoint did not cover the fixed16000 synthetic prefix")
    destination = root / "training" / f"eval-{step:07d}"
    views = {}
    for name in sorted(VIEWS):
        directory = destination if name == "original" else destination / name
        log_prefix = (
            ""
            if name == "original"
            else (
                "resolution/native-640x400/"
                if name == "native-640x400"
                else "dataset/" + name.removeprefix("dataset-") + "/"
            )
        )
        views[name] = compare_saved(
            parent["views"][name]["files"], directory, chosen[0], log_prefix
        )
    names = dict(
        spatial_cosine=("original", "macro/spatial/cosine"),
        summary_cosine=("original", "macro/summary/cosine"),
        lidar_abs_rel=("dataset-scannetpp-lidar", "gt_student/abs_rel"),
    )
    before = {k: views[v]["metrics"][m]["before"] for k, (v, m) in names.items()}
    current = {k: views[v]["metrics"][m]["final"] for k, (v, m) in names.items()}
    archive = {k: parent["against_archive"]["metrics"][k]["before"] for k in names}
    report = dict(
        stage="saved_matched_lr_validation_reviewed",
        step=step,
        new_updates=updates,
        budget_complete=updates == total_updates,
        full_epoch_complete=False,
        learning_rate=plan["learning_rate"],
        parent=plan["source_results"],
        provenance=provenance_ref,
        plan=file_ref(root / "plan.json"),
        views=views,
        actual_inputs=prefix,
        against_parent=joint_result(before, current),
        against_archive=joint_result(archive, current),
        model_inference=False,
        checkpoint_payload_read=False,
        optimizer_updates_by_reviewer=0,
        significance_or_causality_established=False,
        real_world_planarity_improvement_established=False,
        reviewed_at=time.time(),
        reviewer=file_ref(Path(__file__)),
    )
    output.mkdir(exist_ok=False)
    publish_json(output / "report.json", report)
    print(
        json.dumps(
            dict(
                report=file_ref(output / "report.json"),
                step=step,
                coverage=prefix["synthetic_coverage"],
                against_parent=report["against_parent"],
                against_archive=report["against_archive"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.step, args.output.resolve())
