"""Verify already-reviewed periodic epoch results against the online run."""

import argparse
import json
from pathlib import Path
import time

import wandb

from radio_moge.corpus_admission import checked_json, file_ref
from radio_moge.geometry_cache import publish_json
from review_room_lr_arm import endpoints
from verify_room_epoch_online import exact_values, final_validation_record


def verify(root, steps):
    if not steps or len(set(steps)) != len(steps):
        raise ValueError("Distinct reviewed endpoints are required")
    plan = checked_json(file_ref(root / "plan.json"))
    identity_ref = file_ref(root / "training/wandb-run.json")
    identity = checked_json(identity_ref)
    startup_ref = file_ref(root / "online-startup-01/report.json")
    startup = checked_json(startup_ref)
    if startup["run_id"] != identity["id"] or not startup["all_committed_values_numerically_exact"]:
        raise ValueError("Missing unchanged-run online startup proof")
    records = {}
    for line in (root / "training/validation.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["step"] in steps:
            if row["step"] in records:
                raise ValueError("Repeated local validation endpoint")
            records[row["step"]] = row
    if set(records) != set(steps) or not set(steps).issubset(
        endpoints(plan["source_step"], plan.get("actual_batch", 32))
    ):
        raise ValueError("Only complete predefined validation endpoints may be checked")
    reviews = {}
    for step in steps:
        if (root / f"online-validation-{step}-01").exists():
            raise FileExistsError("Online endpoint already checked; reuse its proof")
        reference = file_ref(root / f"validation-{step}-review-01/report.json")
        review = checked_json(reference)
        if review["step"] != step or review["plan"] != file_ref(root / "plan.json"):
            raise ValueError("Saved paired review does not bind this endpoint")
        reviews[step] = reference
    keys = {"validation/" + k for k in records[steps[0]] if k != "step"}
    if any({"validation/" + k for k in row if k != "step"} != keys for row in records.values()):
        raise ValueError("Validation metric coverage changed")
    run = wandb.Api(timeout=45).run("/".join(identity[k] for k in ("entity", "project", "id")))
    rows = list(run.scan_history(keys=["optimizer_step"] + sorted(keys), page_size=1000))
    found = {step: final_validation_record(rows, step, keys)[0] for step in records}
    comparisons = {}
    for step, record in records.items():
        expected = {"validation/" + k: v for k, v in record.items() if k != "step"}
        comparisons[step] = exact_values(
            {k: found[step][k] for k in expected}, expected, f"validation/{step}"
        )
    for step in steps:
        output = root / f"online-validation-{step}-01"
        output.mkdir(exist_ok=False)
        report = dict(
            stage="committed_epoch_validation_online_verified",
            step=step,
            run_id=identity["id"],
            run_state_observed=run.state,
            identity=identity_ref,
            paired_review=reviews[step],
            metadata_startup_proof_reused=startup_ref,
            metrics_count=len(keys),
            all_metrics_numerically_exact=True,
            representation_changes=comparisons[step],
            verifier=file_ref(Path(__file__)),
            new_model_or_checkpoint_read=False,
            training_modified=False,
            checked_at=time.time(),
        )
        publish_json(output / "report.json", report)
        print(
            json.dumps(
                dict(
                    step=step,
                    metrics=len(keys),
                    numeric_exact=True,
                    representation_changes=len(comparisons[step]),
                    report=file_ref(output / "report.json"),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    args = parser.parse_args()
    verify(args.root.resolve(), args.steps)
