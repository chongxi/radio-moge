"""Describe existing supported-depth metadata in already-flagged scene groups.

No RGB, depth payload, model, source mesh or scanner image is read. Bins and
bounded selections describe signatures for review, never physical-error rates,
new exclusions, replacement routes or source acceptance.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np

from radio_moge.corpus_admission import checked_json, file_ref, readonly_db
from radio_moge.geometry_cache import canonical_json, publish_json


DEPTH_EDGES = (0.02, 0.05, 0.1, 0.25, 1.0)
METRICS = ("median_depth_m", "maximum_depth_m", "valid_fraction", "rejected_hit_fraction")


def bin_index(value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Positive finite supported-depth median required")
    return sum(value >= edge for edge in DEPTH_EDGES)


def summarize(records):
    candidates = [r for r in records if r["outcome"] == "native_candidate"]
    bins = []
    for i in range(len(DEPTH_EDGES) + 1):
        selected = [r for r in candidates if bin_index(r["metrics"]["median_depth_m"]) == i]
        bins.append(
            dict(
                lower_inclusive_m=(0.0, *DEPTH_EDGES)[i],
                upper_exclusive_m=(*DEPTH_EDGES, None)[i],
                images=len(selected),
                ordinals=[r["corpus_ordinal"] for r in selected],
                existing_quarantined_ordinals=[
                    r["corpus_ordinal"] for r in selected if r["existing_quarantine"]
                ],
            )
        )
    return dict(
        images=len(records),
        candidates=len(candidates),
        rejected=len(records) - len(candidates),
        source_policy_counts=dict(Counter(r["policy_sha256"] for r in candidates)),
        supported_depth_median_bins=bins,
        per_image_quantiles={
            key: np.quantile(
                [r["metrics"][key] for r in candidates], [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]
            ).tolist()
            for key in METRICS
        }
        if candidates
        else {},
        quantile_probabilities=[0, 0.05, 0.25, 0.5, 0.75, 0.95, 1],
    )


def choose_followup(records, already_reviewed):
    """At most two descriptive signature examples and one median control per group."""
    result = []
    for group in sorted({r["group"] for r in records}):
        candidates = [
            r for r in records if r["group"] == group and r["outcome"] == "native_candidate"
        ]
        available = [r for r in candidates if r["corpus_ordinal"] not in already_reviewed]
        selected = set()

        def take(pool, reason, key):
            ordered = sorted((r for r in pool if r["corpus_ordinal"] not in selected), key=key)
            if ordered:
                row = ordered[0]
                selected.add(row["corpus_ordinal"])
                result.append(
                    dict(
                        **row,
                        selection_reason=reason,
                        eligible_pool_images=len(ordered),
                        absent_from_supplied_reviewed_ordinal_set=True,
                    )
                )

        take(
            available,
            "lowest_supported_depth_median",
            lambda r: (r["metrics"]["median_depth_m"], r["corpus_ordinal"]),
        )
        take(
            [r for r in available if r["metrics"]["median_depth_m"] < 0.1],
            "highest_valid_fraction_among_supported_medians_below_0.1m",
            lambda r: (-r["metrics"]["valid_fraction"], r["corpus_ordinal"]),
        )
        if candidates:
            center = float(np.median([r["metrics"]["median_depth_m"] for r in candidates]))
            take(
                available,
                "nearest_scene_median_of_image_medians_not_assumed_good",
                lambda r: (abs(r["metrics"]["median_depth_m"] - center), r["corpus_ordinal"]),
            )
    return result


def verified_temporary_copy(ref, destination):
    source = Path(ref["path"])
    if any(Path(str(source) + suffix).exists() for suffix in ("-wal", "-journal")):
        raise ValueError("Closed immutable index required")
    before = source.stat()
    checksum = hashlib.sha256()
    with source.open("rb") as incoming, destination.open("xb") as output:
        while chunk := incoming.read(4 * 1024**2):
            checksum.update(chunk)
            output.write(chunk)
    after = source.stat()
    if checksum.hexdigest() != ref["sha256"] or any(
        getattr(before, k) != getattr(after, k)
        for k in ("st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    ):
        raise ValueError("Frozen index hash changed while copying")


def bind_existing_quarantine(records, review, validation, register_values):
    by_ordinal = {r["corpus_ordinal"]: r for r in records}
    reviewed = {r["corpus_ordinal"]: r for r in review["records"]}
    expected = set(review["quarantined_candidate_ordinals"])
    seen = set()
    for binding, register in register_values:
        if (
            register["policy_sha256"] != binding["policy_sha256"]
            or register["packs_index"] != binding["packs_index"]
            or register["stage"] != "candidate_quality_quarantine_recorded"
            or register["training_admission"] is not False
            or register["optimizer_updates"] != 0
            or len(register["flags"]) != binding["count"]
        ):
            raise ValueError("Bound candidate quarantine register differs")
        if sorted(f["corpus_ordinal"] for f in register["flags"]) != sorted(binding["ordinals"]):
            raise ValueError("Quarantine ordinals differ from validated register")
        for flag in register["flags"]:
            ordinal = flag["corpus_ordinal"]
            if ordinal in seen or ordinal not in expected or ordinal not in by_ordinal:
                raise ValueError("Missing or repeated exact quarantined candidate")
            row, visual = by_ordinal[ordinal], reviewed[ordinal]
            if (
                row["outcome"] != "native_candidate"
                or visual["decision"] != "quarantine_candidate"
                or flag["decision"] != "quarantine_candidate"
                or row["policy_sha256"] != register["policy_sha256"]
                or row["policy_sha256"] != visual["policy_sha256"]
                or row["entry_key"] != flag["entry_key"]
                or row["entry_key"] != visual["entry_key"]
                or row["evidence"] != flag["receipt"]
                or row["evidence"] != visual["receipt"]
                or row["group"] != flag["group"]
                or row["group"] != visual["group"]
                or row["image"] != dict(path=flag["image"], sha256=flag["sha256"])
                or row["image"] != visual["image"]
                or flag["member_sha256"] != visual["member_sha256"]
            ):
                raise ValueError("Quarantined candidate image/member/policy identity differs")
            row["existing_quarantine"] = True
            seen.add(ordinal)
    if seen != expected or len(seen) != validation["quarantined_candidates"]:
        raise ValueError("Incomplete existing quarantine overlay")


def run(review_path, validation_path, output):
    review_ref, validation_ref = file_ref(review_path), file_ref(validation_path)
    review, validation = checked_json(review_ref), checked_json(validation_ref)
    audit_ref = review["audit"]
    audit = checked_json(audit_ref)
    if (
        review["stage"] != "visual_review_complete_source_quality_unresolved"
        or validation["stage"] != "exact_candidate_quarantine_validated"
        or validation["visual_review"] != review_ref
        or audit["stage"] != "native_outcomes_accounted_not_accepted"
        or audit["training_admission"] is not False
        or not audit["producer_array_readback_complete"]
    ):
        raise ValueError("Completed native outcomes and existing visual quarantine required")
    groups = sorted(review["affected_sampled_scenes"])
    if not 1 <= len(groups) <= 5 or len(groups) != len(set(groups)):
        raise ValueError("Bounded distinct flagged groups required")
    registers = [
        (binding, checked_json(binding["register"])) for binding in validation["registers"]
    ]
    if any(r["manifest_sha256"] != audit["manifest_sha256"] for _, r in registers):
        raise ValueError("Quarantine manifest differs")
    strata = checked_json(audit["strata"])
    expected = Counter(
        {
            (s["group"], s["policy"], s["outcome"], s["reason"]): s["images"]
            for s in strata
            if s["group"] in groups
        }
    )
    records = []
    with tempfile.TemporaryDirectory(prefix="scannet-scene-scope-") as scratch:
        database = Path(scratch) / "index.sqlite"
        verified_temporary_copy(audit["index"], database)
        with readonly_db(database) as db:
            query = (
                "SELECT ordinal,image,sha256,scene_group,row_json,outcome,reason,policy,evidence_json,entry_key,details_json FROM images WHERE scene_group IN ("
                + ",".join("?" for _ in groups)
                + ") ORDER BY ordinal"
            )
            for (
                ordinal,
                image,
                sha,
                group,
                encoded,
                outcome,
                reason,
                policy,
                evidence,
                key,
                details,
            ) in db.execute(query, groups):
                row, metrics = json.loads(encoded), json.loads(details)
                if (
                    row["image"] != image
                    or row["sha256"] != sha
                    or row["group"] != group
                    or row["dataset"] != "scannetpp_dslr"
                    or row["split"] != "train"
                    or row.get("corpus_ordinal", ordinal) != ordinal
                ):
                    raise ValueError("Frozen identity columns differ")
                if outcome == "native_candidate":
                    if (
                        not all(math.isfinite(metrics[k]) for k in METRICS)
                        or not 0.05 <= metrics["valid_fraction"] <= 1
                        or not 0 <= metrics["rejected_hit_fraction"] <= 1
                        or not 0.01 - 1e-8
                        <= metrics["median_depth_m"]
                        <= metrics["maximum_depth_m"]
                    ):
                        raise ValueError("Candidate metadata range differs")
                elif outcome != "native_rejected":
                    raise ValueError("Unaccounted outcome")
                records.append(
                    dict(
                        corpus_ordinal=ordinal,
                        image=dict(path=image, sha256=sha),
                        group=group,
                        view=row["view"],
                        outcome=outcome,
                        reason=reason,
                        policy_sha256=policy,
                        evidence=json.loads(evidence),
                        entry_key=key,
                        metrics=metrics,
                        existing_quarantine=False,
                    )
                )
    actual = Counter((r["group"], r["policy_sha256"], r["outcome"], r["reason"]) for r in records)
    if actual != expected or len({r["corpus_ordinal"] for r in records}) != len(records):
        raise ValueError("Selected exact scope differs from frozen strata")
    bind_existing_quarantine(records, review, validation, registers)
    reviewed = {r["corpus_ordinal"] for r in review["records"]}
    selections = choose_followup(records, reviewed)
    output.mkdir(exist_ok=False)
    rows_path = output / "rows.jsonl"
    with rows_path.with_suffix(".jsonl.incoming").open("xb") as file:
        for row in records:
            file.write(canonical_json(row) + b"\n")
    rows_path.with_suffix(".jsonl.incoming").rename(rows_path)
    publish_json(output / "followup-candidates.json", selections)
    report = dict(
        stage="flagged_scene_metadata_scope_complete",
        review=review_ref,
        quarantine_validation=validation_ref,
        registers=[b["register"] for b, _ in registers],
        native_audit=audit_ref,
        index=audit["index"],
        strata=audit["strata"],
        manifest_sha256=audit["manifest_sha256"],
        rows=file_ref(rows_path),
        followup_candidates=file_ref(output / "followup-candidates.json"),
        reviewed_ordinal_set=sorted(reviewed),
        existing_quarantined_ordinals=review["quarantined_candidate_ordinals"],
        aggregate=summarize(records),
        scenes=[
            dict(group=g, **summarize([r for r in records if r["group"] == g])) for g in groups
        ],
        followup_images=len(selections),
        sampling_rule="Per scene, at most lowest median; highest support among remaining median<0.1m; closest to scene median of image medians. Exclude supplied reviewed ordinals; ordinal breaks ties.",
        interpretation="Supported-depth metadata signatures, not erroneous-image or erroneous-pixel fractions. Bins and sample rules are not admission/exclusion thresholds; median controls are not presumed physically correct.",
        exact_existing_quarantine_retained=True,
        new_exclusions=0,
        source_quality_accepted=False,
        training_admission=False,
        optimizer_updates=0,
        gpu_used=False,
        raw_RGB_or_geometry_payload_read=False,
        source_arrays_or_inputs_modified=False,
        code=file_ref(Path(__file__)),
    )
    publish_json(output / "report.json", report)
    print(
        json.dumps(
            dict(
                images=len(records),
                candidates=report["aggregate"]["candidates"],
                followup_images=len(selections),
                output=str(output),
            )
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.review.resolve(), args.validation.resolve(), args.output.resolve())
