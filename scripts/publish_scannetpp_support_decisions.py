"""Publish reviewed support-rejection exclusions, keeping RGB and source flags."""

import argparse
import hashlib
import json
from pathlib import Path

from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import canonical_json, publish_json
from review_scannetpp_support_rejections import REASON, attrition


def checked_decision_scope(records, review):
    items = review["records"]
    if (
        not records
        or len(records) != len(items)
        or len({x["corpus_ordinal"] for x in records}) != len(records)
        or len({x["corpus_ordinal"] for x in items}) != len(items)
        or {x["corpus_ordinal"] for x in records} != {x["corpus_ordinal"] for x in items}
    ):
        raise ValueError("Exactly one reviewed decision per rejection required")
    selected = {x["corpus_ordinal"]: x for x in items}
    for rec in records:
        chosen = selected[rec["corpus_ordinal"]]
        row = rec["image"]
        if (
            chosen["decision"] != "exclude_geometry_supervision_retain_rgb"
            or chosen["actual_page_viewed"] is not True
            or any(chosen[k] != rec[k] for k in ("image", "page", "original_rejection"))
            or row["dataset"] != "scannetpp_dslr"
            or row["split"] != "train"
            or rec["reason"] != REASON
            or rec["producer_statistics_reproduced"] is not False
            or rec["final_depth_and_support_retained"] is not False
            or rec["original_mask_hash_recorded"] is not False
        ):
            raise ValueError("Reviewed identity, decision or evidence limitations differ")
        shape = [round(row["height"] * 640 / row["width"]), 640]
        if attrition(rec["producer_statistics"], shape) != rec["attrition_counts"]:
            raise ValueError("Retained rejection accounting differs")


def run(root, output):
    review_ref = file_ref(root / "scannetpp-support-rejection-review-01/visual-review.json")
    review = checked_json(review_ref)
    census = checked_json(review["evidence"])
    records = checked_json(census["records"])
    if (
        review["stage"] != "support_rejection_visual_review_complete"
        or review["all_pages_inspected"] is not True
        or review["reviewed_images"] != 70
        or len(records) != 70
        or review["reviewed_pages"] != len(census["pages"])
        or review["records_evidence"] != census["records"]
        or review["original_final_depth_reviewed"] is not False
        or census["stage"] != "support_rejection_evidence_rendered_not_reviewed"
        or census["all_rejection_identities_bound"] is not True
        or census["original_statistics_reproduced"] is not False
        or review["source_quality_accepted"] is not False
    ):
        raise ValueError("Complete bounded review with original evidence limitations required")
    checked_decision_scope(records, review)
    for proof in census["pages"]:
        checked_path(proof)
    for rec in records:
        original = checked_json(rec["original_rejection"])
        if (
            original["image"] != rec["image"]
            or original["reason"] != REASON
            or original["stats"] != rec["producer_statistics"]
            or original["source_proofs"] != rec["source_proofs"]
        ):
            raise ValueError("Original rejection receipt changed")
    previous_ref = file_ref(root / "tartanair-rejection-review-10/known-geometry-exclusions.jsonl")
    if previous_ref["sha256"] != "81b99a558ea9abe5335e7f8adf5c884a384f4e7170ee5c20c0ae920c1c30a13b":
        raise ValueError("Prior geometry exclusions changed")
    previous = [json.loads(line) for line in checked_path(previous_ref).read_text().splitlines()]
    if len(previous) != 4101 or {x["corpus_ordinal"] for x in previous} & {
        x["corpus_ordinal"] for x in records
    }:
        raise ValueError("Prior and new exclusion scope overlap")
    replay_report_ref = file_ref(root / "rgb-replay-candidates-01/report.json")
    replay = checked_json(replay_report_ref)["manifest"]
    path = Path(replay["path"])
    before = path.stat()
    digest, seen, count = hashlib.sha256(), set(), 0
    wanted = {x["image"]["image"]: x["image"] for x in records}
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            count += 1
            row = json.loads(line)
            if row["image"] in wanted:
                original = wanted[row["image"]]
                if row["image"] in seen or any(
                    row[k] != original[k] for k in ("sha256", "group", "dataset")
                ):
                    raise ValueError("RGB pool identity is duplicated or changed")
                seen.add(row["image"])
    after = path.stat()
    if (
        digest.hexdigest() != replay["sha256"]
        or count != 1637908
        or len(seen) != 70
        or any(
            getattr(before, k) != getattr(after, k)
            for k in ("st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        )
    ):
        raise ValueError("All original RGB must remain in unchanged full candidate pool")
    decisions = []
    for rec in sorted(records, key=lambda x: x["corpus_ordinal"]):
        row = rec["image"]
        decisions.append(
            dict(
                corpus_ordinal=rec["corpus_ordinal"],
                image=row["image"],
                sha256=row["sha256"],
                dataset=row["dataset"],
                group=row["group"],
                decision="exclude",
                scope="geometry_supervision",
                rgb_replay_excluded=False,
                reason="reviewed_" + REASON,
                native_reason=REASON,
                candidate_policy_sha256=rec["policy_sha256"],
                evidence=[
                    review_ref,
                    review["evidence"],
                    census["records"],
                    rec["original_rejection"],
                    rec["page"],
                ],
                interpretation="Insufficient native support under unchanged policy. Current masks are not historical "
                "mask-identity proof; unsaved final geometry was not reconstructed. No physical-error claim.",
            )
        )
    combined = sorted(previous + decisions, key=lambda x: x["corpus_ordinal"])
    if (
        len({x["corpus_ordinal"] for x in combined}) != 4171
        or len({x["image"] for x in combined}) != 4171
        or [x["corpus_ordinal"] for x in combined if x.get("rgb_replay_excluded")] != [210812]
    ):
        raise ValueError("Merged exact geometry/RGB scope differs")
    output.mkdir(exist_ok=False)
    for name, rows in [
        ("geometry-exclusions.jsonl", decisions),
        ("known-geometry-exclusions.jsonl", combined),
    ]:
        tmp = output / (name + ".incoming")
        with tmp.open("xb") as stream:
            for row in rows:
                stream.write(canonical_json(row) + b"\n")
        tmp.rename(output / name)
    publish_json(
        output / "report.json",
        dict(
            stage="explicit_geometry_exclusions_recorded",
            review=review_ref,
            evidence=review["evidence"],
            prior_exclusions=previous_ref,
            new_exclusions=file_ref(output / "geometry-exclusions.jsonl"),
            all_known_exclusions=file_ref(output / "known-geometry-exclusions.jsonl"),
            new_geometry_exclusions=70,
            total_geometry_exclusions=4171,
            actual_rgb_membership=dict(manifest=replay, count=count, verified_retained_images=70),
            candidate_quarantine_unchanged=19,
            rgb_excluded=False,
            whole_scene_exclusions=0,
            generator=file_ref(Path(__file__).resolve()),
            source_quality_accepted=False,
            training_configuration_modified=False,
            training_admission=False,
            optimizer_updates=0,
        ),
    )
    print(json.dumps(file_ref(output / "report.json")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
