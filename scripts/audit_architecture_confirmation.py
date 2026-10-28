"""Check reserved confirmation identities, aggregates and fixed teacher geometry."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

from audit_architecture_screen import atomic, audit_point, digest, lines, read


def aggregate(rows):
    result = {}
    keys = {k for row in rows for k, v in row.items() if type(v) in (int, float)}
    for key in keys:
        values = [
            r[key] for r in rows if type(r.get(key)) in (int, float) and math.isfinite(r[key])
        ]
        if values:
            result[key] = math.fsum(values) / len(values)
            result[key + "/count"] = len(values)
    result["images"] = len(rows)
    return result


def geometry_means(rows):
    result, source_scores = aggregate(rows), []
    sources = sorted({r["dataset"] for r in rows})
    for source in sources:
        group = [r for r in rows if r["dataset"] == source]
        means = aggregate(group)
        for key in (
            "teacher_depth/abs_rel",
            "teacher_depth/rmse",
            "raw/points",
            "raw/normal",
            "scale_log_error",
            "postprocess_failed",
            "images",
        ):
            if key in means:
                result[f"dataset/{source}/{key}"] = means[key]
        if means.get("teacher_depth/abs_rel/count") == len(group):
            source_scores.append(means["teacher_depth/abs_rel"])
    if len(source_scores) == len(sources):
        result["macro/teacher_depth/abs_rel"] = math.fsum(source_scores) / len(sources)
    return result


def run(root):
    plan, shortlist = read(root / "plan.json"), read(root / "shortlist.json")
    records = lines(root / "confirmation.jsonl")
    assert digest(root / "confirmation.jsonl") == plan["manifest_sha256"]["confirmation"]
    status_path = root / "confirmation-status.json"
    if not status_path.exists():
        print('{"stage":"pending"}')
        return
    status = read(status_path)
    assert status["shortlist_sha256"] == digest(root / "shortlist.json")
    assert shortlist["frozen_at"] < status["started_at"] and status["optimizer_updates"] == 0
    teacher_path = root / "confirmation/teacher-per-image.json"
    if not teacher_path.exists():
        print('{"stage":"teacher_pending"}')
        return
    teachers = read(teacher_path)
    counts, expected_ids = Counter(), []
    for i, row in enumerate(records):
        if counts[row["dataset"]] < 4 or row["dataset"] == "nyuv2":
            expected_ids.append(i)
        counts[row["dataset"]] += 1
    assert len(teachers) == len(expected_ids) == 56
    for teacher, i in zip(teachers, expected_ids, strict=True):
        assert teacher["cache_index"] == i
        for key in ("image", "dataset", "group"):
            assert teacher[key] == records[i][key]
        assert teacher["input_image_sha256"] == records[i]["sha256"]
    results = {}
    for selected in shortlist["selected"]:
        name, nfe = selected["name"], selected["nfe"]
        arm = root / "confirmation" / name
        feature_dir = arm / f"features-nfe{nfe}"
        if (
            not (feature_dir / "integrity.json").exists()
            or not (arm / "geometry-metrics.json").exists()
        ):
            continue
        metric = read(feature_dir / "metrics.json")
        feature_proof = audit_point(
            feature_dir,
            records,
            list(range(224)),
            Counter(r["dataset"] for r in records),
            1000,
            nfe,
            metric,
        )
        rows, actual = read(arm / "geometry-per-image.json"), read(arm / "geometry-metrics.json")
        assert len(rows) == 56
        for row, teacher in zip(rows, teachers, strict=True):
            assert all(row.get(k) == v for k, v in teacher.items()), (name, row["cache_index"])
            assert all(math.isfinite(v) for v in row.values() if type(v) in (int, float))
            if row["teacher_failed"] == 0:
                assert row["postprocess_failed"] in (0, 1)
                if row["postprocess_failed"] == 0:
                    for key in (
                        "teacher_depth/abs_rel",
                        "teacher_depth/rmse",
                        "teacher_depth/delta1",
                        "fov_error_deg",
                    ):
                        assert type(row.get(key)) in (int, float) and math.isfinite(row[key]), (
                            name,
                            key,
                        )
                    if row["dataset"] == "nyuv2":
                        assert all(
                            type(row.get("gt_student/" + k)) in (int, float)
                            for k in ("abs_rel", "rmse", "delta1")
                        )
        expected = geometry_means(rows)
        assert set(actual) == set(expected)
        error = max(abs(actual[k] - v) for k, v in expected.items())
        assert error < 1e-8
        result = dict(
            stage="passed",
            checked_at=time.time(),
            feature=feature_proof,
            geometry_images=56,
            exact_geometry_identity_order=True,
            fixed_teacher_rows_identical=True,
            max_geometry_mean_error=error,
            teacher_failures=sum(r["teacher_failed"] for r in rows),
            postprocess_failures=sum(r.get("postprocess_failed", 0) for r in rows),
            plane_failures=sum(r.get("planes/failed", 0) for r in rows),
            preservation_failures=sum(r.get("preservation/failed", 0) for r in rows),
            gt_count=actual.get("gt_student/abs_rel/count", 0),
            input_sha256={
                "geometry-per-image.json": digest(arm / "geometry-per-image.json"),
                "geometry-metrics.json": digest(arm / "geometry-metrics.json"),
            },
        )
        proofpath = arm / "verification.json"
        if proofpath.exists():
            previous = read(proofpath)
            assert previous["input_sha256"] == result["input_sha256"]
            result = previous
        else:
            atomic(proofpath, result)
        results[name] = result
    complete = status["stage"] == "complete" and len(results) == len(shortlist["selected"])
    report = dict(
        stage="passed" if complete else "pending_remaining_candidates",
        checked_at=time.time(),
        optimizer_updates=0,
        frozen_shortlist_before_confirmation=True,
        results=results,
    )
    atomic(root / "confirmation-verification.json", report)
    print(
        json.dumps(
            dict(
                stage=report["stage"],
                verified_candidates=list(results),
                failures={n: p["postprocess_failures"] for n, p in results.items()},
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root)
