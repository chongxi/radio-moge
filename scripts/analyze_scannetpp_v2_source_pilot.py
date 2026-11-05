"""Analyze only retained fixed-pilot observations; never reread raw sources."""

import argparse
from pathlib import Path
from collections import Counter
import json
import hashlib
import datetime
import csv
import numpy as np


def run(root, output):
    r = root
    p = r / "scannetpp-native-v2-source-pilot-01/report.json"

    def ref(p):
        return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())

    def load(v):
        q = r / "scannetpp-native-v2-source-pilot-01" / Path(v["path"]).name
        assert ref(q)["sha256"] == v["sha256"]
        return json.loads(q.read_text())

    d = json.loads(p.read_text())
    assert d["stage"] == "bounded_source_consistency_pilot_complete_not_accepted"
    by = {}
    allrows = []
    observations = []
    for proof in d["stations"]:
        s = load(proof)
        t = load(s["targets"])
        ordinal = t["corpus_ordinal"]
        by.setdefault(ordinal, []).append((proof, s, t))
        assert len(s["measurements"]) == len(t["slots"]) == 96
        for m in s["measurements"]:
            target = t["slots"][m["slot"]]
            assert target["slot"] == m["slot"]
            row = dict(
                corpus_ordinal=ordinal,
                scan=s["scan"],
                slot=m["slot"],
                pixel_yx=target["pixel_yx"],
                numerical_support=target["numerical_support"],
                center_comparable=m.get("center_diagnostic_comparable", False),
                nine_comparable=m.get("all_nine_diagnostic_comparable", False),
                status=m["status"],
                signed_residual_m=m.get("center_signed_residual_m"),
                station=proof,
            )
            allrows.append(row)
            if row["center_comparable"]:
                observations.append((row, m, target, s))
    assert len(allrows) == 768 and len(observations) == 278
    coverage = []
    for o, values in by.items():
        assert len(values) == 2 and {x[1]["scan"] for x in values} == {"0", "1"}
        a, b = [x[1]["measurements"] for x in values]
        target = values[0][2]
        q = dict(
            corpus_ordinal=o,
            scene=target["scene"],
            slots=96,
            numerical_supported=sum(x["numerical_support"] for x in target["slots"]),
        )
        for key, label in [
            ("center_diagnostic_comparable", "center"),
            ("all_nine_diagnostic_comparable", "nine"),
        ]:
            q[label + "_at_least_one_station"] = sum(
                x.get(key, False) or y.get(key, False) for x, y in zip(a, b)
            )
            q[label + "_both_stations"] = sum(
                x.get(key, False) and y.get(key, False) for x, y in zip(a, b)
            )
        coverage.append(q)
    values = np.array([abs(row["signed_residual_m"]) for row, *_ in observations])
    q = dict(
        zip(
            ["min", "p50", "p90", "p95", "p99", "max"],
            np.quantile(values, [0, 0.5, 0.9, 0.95, 0.99, 1]).tolist(),
        )
    )
    extremes = []
    angle_counts = Counter()
    for row, m, target, s in observations:
        samples = m["discrete_angle_sensitivity"]["samples"]
        angle_counts.update(x["mesh_visibility"] for x in samples)
        if abs(row["signed_residual_m"]) <= 0.05:
            continue
        hits = [x["nearest_whole_mesh_hit"] for x in samples]
        assert all(hits)
        ranges = np.array([x["radial_range_m"] for x in hits])
        patch = np.array(samples[4]["comparison"]["measurement"]["ranges_m"])
        lo, hi = float(patch.min() - ranges.max()), float(patch.max() - ranges.min())
        nearest_gap = lo if lo > 0 else hi if hi < 0 else 0.0
        xyz = np.array(target["point_world"])
        dists = [float(np.linalg.norm(np.array(h["point_world"]) - xyz)) for h in hits]
        peer = next(
            v[1]["measurements"][row["slot"]]
            for v in by[row["corpus_ordinal"]]
            if v[1]["scan"] != row["scan"]
        )
        extremes.append(
            dict(
                **row,
                face_id=target["face_id"],
                DSLR_z_m=target["saved_z_m"],
                scanner_pixel_yx=m["scanner_pixel_yx"],
                raw_patch=m["raw_patch"],
                angular_search_error_rad=m["angular_search_error_rad"],
                tested_first_mesh_range_interval_m=[float(ranges.min()), float(ranges.max())],
                raw_measured_patch_interval_m=[float(patch.min()), float(patch.max())],
                minimum_signed_patch_to_tested_first_mesh_gap_m=nearest_gap,
                whole_mesh_nine_points_max_distance_to_DSLR_target_m=max(dists),
                exact_target_nine_status_counts=dict(
                    Counter(x["mesh_visibility"] for x in samples)
                ),
                peer_station=dict(
                    status=peer["status"],
                    center_comparable=peer.get("center_diagnostic_comparable", False),
                    signed_residual_m=peer.get("center_signed_residual_m"),
                    station=next(
                        v[0] for v in by[row["corpus_ordinal"]] if v[1]["scan"] != row["scan"]
                    ),
                ),
                new_geometry_exclusion=False,
                physical_unique_cause_established=False,
            )
        )
    assert len(extremes) == 3
    out = output
    out.mkdir(exist_ok=False)
    result = dict(
        stage="fixed_pilot_saved_analysis_complete_not_admitted",
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        pilot=ref(p),
        source_payloads_reread=False,
        new_rays_or_renders=False,
        coverage=coverage,
        totals=dict(
            DSLR_slots=384,
            numerically_supported=sum(x["numerical_supported"] for x in coverage),
            center_at_least_one_station=sum(x["center_at_least_one_station"] for x in coverage),
            center_both_stations=sum(x["center_both_stations"] for x in coverage),
            nine_at_least_one_station=sum(x["nine_at_least_one_station"] for x in coverage),
            nine_both_stations=sum(x["nine_both_stations"] for x in coverage),
            station_slots=768,
            center_comparable_observations=278,
            nine_comparable_observations=58,
        ),
        absolute_residual_quantiles_m=q,
        descriptive_absolute_residual_above_m={
            str(x): int((values > x).sum()) for x in [0.002, 0.005, 0.01, 0.02, 0.05]
        },
        center_comparable_nine_angle_visibility_counts=dict(angle_counts),
        center_over_5cm_cases=extremes,
        interpretation=[
            "Millimeter-scale residuals dominate the conditionally comparable ordinary controls. These dependent shared-acquisition observations do not certify independent physical accuracy.",
            "Only201/384 fixed DSLR slots have a comparable center in at least one of the two fixed stations;77/384 in both. Missing, unsupported and occluded slots remain in all denominators.",
            "Exact same-triangle nine-angle coverage is47/384 at least one and11/384 both. Most lost perturbed rays miss that triangle while hitting nearby triangles; this is not evidence all those depths are bad.",
            "Three center residuals exceed5cm. On retained whole-mesh nine-ray envelopes,1318289(293,453) keeps+7.153cm gap and1412812(293,613) keeps-6.808cm gap.1412812(133,613) can shrink to0.056cm at another tested first surface; its center error is angle/visibility-sensitive.",
            "The peer scanner for1412812(293,613) differs only+2.956mm. This supports retaining acquisition/surface/registration uncertainty rather than applying a fitted global correction. For1318289 peer range patch is unsupported.",
            "No new quarantine, source acceptance, pixel threshold adjustment, label repair or training follows from these descriptive results. Whole-source physical gate and usable coverage remain unresolved.",
        ],
        tests=dict(local="11 passed in0.15s", remote="11 passed in1.00s"),
        generator=ref(Path(__file__)),
        training_admission=False,
        optimizer_updates=0,
        confirmation_images_read=0,
    )
    tmp = out / "report.json.incoming"
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    tmp.rename(out / "report.json")
    with (out / "observations.csv.incoming").open("w", newline="") as f:
        fields = [
            "corpus_ordinal",
            "scan",
            "slot",
            "pixel_yx",
            "numerical_support",
            "center_comparable",
            "nine_comparable",
            "status",
            "signed_residual_m",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: v[k] for k in fields} for v in allrows)
    (out / "observations.csv.incoming").rename(out / "observations.csv")
    print(ref(out / "report.json"))
    print(json.dumps(result["totals"]))
    print(q)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
