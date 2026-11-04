"""Recheck saved scanner rays with full published transforms, without raw reads.

Saved pixel selection is retained. This does not assert the corrected inverse
would select the same pixel from the full angle grid, or establish a continuous
angular/pose uncertainty bound. Never overwrite the previous comparisons.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

import probe_scannetpp_triangle_scan_support as support
from radio_moge.corpus_admission import checked_json, file_ref
from radio_moge.geometry_cache import publish_json


def identity(record):
    return (record["scene"], str(record["scan"]), *record["pixel_yx"], record["target"]["face_id"])


def robust(sensitivity):
    gap = sensitivity["minimum_measured_patch_minus_maximum_tested_triangle_m"]
    return (
        gap is not None
        and gap > 0.05
        and all(
            x["comparison"]["diagnostic"] == "measurement_beyond_triangle"
            for x in sensitivity["samples"]
        )
    )


def audit(source_path, output, previous_sensitivity=None):
    source_ref = file_ref(source_path)
    source = checked_json(source_ref)
    if (
        source["stage"] != "bounded_triangle_scanner_support_complete"
        or source["training_admission"] is not False
        or source["optimizer_updates"] != 0
    ):
        raise ValueError("Completed bounded scanner comparisons required")
    previous_ref, previous = None, {}
    if previous_sensitivity:
        previous_ref = file_ref(previous_sensitivity)
        old = checked_json(previous_ref)
        if old["source"] != source_ref:
            raise ValueError("Previous sensitivity belongs to different source evidence")
        for record in old["records"]:
            key = identity(record)
            if key in previous:
                raise ValueError("Repeated previous comparison identity")
            previous[key] = record
    records, seen, pose_refs = [], {}, {}
    target_shifts, saved_target_selections = [], 0
    nominal_deltas, sensitivity_deltas = [], []
    for scan_ref in source["scans"]:
        scan = checked_json(scan_ref)
        pose_ref = scan["scanner_poses"]
        pose_refs[pose_ref["path"]] = pose_ref
        pose = np.asarray(checked_json(pose_ref)[int(scan["scan"])], dtype=np.float64)
        support.scanner_pose_diagnostics(pose)
        for saved in scan["measurements"]:
            target, old_comparison = saved["target"], saved["comparison"]
            local_ray = support.directions(
                saved["stored_azimuth_millirad"], saved["stored_polar_millirad"]
            )
            measurement = old_comparison["measurement"]
            current = support.compare_triangle(
                target["vertices_world"], pose, local_ray, measurement
            )
            sensitivity = support.angle_sensitivity(
                target["vertices_world"],
                pose,
                saved["stored_azimuth_millirad"],
                saved["stored_polar_millirad"],
                measurement,
            )
            local = support.scanner_local_target(pose, target["point_world"])
            old_local = (np.asarray(target["point_world"]) - pose[:3, 3]) @ pose[:3, :3]
            chord = np.linalg.norm(local_ray - local / np.linalg.norm(local))
            current_angle = float(2 * np.arcsin(min(1, chord / 2)))
            record = dict(
                scene=scan["scene"],
                scan=scan["scan"],
                pixel_yx=saved["pixel_yx"],
                source=scan_ref,
                target=target,
                original_comparison=old_comparison,
                comparison=current,
                sensitivity=sensitivity,
                all_nine_tested_angles_retain_range_disagreement=robust(sensitivity),
                original_angular_error_rad=saved["angular_error_rad"],
                corrected_angular_error_for_same_saved_pixel_rad=current_angle,
                inverse_vs_transpose_target_shift_m=float(np.linalg.norm(local - old_local)),
                nominal_diagnostic_changed=old_comparison["diagnostic"] != current["diagnostic"],
                nominal_hit_changed=old_comparison["intersects_exact_triangle"]
                != current["intersects_exact_triangle"],
                equivalent_target_selections=[],
            )
            saved_target_selections += 1
            target_shifts.append(record["inverse_vs_transpose_target_shift_m"])
            key = identity(record)
            if key in seen:
                existing = seen[key]
                if (
                    existing["source"] != scan_ref
                    or existing["comparison"] != current
                    or existing["sensitivity"] != sensitivity
                    or existing["original_comparison"] != old_comparison
                    or existing["target"]["vertices_world"] != target["vertices_world"]
                ):
                    raise ValueError("Conflicting measurements for repeated scanner ray and face")
                # Different DSLR targets can select the same scanner pixel/face.
                # Retain their provenance, but count the measured ray only once.
                existing["equivalent_target_selections"].append(
                    dict(
                        target=target,
                        corrected_angular_error_for_same_saved_pixel_rad=current_angle,
                        original_angular_error_rad=saved["angular_error_rad"],
                        inverse_vs_transpose_target_shift_m=record[
                            "inverse_vs_transpose_target_shift_m"
                        ],
                    )
                )
                continue
            seen[key] = record
            if current["intersects_exact_triangle"] and old_comparison["intersects_exact_triangle"]:
                delta = current["triangle_range_m"] - old_comparison["triangle_range_m"]
                nominal_deltas.append(abs(delta))
                record["nominal_local_radial_minus_previous_range_m"] = delta
            if previous_ref:
                old = previous.pop(key)
                if old["source"] != scan_ref or old["target"] != target:
                    raise ValueError("Previous sensitivity target/source identity differs")
                record["previous_robust_nine_angle"] = old[
                    "all_nine_tested_angles_retain_range_disagreement"
                ]
                record["robust_nine_angle_changed"] = record[
                    "previous_robust_nine_angle"
                ] != robust(sensitivity)
                old_samples = {
                    (s["azimuth_offset_millirad"], s["polar_offset_millirad"]): s["comparison"]
                    for s in old["sensitivity"]["samples"]
                }
                if len(old_samples) != 9:
                    raise ValueError("Expected all nine unique previous angles")
                changes = []
                for sample in sensitivity["samples"]:
                    offset = sample["azimuth_offset_millirad"], sample["polar_offset_millirad"]
                    before, after = old_samples[offset], sample["comparison"]
                    if before["diagnostic"] != after["diagnostic"]:
                        changes.append(
                            dict(
                                offset=list(offset),
                                before=before["diagnostic"],
                                after=after["diagnostic"],
                            )
                        )
                    if before["intersects_exact_triangle"] and after["intersects_exact_triangle"]:
                        sensitivity_deltas.append(
                            abs(after["triangle_range_m"] - before["triangle_range_m"])
                        )
                record["angle_diagnostic_changes"] = changes
            records.append(record)
    if previous:
        raise ValueError("Previous sensitivity has extra observations")
    if not records:
        raise ValueError("No saved comparisons")
    robust_records = [r for r in records if r["all_nine_tested_angles_retain_range_disagreement"]]
    gaps = [
        r["sensitivity"]["minimum_measured_patch_minus_maximum_tested_triangle_m"]
        for r in robust_records
    ]
    report = dict(
        stage="saved_scanner_transform_and_angle_audit_complete",
        source=source_ref,
        previous_sensitivity=previous_ref,
        records=records,
        unique_observations=len(records),
        saved_target_selections=saved_target_selections,
        duplicate_target_selections=saved_target_selections - len(records),
        scanner_poses=list(pose_refs.values()),
        nominal_diagnostic_counts=dict(Counter(r["comparison"]["diagnostic"] for r in records)),
        nominal_diagnostic_changes=sum(r["nominal_diagnostic_changed"] for r in records),
        nominal_hit_changes=sum(r["nominal_hit_changed"] for r in records),
        robust_nine_angle_observations=len(robust_records),
        robust_distinct_triangles=len(
            {(r["scene"], r["target"]["face_id"]) for r in robust_records}
        ),
        robust_gap_min_m=min(gaps) if gaps else None,
        robust_gap_max_m=max(gaps) if gaps else None,
        nominal_compared_intersections=len(nominal_deltas),
        tested_angle_compared_intersections=len(sensitivity_deltas) if previous_ref else None,
        nominal_range_change_max_abs_m=max(nominal_deltas, default=None),
        tested_angle_range_change_max_abs_m=max(sensitivity_deltas, default=None)
        if previous_ref
        else None,
        robust_nine_angle_changes=sum(r["robust_nine_angle_changed"] for r in records)
        if previous_ref
        else None,
        angle_diagnostic_changes=sum(len(r["angle_diagnostic_changes"]) for r in records)
        if previous_ref
        else None,
        inverse_vs_transpose_target_shift_max_m=max(target_shifts),
        same_saved_pixels_used=True,
        corrected_full_grid_pixel_selection_verified=False,
        continuous_angular_bound_proven=False,
        scanner_pose_uncertainty_tested=False,
        raw_images_reread=False,
        mesh_rerendered=False,
        geometry_or_camera_edited=False,
        source_quality_accepted=False,
        independent_physical_calibration=False,
        training_admission=False,
        optimizer_updates=0,
        gpu_used=False,
        code=file_ref(Path(__file__)),
        comparison_code=file_ref(Path(support.__file__)),
    )
    output.mkdir(exist_ok=False)
    publish_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--previous-sensitivity", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.source.resolve(), args.output.resolve(), args.previous_sensitivity)
    print({k: v for k, v in result.items() if k not in ("records", "scanner_poses")}, flush=True)
