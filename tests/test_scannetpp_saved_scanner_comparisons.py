import json

import numpy as np
import pytest

from audit_scannetpp_saved_scanner_comparisons import audit
from radio_moge.corpus_admission import file_ref
from probe_scannetpp_triangle_scan_support import (
    angle_sensitivity,
    compare_triangle,
    directions,
    range_neighborhood,
)
from probe_scannetpp_native_near_surfaces import analytic_hits


def fixture(tmp_path):
    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return file_ref(path)

    pose = np.eye(4)
    pose[0, 0] = 1.0002
    pose_ref = save("poses.json", [pose.tolist()])
    vertices = (np.array([[2, -1, -1], [2, 1, -1], [2, 0, 1]], float) @ pose[:3, :3].T).tolist()
    measurement = range_neighborhood(
        np.full((3, 3), 4000, np.uint16), np.full((3, 3), 255, np.uint8), 1, 1
    )
    comparison = compare_triangle(vertices, pose, directions(0, 1571), measurement)
    # Emulate old world-normalized intersection parameter, preserve old source.
    comparison["triangle_range_m"] = comparison["triangle_world_distance_m"]
    target = dict(
        corpus_ordinal=7, face_id=3, vertices_world=vertices, point_world=comparison["point_world"]
    )
    scan_ref = save(
        "scan.json",
        dict(
            scene="fixture",
            scan="0",
            scanner_poses=pose_ref,
            inputs={"range": {"path": "/not-readable/raw-range.png", "sha256": "a" * 64}},
            measurements=[
                dict(
                    target=target,
                    comparison=comparison,
                    pixel_yx=[1, 1],
                    stored_azimuth_millirad=0,
                    stored_polar_millirad=1571,
                    angular_error_rad=0,
                )
            ],
        ),
    )
    source_ref = save(
        "source.json",
        dict(
            stage="bounded_triangle_scanner_support_complete",
            training_admission=False,
            optimizer_updates=0,
            scans=[scan_ref],
        ),
    )
    return tmp_path / "source.json", source_ref


def test_saved_audit_uses_only_saved_metadata_and_reports_numeric_effect(tmp_path):
    source_path, _ = fixture(tmp_path)
    original = {p: p.read_bytes() for p in tmp_path.iterdir()}
    report = audit(source_path, tmp_path / "out")
    assert report["unique_observations"] == report["robust_nine_angle_observations"] == 1
    assert 0.0003 < report["nominal_range_change_max_abs_m"] < 0.0005
    assert report["nominal_diagnostic_changes"] == 0
    assert report["robust_gap_min_m"] > 1.99
    assert report["raw_images_reread"] is False
    assert report["corrected_full_grid_pixel_selection_verified"] is False
    for path, data in original.items():
        assert path.read_bytes() == data


def test_saved_sensitivity_cannot_be_bound_to_other_source(tmp_path):
    source_path, source_ref = fixture(tmp_path)
    source_ref["sha256"] = "f" * 64
    previous = tmp_path / "wrong.json"
    previous.write_text(json.dumps(dict(source=source_ref, records=[])))
    with pytest.raises(ValueError, match="different source"):
        audit(source_path, tmp_path / "out", previous)
    assert not (tmp_path / "out").exists()


def test_valid_previous_nine_angles_report_changed_decisions_at_threshold(tmp_path):
    source_path, _ = fixture(tmp_path)
    source = json.loads(source_path.read_text())
    scan_path = tmp_path / "scan.json"
    scan = json.loads(scan_path.read_text())
    saved = scan["measurements"][0]
    pose = np.array(json.loads((tmp_path / "poses.json").read_text())[0])
    vertices = saved["target"]["vertices_world"]
    measurement = dict(stable_supported=True, range_m=2.0502, ranges_m=[[2.0502] * 3] * 3)
    sensitivity = angle_sensitivity(vertices, pose, 0, 1571, measurement)
    # Reproduce historical world-normalized parameterization independently.
    for sample in sensitivity["samples"]:
        d = pose[:3, :3] @ directions(
            sample["azimuth_offset_millirad"], 1571 + sample["polar_offset_millirad"]
        )
        d /= np.linalg.norm(d)
        hit = analytic_hits(vertices, np.array([[0, 1, 2]]), pose[:3, 3], d)["first_hits"][0]
        c = sample["comparison"]
        c["triangle_range_m"] = hit["t"]
        c["measured_minus_triangle_range_m"] = measurement["range_m"] - hit["t"]
        assert 0 < c["measured_minus_triangle_range_m"] < 0.05
        c["diagnostic"] = "measurement_near_triangle"
        if sample["azimuth_offset_millirad"] == sample["polar_offset_millirad"] == 0:
            saved["comparison"] = c
    scan_path.write_text(json.dumps(scan))
    source["scans"] = [file_ref(scan_path)]
    source_path.write_text(json.dumps(source))
    previous = tmp_path / "previous.json"
    previous.write_text(
        json.dumps(
            dict(
                source=file_ref(source_path),
                records=[
                    dict(
                        scene=scan["scene"],
                        scan=scan["scan"],
                        pixel_yx=saved["pixel_yx"],
                        target=saved["target"],
                        source=file_ref(scan_path),
                        sensitivity=sensitivity,
                        all_nine_tested_angles_retain_range_disagreement=False,
                    )
                ],
            )
        )
    )
    result = audit(source_path, tmp_path / "out", previous)
    assert result["nominal_diagnostic_changes"] == result["robust_nine_angle_changes"] == 1
    assert result["angle_diagnostic_changes"] == result["tested_angle_compared_intersections"] == 9
    assert 0.0003 < result["tested_angle_range_change_max_abs_m"] < 0.0005
    assert result["robust_nine_angle_observations"] == 1
