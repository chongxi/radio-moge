import json

import numpy as np
import pytest

from audit_scannetpp_saved_scanner_comparisons import audit
from radio_moge.corpus_admission import file_ref
from probe_scannetpp_triangle_scan_support import compare_triangle, directions, range_neighborhood


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
