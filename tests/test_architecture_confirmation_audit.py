import importlib.util
from pathlib import Path
import sys

import pytest

scripts = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location(
    "confirmation_audit", scripts / "audit_architecture_confirmation.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_gt_mean_uses_only_images_with_ground_truth():
    rows = [
        {"dataset": "nyu", "gt_student/abs_rel": 0.1, "teacher_depth/abs_rel": 0.2},
        {"dataset": "nyu", "gt_student/abs_rel": 0.3, "teacher_depth/abs_rel": 0.2},
        {"dataset": "other", "teacher_depth/abs_rel": 0.8},
    ]
    result = audit.geometry_means(rows)
    assert result["images"] == 3
    assert result["gt_student/abs_rel/count"] == 2
    assert result["gt_student/abs_rel"] == pytest.approx(0.2)
    assert result["teacher_depth/abs_rel"] == pytest.approx(0.4)
    assert result["macro/teacher_depth/abs_rel"] == pytest.approx(0.5)


def test_teacher_macro_is_absent_if_one_source_has_no_metric():
    result = audit.geometry_means(
        [
            {"dataset": "a", "teacher_depth/abs_rel": 0.2},
            {"dataset": "b", "teacher_failed": 1},
        ]
    )
    assert "macro/teacher_depth/abs_rel" not in result
