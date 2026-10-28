import importlib.util
import json
from pathlib import Path

import pytest


path = Path(__file__).parents[1] / "scripts/audit_architecture_screen.py"
spec = importlib.util.spec_from_file_location("screen_audit", path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_source_macro_does_not_weight_larger_sources_more():
    rows = [
        dict(dataset=source, **dict.fromkeys(audit.METRICS, value))
        for source, value in [("a", 0.9), ("a", 0.9), ("b", 0.3)]
    ]
    result = audit.aggregate(rows)
    assert result["spatial/cosine"] == pytest.approx(0.7)
    assert result["macro/spatial/cosine"] == pytest.approx(0.6)


def test_valid_checksums_cannot_hide_wrong_image_identity(tmp_path):
    records = [dict(image="expected.png", group="scene", dataset="a", screen_id="a1")]
    rows = [dict(records[0], image="wrong.png", cache_index=0, **dict.fromkeys(audit.METRICS, 0.5))]
    (tmp_path / "per_image.json").write_text(json.dumps(rows))
    metrics = dict(step=100, nfe=1, images=1, **audit.aggregate(rows))
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    (tmp_path / "integrity.json").write_text(
        json.dumps(dict(per_image_sha256=audit.digest(tmp_path / "per_image.json"), image_count=1))
    )
    with pytest.raises(AssertionError):
        audit.audit_point(tmp_path, records, [0], {"a": 1}, 100, 1, metrics)


def test_partial_log_line_is_not_a_completed_evaluation(tmp_path):
    p = tmp_path / "validation.jsonl"
    p.write_text('{"step": 0}\n{"step": 100')
    assert audit.lines(p) == [{"step": 0}]
