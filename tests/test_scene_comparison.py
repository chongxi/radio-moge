import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "scene_comparison", Path(__file__).resolve().parents[1] / "scripts/compare_scene_metrics.py"
)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_cluster_bootstrap_keeps_correlated_frames_together():
    records = [dict(image=str(i), dataset="a", group="long") for i in range(100)]
    records.append(dict(image="100", dataset="a", group="short"))
    before = [dict(image=r["image"], error=1.0, postprocess_failed=0) for r in records]
    after = [
        dict(image=r["image"], error=0.0 if r["group"] == "long" else 2.0, postprocess_failed=0)
        for r in records
    ]
    report = comparison.paired_report(before, after, records, metrics=["error"], repeats=1000)
    metric = report["metrics"]["error"]
    assert metric["image_mean_difference"] == pytest.approx(-99 / 101)
    assert metric["macro_difference_ci95"] == [-1.0, 1.0]
    assert metric["sources"]["a"]["scenes_improved"] == 1


def test_missing_metric_is_not_silently_dropped_and_pairs_must_match():
    records = [dict(image="a", dataset="x", group="one")]
    before = [dict(image="a", error=1.0, postprocess_failed=0)]
    after = [dict(image="a", error=None, postprocess_failed=1)]
    report = comparison.paired_report(before, after, records, metrics=["error"], repeats=100)
    assert "macro_difference" not in report["metrics"]["error"]
    assert report["postprocess_failures"]["candidate"] == 1
    with pytest.raises(ValueError, match="same manifest"):
        comparison.paired_report(before, [], records)
