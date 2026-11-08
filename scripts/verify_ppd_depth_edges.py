"""Compare saved edge masks to the pinned MoGe dependency, on CPU only."""

import argparse
import importlib.metadata
import inspect
import json
from pathlib import Path

import numpy as np
import utils3d_moge

from export_ppd_depth_edges import METHODS, THRESHOLDS, digest, publish
from radio_moge.depth_edges import depth_edge_keep_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.root / "report.json").read_text())
    revision = json.loads(importlib.metadata.distribution("utils3d-moge").read_text("direct_url.json"))
    assert revision["vcs_info"]["commit_id"] == "62f09d58509485564e24d5d9f6aac9ee9ebc0c37"
    rows = []
    for result in report["results"]:
        name = result["id"]
        raw_path = args.source / "results-01" / (name + ".npz")
        geo_path = args.source / "pointclouds-02" / (name + ".npz")
        masks_path = args.root / "masks" / (name + ".npz")
        assert digest(raw_path) == result["source_depth_sha256"]
        assert digest(geo_path) == result["source_geometry_sha256"]
        assert digest(masks_path) == result["masks_sha256"]
        with np.load(raw_path) as d:
            moge = d["moge_adapter"].copy()
        with np.load(geo_path) as d:
            ppd, base = d["ppd_adapter"].copy(), d["adapter_mask"].copy()
        with np.load(masks_path) as masks:
            for key, depth in zip(METHODS, [moge, ppd]):
                for index, threshold in enumerate(THRESHOLDS):
                    official = base & ~utils3d_moge.np.depth_map_edge(depth, rtol=threshold)
                    saved, local = masks[f"{key}_keep_{index}"], depth_edge_keep_mask(depth, base, threshold)
                    assert np.array_equal(official, saved) and np.array_equal(official, local), (name, key, threshold)
                    rows.append({"sample": name, "method": key, "threshold": threshold,
                                 "pixels_compared": int(depth.size), "different_pixels": 0})
    fixtures = []
    for name, depth in [
        ("constant_and_image_boundaries", np.ones((5, 7), np.float32)),
        ("sharp_depth_step", np.array([[1, 1, 1, 2, 2]] * 5, np.float32)),
        ("invalid_neighbors", np.array([[np.nan, 1, 1], [1, np.inf, 2], [1, 2, 0]], np.float32)),
    ]:
        valid = np.isfinite(depth) & (depth > 0)
        for threshold in THRESHOLDS:
            expected = valid & ~utils3d_moge.np.depth_map_edge(depth, rtol=threshold)
            assert np.array_equal(expected, depth_edge_keep_mask(depth, valid, threshold))
        fixtures.append(name)
    publish(args.root / "upstream-parity.json", {
        "passed": True, "cases": 32, "methods": 2, "thresholds": THRESHOLDS,
        "mask_comparisons": len(rows), "total_pixel_comparisons": sum(r["pixels_compared"] for r in rows),
        "different_pixels": 0, "fixtures": fixtures, "rows": rows,
        "utils3d_revision": revision["vcs_info"]["commit_id"],
        "upstream_helpers_sha256": digest(inspect.getfile(utils3d_moge.np.depth_map_edge)),
        "report_sha256": digest(args.root / "report.json"), "model_inference": False,
    })
    print(json.dumps({"passed": True, "mask_comparisons": len(rows), "different_pixels": 0}))


if __name__ == "__main__":
    main()
