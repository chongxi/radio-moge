"""Measure flatness on teacher-selected image windows, without running a model."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from build_updated_twenty import planes, read_export
from radio_moge.checkpoint import atomic_json


def backproject(depth, intrinsics):
    h, w = depth.shape
    y, x = np.mgrid[:h, :w]
    k = np.asarray(intrinsics)
    return np.stack(
        (
            ((x + 0.5) / w - k[0, 2]) / k[0, 0] * depth,
            ((y + 0.5) / h - k[1, 2]) / k[1, 1] * depth,
            depth,
        ),
        axis=-1,
    )


def plane_residual(points):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 20 or not np.isfinite(points).all():
        raise ValueError("A plane fit needs at least20 finite points")
    center = points.mean(axis=0)
    centered = points - center
    eigenvalues, vectors = np.linalg.eigh(centered.T @ centered / len(points))
    normal = vectors[:, 0]
    distances = np.abs(centered @ normal)
    extent = float(np.sqrt(np.maximum(eigenvalues[1:], 0).sum()))
    if extent <= 1e-8:
        raise ValueError("Degenerate plane support")
    rms = float(np.sqrt(np.mean(distances**2)))
    return dict(
        rms_m=rms,
        normalized_rms=rms / extent,
        p95_m=float(np.quantile(distances, 0.95)),
        extent_m=extent,
        normal=normal.tolist(),
        offset=float(-center @ normal),
    )


def audit(baseline, candidate, output):
    _, left = read_export(baseline)
    _, right = read_export(candidate)
    assert list(left) == list(right)
    rows = []
    for sid, (b, gb) in right.items():
        a, ga = left[sid]
        da, db = planes(baseline, a), planes(candidate, b)
        np.testing.assert_allclose(da[2], db[2], rtol=1e-5, atol=1e-5, equal_nan=True)
        teacher_valid = np.asarray(Image.open(candidate / gb["teacher"]["mask"])) > 0
        valid = [
            np.asarray(Image.open(root / g["after"]["mask"])) > 0
            for root, g in [(baseline, ga), (candidate, gb)]
        ]
        xyz = [
            backproject(depth, camera["intrinsics"])
            for depth, camera in [
                (da[1], ga["after"]),
                (db[1], gb["after"]),
                (db[2], gb["teacher"]),
            ]
        ]
        h, w = db.shape[-2:]
        # Nonoverlapping windows within each scale; not semantic wall segmentation.
        for size in [128, 256]:
            for y in range(0, h - size + 1, size):
                for x in range(0, w - size + 1, size):
                    region = np.s_[y : y + size : 2, x : x + size : 2]
                    mask = teacher_valid[region] & np.isfinite(xyz[2][region]).all(axis=-1)
                    if mask.mean() < 0.98:
                        continue
                    teacher = plane_residual(xyz[2][region][mask])
                    if (
                        teacher["extent_m"] < 0.15
                        or teacher["normalized_rms"] > 0.005
                        or teacher["p95_m"] > 0.02
                    ):
                        continue
                    row = dict(id=sid, scene=b["scene"], window=[x, y, size, size], teacher=teacher)
                    for name, j in [("fusion", 0), ("attention", 1)]:
                        if (
                            not valid[j][region][mask].all()
                            or not np.isfinite(xyz[j][region][mask]).all()
                        ):
                            row[name] = {"status": "invalid_student_support"}
                        else:
                            row[name] = dict(
                                status="complete", **plane_residual(xyz[j][region][mask])
                            )
                    rows.append(row)
    by_scene = []
    for sid in right:
        selected = [r for r in rows if r["id"] == sid]
        complete = [
            r
            for r in selected
            if all(r[k].get("status") == "complete" for k in ["fusion", "attention"])
        ]
        if not selected:
            continue
        by_scene.append(
            dict(
                id=sid,
                selected=len(selected),
                complete=len(complete),
                median_normalized_rms={
                    k: float(np.median([r[k]["normalized_rms"] for r in complete]))
                    if complete
                    else None
                    for k in ["teacher", "fusion", "attention"]
                },
            )
        )
    report = dict(
        status="complete",
        windows=rows,
        by_scene=by_scene,
        inputs={"baseline": str(baseline), "candidate": str(candidate)},
        selection="Teacher only:128/256px windows;>=98% valid, extent>=0.15m, RMS/extent<=0.005 and p95 distance<=0.02m. Thresholds fixed before inspection.",
        metric="RMS distance to each prediction own best-fit plane, divided by its spatial extent. Own recovered intrinsics; unit-scale invariant.",
        limits="Exploratory fixed20 audit. Teacher-planar windows are not semantic wall labels or sensor ground truth. Windows/scales are dependent; no significance claim. Flatness alone does not establish correct depth or plane pose. Different training budgets.",
    )
    atomic_json(report, output)
    print(
        json.dumps({"selected_windows": len(rows), "scenes": len(by_scene), "by_scene": by_scene})
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["baseline", "candidate", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    audit(a.baseline, a.candidate, a.output)
