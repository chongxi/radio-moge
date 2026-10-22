import importlib.util
from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location(
    "planarity_audit", Path(__file__).resolve().parents[1] / "scripts/audit_exported_planarity.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_tilted_plane_has_varying_depth_but_zero_residual():
    y, x = np.mgrid[:128, :128]
    xn, yn = (x + 0.5) / 128 - 0.5, (y + 0.5) / 128 - 0.5
    depth = 1 / (0.4 + 0.08 * xn - 0.05 * yn)
    xyz = audit.backproject(depth, [[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]])
    assert np.ptp(depth) > 0.5
    assert audit.plane_residual(xyz)["normalized_rms"] < 1e-12


def test_bowing_is_detected_and_metric_is_scale_invariant():
    y, x = np.mgrid[:128, :128]
    depth = 2 + 0.4 * ((x - 64) / 64) ** 2
    xyz = audit.backproject(depth, [[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]])
    a, b = audit.plane_residual(xyz), audit.plane_residual(xyz * 3)
    assert a["normalized_rms"] > 0.05
    np.testing.assert_allclose(a["normalized_rms"], b["normalized_rms"])
    np.testing.assert_allclose(a["rms_m"] * 3, b["rms_m"])
