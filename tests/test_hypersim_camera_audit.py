import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from audit_hypersim_cameras import pose_record  # noqa: E402


def test_native_shear_is_quarantined_not_repaired():
    # Released ai_001_006/cam_01/frame0, not a rounding-error rotation.
    matrix = np.array([[.2842122316, .08111450076, -.9572415352],
                       [-.8916642666, .3518683314, -.2205416113],
                       [.3507648706, .9244167805, .2263512611]])
    result = pose_record(matrix)
    assert not result['usable'] and result['orthogonality_residual'] > .05
    assert result['source_matrix'] == matrix.tolist()


def test_uniform_scale_passes_but_reflection_and_nonfinite_do_not():
    assert pose_record(np.eye(3) * .25)['usable']
    assert not pose_record(np.diag([1, 1, -1]))['usable']
    assert not pose_record(np.full((3, 3), np.nan))['usable']
