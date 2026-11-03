from copy import deepcopy

import numpy as np
import pytest

from review_teacher_preflight_parity import check_pair


def test_only_policy_identity_may_change_across_exact_teacher_arrays():
    row = dict(image='/rgb.png', sha256='a' * 64, scene='reviewed')
    old = (np.ones((8, 8), np.float32), np.ones((8, 8), bool), np.eye(3),
           dict(policy_sha256='old', source_needs_review=True))
    new = (*old[:3], dict(old[3], policy_sha256='new'))
    check_pair(row, row, old, new)
    with pytest.raises(ValueError, match='RGB'):
        check_pair(row, dict(row, scene='other'), old, new)
    with pytest.raises(ValueError, match='provenance'):
        check_pair(row, row, old, (*new[:3], dict(new[3], source_needs_review=False)))
    for index in range(3):
        changed = list(deepcopy(new))
        changed[index].flat[0] = 0
        with pytest.raises(ValueError, match='arrays differ'):
            check_pair(row, row, old, changed)
    changed = list(new)
    changed[0] = changed[0].astype(np.float64)
    with pytest.raises(ValueError, match='arrays differ'):
        check_pair(row, row, old, changed)
    changed = list(deepcopy(new))
    changed[2][0, 1] = -0.0
    with pytest.raises(ValueError, match='arrays differ'):
        check_pair(row, row, old, changed)
