import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


spec = importlib.util.spec_from_file_location(
    'plane_diagnostics', Path(__file__).parents[1] / 'scripts/diagnose_plane_objectives.py')
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


def test_fixed_teacher_axis_penalizes_tilt_even_when_student_is_perfectly_flat():
    y, x = np.mgrid[-1:1:20j, -1:1:20j]
    points = np.stack([x, y, 3 + .4*x], -1).reshape(-1, 3)
    a = diagnostics.variance_decomposition(points, np.array([0., 0., 1.]))
    assert a['own_plane_normalized_rms'] < 1e-7
    assert a['flatness_loss'] > .01
    assert a['orientation_fraction'] == pytest.approx(1.)
    b = diagnostics.variance_decomposition(points * 3, np.array([0., 0., 1.]))
    assert a['flatness_loss'] == pytest.approx(b['flatness_loss'])


def test_gradient_cosine_sign_and_total_conflict():
    result = diagnostics.gram_report({'a': torch.tensor([2., 0.]), 'b': torch.tensor([-1., 0.])})
    assert result['pairwise_cosine']['a:b'] == pytest.approx(-1.)
    assert result['cosine_to_total']['a'] == pytest.approx(1.)
    assert result['cosine_to_total']['b'] == pytest.approx(-1.)
