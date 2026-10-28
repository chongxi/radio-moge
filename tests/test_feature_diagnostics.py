import pytest
import torch

from radio_moge.adapters import EncoderFeatures
from radio_moge.feature_diagnostics import (
    factorized_features, feature_metrics, replace_scale, rescale_summary, scene_difference,
)
from radio_moge.losses import FeatureStatistics


def test_cosine_does_not_hide_norm_error_in_diagnostics():
    teacher = EncoderFeatures(torch.tensor([[[[1.0]], [[2.0]]]]), torch.tensor([[3.0, 4.0]]))
    prediction = EncoderFeatures(teacher.spatial * 2, teacher.summary * 0.5)
    result = feature_metrics(prediction, teacher, FeatureStatistics(2, 2))
    assert result['spatial/cosine'] == pytest.approx(1)
    assert result['summary/cosine'] == pytest.approx(1)
    assert result['spatial/norm_ratio'] == pytest.approx(2)
    assert result['summary/norm_ratio'] == pytest.approx(0.5)
    assert result['spatial/relative_l2'] == pytest.approx(1)
    assert result['summary/relative_l2'] == pytest.approx(0.5)


def test_antialigned_summary_is_distinguished():
    teacher = EncoderFeatures(torch.ones(1, 2, 2, 2), torch.ones(1, 2))
    result = feature_metrics(EncoderFeatures(teacher.spatial, -teacher.summary),
                             teacher, FeatureStatistics(2, 2))
    assert result['spatial/cosine'] == pytest.approx(1)
    assert result['summary/cosine'] == pytest.approx(-1)
    assert result['summary/norm_ratio'] == pytest.approx(1)


def test_scale_replacement_preserves_geometry_and_inputs():
    student = {'points': torch.ones(1, 2, 2, 3), 'metric_scale': torch.tensor([2.0])}
    teacher = {'points': torch.zeros(1, 2, 2, 3), 'metric_scale': torch.tensor([3.0])}
    result = replace_scale(student, teacher)
    assert result['points'] is student['points']
    assert result['metric_scale'] is teacher['metric_scale']
    assert student['metric_scale'].item() == 2


def test_scene_bootstrap_uses_paired_difference_and_image_mean():
    rows = [{'group': g, 'baseline': 5, 'candidate': 3} for g in ['a', 'a', 'b']]
    result = scene_difference(rows, 'candidate', 'baseline', repeats=100)
    assert result['difference'] == -2
    assert result['scene_bootstrap_ci95'] == [-2, -2]
    assert result['scenes'] == 2
    assert result['images_improved'] == 3


def test_factorized_replacements_isolate_norm_and_direction():
    student = EncoderFeatures(torch.tensor([[[[3.0]], [[0.0]]]]), torch.tensor([[0.0, 4.0]]))
    teacher = EncoderFeatures(torch.tensor([[[[0.0]], [[2.0]]]]), torch.tensor([[5.0, 0.0]]))
    norm = factorized_features(student, teacher, 'spatial_norm')
    direction = factorized_features(student, teacher, 'spatial_direction')
    torch.testing.assert_close(norm.spatial, torch.tensor([[[[2.0]], [[0.0]]]]))
    torch.testing.assert_close(direction.spatial, torch.tensor([[[[0.0]], [[3.0]]]]))
    assert norm.summary is student.summary
    torch.testing.assert_close(factorized_features(student, teacher, 'summary_norm').summary,
                               torch.tensor([[0.0, 5.0]]))
    torch.testing.assert_close(factorized_features(student, teacher, 'summary_direction').summary,
                               torch.tensor([[4.0, 0.0]]))


def test_factorized_replacements_reject_zero_direction():
    student = EncoderFeatures(torch.zeros(1, 2, 1, 1), torch.ones(1, 2))
    teacher = EncoderFeatures(torch.ones(1, 2, 1, 1), torch.ones(1, 2))
    with pytest.raises(ValueError, match='zero vectors'):
        factorized_features(student, teacher, 'spatial_direction')


def test_summary_scaling_preserves_direction_spatial_and_input():
    student = EncoderFeatures(torch.ones(1, 2, 1, 1), torch.tensor([[3.0, -4.0]]))
    result = rescale_summary(student, 1.04)
    assert result.spatial is student.spatial
    torch.testing.assert_close(student.summary, torch.tensor([[3.0, -4.0]]))
    assert result.summary.norm().item() == pytest.approx(5.2)
    assert torch.nn.functional.cosine_similarity(result.summary, student.summary).item() == (
        pytest.approx(1))


@pytest.mark.parametrize('multiplier', [0, -1, float('nan'), float('inf')])
def test_summary_scaling_rejects_nonpositive_or_nonfinite(multiplier):
    student = EncoderFeatures(torch.ones(1, 2, 1, 1), torch.ones(1, 2))
    with pytest.raises(ValueError, match='positive and finite'):
        rescale_summary(student, multiplier)
