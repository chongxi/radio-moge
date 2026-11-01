import numpy as np
import pytest

from radio_moge.gt_geometry_metrics import metric_camera_points, physical_geometry_metrics


def sample():
    depth = np.full((32, 48), 2.)
    labels = np.zeros(depth.shape, dtype=np.uint8)
    labels[:, :32] = 1
    planes = np.zeros((6, 4))
    planes[0] = [0, 0, 1, -2]
    return dict(depth=depth, valid=np.ones(depth.shape, bool), plane_labels=labels,
                planes=planes, intrinsics=np.array([[.8, 0, .5], [0, 1.2, .5], [0, 0, 1.]]))


def evaluate(depth, target, raw=None):
    if raw is None:
        raw = metric_camera_points(depth, target['intrinsics'])
    return physical_geometry_metrics(depth, raw, target)


def test_exact_geometry_has_zero_physical_errors():
    target = sample()
    result = evaluate(target['depth'], target)
    assert result['all']['depth_rmse_m'] == 0
    assert result['all']['raw_xyz_rmse_m'] == 0
    assert result['projected_normals']['all']['mean_angle_deg'] < 1e-6
    assert result['planes'][0]['fixed_gt_plane_rmse_m'] == 0
    assert result['planes'][0]['self_fit_rmse_m'] < 1e-10


def test_wrong_scale_stays_flat_but_metric_errors_are_not_aligned_away():
    target = sample()
    result = evaluate(target['depth'] * 1.5, target)
    assert result['all']['depth_rmse_m'] == 1
    assert result['all']['abs_rel'] == .5
    assert result['all']['raw_xyz_rmse_m'] > 1
    assert result['planes'][0]['self_fit_rmse_m'] < 1e-10
    assert result['planes'][0]['fixed_gt_plane_rmse_m'] == 1
    assert result['planes'][0]['fit_offset_error_m'] == pytest.approx(1)


def test_bowed_wall_is_penalized_by_flatness_and_normals():
    target = sample()
    x = np.linspace(-1, 1, target['depth'].shape[1])
    depth = target['depth'] + .3 * x[None] ** 2
    result = evaluate(depth, target)
    assert result['planes'][0]['self_fit_rmse_m'] > .02
    assert result['projected_normals']['all']['mean_angle_deg'] > 2


def test_raw_xy_error_is_visible_even_when_depth_is_exact():
    target = sample()
    depth = target['depth']
    raw = metric_camera_points(depth, target['intrinsics'])
    raw[..., 0] += .1
    result = evaluate(depth, target, raw)
    assert result['all']['raw_xyz_rmse_m'] == pytest.approx(.1)
    assert result['all']['projected_xyz_rmse_m'] == 0


def test_nonplane_objects_contribute_to_whole_image_and_nonplane_metrics():
    target = sample()
    depth = target['depth'].copy()
    depth[:, 32:] += .5
    result = evaluate(depth, target)
    assert result['plane_region']['depth_rmse_m'] == 0
    assert result['nonplane_region']['depth_rmse_m'] == .5
    assert result['all']['depth_rmse_m'] > 0


def test_holes_are_excluded_but_nonfinite_valid_prediction_is_an_error():
    target = sample()
    target['valid'][10:15, 10:15] = False
    target['plane_labels'][~target['valid']] = 0
    depth = target['depth'].copy()
    depth[~target['valid']] = np.nan
    result = evaluate(depth, target)
    assert result['all']['depth_rmse_m'] == 0
    assert result['projected_normals']['all']['stencils'] < 30 * 46
    depth[0, 0] = np.nan
    with pytest.raises(FloatingPointError):
        evaluate(depth, target)


def test_gt_discontinuities_excluded_from_normal_stencils():
    target = sample()
    target['plane_labels'][:] = 0
    target['depth'][:, 24:] = 4
    result = evaluate(target['depth'], target)
    assert result['projected_normals']['all']['stencils'] == 30 * (46 - 2)
    assert result['projected_normals']['all']['mean_angle_deg'] < 1e-6


def test_collapsed_prediction_cannot_earn_perfect_normals_or_identifiable_plane():
    target = sample()
    result = evaluate(np.zeros_like(target['depth']), target)
    assert result['all']['nonpositive_depth_fraction'] == 1
    assert result['projected_normals']['all']['mean_angle_deg'] == 180
    assert not result['planes'][0]['fit_identifiable']
    assert result['planes'][0]['fit_normal_error_deg'] is None
