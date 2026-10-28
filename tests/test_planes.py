import numpy as np
import pytest
import torch

from radio_moge.decoder import camera_points
from radio_moge.planes import PlaneConfig, PlanePatch, detect_planes, plane_losses, planarity_metrics
from radio_moge.pointcloud import unproject
from radio_moge.planes import preservation_metrics


def surface(size=64):
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    return np.stack((x, y, 3 + .1 * x + .15 * y), -1)


def test_edge_checks_catch_flattening_but_ignore_uniform_scale_for_edges():
    t = np.full((32, 32), 2.0)
    t[:, 16:] = 4
    valid = np.ones_like(t, bool)
    exact = preservation_metrics(t, t, valid, [])
    scaled = preservation_metrics(t * 2, t, valid, [])
    collapsed = preservation_metrics(np.full_like(t, 3), t, valid, [])
    assert exact['depth_edges/log_gradient_mae'] == 0
    assert scaled['depth_edges/log_gradient_mae'] < 1e-12
    assert scaled['outside_plane/teacher_abs_rel'] == 1
    assert collapsed['depth_edges/log_gradient_mae'] > .5
    assert collapsed['depth_edges/retention_ratio'] == 0


def patch(points):
    mask = np.ones(points.shape[:2], bool)
    normal = np.array([-.1, -.15, 1.0])
    normal /= np.linalg.norm(normal)
    return PlanePatch(mask, normal, -3 * normal[2], 0, 1, 1)


def test_teacher_planes_separate_disconnected_surfaces_and_occluders():
    p = surface()
    p[:, 32:, 2] += 1
    p[24:40, 24:40, 2] = 1
    p[0, 0] = np.nan
    planes = detect_planes(p)
    assert len(planes) >= 2
    for plane in planes:
        assert not (plane.mask[:, :29].any() and plane.mask[:, 35:].any())
        assert not plane.mask[24:40, 24:40].any()
        assert plane.rms < 1e-8
    assert np.logical_or.reduce([x.mask for x in planes]).sum() > 64 * 64 * .4


def test_large_curved_surface_is_not_a_large_plane():
    p = surface(80)
    p[..., 2] += .6 * (p[..., 0] ** 2 + p[..., 1] ** 2)
    cfg = PlaneConfig(min_area_ratio=.15, normal_degrees=10)
    assert detect_planes(p, config=cfg) == []


def test_plane_fitting_is_deterministic_and_scale_equivariant():
    p = surface()
    a, b, c = detect_planes(p), detect_planes(p), detect_planes(p * 7)
    assert len(a) == len(b) == len(c) == 1
    np.testing.assert_array_equal(a[0].mask, b[0].mask)
    np.testing.assert_array_equal(a[0].mask, c[0].mask)
    np.testing.assert_allclose(a[0].normal, c[0].normal, atol=1e-8)
    assert c[0].offset == pytest.approx(a[0].offset * 7)
    assert not a[0].mask[:3].any()


def test_no_plane_for_invalid_or_thin_collinear_support():
    p = surface()
    assert detect_planes(np.full_like(p, np.nan)) == []
    valid = np.zeros(p.shape[:2], bool)
    valid[:, 32] = True
    assert detect_planes(p, valid) == []


def test_losses_detect_bending_and_gradients_reduce_it():
    points = surface(32)
    teacher = torch.tensor(points[None], dtype=torch.float32)
    amplitude = torch.tensor(.2, requires_grad=True)
    bend = torch.zeros_like(teacher)
    bend[..., 2] = teacher[..., 0].square()
    student = teacher + amplitude * bend
    losses = plane_losses(student, teacher, [[patch(points)]])
    assert losses['plane_flatness'] > 0
    assert losses['plane_normal'] > 0
    sum(losses.values()).backward()
    assert torch.isfinite(amplitude.grad) and amplitude.grad > 0
    perfect = plane_losses(teacher, teacher, [[patch(points)]])
    assert perfect['plane_flatness'] < 1e-12
    assert perfect['plane_anchor'] == 0


def test_flatness_is_scale_invariant_but_anchor_blocks_scale_and_collapse():
    p = surface(32)
    teacher = torch.tensor(p[None], dtype=torch.float32)
    student = teacher.clone()
    student[..., 2] += .2 * student[..., 0].square()
    a = plane_losses(student, teacher, [[patch(p)]])
    b = plane_losses(student * .1, teacher, [[patch(p)]])
    torch.testing.assert_close(a['plane_flatness'], b['plane_flatness'])
    torch.testing.assert_close(a['plane_normal'], b['plane_normal'])
    assert b['plane_anchor'] > a['plane_anchor']
    collapsed = plane_losses(torch.zeros_like(student), teacher, [[patch(p)]])
    assert collapsed['plane_anchor'] > 0


def test_teacher_mask_is_fixed_and_gradients_do_not_smooth_other_objects():
    p = surface(32)
    plane = patch(p)
    plane.mask[:, 16:] = False
    teacher = torch.tensor(p[None], dtype=torch.float32)
    student = (teacher + .2).requires_grad_()
    values = plane_losses(student, teacher, [[plane]])
    sum(values.values()).backward()
    assert student.grad[:, :, 16:].abs().max() == 0
    assert student.grad[:, :, :16].abs().max() > 0
    broken = student.detach().clone()
    broken[0, 10, 10] = torch.nan
    with pytest.raises(FloatingPointError):
        plane_losses(broken, teacher, [[plane]])


def test_empty_patches_are_finite_differentiable_zero():
    p = torch.randn(1, 12, 12, 3, requires_grad=True)
    values = plane_losses(p, p.detach(), [[]])
    sum(values.values()).backward()
    assert p.grad.abs().sum() == 0


def test_differentiable_camera_points_match_viewer_and_keep_depth_gradient():
    depth = torch.tensor([[[1., 2., 3.], [3., 4., 5.]]], requires_grad=True)
    k = torch.tensor([[[.8, 0, .4], [0, 1.2, .6], [0, 0, 1.]]])
    p = camera_points(depth, k)
    np.testing.assert_allclose(p.detach()[0].numpy(), unproject(depth.detach()[0].numpy(), k[0]),
                               atol=1e-6)
    p.square().mean().backward()
    assert torch.isfinite(depth.grad).all() and depth.grad.abs().min() > 0


def test_bad_detector_config_fails_before_running():
    with pytest.raises(ValueError):
        PlaneConfig(distance_ratio=float('nan'))
    with pytest.raises(ValueError):
        PlaneConfig(min_area_ratio=0)


def test_evaluation_flatness_is_scale_invariant_and_does_not_reward_missing_planes():
    p = surface()
    regions = detect_planes(p)
    p[..., 2] += .3 * p[..., 0] ** 2
    a, b = planarity_metrics(p, regions), planarity_metrics(p * 2, regions)
    assert a['normalized_rms'] > .01
    assert a['normalized_rms'] == pytest.approx(b['normalized_rms'])
    assert planarity_metrics(p, []) == {}
