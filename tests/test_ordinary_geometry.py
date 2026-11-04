"""Synthetic CPU checks; these do not certify a real-model geometry training run."""

import random

import numpy as np
from PIL import Image, ImageOps
import pytest
import torch

from radio_moge.data import TrainingImages, ordinary_crop_parameters, oriented_rgb
from radio_moge.geometry_sampling import select_observations
from radio_moge.geometry_sources import pinhole_rays
from radio_moge.ordinary_geometry import ORDINARY_CAMERA_CONTRACT, ordinary_view


def archived_rgb_pixels(row, seed, h, w, augment, compact):
    # Literal pre-extraction path, intentionally independent of the new helper.
    rng = random.Random(seed)
    image = oriented_rgb(row)
    iw, ih = image.size
    cw, ch = min(iw, max(1, round(ih * w / h))), min(ih, max(1, round(iw * h / w)))
    left = rng.randint(0, iw - cw) if augment else (iw - cw) // 2
    top = rng.randint(0, ih - ch) if augment else (ih - ch) // 2
    image = image.crop((left, top, left + cw, top + ch))
    if augment and rng.random() < .5:
        image = ImageOps.mirror(image)
    image = image.resize((w, h), Image.Resampling.BILINEAR)
    array = np.array(image, dtype=np.uint8 if compact else np.float32, copy=True)
    pixels = torch.from_numpy(array).permute(2, 0, 1)
    return pixels if compact else pixels / 255


@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
@pytest.mark.parametrize('augment', [False, True])
@pytest.mark.parametrize('compact', [False, True])
def test_feature_pixels_bitwise_equal_archived_crop_path(tmp_path, rotation, augment, compact):
    path = tmp_path / 'rgb.png'
    Image.fromarray(np.random.default_rng(431).integers(0, 256, (61, 97, 3), dtype=np.uint8)).save(path)
    row = dict(image=str(path), image_rotation_cw=rotation)
    dataset = TrainingImages([row], augment=augment, compact_transport=compact)
    for h, w in ((400, 640), (480, 640), (640, 480)):
        for seed in (0, 1, 19, 2**63 - 1):
            expected = archived_rgb_pixels(row, seed, h, w, augment, compact)
            actual = dataset[(0, seed, h, w)]
            assert actual.dtype == expected.dtype
            assert torch.equal(actual, expected)


def test_descriptor_retains_zero_range_rng_draws_and_centered_nonaugmentation():
    shape = (48, 64)  # Both randint ranges are zero; both still consume RNG.
    for seed in range(32):
        rng = random.Random(seed)
        rng.randint(0, 0)
        rng.randint(0, 0)
        expected_flip = rng.random() < .5
        assert ordinary_crop_parameters(shape, shape, seed) == ((0, 0, 64, 48), expected_flip)
    for seed in (0, 99):
        assert ordinary_crop_parameters((49, 65), (40, 64), seed, False) == ((0, 4, 65, 45), False)


@pytest.mark.parametrize('flip', [False, True])
@pytest.mark.parametrize('output', [(400, 640), (480, 640), (640, 480)])
def test_pil_boundary_coordinates_match_normalized_camera_centers(flip, output):
    shape, box = (47, 71), (5, 3, 60, 44)
    k = np.array([[.8, .023, -.15], [0., 1.2, 1.1], [0., 0., 1.]])
    original = k.copy()
    view = ordinary_view(k, shape, output, box, flip)
    assert view['camera_contract'] == ORDINARY_CAMERA_CONTRACT
    np.testing.assert_array_equal(k, original)
    oh, ow = output
    xy = np.array([[0, 0], [ow - 1, oh - 1], [ow // 3, oh // 3]], np.float64)
    uv = (xy + .5) / [ow, oh]
    source = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(view['rgb_to_view_uv']).T
    expected = np.column_stack(((5 + 55 * (1 - uv[:, 0] if flip else uv[:, 0])) / 71,
                                (3 + 41 * uv[:, 1]) / 47))
    np.testing.assert_allclose(source[:, :2] / source[:, 2:], expected, atol=2e-16)
    np.testing.assert_allclose(view['intrinsics'] @ view['camera_transform'] @ np.linalg.inv(k),
                               view['rgb_to_view_uv'], atol=5e-16)
    np.testing.assert_allclose(view['camera_transform'].T @ view['camera_transform'], np.eye(3))
    assert np.linalg.det(view['camera_transform']) == (-1 if flip else 1)
    assert view['intrinsics'][0, 0] > 0 and view['intrinsics'][1, 1] > 0
    assert view['intrinsics'][0, 1] == pytest.approx((-.023 if flip else .023) * 71 / 55)
    # Actual principal points remain off center, even outside the visible image.
    assert not np.allclose(view['intrinsics'][:2, 2], [.5, .5])


def test_mirror_twice_restores_skew_offcenter_camera_and_identity_crop():
    k = np.array([[.9, .037, .34], [0., 1.15, 1.2], [0., 0., 1.]])
    unchanged = ordinary_view(k, (48, 64), (48, 64), (0, 0, 64, 48))
    np.testing.assert_array_equal(unchanged['intrinsics'], k)
    np.testing.assert_array_equal(unchanged['rgb_to_view_uv'], np.eye(3))
    a = ordinary_view(k, (48, 64), (48, 64), (0, 0, 64, 48), True)
    b = ordinary_view(a['intrinsics'], (48, 64), (48, 64), (0, 0, 64, 48), True)
    np.testing.assert_allclose(b['intrinsics'], k, atol=2e-16)
    assert a['intrinsics'][0, 2] == pytest.approx(1 - k[0, 2])


def test_integer_crop_rounding_preserves_actual_anisotropic_resize():
    shape, output = (49, 65), (40, 64)
    box, flip = ordinary_crop_parameters(shape, output, 0, False)
    # Original square pixels: equal pixel focal lengths, unequal normalized ones.
    k = np.array([[60 / 65, 0, .5], [0, 60 / 49, .5], [0, 0, 1.]])
    view = ordinary_view(k, shape, output, box, flip)
    fx, fy = view['intrinsics'][0, 0] * 64, view['intrinsics'][1, 1] * 40
    assert fx == pytest.approx(60 * 64 / 65)
    assert fy == pytest.approx(60 * 40 / 41)
    assert abs(fx - fy) > .5  # Do not silently force a square-pixel camera.


@pytest.mark.parametrize('flip', [False, True])
def test_multigrid_native_points_are_exact_and_holes_never_supervised(flip):
    kd = np.array([[.84, .01, .511], [0., 1.1, .489], [0., 0., 1.]])
    kr = np.array([[.83, .015, .5005], [0., 1.13, .5005], [0., 0., 1.]])
    dh, dw = 48, 64
    depth = np.arange(dh * dw, dtype=np.float32).reshape(dh, dw) / 1000 + 1
    mask = np.ones((dh, dw), bool)
    mask[13:27, 19:39] = False
    view = ordinary_view(kr, (481, 641), (100, 160), (3, 37, 640, 435), flip)
    target = select_observations(depth, mask, kd, view)
    valid, ids = target['valid'], target['native_index'][target['valid']]
    assert 500 < len(ids) <= mask.sum()
    assert len(ids) == len(np.unique(ids)) and mask.ravel()[ids].all()
    native = (pinhole_rays(kd, dh, dw) * depth[..., None]).reshape(-1, 3)[ids]
    expected = native @ view['camera_transform'].T
    np.testing.assert_allclose(target['points'][valid], expected, rtol=1e-7, atol=2e-7)
    np.testing.assert_array_equal(target['points'][valid, 2], depth.ravel()[ids])
    np.testing.assert_allclose(np.linalg.norm(target['points'][valid], axis=1),
                               np.linalg.norm(native, axis=1), rtol=2e-7)
    # An independent native-grid -> RGB-grid -> crop projection.
    native_uv = np.column_stack(((ids % dw + .5) / dw, (ids // dw + .5) / dh, np.ones(len(ids))))
    projection = native_uv @ (view['rgb_to_view_uv'] @ kr @ np.linalg.inv(kd)).T
    np.testing.assert_allclose(target['query_uv'][valid], projection[:, :2] / projection[:, 2:], atol=1e-7)
    assert np.all(target['query_uv'][valid] >= [.5 / 160, .5 / 100])
    assert np.all(target['query_uv'][valid] <= [1 - .5 / 160, 1 - .5 / 100])
    changed = depth.copy()
    changed[~mask] = np.nan
    other = select_observations(changed, mask, kd, view)
    for name in ('points', 'query_uv', 'valid', 'native_index'):
        np.testing.assert_array_equal(other[name], target[name])


@pytest.mark.parametrize('flip', [False, True])
def test_exact_output_center_boundaries_and_empty_support(flip):
    k = np.array([[1., 0., .5], [0., 1., .5], [0., 0., 1.]])
    view = ordinary_view(k, (8, 8), (8, 8), (0, 0, 8, 8), flip)
    depth, mask = np.ones((8, 8), np.float32), np.ones((8, 8), bool)
    target = select_observations(depth, mask, k, view)
    assert target['visible_native_count'] == 64
    assert target['valid'].sum() == 64
    np.testing.assert_allclose(target['query_uv'].min(axis=(0, 1)), [.5 / 8, .5 / 8])
    np.testing.assert_allclose(target['query_uv'].max(axis=(0, 1)), [1 - .5 / 8, 1 - .5 / 8])
    empty = select_observations(np.full((8, 8), np.inf), np.zeros((8, 8), bool), k, view)
    assert not empty['valid'].any() and empty['visible_native_count'] == 0
    assert np.all(empty['native_index'] == -1)


@pytest.mark.parametrize('box', [(-1, 0, 8, 8), (0, 0, 9, 8), (4, 0, 4, 8), (0, 0, 8, 0),
                               (0., 0, 8, 8), (False, 0, 8, 8)])
def test_bad_crop_descriptor_fails(box):
    with pytest.raises(ValueError):
        ordinary_view(np.eye(3), (8, 8), (8, 8), box)


def test_bad_shapes_and_flip_fail():
    for shape, output, flip in (((0, 8), (8, 8), False), ((8, 8), (8., 8), False),
                                ((8, 8), (8, 8), 1)):
        with pytest.raises(ValueError):
            ordinary_view(np.eye(3), shape, output, (0, 0, 8, 8), flip)
