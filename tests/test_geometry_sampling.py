import hashlib

import numpy as np
from PIL import Image
import pytest

from radio_moge.geometry_cache import GeometryPackWriter
from radio_moge.geometry_sampling import (centered_view, load_candidate,
    select_observations, source_supervision, warp_rgb)
from radio_moge.geometry_sources import pinhole_rays


def test_offcenter_camera_rectifies_exact_rays_without_changing_metric_length():
    k = np.array([[.7, .01, .45], [0, 1.1, 1.2], [0, 0, 1.]])
    view = centered_view(k, (72, 96), (40, 64))
    np.testing.assert_allclose(view['intrinsics'][:2, 2], [.5, .5])
    r = view['camera_transform']
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1)
    depth = np.full((72, 96), 3., np.float32)
    target = select_observations(depth, np.ones(depth.shape, bool), k, view)
    valid = target['valid']
    assert valid.sum() > 300
    exact = (pinhole_rays(k, 72, 96) * 3).reshape(-1, 3)[target['native_index'][valid]]
    np.testing.assert_allclose(target['points'][valid], exact @ r.T, atol=4e-7)
    np.testing.assert_allclose(np.linalg.norm(target['points'][valid].astype(np.float64), axis=1),
                               np.linalg.norm(exact, axis=1), rtol=1e-7)
    projected = target['points'][valid] @ view['intrinsics'].T
    np.testing.assert_allclose(projected[:, :2] / projected[:, 2:],
                               target['query_uv'][valid], atol=1e-7)
    assert not np.allclose(target['points'][valid, 2], 3.)  # Correctly rotated Z.


def test_multigrid_does_not_upsample_duplicate_or_fill_native_depth():
    kd = np.array([[.8, 0, .51], [0, 1.05, .49], [0, 0, 1.]])
    kr = np.array([[.8, 0, .5005], [0, 1.05, .5005], [0, 0, 1.]])
    depth = np.full((48, 64), 2.5, np.float32)
    mask = np.ones(depth.shape, bool)
    mask[10:20, 20:30] = False
    view = centered_view(kr, (480, 640), (100, 160))
    a = select_observations(depth, mask, kd, view)
    changed = depth.copy()
    changed[~mask] = np.nan
    b = select_observations(changed, mask, kd, view)
    for key in ('points', 'query_uv', 'valid', 'native_index'):
        np.testing.assert_array_equal(a[key], b[key])
    assert a['valid'].shape[0] <= 48 and a['valid'].shape[1] <= 64
    ids = a['native_index'][a['valid']]
    assert len(ids) == len(np.unique(ids)) <= mask.sum()
    assert mask.ravel()[ids].all()
    native = (pinhole_rays(kd, 48, 64) * 2.5).reshape(-1, 3)[ids]
    np.testing.assert_allclose(a['points'][a['valid']],
                               native @ view['camera_transform'].T, atol=2e-7)


def test_rgb_and_geometry_share_crop_rotation_flip_homography():
    h, w = 60, 80
    k = np.array([[.8, 0, .5], [0, 1., .7], [0, 0, 1.]])
    u, v = np.meshgrid((np.arange(w) + .5) / w, (np.arange(h) + .5) / h)
    rgb = np.stack((u, v, u * 0 + .5), -1)
    rgb = np.rint(255 * rgb).astype(np.uint8)
    view = centered_view(k, (h, w), (40, 64), center=(.55, .45), zoom=1.1, flip=True)
    assert np.linalg.det(view['camera_transform']) == pytest.approx(-1)
    output = warp_rgb(rgb, view)
    oy, ox = 20, 32
    source = np.linalg.solve(view['rgb_to_view_uv'], [(ox + .5) / 64, (oy + .5) / 40, 1])
    np.testing.assert_allclose(output[oy, ox, :2] / 255, source[:2] / source[2], atol=1.5 / 255)
    target = select_observations(np.ones((h, w), np.float32), np.ones((h, w), bool), k, view)
    mask = target['valid']
    ids = target['native_index'][mask]
    src_uv = np.stack((u.ravel()[ids], v.ravel()[ids], np.ones(len(ids))), 1)
    projected = src_uv @ view['rgb_to_view_uv'].T
    np.testing.assert_allclose(target['query_uv'][mask], projected[:, :2] / projected[:, 2:], atol=1e-7)


def test_support_is_fixed_and_empty_stays_empty():
    k = np.array([[1., 0, .5], [0, 1., .5], [0, 0, 1.]])
    view = centered_view(k, (8, 8), (8, 8))
    depth = np.full((8, 8), np.nan, np.float32)
    mask = np.zeros((8, 8), bool)
    assert not select_observations(depth, mask, k, view)['valid'].any()
    mask[3, 3] = True
    with pytest.raises(ValueError, match='supported native'):
        select_observations(depth, mask, k, view)


def test_zoom_does_not_inflate_loss_grid_beyond_native_angular_density():
    k = np.array([[.8, 0, .5], [0, 1.05, .5], [0, 0, 1.]])
    depth = np.full((192, 256), 2., np.float32)
    valid = np.ones(depth.shape, bool)
    a = select_observations(depth, valid, k, centered_view(k, (480, 640), (400, 640)))
    b = select_observations(depth, valid, k, centered_view(k, (480, 640), (400, 640), zoom=2))
    assert b['valid'].shape[0] < a['valid'].shape[0] * .6
    assert b['valid'].shape[1] < a['valid'].shape[1] * .6
    assert b['valid'].mean() > .99
    assert b['valid'].sum() <= valid.sum()


def test_provenance_scale_and_unresolved_review_are_fail_closed():
    pseudo = dict(label_kind='teacher_pseudo', camera_kind='teacher_estimated',
                  metric_calibrated=False, teacher_revision='pinned')
    assert source_supervision(pseudo) == 'relative_pseudo'
    with pytest.raises(ValueError, match='cannot claim'):
        source_supervision(dict(pseudo, metric_calibrated=True))
    with pytest.raises(ValueError, match='Unresolved'):
        source_supervision(dict(pseudo, source_needs_review=True))
    with pytest.raises(ValueError, match='calibrated source'):
        source_supervision(dict(pseudo, label_kind='measured_sensor'))


def packed_fixture(tmp_path, rotation=0, multigrid=True):
    path = tmp_path / 'rgb.png'
    Image.fromarray(np.full((24, 32, 3), 128, np.uint8)).save(path)
    rgb_shape = (32, 24) if rotation in (90, 270) else (24, 32)
    row = dict(image=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               dataset='arkitscenes', split='train', image_rotation_cw=rotation,
               height=rgb_shape[0], width=rgb_shape[1])
    kd = np.array([[1., 0, .51], [0, 1., .51], [0, 0, 1.]])
    kr = np.array([[1., 0, .5], [0, 1., .5], [0, 0, 1.]])
    shape = tuple(x // 2 for x in rgb_shape) if multigrid else rgb_shape
    writer = GeometryPackWriter(tmp_path / 'packs', dict(policy='test'))
    metadata = dict(image_sha256=row['sha256'], dataset=row['dataset'], source_id='test',
        label_kind='measured_sensor', camera_kind='calibrated_sensor', metric_calibrated=True,
        conversion='test', policy_sha256=writer.policy_hash,
        source_proofs=[dict(source=str(path), sha256=row['sha256'])], image_rotation_cw=rotation,
        rgb_shape=list(rgb_shape), rgb_intrinsics_normalized=kr.tolist(),
        depth_to_rgb_uv_transform=(kr @ np.linalg.inv(kd)).tolist())
    writer.add(row, np.full(shape, 2., np.float32), np.ones(shape, bool), kd, metadata)
    entry = writer.entries[0]
    writer.close()
    return tmp_path / 'packs/pack-000000.tar', entry


@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
def test_actual_packed_reader_checks_orientation_and_is_deterministic(tmp_path, rotation):
    pack, entry = packed_fixture(tmp_path, rotation)
    a = load_candidate(pack, entry, (20, 32), seed=19, augment=True)
    b = load_candidate(pack, entry, (20, 32), seed=19, augment=True)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1]['points'], b[1]['points'])
    assert a[1]['supervision'] == 'metric_native'
    assert a[1]['valid'].sum() > 0
    Image.fromarray(np.zeros((24, 32, 3), np.uint8)).save(entry['image']['image'])
    with pytest.raises(ValueError, match='bytes changed'):
        load_candidate(pack, entry)


def test_hash_bound_rgb_without_original_dimension_fields_is_not_resized(tmp_path):
    pack, entry = packed_fixture(tmp_path)
    a = load_candidate(pack, entry, (20, 32))
    entry['image'].pop('height')
    entry['image'].pop('width')
    b = load_candidate(pack, entry, (20, 32))
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1]['points'], b[1]['points'])
    entry['image']['width'] = 32  # Partial frozen dimension metadata is invalid.
    with pytest.raises(ValueError, match='dimensions differ'):
        load_candidate(pack, entry)
