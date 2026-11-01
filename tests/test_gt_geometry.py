from copy import deepcopy
import hashlib
import json

import numpy as np
from PIL import Image
import pytest
import torch

from radio_moge.data import read_manifest
from radio_moge.decoder import camera_points
from radio_moge.gt_data import GroundTruthImages, validate_gt_records
from radio_moge.gt_geometry import GT_LOSS_KEYS, calibrated_geometry, ground_truth_losses
from radio_moge.scannetpp_panorama import PanoramaProjector, perspective_camera, scanner_rays
from radio_moge.train import train, validate_config
from test_training import config_for, tiny_system


def plane_target(batch=1, height=32, width=48):
    intrinsics = torch.tensor([[.8, 0, .5], [0, 1.2, .5], [0, 0, 1.]])[None].repeat(batch, 1, 1)
    depth = torch.full((batch, height, width), 3.)
    planes = torch.zeros(batch, 6, 4)
    planes[:, 0, 2], planes[:, 0, 3] = 1, -3
    return {'depth': depth, 'valid': torch.ones_like(depth, dtype=torch.bool),
            'intrinsics': intrinsics, 'planes': planes,
            'plane_labels': torch.ones_like(depth, dtype=torch.uint8)}


def raw_for(target, scale=1., shift=0.):
    points = camera_points(target['depth'], target['intrinsics']) / scale
    points[..., 2] -= shift
    return {'points': points.requires_grad_(), 'metric_scale': torch.full(
        (len(points),), scale, requires_grad=True), 'mask': torch.zeros_like(target['depth'])}


def test_known_camera_recovers_shift_and_metric_scale_without_gt_depth():
    target = plane_target()
    raw = raw_for(target, scale=2., shift=.7)
    output = calibrated_geometry(raw, target['intrinsics'], target['valid'])
    torch.testing.assert_close(output['depth'], target['depth'])
    torch.testing.assert_close(output['shift'], torch.tensor([.7]))
    losses = ground_truth_losses(raw, target)
    assert all(abs(losses[key].item()) < 1e-6 for key in GT_LOSS_KEYS)


def test_bent_wall_and_wrong_scale_are_penalized_with_gradients():
    target = plane_target()
    raw = raw_for(target)
    x = torch.linspace(-1, 1, 48)
    with torch.no_grad():
        raw['points'][..., 2] += .3 * x.square()
    losses = ground_truth_losses(raw, target)
    assert all(losses[key] > 0 for key in GT_LOSS_KEYS)
    sum(losses[key] for key in GT_LOSS_KEYS).backward()
    assert torch.isfinite(raw['points'].grad).all() and raw['points'].grad.abs().sum() > 0
    assert torch.isfinite(raw['metric_scale'].grad).all() and raw['metric_scale'].grad.abs().sum() > 0
    wrong = raw_for(target)
    wrong['metric_scale'] = torch.tensor([1.5], requires_grad=True)
    assert ground_truth_losses(wrong, target)['gt_depth'] > .4


def test_raw_xy_errors_cannot_hide_behind_known_camera_projection():
    target = plane_target()
    raw = raw_for(target)
    with torch.no_grad():
        raw['points'][..., 0] += .4  # Symmetric rays make its fitted z shift zero.
    losses = ground_truth_losses(raw, target)
    assert losses['gt_depth'] < 1e-6
    assert losses['gt_points'] > .01
    losses['gt_points'].backward()
    assert raw['points'].grad[..., 0].abs().sum() > 0


def test_invalid_gt_does_not_contaminate_backward_and_student_mask_cannot_hide_errors():
    target = plane_target()
    raw = raw_for(target)
    target['depth'][:, :8] = float('nan')
    target['valid'][:, :8] = False
    raw['metric_scale'] = torch.tensor([1.2], requires_grad=True)
    losses = ground_truth_losses(raw, target)
    raw['mask'].fill_(1)
    with_mask = ground_truth_losses(raw, target)
    for key in GT_LOSS_KEYS:
        torch.testing.assert_close(losses[key], with_mask[key], rtol=0, atol=0)
    sum(losses[key] for key in GT_LOSS_KEYS).backward()
    assert torch.isfinite(raw['points'].grad).all()
    target['valid'].fill_(False)
    with pytest.raises(ValueError, match='Every GT image'):
        ground_truth_losses(raw, target)


def test_gt_edges_and_holes_excluded_from_normal_gradient_stencils():
    target = plane_target()
    target['depth'][..., 24:] = 5
    target['plane_labels'].zero_()
    raw = raw_for(target)
    losses = ground_truth_losses(raw, target)
    assert losses['gt_gradient'].item() < 1e-7
    assert losses['gt_normal'].item() < 1e-6


def test_geometry_batch_uses_individual_gt_depth_scales():
    target = plane_target(batch=2)
    target['depth'][1] *= 10
    target['planes'][1, 0, 3] *= 10
    raw = raw_for(target)
    raw['metric_scale'] = torch.tensor([1.2, 1.2], requires_grad=True)
    batch = ground_truth_losses(raw, target)
    single = ground_truth_losses({k: v[:1] for k, v in raw.items()},
                                 {k: v[:1] for k, v in target.items()})
    for key in GT_LOSS_KEYS:
        torch.testing.assert_close(batch[key], single[key], atol=1e-6, rtol=1e-5)


def publish_pair(tmp_path, name='pair', height=32, width=48):
    target = plane_target(height=height, width=width)
    image_path, geometry_path = tmp_path / (name + '.png'), tmp_path / (name + '.npz')
    Image.fromarray(np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)).save(image_path)
    np.savez(geometry_path, **{k: v[0].numpy() for k, v in target.items()})
    scene = next(str(i) for i in range(100) if int(hashlib.sha256(
        ('scannetpp-internal-val-v1:' + str(i)).encode()).hexdigest()[:8], 16) % 20)
    row = {'image': str(image_path), 'geometry': str(geometry_path),
           'geometry_sha256': hashlib.sha256(geometry_path.read_bytes()).hexdigest(),
           'group': 'scannetpp:' + scene, 'scene': scene, 'split': 'train',
           'official_split': 'nvs_sem_train', 'supervision': 'scannetpp_panocam_laser'}
    return row


def test_paired_augmentation_keeps_camera_depth_and_plane_aligned(tmp_path):
    row = publish_pair(tmp_path, height=40, width=60)
    target = plane_target(height=40, width=60)
    rays = camera_points(torch.ones_like(target['depth']), target['intrinsics'])
    normal = torch.tensor([.3, .1, 1.])
    normal /= normal.norm()
    target['depth'] = 3 / (rays * normal).sum(-1)
    target['planes'][:, 0, :3] = normal
    np.savez(row['geometry'], **{k: v[0].numpy() for k, v in target.items()})
    loader = GroundTruthImages([row], augment=True)
    signs = set()
    for seed in range(8):
        a = loader[0, seed, 40, 40]
        b = loader[0, seed, 40, 40]
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        points = camera_points(a['depth'][None], a['intrinsics'][None])[0]
        plane = a['planes'][0]
        assert ((points * plane[:3]).sum(-1) + plane[3]).abs().max() < 1e-6
        assert a['intrinsics'][0, 0] == pytest.approx(.8 * 60 / 40)
        signs.add(int(torch.sign(plane[0])))
    assert signs == {-1, 1}


def test_gt_admission_rejects_heldout_or_unverified_data(tmp_path):
    row = publish_pair(tmp_path)
    validate_gt_records([row])
    for key, value in [('official_split', 'nvs_sem_val'), ('geometry_sha256', ''),
                       ('supervision', 'teacher_prediction'), ('split', 'val')]:
        changed = dict(row, **{key: value})
        with pytest.raises(ValueError, match='eligible training scenes'):
            validate_gt_records([changed])


def test_polar_ray_and_perspective_projection_geometry():
    rays = scanner_rays(np.array([0, np.pi / 2, 0]), np.array([np.pi / 2, np.pi / 2, 0]))
    np.testing.assert_allclose(rays, np.eye(3), atol=1e-7)
    k, rotation, wanted = perspective_camera(32, 48, 0, fov_degrees=80)
    azim = np.mod(np.arctan2(wanted[..., 1], wanted[..., 0]), 2 * np.pi).astype(np.float32)
    polar = np.arccos(wanted[..., 2]).astype(np.float32)
    camera_ray = wanted @ rotation
    distance = 3 / camera_ray[..., 2]
    rgb = np.zeros((32, 48, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(48)
    result = PanoramaProjector(azim, polar, stride=4).project(
        rgb, distance, np.full((32, 48), 255, np.uint8), height=32, width=48,
        yaw=0, fov_degrees=80)
    image, depth, valid, actual_k, report = result
    np.testing.assert_allclose(depth[valid], 3, atol=1e-5)
    np.testing.assert_array_equal(image[valid], rgb[valid])
    np.testing.assert_array_equal(actual_k, k)
    assert report['valid_pixel_error_p50_p95_max'][-1] < 1e-4
    assert np.linalg.det(rotation) == pytest.approx(1)


def gt_config(tmp_path):
    config = config_for(tmp_path)
    row = publish_pair(tmp_path)
    manifest = tmp_path / 'gt-train.jsonl'
    manifest.write_text(json.dumps(row) + '\n')
    config['data'].update(train_manifest=str(manifest), val_manifest=None)
    config['geometry_supervision'] = {'target': 'ground_truth', 'camera': 'known_intrinsics'}
    config['loss'] = {'spatial': .1, 'summary': .1, 'cosine': .1, 'summary_cosine': .1,
                      'gt_depth': 1., 'gt_points': 1., 'gt_gradient': .5,
                      'gt_normal': .2, 'gt_plane': .2}
    config['train'].update(steps=2, max_seconds=120, lr=1e-5, lr_schedule='constant',
                           batch_size=1, accumulation=1, eval_at_start=False)
    config['output_dir'] = str(tmp_path / 'gt-run')
    return config


def test_gt_training_updates_adapter_freezes_system_and_resumes_exactly(tmp_path):
    config = gt_config(tmp_path)
    system, adapter = tiny_system()
    frozen = {name: {k: v.clone() for k, v in getattr(system, name).state_dict().items()}
              for name in ('radio', 'encoder', 'decoder')}
    initial_adapter = {k: v.clone() for k, v in adapter.state_dict().items()}
    final = torch.load(train(config, system_adapter=(system, adapter)), weights_only=True)
    assert any(not torch.equal(v, final['adapter'][k]) for k, v in initial_adapter.items())
    for name in frozen:
        for key, value in getattr(system, name).state_dict().items():
            torch.testing.assert_close(value, frozen[name][key], rtol=0, atol=0)
    rows = [json.loads(s) for s in (tmp_path / 'gt-run/train.jsonl').read_text().splitlines()]
    assert rows[-1]['geometry_weighted'] > 0
    assert all(np.isfinite(rows[-1][key]) for key in GT_LOSS_KEYS)
    split = deepcopy(config)
    split['output_dir'] = str(tmp_path / 'gt-resume')
    path = train(split, stop_after=1, system_adapter=tiny_system())
    resumed = torch.load(train(split, resume=path, system_adapter=tiny_system()), weights_only=True)
    for key in final['adapter']:
        torch.testing.assert_close(final['adapter'][key], resumed['adapter'][key], rtol=0, atol=0)
    # Replacing labels must invalidate exact resume even when RGB is unchanged.
    row = read_manifest(config['data']['train_manifest'])[0]
    from pathlib import Path
    path = Path(row['geometry'])
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='identical config and manifests'):
        train(split, resume=tmp_path / 'gt-resume/latest.pt', system_adapter=tiny_system())


def test_gt_config_rejects_teacher_geometry_or_missing_anchors(tmp_path):
    config = gt_config(tmp_path)
    validate_config(config)
    config['loss']['points'] = .5
    with pytest.raises(ValueError, match='pseudo-geometry'):
        validate_config(config)
    config['loss']['points'] = 0
    config['loss']['gt_points'] = 0
    with pytest.raises(ValueError, match='anchors'):
        validate_config(config)


def test_gt_init_preserves_weights_statistics_and_does_not_inherit_rgb_pipeline(tmp_path):
    parent_config = config_for(tmp_path)
    parent_config['data']['val_manifest'] = None
    parent_config['output_dir'] = str(tmp_path / 'feature-parent')
    source_path = train(parent_config, stop_after=1, system_adapter=tiny_system())
    source = torch.load(source_path, weights_only=True)
    source['performance'] = {'data_pipeline': 'prepared', 'compact_transport': True}
    torch.save(source, source_path)
    before = hashlib.sha256(source_path.read_bytes()).hexdigest()
    config = gt_config(tmp_path)
    final_path = train(config, init=source_path, system_adapter=tiny_system())
    initial = torch.load(final_path.parent / 'initial.pt', weights_only=True)
    final = torch.load(final_path, weights_only=True)
    for section in ('adapter', 'statistics'):
        for key in source[section]:
            torch.testing.assert_close(initial[section][key], source[section][key], rtol=0, atol=0)
    assert initial['step'] == 0 and initial['optimizer']['state'] == {}
    assert final['performance'] is None
    assert final['initialization_provenance']['source_step'] == 1
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == before
