from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from radio_moge.checkpoint import rng_state, restore_rng
from radio_moge.data import TrainingImages
from radio_moge.losses import FeatureStatistics
from radio_moge.ordinary_geometry_data import GeometrySidecar, GeometryAttachedImages, collate_geometry_attached
from radio_moge.ordinary_model_probe import (state_digest, compare_feature_window, geometry_window, module_states,
                                            verify_scale_routing, verify_target, compare_feature_forward)
from radio_moge.ordinary_training import equal_state
from radio_moge.prefetch import normalize_uint8
from radio_moge.rgb_input_audit import load_reference_records
from test_ordinary_training import setup
from test_training import tiny_system
from test_ordinary_geometry_losses import small_official  # noqa: F401


def fixture(tmp_path):
    cfg, ref = setup(tmp_path, payloads=True)
    rows = load_reference_records(cfg['data']['train_manifest'])
    dataset = GeometryAttachedImages(TrainingImages(rows, True, compact_transport=True), GeometrySidecar(ref))
    groups = [normalize_uint8(collate_geometry_attached([dataset[(i, 43+j, 32, 48)]
              for j, i in enumerate(indices)])) for indices in [(0, 1, 3, 4), (4, 4, 0, 0)]]
    system, adapter = tiny_system()
    stats = FeatureStatistics(6, 6)
    stats.spatial_scale.fill_(2)
    stats.summary_scale.fill_(3)
    return cfg, groups, system, adapter, stats


@pytest.mark.usefixtures('small_official')
def test_exact_g0_and_actual_geometry_connectivity_without_an_optimizer(tmp_path, monkeypatch):
    cfg, groups, system, adapter, stats = fixture(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('Model gate must not construct an optimizer')
    monkeypatch.setattr(torch.optim, 'AdamW', forbidden)
    before, random_before = module_states(system, adapter, stats), rng_state()
    forward = compare_feature_forward(system, adapter, stats, [x.image for x in groups], groups, cfg)
    assert forward['runtime_feature_pixels_losses_exact'] and not forward['backward_performed']
    assert all(p.grad is None for p in adapter.parameters())
    proof = compare_feature_window(system, adapter, stats, [x.image for x in groups], groups, cfg,
                                   same_path_check=True)
    assert proof['feature_pixels_losses_gradients_exact'] and proof['original_vs_original_exact']
    result = geometry_window(system, adapter, stats, groups, cfg, 8)
    assert result['counts']['gt_supervised_images'] == 4
    assert result['counts']['gt_skipped_sparse_images'] == 1
    assert result['counts']['gt_missing_target_images'] == 3
    assert all(r['geometry_gradient_norm'] > 0 for r in result['groups'])
    assert before == module_states(system, adapter, stats)
    adapter.zero_grad(set_to_none=True)
    restore_rng(random_before)
    assert equal_state(random_before, rng_state())


def test_g0_parity_cannot_hide_changed_cuda_payload_values(tmp_path):
    cfg, groups, system, adapter, stats = fixture(tmp_path)
    altered = list(groups)
    image = altered[0].image.clone()
    image[0, 0, 0, 0] += .1
    altered[0] = replace(altered[0], image=image)
    with pytest.raises(ValueError, match='groups/pixels'):
        compare_feature_window(system, adapter, stats, [x.image for x in groups], altered, cfg)


def test_nested_state_hash_and_chunk_comparison_detect_change_and_nonfinite():
    value = {'adam': {1: dict(step=torch.tensor(232288.), exp_avg=torch.ones(1100000))}, 'rng': [1, 2]}
    copy = deepcopy(value)
    assert state_digest(copy) == state_digest(value) and equal_state(copy, value)
    copy['adam'][1]['exp_avg'][-1] += 1
    assert state_digest(copy) != state_digest(value) and not equal_state(copy, value)
    copy['adam'][1]['exp_avg'][0] = float('nan')
    assert not equal_state(copy, copy)
    with pytest.raises(ValueError, match='Nonfinite'):
        state_digest(copy)


def test_mixed_batch_cannot_hide_pseudo_scale_gradient_or_lost_target():
    targets = [dict(supervision='metric_native', valid=torch.ones(32, dtype=torch.bool)),
               dict(supervision='relative_pseudo', valid=torch.ones(32, dtype=torch.bool)), None]
    scale = torch.ones(3, requires_grad=True)
    verify_scale_routing(scale[0].square(), scale, targets)
    with pytest.raises(ValueError, match='slot acquired'):
        verify_scale_routing(scale.sum(), scale, targets)
    with pytest.raises(ValueError, match='availability'):
        verify_target(None, dict(geometry_available=True))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Actual deployment gate also verifies device parity')
def test_device_independent_checkpoint_comparison():
    original = torch.arange(1100000, dtype=torch.float32)
    device = original.cuda()
    assert equal_state(original, device) and equal_state(device, original)
    assert state_digest(original) == state_digest(device)
    device[-1] += 1
    assert not equal_state(device, original)
