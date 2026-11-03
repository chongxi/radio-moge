from copy import deepcopy
from pathlib import Path

import pytest

from freeze_arkit_training_stage import COUNTS, LABELS, WEIGHTS, stage_config, validate_new_evidence


def evidence():
    return dict(accounting=dict(selected_images=598298, total_images=602337, unresolved=0, selected_counts=COUNTS),
        isolation=dict(stage='expanded_selection_bound_to_full_canonical_comparison',
            original_geometry_scope=602337, selected_geometry=598298,
            exact_pixel_overlap_pairs=0, forbidden_shared_scene_rows=0),
        crop=dict(stage='candidate_crop_support_observed_not_admitted', images=127, groups=99, views=381,
            sparse_views=6, rotation_images={k: 1 for k in ('0', '90', '180', '270')},
            actual_production_loader_used=True, original_support_unchanged=True,
            max_native_to_target_relative_error=5.94e-8, sparse_ordinals=[36153, 168525]),
        source=dict(stage='arkit_source_review_and_routes_complete_not_corpus_admission', original_images=197542,
            source_groups=1478, source_disposition_counts=dict(native_source_eligible=194916, excluded_geometry=2626),
            original_source_flags_retained=True), coverage=dict(label_counts=LABELS),
        finish=dict(stage='completed_training_checkpoint_verified', step=2895, new_optimizer_updates=895),
        stop=dict(step=2895, final_evaluation='complete', reason='wall_clock_budget'))


def test_actual_wall_clock_end_between_validation_intervals_is_a_valid_parent():
    validate_new_evidence(**evidence())


@pytest.mark.parametrize('section,key,value', [
    ('stop', 'final_evaluation', 'pending'), ('stop', 'step', 3000), ('stop', 'reason', 'signal'),
    ('finish', 'new_optimizer_updates', 1000), ('accounting', 'selected_images', 598299),
    ('accounting', 'unresolved', 1), ('isolation', 'exact_pixel_overlap_pairs', 1),
    ('source', 'original_source_flags_retained', False), ('crop', 'original_support_unchanged', False),
    ('coverage', 'label_counts', dict(LABELS, measured_sensor=0)),
])
def test_incomplete_or_changed_source_and_parent_cannot_publish_readiness(section, key, value):
    values = deepcopy(evidence())
    values[section][key] = value
    with pytest.raises(ValueError):
        validate_new_evidence(**values)


def test_new_stage_preserves_loss_replay_and_optimizer_settings_with_exact_parent_baselines():
    parent = dict(data=dict(geometry_corpus={'sealed': True}, mixture_weights={}, buckets=[[400, 640]]),
        train=dict(steps=3000, max_seconds=7200, lr=1e-5, accumulation=16),
        evaluation=dict(checkpoint_metrics={'nyu': dict(metric='depth', mode='min', baseline=.07)}),
        tracking={}, feature_replay=dict(manifest='/fixed', fraction=.5), loss=dict(gt_scale=.1))
    original = deepcopy(parent)
    config = stage_config(parent, dict(step=2895, depth=.068), Path('/scope'), 2895)
    assert parent == original and config['train']['steps'] == 3895
    assert config['feature_replay'] == parent['feature_replay'] and config['loss'] == parent['loss']
    assert config['train']['lr'] == parent['train']['lr'] and config['train']['accumulation'] == 16
    assert 'geometry_corpus' not in config['data'] and config['data']['mixture_weights'] == WEIGHTS
    assert sum(WEIGHTS.values()) == pytest.approx(1) and WEIGHTS['arkitscenes'] == .2
    assert config['evaluation']['checkpoint_metrics']['nyu']['baseline'] == .068
    with pytest.raises(ValueError, match='actual final parent'):
        stage_config(parent, dict(step=2750, depth=.068), Path('/scope'), 2895)
    with pytest.raises(ValueError, match='nonfinite'):
        stage_config(parent, dict(step=2895, depth=float('nan')), Path('/scope'), 2895)
