import json

import pytest

from radio_moge.corpus_admission import file_ref
from review_feature_reference import review
from test_corpus_controls_review import completed_fixture, save


def fixture(root):
    completed_fixture(root)
    original = json.loads((root/'f1.0g0/config.json').read_text())
    training = root/'f1.0g0/training'
    (training/'eval-0000250').rename(training/'eval-0232538')
    value = json.loads((training/'validation.jsonl').read_text())
    value['step'] = 232538
    (training/'validation.jsonl').write_text(json.dumps(value)+'\n')
    config = dict(output_dir=str(training), train=dict(steps=232538), loss=original['loss'])
    config_ref = save(root, 'reference-config.json', config)
    plan = dict(planned_source_counts=dict(source=8000), planned_resolution_counts={'400x640':8000},
        primary_joint_rule=dict(spatial_cosine_gt=.96, summary_cosine_gt=.93, scannet_lidar_absrel_lt=.1))
    request = save(root, 'reference-request.json', dict(config=config_ref, plan=save(root, 'reference-plan.json', plan)))
    contract = save(root, 'evaluation-contract.json', dict(endpoint=232538, reference_config=config_ref,
                                                        reference_validation=file_ref(root/'baseline-review.json')))
    audit = save(root, 'reference-input-audit.json', dict(optimizer_updates=250, cursor=2000, presentations=8000,
        records=file_ref(root/'f1.0g0/actual-inputs.json'), source_counts=plan['planned_source_counts'],
        resolution_counts=plan['planned_resolution_counts'], planned_keys_matched=True, actual_compute_groups_verified=True))
    final = dict(path=str(root/'fixed-final.pt'), sha256='f'*64)
    save(root, 'completed.json', dict(stage='fixed250_reference_training_complete',optimizer_updates=250,
        final_state_finite=True, fixed_statistics_unchanged=True, request=request,
        actual_inputs=audit, final_checkpoint=final))
    save(root, 'laser-report.json', dict(stage='reference_laser_evaluation_complete',
        contract=contract, fixed_checkpoint=final, heldout_digests=dict(geometry_val='a'*64,geometry_val_rgb='a'*64)))
    return root/'evaluation-contract.json', root/'laser-report.json'


def test_fixed_reference_reviews_all_views_with_predefined_joint_rule(tmp_path):
    contract, laser = fixture(tmp_path)
    result = review(contract, tmp_path/'result', laser)
    assert result['primary']['all_three_improve']
    assert len(result['views']) == 4 and result['fixed_final_not_best_selected']
    assert result['heldout_digest_keys_checked'] == ['val','geometry_val','geometry_val_rgb',
        'dataset/scannetpp-dslr','dataset/scannetpp-lidar']


@pytest.mark.parametrize('changed', ['updates', 'stream', 'laser_digest', 'endpoint', 'teacher'])
def test_reference_review_rejects_incomplete_or_mismatched_comparison(tmp_path, changed):
    contract, laser = fixture(tmp_path)
    if changed in ('updates', 'endpoint'):
        path = tmp_path/'completed.json' if changed == 'updates' else contract
        data = json.loads(path.read_text())
        data['optimizer_updates' if changed == 'updates' else 'endpoint'] -= 1
        path.write_text(json.dumps(data))
    elif changed == 'stream':
        path = tmp_path/'reference-input-audit.json'
        data = json.loads(path.read_text())
        data['presentations'] = 7999
        path.write_text(json.dumps(data))
        data = json.loads((tmp_path/'completed.json').read_text())
        data['actual_inputs'] = file_ref(path)
        (tmp_path/'completed.json').write_text(json.dumps(data))
    elif changed == 'laser_digest':
        data = json.loads(laser.read_text())
        data['heldout_digests']['geometry_val'] = 'b'*64
        laser.write_text(json.dumps(data))
    else:
        path = tmp_path/'f1.0g0/training/eval-0232538/per_image.json'
        data = json.loads(path.read_text())
        data[0]['gt_teacher/abs_rel'] = .9
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        review(contract, tmp_path/'invalid-result', laser)
    assert not (tmp_path/'invalid-result').exists()
