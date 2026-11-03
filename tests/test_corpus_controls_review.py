from copy import deepcopy
import json

import pytest

from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import publish_json
from review_corpus_controls import factorial_comparisons, joint_result, laser_review, run, view_review
from review_corpus_validation import HELDOUT_KEYS


def save(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    publish_json(path, value)
    return file_ref(path)


def test_joint_criterion_cannot_be_replaced_with_one_improved_metric():
    baseline = dict(spatial_cosine=.96, summary_cosine=.93, lidar_abs_rel=.09)
    improved = dict(spatial_cosine=.961, summary_cosine=.931, lidar_abs_rel=.089)
    assert joint_result(baseline, improved)['all_three_improve']
    for key in baseline:
        assert not joint_result(baseline, dict(improved, **{key: baseline[key]}))['all_three_improve']
    with pytest.raises(ValueError):
        joint_result(baseline, dict(improved, nyu_abs_rel=.01))
    with pytest.raises(ValueError):
        joint_result(baseline, dict(improved, lidar_abs_rel=float('nan')))


def variants():
    return {f'f{f}g{g}': dict(feature_coefficients=dict.fromkeys(
        ('spatial', 'summary', 'cosine', 'summary_cosine'), f), geometry_multiplier=g)
        for f in (.1, 1.) for g in (0, 1)}


def test_factorial_reports_both_directions_without_best_arm_selection():
    factors = variants()
    values = {name: dict(spatial_cosine=.95+i*.001, summary_cosine=.93, lidar_abs_rel=.1-i*.001)
              for i, name in enumerate(factors)}
    result = factorial_comparisons(factors, values)
    assert len(result) == 4
    assert result[0]['reference'] == 'f0.1g0' and result[0]['treatment'] == 'f0.1g1'
    assert result[0]['primary']['spatial_cosine']['difference'] == pytest.approx(.001)
    assert result[2]['reference'] == 'f0.1g0' and result[2]['treatment'] == 'f1.0g0'
    with pytest.raises(ValueError):
        factorial_comparisons(factors, {k:v for k,v in values.items() if k != 'f1.0g1'})
    factors['f1.0g1'] = deepcopy(factors['f1.0g0'])
    with pytest.raises(ValueError, match='Duplicate'):
        factorial_comparisons(factors, values)


def make_view(root, name, spatial=.96, summary=.93, depth=.1):
    names = ('spatial/cosine', 'summary/cosine', 'teacher_depth/abs_rel',
             'gt_student/abs_rel', 'gt_student/rmse', 'gt_student/delta1')
    values = dict(zip(names, (spatial, summary, .2, depth, .3, .8)))
    rows = [dict(image='one.png', dataset='source', **values, **{'gt_teacher/abs_rel': .15})]
    metrics = dict(values, images=1, **{'macro/'+k:values[k] for k in names[:3]})
    directory = root / name
    refs = {k: save(directory, k+'.json', value) for k,value in
            dict(metrics=metrics, per_image=rows, protocol=dict(images=1, max_side=640)).items()}
    old = dict(before=refs['metrics'], before_per_image=refs['per_image'],
        protocol_file=refs['protocol'], protocol=dict(images=1, max_side=640), images=1,
        manifest=save(directory, 'manifest.json', ['one.png']),
        datasets=dict(source=dict(images=1, metrics={k:dict(before=v) for k,v in values.items()})))
    return directory, refs, old, metrics


def test_view_review_binds_images_teacher_protocol_all_metrics_and_baseline(tmp_path):
    directory, refs, old, metrics = make_view(tmp_path, 'view')
    result = view_review(old, directory, 'original', metrics)
    assert set(result['metrics']) == set(metrics)
    assert all(v['difference'] == 0 for v in result['metrics'].values())
    with pytest.raises(ValueError, match='saved metrics'):
        view_review(old, directory, 'original', dict(metrics, **{'macro/spatial/cosine': .8}))
    changed = deepcopy(old)
    changed['datasets']['source']['metrics']['spatial/cosine']['before'] = .8
    with pytest.raises(ValueError, match='per-source baseline'):
        view_review(changed, directory, 'original', metrics)
    rows = json.loads((directory/'per_image.json').read_text())
    rows[0]['gt_teacher/abs_rel'] = .9
    target = tmp_path/'changed'
    target.mkdir()
    for filename in ('metrics.json', 'protocol.json'):
        (target/filename).write_bytes((directory/filename).read_bytes())
    publish_json(target/'per_image.json', rows)
    with pytest.raises(ValueError, match='Frozen teacher'):
        view_review(old, target, 'original', metrics)


def make_laser(root):
    row = dict(id='a', scene='scene', gt_valid_fraction=.8, gt_plane_fraction=.5, abs_rel=.1, rmse=.2)
    metrics = dict(images=1, abs_rel=.1, rmse=.2, plane_images=0)
    refs = {k:save(root, k+'.json', v) for k,v in
            dict(metrics=metrics, per_image=[row], protocol=dict(camera='known')).items()}
    return dict(current=refs, images=1), metrics


def test_laser_review_rejects_changed_support_even_when_metric_improves(tmp_path):
    old, metrics = make_laser(tmp_path/'before')
    current, _ = make_laser(tmp_path/'after')
    record = {'gt_laser_known_camera/'+k:v for k,v in metrics.items()}
    assert laser_review(old, tmp_path/'after', record)['fixed_gt_support_same']
    rows = json.loads((tmp_path/'after/per_image.json').read_text())
    rows[0]['gt_valid_fraction'] = .6
    (tmp_path/'after/per_image.json').write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='fixed GT support'):
        laser_review(old, tmp_path/'after', record)


def completed_fixture(root):
    factors = variants()
    view_names = ('original', 'native-640x400', 'dataset-scannetpp-lidar', 'dataset-scannetpp-dslr')
    old_views = {name:make_view(root/'before', name)[2] for name in view_names}
    laser, _ = make_laser(root/'before/laser')
    provenance = dict(revisions=dict(frozen='exact'), adapter=dict(kind='adapter'),
                      data_digests=dict.fromkeys(HELDOUT_KEYS, 'a'*64))
    prior = save(root, 'baseline-review.json', dict(views=old_views, known_camera_laser=laser,
                 provenance=save(root, 'prior-provenance.json', provenance)))
    plan = dict(variants=factors, evaluation=dict(reference=prior), train=dict(steps=250, seed=43),
                feature_replay=dict(seed=100043))
    experiment = save(root, 'experiment.json', dict(plan=save(root, 'plan.json', plan)))
    completed = {}
    for name, factor in factors.items():
        directory = root/name
        training = directory/'training'
        output = training/'eval-0000250'
        record = dict(step=250)
        for view in view_names:
            destination = output if view == 'original' else output/view
            _, _, _, metrics = make_view(destination, '.', spatial=.961, summary=.931, depth=.099)
            prefix = '' if view == 'original' else ('resolution/native-640x400/' if view == 'native-640x400'
                      else 'dataset/'+view.removeprefix('dataset-')+'/')
            record.update({prefix+k:v for k,v in metrics.items()})
        _, metrics = make_laser(output/'gt-laser-known-camera')
        record.update({'gt_laser_known_camera/'+k:v for k,v in metrics.items()})
        (training/'validation.jsonl').write_text(json.dumps(record)+'\n')
        save(training, 'provenance.json', provenance)
        actual = save(directory, 'actual-inputs.json', ['identical actual pixels and targets'])
        config = dict(output_dir=str(training), train=plan['train'], feature_replay=plan['feature_replay'],
                      loss=factor['feature_coefficients'], corpus_loss_control=dict(geometry_multiplier=factor['geometry_multiplier']))
        result = dict(stage='fixed_budget_control_completed', optimizer_updates=250,
            fixed_final_step_selected=True, input_sha256=actual['sha256'],
            actual_input_audit=save(directory, 'audit.json', dict(records=actual, microbatches=4000, replay_microbatches=2000)),
            initial_verification=save(directory, 'initial-proof.json', dict(source_step=232288, step=0, cursor=0,
                adapter_statistics_bitwise_archive=True, optimizer_state_entries=0,
                first_arm_reference=False, optimizer_scheduler_rng_bitwise_first_arm=True)),
            config=save(directory, 'config.json', config), validation_log=file_ref(training/'validation.jsonl'))
        completed[name] = save(directory, 'completed.json', result)
    gates = save(root, 'model-gates.json', dict(stage='all_four_actual_model_gates_complete',
        same_initial_model_states=True, same_selected_views=True, optimizer_updates=0))
    save(root, 'report.json', dict(stage='four_fixed_budget_controls_completed', total_optimizer_updates=1000,
        updates_per_arm=250, completed=completed, experiment=experiment, model_gates=gates,
        actual_input_sha256=actual['sha256']))
    return root/'report.json'


def test_four_arm_review_is_separate_from_production_step_history_and_fails_closed(tmp_path):
    source = completed_fixture(tmp_path)
    report = run(source, tmp_path/'review')
    assert report['source_step'] == 232288 and report['control_updates'] == 250
    assert set(report['primary_joint_candidates']) == set(variants())
    assert len(report['factorial_comparisons']) == 4 and not report['model_inference_repeated']
    data = json.loads(source.read_text())
    data['completed'].pop('f1.0g1')
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Every fixed-budget control'):
        run(source, tmp_path/'incomplete-review')
    assert not (tmp_path/'incomplete-review').exists()


def test_different_actual_inputs_cannot_produce_a_controlled_comparison(tmp_path):
    source = completed_fixture(tmp_path)
    path = tmp_path/'f1.0g1/completed.json'
    data = json.loads(path.read_text())
    data['input_sha256'] = 'f'*64
    path.write_text(json.dumps(data))
    completed = json.loads(source.read_text())
    completed['completed']['f1.0g1'] = file_ref(path)
    source.write_text(json.dumps(completed))
    with pytest.raises(ValueError, match='input sequence'):
        run(source, tmp_path/'not-matched')
    assert not (tmp_path/'not-matched').exists()
