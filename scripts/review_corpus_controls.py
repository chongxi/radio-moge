"""Review all four fixed-final controls from saved evaluations, without inference.

Control step 250 is not geometry-production step 250. Each arm is independently
paired with the archived feature baseline and uses the frozen factorial plan.
No best checkpoint is selected and an incomplete factorial cannot be reviewed.
"""
import argparse
import math
from pathlib import Path
from statistics import mean
import time

from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import publish_json
from review_corpus_validation import HELDOUT_KEYS, check_aggregates, paired_rows


def change(before, after):
    if not all(math.isfinite(v) for v in (before, after)):
        raise ValueError('Nonfinite comparison value')
    return dict(before=before, final=after, difference=after-before,
                percent=100*(after/before-1) if before else None)


def joint_result(baseline, current):
    if set(baseline) != {'spatial_cosine', 'summary_cosine', 'lidar_abs_rel'} or set(current) != set(baseline):
        raise ValueError('All three predefined primary metrics are required')
    changes = {k: change(baseline[k], current[k]) for k in baseline}
    passed = {k: current[k] > baseline[k] if k.endswith('cosine') else current[k] < baseline[k]
              for k in baseline}
    return dict(metrics=changes, criteria=passed, all_three_improve=all(passed.values()),
                significance_established=False, replication_required=True)


def factorial_comparisons(variants, results):
    factors = {}
    for name, variant in variants.items():
        coefficient = set(variant['feature_coefficients'].values())
        if len(coefficient) != 1:
            raise ValueError('Nonuniform feature coefficients')
        key = (coefficient.pop(), variant['geometry_multiplier'])
        if key in factors:
            raise ValueError('Duplicate control factor')
        factors[key] = name
    if set(factors) != {(f, g) for f in (.1, 1.) for g in (0, 1)} or set(results) != set(variants):
        raise ValueError('All four completed factors are required')
    pairs = [('add_geometry', factors[f, 0], factors[f, 1]) for f in (.1, 1.)]
    pairs += [('increase_feature_coefficient', factors[.1, g], factors[1., g]) for g in (0, 1)]
    return [dict(factor=factor, reference=a, treatment=b,
                 primary={k: change(results[a][k], results[b][k]) for k in results[a]})
            for factor, a, b in pairs]


def view_review(old, destination, name, record):
    refs = {n: file_ref(destination / (n + '.json')) for n in ('metrics', 'per_image', 'protocol')}
    metrics, rows, protocol = (checked_json(refs[n]) for n in ('metrics', 'per_image', 'protocol'))
    before, before_rows = checked_json(old['before']), checked_json(old['before_per_image'])
    checked_path(old['manifest'])
    if protocol != old['protocol'] or protocol != checked_json(old['protocol_file']):
        raise ValueError('Control evaluation protocol differs from the archived baseline')
    paired_rows(before_rows, rows)
    check_aggregates(before_rows, before)
    check_aggregates(rows, metrics)
    if len(rows) != old['images'] or set(metrics) != set(before):
        raise ValueError('Control view image count or metric coverage differs')
    prefix = '' if name == 'original' else ('resolution/native-640x400/'
        if name == 'native-640x400' else 'dataset/' + name.removeprefix('dataset-') + '/')
    for key, value in metrics.items():
        if record.get(prefix + key) != value:
            raise ValueError('Completed validation record differs from saved metrics: ' + prefix + key)
    datasets = {}
    if {row['dataset'] for row in rows} != set(old['datasets']):
        raise ValueError('Control source coverage differs')
    for dataset, prior in old['datasets'].items():
        subset = [row for row in rows if row['dataset'] == dataset]
        initial = [row for row in before_rows if row['dataset'] == dataset]
        if len(subset) != prior['images']:
            raise ValueError('Control dataset image count differs')
        values = {}
        for key, history in prior['metrics'].items():
            a, b = mean(row[key] for row in initial if key in row), mean(row[key] for row in subset if key in row)
            if not math.isclose(a, history['before'], rel_tol=0, abs_tol=1e-12):
                raise ValueError('Archived per-source baseline differs from actual rows')
            values[key] = change(a, b)
        datasets[dataset] = dict(images=len(subset), metrics=values)
    return dict(files=refs, baseline=old['before'], baseline_per_image=old['before_per_image'],
        images=len(rows), protocol=protocol, same_image_order=True, frozen_teacher_gt_unchanged=True,
        aggregate_recomputed=True, metrics={k: change(before[k], metrics[k]) for k in metrics},
        datasets=datasets)


def laser_review(old, destination, record):
    refs = {n: file_ref(destination / (n + '.json')) for n in ('metrics', 'per_image', 'protocol')}
    metrics, rows, protocol = (checked_json(refs[n]) for n in ('metrics', 'per_image', 'protocol'))
    if protocol != checked_json(old['current']['protocol']):
        raise ValueError('Known-camera evaluation protocol differs')
    paired_rows(checked_json(old['current']['per_image']), rows,
                ('id', 'scene'), ('gt_valid_fraction', 'gt_plane_fraction'))
    if len(rows) != old['images'] or metrics['images'] != len(rows):
        raise ValueError('Known-camera image count differs')
    for key, value in metrics.items():
        if record.get('gt_laser_known_camera/' + key) != value:
            raise ValueError('Known-camera validation record differs')
        if key == 'plane_images':
            if value != sum('plane_rmse_m' in row for row in rows):
                raise ValueError('Known-camera plane image count differs')
        elif key != 'images' and not math.isclose(mean(row[key] for row in rows if key in row), value,
                                                 rel_tol=0, abs_tol=1e-12):
            raise ValueError('Known-camera aggregate differs from per-image values')
    return dict(files=refs, images=len(rows), metrics=metrics, same_image_order=True,
                fixed_gt_support_same=True, protocol=protocol,
                archived_baseline_available_in_reference=False)


def run(completion_path, output):
    import json
    completion_ref = file_ref(completion_path)
    completion = checked_json(completion_ref)
    experiment = checked_json(completion['experiment'])
    plan = checked_json(experiment['plan'])
    reference_ref = plan['evaluation']['reference']
    reference = checked_json(reference_ref)
    reference_provenance = checked_json(reference['provenance'])
    model_gates = checked_json(completion['model_gates'])
    if (completion['stage'] != 'four_fixed_budget_controls_completed'
            or completion['total_optimizer_updates'] != 1000 or completion['updates_per_arm'] != 250
            or set(completion['completed']) != set(plan['variants']) or len(completion['completed']) != 4
            or model_gates['stage'] != 'all_four_actual_model_gates_complete'
            or model_gates['same_initial_model_states'] is not True
            or model_gates['same_selected_views'] is not True or model_gates['optimizer_updates'] != 0):
        raise ValueError('Every fixed-budget control and actual model gate must be complete')
    arms, primary = {}, {}
    for name, result_ref in completion['completed'].items():
        result = checked_json(result_ref)
        initial = checked_json(result['initial_verification'])
        audit = checked_json(result['actual_input_audit'])
        checked_path(audit['records'])
        config = checked_json(result['config'])
        training = Path(config['output_dir'])
        provenance_ref = file_ref(training / 'provenance.json')
        provenance = checked_json(provenance_ref)
        if (result['stage'] != 'fixed_budget_control_completed' or result['optimizer_updates'] != 250
                or result['fixed_final_step_selected'] is not True
                or result['input_sha256'] != completion['actual_input_sha256']
                or audit['records']['sha256'] != result['input_sha256']
                or audit['microbatches'] != 4000 or audit['replay_microbatches'] != 2000
                or initial['source_step'] != 232288 or initial['step'] != 0 or initial['cursor'] != 0
                or initial['adapter_statistics_bitwise_archive'] is not True
                or initial['optimizer_state_entries'] != 0
                or not (initial['first_arm_reference'] or initial['optimizer_scheduler_rng_bitwise_first_arm'])
                or config['train'] != plan['train'] or config['feature_replay'] != plan['feature_replay']
                or config['corpus_loss_control']['geometry_multiplier'] != plan['variants'][name]['geometry_multiplier']
                or any(config['loss'][k] != v for k,v in plan['variants'][name]['feature_coefficients'].items())
                or provenance['revisions'] != reference_provenance['revisions']
                or provenance['adapter'] != reference_provenance['adapter']
                or any(provenance['data_digests'][k] != reference_provenance['data_digests'][k] for k in HELDOUT_KEYS)):
            raise ValueError('Control initialization, budget, input sequence or heldout identity differs')
        records = [json.loads(line) for line in checked_path(result['validation_log']).read_text().splitlines()]
        if len(records) != 1 or records[0]['step'] != 250:
            raise ValueError('Exactly one fixed-final evaluation is required')
        record = records[0]
        destination = training / 'eval-0000250'
        views = {view: view_review(old, destination if view == 'original' else destination/view, view, record)
                 for view, old in reference['views'].items()}
        laser = laser_review(reference['known_camera_laser'], destination/'gt-laser-known-camera', record)
        keys = dict(spatial_cosine=('original', 'macro/spatial/cosine'),
                    summary_cosine=('original', 'macro/summary/cosine'),
                    lidar_abs_rel=('dataset-scannetpp-lidar', 'gt_student/abs_rel'))
        baseline = {k: views[v]['metrics'][metric]['before'] for k,(v,metric) in keys.items()}
        primary[name] = {k: views[v]['metrics'][metric]['final'] for k,(v,metric) in keys.items()}
        arms[name] = dict(completion=result_ref, provenance=provenance_ref, views=views,
            known_camera_laser=laser, primary=joint_result(baseline, primary[name]))
    comparison = factorial_comparisons(plan['variants'], primary)
    checked_path(completion_ref)
    checked_path(reference_ref)
    output.mkdir(exist_ok=False)
    report = dict(stage='four_fixed_final_controls_reviewed', created_at=time.time(),
        completion=completion_ref, baseline_review=reference_ref, plan=experiment['plan'], arms=arms,
        factorial_comparisons=comparison, input_sequence_identical=True, source_step=232288,
        control_updates=250, selection='Every predefined arm at final250, without best-checkpoint selection',
        primary_joint_candidates=[name for name, value in arms.items() if value['primary']['all_three_improve']],
        protocols_kept_separate=True, heldout_digest_keys_checked=list(HELDOUT_KEYS),
        optimizer_updates_by_reviewer=0, model_inference_repeated=False, checkpoint_payloads_reread=False,
        online_log_verification_pending=True, code=file_ref(Path(__file__)),
        helper=file_ref(Path(__file__).with_name('review_corpus_validation.py')),
        interpretation='Single-seed fixed-budget contrasts on reused validation. A joint-positive result '
                       'requires replication; these controls do not establish the cause of the sequential production regression.')
    publish_json(output/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completion', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.completion.resolve(), args.output.resolve())
