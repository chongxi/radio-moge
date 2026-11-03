"""Bind reviewed sensor supervision to the next bounded corpus continuation.

The parent may finish between evaluation intervals at its wall-clock limit.
Its actual final checkpoint and completed paired validation define this fork.
This publishes CPU readiness only; the launcher still requires a real backward
probe of every source/label stratum before issuing acceptance or training.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

from radio_moge.corpus_admission import (checked_json, checked_path, file_ref,
    implementation_hashes, validate_pre_model_evidence)
from radio_moge.data import MixtureBatches
from radio_moge.geometry_cache import publish_json
from radio_moge.train import validate_config


WEIGHTS = dict(arkitscenes=.20, coco2017=.1792, hypersim=.1792, irs=.128,
               nyuv2=.0256, structured3d=.128, scannetpp_dslr=.16)
COUNTS = dict(arkitscenes=194916, coco2017=39984, hypersim=59140, irs=133869,
              nyuv2=1043, structured3d=133563, scannetpp_dslr=35783)
LABELS = dict(measured_sensor=194916, synthetic_renderer=189025, teacher_pseudo=214357)


def validate_new_evidence(accounting, isolation, crop, source, coverage, finish, stop):
    if (accounting['selected_images'] != 598298 or accounting['total_images'] != 602337
            or accounting['unresolved'] != 0 or accounting['selected_counts'] != COUNTS
            or coverage['label_counts'] != LABELS
            or isolation['stage'] != 'expanded_selection_bound_to_full_canonical_comparison'
            or isolation['original_geometry_scope'] != 602337 or isolation['selected_geometry'] != 598298
            or isolation['exact_pixel_overlap_pairs'] != 0 or isolation['forbidden_shared_scene_rows'] != 0
            or crop['stage'] != 'candidate_crop_support_observed_not_admitted'
            or (crop['images'], crop['groups'], crop['views'], crop['sparse_views']) != (127, 99, 381, 6)
            or set(crop['rotation_images']) != {'0', '90', '180', '270'}
            or crop['actual_production_loader_used'] is not True
            or crop['original_support_unchanged'] is not True
            or crop['max_native_to_target_relative_error'] > 1e-6
            or crop['sparse_ordinals'] != [36153, 168525]
            or source['stage'] != 'arkit_source_review_and_routes_complete_not_corpus_admission'
            or source['original_images'] != 197542 or source['source_groups'] != 1478
            or source['source_disposition_counts'] != dict(native_source_eligible=194916, excluded_geometry=2626)
            or source['original_source_flags_retained'] is not True):
        raise ValueError('Exact reviewed scope, qualified sensor source, isolation and crop evidence required')
    if (finish['stage'] != 'completed_training_checkpoint_verified'
            or type(finish['step']) is not int or not 2000 < finish['step'] <= 3000
            or finish['new_optimizer_updates'] != finish['step'] - 2000
            or stop['step'] != finish['step'] or stop['final_evaluation'] != 'complete'
            or stop['reason'] not in ('step_limit', 'wall_clock_budget')):
        raise ValueError('The actual bounded parent and its final evaluation must be complete')


def stage_config(parent, validation, root, step):
    if validation['step'] != step:
        raise ValueError('Selection baselines must come from the actual final parent step')
    config = deepcopy(parent)
    config['data'].pop('geometry_corpus')
    config['data']['mixture_weights'] = dict(WEIGHTS)
    config['output_dir'] = str(root / 'reviewed-stage4-training-01')
    config['train'].update(steps=step + 1000, max_seconds=7200)
    for entry in config['evaluation']['checkpoint_metrics'].values():
        baseline = validation[entry['metric']]
        if not math.isfinite(baseline):
            raise ValueError('Parent validation baseline is nonfinite')
        entry['baseline'] = baseline
    config['tracking'].update(group='geometry-stage4-20261103', name='geometry-reviewed-stage4-598298-20261103',
        tags=['geometry-reviewed-stage4', 'seven-geometry-sources', 'seven-rgb-replay-sources',
              'adapter-only', f'parent-geometry-step{step}', 'moge2-core', 'qualified-sensor-depth',
              'relative-pseudo-separated', 'source-scene-balanced', 'constant-lr1e-5', 'preserved-adam'])
    return config


def run(root):
    data = root / 'reviewed-stage4-data-01'
    accounting_ref = file_ref(data / 'accounting/report.json')
    accounting = checked_json(accounting_ref)
    isolation_ref = file_ref(data / 'isolation-report.json')
    isolation = checked_json(isolation_ref)
    crop_ref = file_ref(root / 'arkit-training-crops-01/report.json')
    crop = checked_json(crop_ref)
    source_ref = file_ref(root / 'arkit-source-review-02/report.json')
    source = checked_json(source_ref)
    coverage_ref = file_ref(data / 'coverage-summary.json')
    coverage = checked_json(coverage_ref)
    finish_ref = file_ref(root / 'reviewed-stage3-finish-01/report.json')
    finish = checked_json(finish_ref)
    stop = checked_json(finish['stop_report'])
    validate_new_evidence(accounting, isolation, crop, source, coverage, finish, stop)
    if isolation['accounting'] != accounting_ref:
        raise ValueError('Isolation is bound to another accounting report')
    previous_checks = checked_json(file_ref(root / 'reviewed-stage3-readiness-01/plan.json'))['checks']
    loss = checked_json(previous_checks['loss_tests'])
    if loss['implementation_sha256'] != implementation_hashes():
        raise ValueError('Previously tested loss/training implementation changed')
    parent = checked_json(finish['config'])
    checked_path(finish['checkpoint'])
    paired = checked_json(finish['paired_validation'])
    validation_ref = file_ref(Path(finish['paired_validation']['path']).parent / 'validation-record.json')
    validation = checked_json(validation_ref)
    if validation_ref not in paired.values():
        raise ValueError('Actual final validation was not included in the paired review')
    config = stage_config(parent, validation, root, finish['step'])
    validate_config(config, corpus_preflight=True)
    replay = checked_json(isolation['unchanged_replay'])
    if (config['feature_replay'] != parent['feature_replay'] or replay['images'] != 796477
            or file_ref(config['feature_replay']['manifest']) != replay['replay']):
        raise ValueError('Exact previously isolated RGB replay must remain unchanged')
    request_dir, ready_dir = root / 'reviewed-stage4-request-01', root / 'reviewed-stage4-readiness-01'
    if any(p.exists() for p in (request_dir, ready_dir, Path(config['output_dir']), root / 'reviewed-stage4-launch-01')):
        raise FileExistsError('Stage output already exists; inspect it instead of relaunching')
    with tempfile.TemporaryDirectory(prefix='seven-source-sampling-') as temporary:
        local = Path(temporary) / 'accounting.sqlite'
        shutil.copyfile(checked_path(accounting['index']), local)
        if file_ref(local)['sha256'] != accounting['index']['sha256']:
            raise ValueError('Local sampling index copy differs')
        with sqlite3.connect(local) as db:
            rows = [dict(dataset=d, group=g) for d, g in db.execute(
                "SELECT dataset,scene_group FROM images WHERE disposition='selected' ORDER BY ordinal")]
        counts, groups = Counter(), Counter()
        for batch in MixtureBatches(rows, WEIGHTS, 1, [[400, 640]], 43, 0, 100000):
            row = rows[batch[0][0]]
            counts[row['dataset']] += 1
            groups[(row['dataset'], row['group'])] += 1
        observed = {k: counts[k] / 100000 for k in WEIGHTS}
        if (any(abs(observed[k] - v) > .005 for k, v in WEIGHTS.items())
                or sum(d == 'arkitscenes' for d, _ in groups) < .99 * 1478
                or sum(d == 'scannetpp_dslr' for d, _ in groups) != 25):
            raise ValueError('Actual sampler does not cover the reviewed mixture')
    sampling_path = data / 'sampling-probe-01.json'
    publish_json(sampling_path, dict(stage='sampling_observed_not_admitted', draws=100000,
        weights=WEIGHTS, observed_source_fractions=observed,
        distinct_groups={d: sum(source == d for source, _ in groups) for d in WEIGHTS},
        expected_selection=accounting['selected_counts'], accounting=accounting_ref,
        algorithm='Actual MixtureBatches: source weighted, scene uniform, frame uniform; with replacement',
        replay_unchanged=True, training_admission=False, optimizer_updates=0,
        implementation_sha256=implementation_hashes(), code=file_ref(Path(__file__))))
    request_dir.mkdir()
    publish_json(request_dir / 'training-config.json', config)
    config_ref = file_ref(request_dir / 'training-config.json')
    request = dict(scope_kind='reviewed_source_subset', start_mode='corpus_data_fork',
        training_authorized_after_checks=True, training_config=config_ref, initial_checkpoint=finish['checkpoint'],
        source_step=finish['step'], accounting_report=accounting_ref, expected_counts=accounting['expected_counts'],
        readiness_plan=str(ready_dir / 'plan.json'), execution_root=str(root / 'code-reviewed-continuation-v1'),
        execution_commit='8332c8c', launch_directory=str(root / 'reviewed-stage4-launch-01'),
        gpu_lock=str(root.parent / 'data-expansion-attention-20261022/attention-24h/.training.lock'),
        parent_finish=finish_ref, parent_config=finish['config'], paired_validation=finish['paired_validation'],
        requested_additional_updates=1000, requested_max_seconds=7200, training_started=False, optimizer_updates=0,
        changes='Add 194916 qualified ARKit measured-sensor targets at20% of geometry draws; old shares x0.8. '
                'Preserve Adam, LR, losses, replay and validation. New stream starts at0; global step continues '
                f'from actual bounded parent {finish["step"]}.')
    publish_json(request_dir / 'request.json', request)
    ready_dir.mkdir()
    base = dict(stage='complete', passed=True, accounting_report_sha256=accounting_ref['sha256'],
        index_sha256=accounting['index']['sha256'], manifest_sha256=accounting['manifest_sha256'],
        training_admission=False, optimizer_updates=0, global_full_corpus_acceptance=False, created_at=time.time())
    values = dict(
        source_quality=dict(unresolved=0, reviewed_images=602337,
            evidence=[accounting_ref, accounting['plan'], previous_checks['source_quality'], source_ref,
                      source['source_review'], source['candidate_quarantine'], source['source_routes']],
            review='Seven scoped sources reviewed; all native/external flags and4039 exclusions retained. '
                   '194916 noisy measured sensor,189025 synthetic and214357 relative pseudo labels stay distinct. '
                   'FARO comparisons do not establish whole-source physical accuracy; no residual bias fitted.'),
        coverage=dict(reviewed_counts=accounting['expected_counts'],
            evidence=[coverage_ref, previous_checks['coverage'], crop_ref,
                      file_ref(root / 'arkit-sparse-crop-visuals-01/visual-review.json'),
                      file_ref(sampling_path), config_ref, isolation['unchanged_replay']],
            review='127 ARKit images from99 visits,381 production crops,all4 rotations and flips reviewed. '
                   'Six sparse views from36153/168525 retain exact support: fewer than32 observations gives zero '
                   'geometry and retains feature supervision. Prior six-source crops reused. '
                   'No hole filling or prevalence/physical accuracy claim; scene counts are not independence counts.'),
        heldout_isolation=dict(overlaps=0, evidence=[isolation_ref, isolation['full_comparison'],
            isolation['unchanged_replay'], isolation['replay_original_comparison']],
            review='All602337 scoped originals checked against complete1637909 RGB and190954 heldout memberships. '
                   'Exact unchanged796477 replay retains completed canonical proof; no full near-duplicate claim.'),
        loss_tests=dict(implementation_sha256=implementation_hashes(),
            evidence=[previous_checks['loss_tests'], *loss['evidence']], actual_model_forward_backward=False,
            prior_tests_reused_without_rerun=True,
            review='Exact loss/training hashes match passed loss and continuation tests. '
                   'Actual final-parent model probe including new measured-sensor stratum remains mandatory.'))
    checks = {}
    for gate, value in values.items():
        for ref in value['evidence']:
            checked_path(ref)
        publish_json(ready_dir / (gate + '.json'), dict(base, gate=gate, **value))
        checks[gate] = file_ref(ready_dir / (gate + '.json'))
    validate_pre_model_evidence(accounting_ref, checks)
    publish_json(ready_dir / 'plan.json', dict(accounting_report=accounting_ref, checks=checks,
        training_config=config_ref, initial_checkpoint=finish['checkpoint'], source_step=finish['step'],
        start_mode='corpus_data_fork'))
    print(json.dumps(dict(stage='nonmodel_readiness_complete', request=file_ref(request_dir / 'request.json'),
        plan=file_ref(ready_dir / 'plan.json'), actual_model_probe_complete=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    run(args.root.resolve())
