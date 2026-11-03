"""Freeze the reviewed six-source continuation, without starting an optimizer.

The previous completed stage supplies unchanged loss implementation evidence.
New source, crop, sampling and canonical isolation evidence are required before
publishing readiness. The launcher still requires an actual model backward probe.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
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


WEIGHTS = dict(coco2017=.224, hypersim=.224, irs=.16, nyuv2=.032,
               structured3d=.16, scannetpp_dslr=.20)


def run(root):
    data = root / 'reviewed-stage3-data-01'
    accounting_ref = file_ref(data / 'accounting/report.json')
    accounting = checked_json(accounting_ref)
    isolation_ref = file_ref(data / 'isolation-report.json')
    isolation = checked_json(isolation_ref)
    crop_ref = file_ref(root / 'scannetpp-teacher-training-crops-01/report.json')
    crop = checked_json(crop_ref)
    source_ref = file_ref(root / 'scannetpp-teacher-source-reviews-01/report.json')
    source = checked_json(source_ref)
    finish_ref = file_ref(root / 'reviewed-stage2-finish-01/report.json')
    finish = checked_json(finish_ref)
    previous_checks = checked_json(file_ref(root / 'reviewed-stage2-readiness-01/plan.json'))['checks']
    loss = checked_json(previous_checks['loss_tests'])
    if (accounting['selected_images'] != 403382 or accounting['total_images'] != 404795
            or accounting['unresolved'] != 0 or accounting['selected_counts']['scannetpp_dslr'] != 35783
            or isolation['stage'] != 'expanded_selection_bound_to_full_canonical_comparison'
            or isolation['accounting'] != accounting_ref or isolation['original_geometry_scope'] != 404795
            or isolation['exact_pixel_overlap_pairs'] != 0 or isolation['forbidden_shared_scene_rows'] != 0
            or crop['stage'] != 'candidate_crop_support_observed_not_admitted'
            or (crop['images'], crop['groups'], crop['views'], crop['sparse_geometry_views']) != (100, 25, 300, 0)
            or not crop['actual_flip_exercised_per_image'] or not crop['original_mask_unchanged']
            or crop['max_native_to_target_relative_error'] > 1e-6
            or source['stage'] != 'scannetpp_teacher_source_reviews_complete_not_admitted'
            or source['source_eligible_images'] != 35783 or source['native_source_flags_cleared'] is not False
            or finish['stage'] != 'completed_training_checkpoint_verified' or finish['step'] != 2000
            or loss['implementation_sha256'] != implementation_hashes()):
        raise ValueError('Completed exact new source, crop, isolation, parent and unchanged implementation required')
    parent = checked_json(finish['config'])
    checked_path(finish['checkpoint'])
    paired = checked_json(finish['paired_validation'])
    validation_ref = file_ref(root / 'reviewed-stage2-validation-2000-review-01/validation-record.json')
    validation = checked_json(validation_ref)
    if validation['step'] != 2000 or validation_ref not in paired.values():
        raise ValueError('Checkpoint selection baseline must be the already reviewed parent validation')
    config = deepcopy(parent)
    config['data'].pop('geometry_corpus')
    config['data']['mixture_weights'] = WEIGHTS
    config['output_dir'] = str(root / 'reviewed-stage3-training-01')
    config['train'].update(steps=3000, max_seconds=7200)
    for entry in config['evaluation']['checkpoint_metrics'].values():
        entry['baseline'] = validation[entry['metric']]
    config['tracking'].update(group='geometry-stage3-20261103', name='geometry-reviewed-stage3-403382-20261103',
        tags=['geometry-reviewed-stage3', 'six-geometry-sources', 'seven-rgb-replay-sources',
              'adapter-only', 'parent-geometry-step2000', 'moge2-core', 'relative-pseudo-separated',
              'source-scene-balanced', 'constant-lr1e-5', 'preserved-adam'])
    validate_config(config, corpus_preflight=True)
    replay = checked_json(isolation['unchanged_replay'])
    if (config['feature_replay'] != parent['feature_replay'] or replay['images'] != 796477
            or file_ref(config['feature_replay']['manifest']) != replay['replay']):
        raise ValueError('Exact previously isolated RGB replay must remain unchanged')
    request_dir, ready_dir = root / 'reviewed-stage3-request-01', root / 'reviewed-stage3-readiness-01'
    if any(p.exists() for p in (request_dir, ready_dir, Path(config['output_dir']), root / 'reviewed-stage3-launch-01')):
        raise FileExistsError('Stage output already exists; inspect it instead of relaunching')
    with tempfile.TemporaryDirectory(prefix='six-source-sampling-') as temporary:
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
        source_step=2000, accounting_report=accounting_ref, expected_counts=accounting['expected_counts'],
        readiness_plan=str(ready_dir / 'plan.json'), execution_root=str(root / 'code-reviewed-continuation-v1'),
        execution_commit='8332c8c', launch_directory=str(root / 'reviewed-stage3-launch-01'),
        gpu_lock=str(root.parent / 'data-expansion-attention-20261022/attention-24h/.training.lock'),
        parent_finish=finish_ref, parent_config=finish['config'], paired_validation=finish['paired_validation'],
        requested_additional_updates=1000, requested_max_seconds=7200, training_started=False, optimizer_updates=0,
        changes='Add only 35783 reviewed ScanNet relative teacher labels at20% of geometry draws; old shares x0.8. '
                'Preserve Adam, LR, losses, replay and validation. New stream starts at0, global step continues2000.')
    publish_json(request_dir / 'request.json', request)
    ready_dir.mkdir()
    base = dict(stage='complete', passed=True, accounting_report_sha256=accounting_ref['sha256'],
        index_sha256=accounting['index']['sha256'], manifest_sha256=accounting['manifest_sha256'],
        training_admission=False, optimizer_updates=0, global_full_corpus_acceptance=False, created_at=time.time())
    values = dict(
        source_quality=dict(unresolved=0, reviewed_images=404795,
            evidence=[accounting_ref, accounting['plan'], previous_checks['source_quality'], source_ref,
                      *source['source_reviews']],
            review='All six scoped sources reviewed; native/external quarantines and1413 exclusions retained. '
                   '189025 native synthetic and214357 relative teacher labels kept distinct; native ScanNet flags not cleared.'),
        coverage=dict(reviewed_counts=accounting['expected_counts'],
            evidence=[file_ref(data / 'coverage-summary.json'), previous_checks['coverage'], crop_ref,
                      file_ref(sampling_path), config_ref, isolation['unchanged_replay']],
            review='Reviewed25 added scenes,100 actual payload views and300 production crops with flips. '
                   'Minimum213306 exact original observations; no sparse views in selected ScanNet extremes. '
                   'Previous sparse crops retain zero geometry below32 observations and keep feature branch. '
                   'Scene counts do not assert independent captures; selection is not a prevalence estimate.'),
        heldout_isolation=dict(overlaps=0, evidence=[isolation_ref, isolation['full_comparison'],
            isolation['unchanged_replay'], isolation['replay_original_comparison']],
            review='All404795 scoped originals including excluded geometry checked against complete1637909 RGB '
                   'and190954 heldout memberships. Exact unchanged796477 replay reuses its completed canonical proof.'),
        loss_tests=dict(implementation_sha256=implementation_hashes(),
            evidence=[previous_checks['loss_tests'], *loss['evidence']], actual_model_forward_backward=False,
            prior_tests_reused_without_rerun=True,
            review='Exact implementation hashes match previously passed real losses and continuation tests. '
                   'New actual archived-model backward probe is mandatory before seal or optimizer.'))
    checks = {}
    for gate, value in values.items():
        for ref in value['evidence']:
            checked_path(ref)
        publish_json(ready_dir / (gate + '.json'), dict(base, gate=gate, **value))
        checks[gate] = file_ref(ready_dir / (gate + '.json'))
    validate_pre_model_evidence(accounting_ref, checks)
    publish_json(ready_dir / 'plan.json', dict(accounting_report=accounting_ref, checks=checks,
        training_config=config_ref, initial_checkpoint=finish['checkpoint'], source_step=2000,
        start_mode='corpus_data_fork'))
    print(json.dumps(dict(stage='nonmodel_readiness_complete', request=file_ref(request_dir / 'request.json'),
        plan=file_ref(ready_dir / 'plan.json'), actual_model_probe_complete=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    run(args.root.resolve())
