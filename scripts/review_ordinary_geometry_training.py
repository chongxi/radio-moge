"""Review the fixed ordinary-geometry endpoint against archive and RGB-only reference.

The review consumes saved RGB evaluations. A separate explicit laser mode evaluates
only the new final checkpoint; no archived/reference inference is repeated.
"""
import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import time

import torch

from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.data import manifest_digest, read_manifest
from radio_moge.geometry_cache import publish_json
from radio_moge.gt_data import geometry_digest
from radio_moge.gt_evaluate import evaluate_ground_truth
from radio_moge.system import build_system
from review_corpus_controls import change, joint_result, laser_review, view_review
from review_corpus_validation import HELDOUT_KEYS


def reference_descriptor(archived, reviewed):
    value = deepcopy(archived)
    value.update(before=reviewed['files']['metrics'], before_per_image=reviewed['files']['per_image'],
                 protocol_file=reviewed['files']['protocol'], protocol=reviewed['protocol'])
    if value['images'] != reviewed['images'] or set(value['datasets']) != set(reviewed['datasets']):
        raise ValueError('RGB-only reference scope differs from archive')
    value['datasets'] = {name: dict(images=row['images'],
        metrics={key: dict(before=metric['final']) for key, metric in row['metrics'].items()})
        for name, row in reviewed['datasets'].items()}
    return value


def compare_input_rows(reference, current, expected_updates=250):
    if len(reference) != expected_updates or len(current) != expected_updates:
        raise ValueError('Fixed-budget actual input accounting is incomplete')
    for original, annotated in zip(reference, current, strict=True):
        value = deepcopy(annotated)
        for row in value['logical_images']:
            annotation = row.pop('ordinary_geometry', None)
            if not isinstance(annotation, dict) or annotation.get('augment') is not True:
                raise ValueError('Actual ordinary input annotation is missing')
        if value != original:
            raise ValueError('Actual RGB pixels, identities or prepared groups differ from reference02')


def context(contract_path):
    contract = checked_json(file_ref(contract_path))
    completion_ref = file_ref(contract_path.parent/'completed.json')
    completed = checked_json(completion_ref)
    request = checked_json(contract['training_request'])
    config = checked_json(request['config'])
    initial = checked_json(completed['actual_initial'])
    reference = checked_json(contract['feature_reference_review'])
    archived = checked_json(reference['baseline_review'])
    if (completed['stage'] != 'ordinary_fixed250_training_complete'
            or completed['request'] != contract['training_request']
            or completed['optimizer_updates'] != 250 or completed['final_state_finite'] is not True
            or completed['fixed_statistics_unchanged'] is not True
            or config['train']['steps'] != 232538 or contract['endpoint'] != 232538
            or config['ordinary_geometry']['multiplier'] != 1
            or initial['stage'] != 'ordinary_actual_initial_state_verified'
            or initial['adapter_rms_adam_rng_exact'] is not True or initial['source_step'] != 232288
            or reference['stage'] != 'fixed250_original_feature_reference_reviewed'
            or contract['selection'] != 'fixed_final_not_best_selected'):
        raise ValueError('Complete sealed fixed250 ordinary run and original reference required')
    expected = {key: value['before'] for key, value in reference['primary']['metrics'].items()}
    if contract['primary_archive_baseline'] != expected:
        raise ValueError('Predeclared joint baseline changed')
    return contract, completion_ref, completed, request, config, reference, archived


def run_laser(contract_path, output, lock_path):
    contract, _, completed, request, config, reference, archived = context(contract_path)
    records = read_manifest(checked_path(contract['laser_manifest']))
    digests = dict(geometry_val=geometry_digest(records), geometry_val_rgb=manifest_digest(records))
    prior = checked_json(archived['provenance'])
    if len(records) != 18 or any(v != prior['data_digests'][k] for k, v in digests.items()):
        raise ValueError('Known-camera heldout identities changed')
    checkpoint = checked_path(completed['final_checkpoint'])
    with lock_path.open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(exist_ok=False)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        if state['config'] != config or state['step'] != 232538 or state['revisions'] != prior['revisions']:
            raise ValueError('Wrong fixed final ordinary checkpoint or model revisions')
        system, adapter = build_system(config, torch.device('cuda'))
        if adapter.config != state['adapter_config'] or system.model_config != state['model_config']:
            raise ValueError('Evaluator architecture differs')
        adapter.load_state_dict(state['adapter'])
        metrics = evaluate_ground_truth(system, adapter, records, config, output/'evaluation')
        paired = laser_review(archived['known_camera_laser'], output/'evaluation',
                             {'gt_laser_known_camera/'+k: v for k, v in metrics.items()})
        publish_json(output/'report.json', dict(stage='ordinary_final_laser_evaluation_complete',
            fixed_checkpoint=completed['final_checkpoint'], contract=file_ref(contract_path),
            config=request['config'], heldout_digests=digests, review=paired,
            model_revisions=state['revisions'], optimizer_updates=0, optimizer_constructed=False,
            archived_laser_baseline_available=False, completed_at=time.time()))


def review(contract_path, output, laser_path):
    contract, completion_ref, completed, request, config, reference, archived = context(contract_path)
    directory = Path(config['output_dir'])
    provenance_ref = file_ref(directory/'provenance.json')
    provenance, prior = checked_json(provenance_ref), checked_json(archived['provenance'])
    reference_provenance = checked_json(reference['provenance'])
    laser_ref = file_ref(laser_path)
    laser = checked_json(laser_ref)
    combined = dict(provenance['data_digests'], **laser['heldout_digests'])
    if (laser['stage'] != 'ordinary_final_laser_evaluation_complete'
            or laser['contract'] != file_ref(contract_path)
            or laser['fixed_checkpoint'] != completed['final_checkpoint']
            or provenance['revisions'] != prior['revisions']
            or provenance['revisions'] != reference_provenance['revisions']
            or provenance['adapter'] != prior['adapter']
            or provenance['adapter'] != reference_provenance['adapter']
            or any(combined[k] != prior['data_digests'][k] for k in HELDOUT_KEYS)):
        raise ValueError('Actual model, evaluation identities or fixed endpoint differs')
    audit, old_audit = checked_json(completed['actual_inputs']), checked_json(reference['actual_input_audit'])
    for proof in (audit, old_audit):
        if (proof['optimizer_updates'] != 250 or proof['cursor'] != 2000 or proof['presentations'] != 8000
                or proof['planned_keys_matched'] is not True or proof['actual_compute_groups_verified'] is not True):
            raise ValueError('Actual fixed250 input audit is incomplete')
    def lines(ref):
        return [json.loads(row) for row in checked_path(ref).read_text().splitlines()]
    compare_input_rows(lines(old_audit['records']), lines(audit['records']))
    records = lines(file_ref(directory/'validation.jsonl'))
    if len(records) != 1 or records[0]['step'] != 232538:
        raise ValueError('Exactly one completed fixed-final RGB evaluation required')
    destination = directory/'eval-0232538'
    comparisons = {}
    keys = dict(spatial_cosine=('original', 'macro/spatial/cosine'),
                summary_cosine=('original', 'macro/summary/cosine'),
                lidar_abs_rel=('dataset-scannetpp-lidar', 'gt_student/abs_rel'))
    for label in ('archive', 'reference02'):
        descriptors = archived['views'] if label == 'archive' else {
            name: reference_descriptor(old, reference['views'][name]) for name, old in archived['views'].items()}
        views = {name: view_review(old, destination if name == 'original' else destination/name,
                                  name, records[0]) for name, old in descriptors.items()}
        for view in views.values():
            view.pop('aggregate_recomputed')
            view.update(selected_aggregates_recomputed=True, all_saved_metrics_checked_against_validation_log=True)
        before = {key: views[view]['metrics'][metric]['before'] for key, (view, metric) in keys.items()}
        final = {key: views[view]['metrics'][metric]['final'] for key, (view, metric) in keys.items()}
        comparisons[label] = dict(views=views, primary=joint_result(before, final))
    old_laser = checked_json(reference['known_camera_laser'])['review']
    old_metrics, new_metrics = old_laser['metrics'], laser['review']['metrics']
    if set(old_metrics) != set(new_metrics):
        raise ValueError('Known-camera laser metric coverage differs')
    # Validate reference02 identities/support directly as well as the archived protocol.
    laser_review(dict(current=old_laser['files'], images=old_laser['images']),
        Path(laser['review']['files']['metrics']['path']).parent,
        {'gt_laser_known_camera/'+k: v for k, v in new_metrics.items()})
    output.mkdir(exist_ok=False)
    report = dict(stage='fixed250_ordinary_geometry_reviewed', completed_at=time.time(),
        contract=file_ref(contract_path), completion=completion_ref, provenance=provenance_ref,
        baseline_reviews=dict(archive=reference['baseline_review'], reference02=contract['feature_reference_review']),
        comparisons=comparisons, actual_input_audit=completed['actual_inputs'],
        actual_rgb_pixels_keys_and_groups_match_reference02=True,
        known_camera_laser=dict(report=laser_ref, archived_baseline_available=False,
            versus_reference02={k: change(old_metrics[k], new_metrics[k]) for k in old_metrics}),
        fixed_final_not_best_selected=True, heldout_digest_keys_checked=list(HELDOUT_KEYS),
        model_inference_repeated=False, checkpoint_payloads_reread=False, optimizer_updates_by_reviewer=0,
        online_verification_pending=True, reviewer=file_ref(Path(__file__)),
        interpretation='Single-seed fixed250 comparison with identical actual RGB presentations. '
            'Both original-resolution cosines must rise and independent LiDAR AbsRel must fall. '
            'Known-camera laser is separate; no significance or historical root-cause claim.')
    publish_json(output/'report.json', report)
    print(json.dumps({k: v['primary'] for k, v in comparisons.items()}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('laser', 'review'))
    parser.add_argument('--contract', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--lock', type=Path)
    parser.add_argument('--laser-report', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == 'laser':
        if args.lock is None:
            parser.error('Existing GPU lock required')
        run_laser(args.contract.resolve(), args.output.resolve(), args.lock.resolve())
    else:
        if args.laser_report is None:
            parser.error('Completed candidate laser report required')
        review(args.contract.resolve(), args.output.resolve(), args.laser_report.resolve())
