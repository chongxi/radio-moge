"""One fixed-final RGB reference review; laser is separate from its training loss."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import torch

from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.data import read_manifest, manifest_digest
from radio_moge.geometry_cache import publish_json
from radio_moge.gt_data import geometry_digest
from radio_moge.gt_evaluate import evaluate_ground_truth
from radio_moge.system import build_system
from review_corpus_controls import view_review, laser_review, joint_result
from review_corpus_validation import HELDOUT_KEYS


def context(contract_path):
    contract = checked_json(file_ref(contract_path))
    completed = checked_json(file_ref(contract_path.parent / 'completed.json'))
    request = checked_json(completed['request'])
    config = checked_json(request['config'])
    reference = checked_json(contract['reference_validation'])
    if (completed['stage'] != 'fixed250_reference_training_complete'
            or completed['optimizer_updates'] != 250 or not completed['final_state_finite']
            or not completed['fixed_statistics_unchanged'] or contract['endpoint'] != 232538
            or request['config'] != contract['reference_config']
            or config['train']['steps'] != contract['endpoint']):
        raise ValueError('Complete fixed250 reference with its predeclared evaluator required')
    return contract, completed, request, config, reference


def run_laser(contract_path, output, lock_path):
    contract, completed, request, config, reference = context(contract_path)
    records = read_manifest(checked_path(contract['laser_manifest']))
    digests = dict(geometry_val=geometry_digest(records), geometry_val_rgb=manifest_digest(records))
    prior = checked_json(reference['provenance'])
    if len(records) != 18 or any(v != prior['data_digests'][k] for k, v in digests.items()):
        raise ValueError('Known-camera heldout identities changed')
    checkpoint = checked_path(completed['final_checkpoint'])
    with lock_path.open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(exist_ok=False)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        if state['config'] != config or state['step'] != 232538 or state['revisions'] != prior['revisions']:
            raise ValueError('Wrong fixed final reference checkpoint/revisions')
        system, adapter = build_system(config, torch.device('cuda'))
        if adapter.config != state['adapter_config'] or system.model_config != state['model_config']:
            raise ValueError('Evaluator architecture differs')
        adapter.load_state_dict(state['adapter'])
        metrics = evaluate_ground_truth(system, adapter, records, config, output / 'evaluation')
        paired = laser_review(reference['known_camera_laser'], output / 'evaluation',
                             {'gt_laser_known_camera/' + k: v for k, v in metrics.items()})
        publish_json(output / 'report.json', dict(stage='reference_laser_evaluation_complete',
            fixed_checkpoint=completed['final_checkpoint'], contract=file_ref(contract_path),
            config=request['config'], heldout_digests=digests, review=paired,
            model_revisions=state['revisions'], optimizer_updates=0, optimizer_constructed=False,
            archived_laser_baseline_available=False, completed_at=time.time()))


def review(contract_path, output, laser_path):
    contract, completed, request, config, reference = context(contract_path)
    plan = checked_json(request['plan'])
    directory = Path(config['output_dir'])
    provenance_ref = file_ref(directory / 'provenance.json')
    provenance, prior = checked_json(provenance_ref), checked_json(reference['provenance'])
    laser_ref = file_ref(laser_path)
    laser = checked_json(laser_ref)
    combined = dict(provenance['data_digests'], **laser['heldout_digests'])
    if (provenance['revisions'] != prior['revisions'] or provenance['adapter'] != prior['adapter']
            or any(combined[k] != prior['data_digests'][k] for k in HELDOUT_KEYS)
            or laser['fixed_checkpoint'] != completed['final_checkpoint']
            or laser['stage'] != 'reference_laser_evaluation_complete'
            or laser['contract'] != file_ref(contract_path)):
        raise ValueError('Reference model, evaluation identity or frozen contract differs')
    audit = checked_json(completed['actual_inputs'])
    checked_path(audit['records'])
    if (audit['optimizer_updates'] != 250 or audit['cursor'] != 2000 or audit['presentations'] != 8000
            or audit['source_counts'] != plan['planned_source_counts']
            or audit['resolution_counts'] != plan['planned_resolution_counts']
            or not audit['planned_keys_matched'] or not audit['actual_compute_groups_verified']):
        raise ValueError('Actual reference input consumption differs from predeclared stream')
    rows = [json.loads(x) for x in (directory / 'validation.jsonl').read_text().splitlines()]
    if len(rows) != 1 or rows[0]['step'] != 232538:
        raise ValueError('Exactly one fixed final reference evaluation required')
    destination = directory / 'eval-0232538'
    views = {name: view_review(old, destination if name == 'original' else destination / name,
                              name, rows[0]) for name, old in reference['views'].items()}
    keys = dict(spatial_cosine=('original', 'macro/spatial/cosine'),
                summary_cosine=('original', 'macro/summary/cosine'),
                lidar_abs_rel=('dataset-scannetpp-lidar', 'gt_student/abs_rel'))
    baseline = {k: views[v]['metrics'][m]['before'] for k, (v, m) in keys.items()}
    after = {k: views[v]['metrics'][m]['final'] for k, (v, m) in keys.items()}
    expected = dict(spatial_cosine=plan['primary_joint_rule']['spatial_cosine_gt'],
                    summary_cosine=plan['primary_joint_rule']['summary_cosine_gt'],
                    lidar_abs_rel=plan['primary_joint_rule']['scannet_lidar_absrel_lt'])
    if baseline != expected:
        raise ValueError('Predeclared primary baseline changed')
    result = dict(stage='fixed250_original_feature_reference_reviewed', completed_at=time.time(),
        contract=file_ref(contract_path), completion=file_ref(contract_path.parent / 'completed.json'),
        provenance=provenance_ref, baseline_review=contract['reference_validation'],
        actual_input_audit=completed['actual_inputs'], views=views, known_camera_laser=laser_ref,
        primary=joint_result(baseline, after), optimizer_updates_by_reviewer=0,
        rgb_inference_repeated=False, fixed_final_not_best_selected=True,
        heldout_digest_keys_checked=list(HELDOUT_KEYS), online_verification_pending=True,
        interpretation='Original feature recipe restored together, with preserved Adam/model RNG and a '
            'fresh clean-manifest cursor. This tests recipe recovery, not individual-factor causality. '
            'Single-seed fixed250 result on reused validation; a positive requires replication.')
    output.mkdir(exist_ok=False)
    publish_json(output / 'report.json', result)
    print(json.dumps(result['primary']), flush=True)
    return result


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
            parser.error('Completed fixed-final known-camera evaluation required')
        review(args.contract.resolve(), args.output.resolve(), args.laser_report.resolve())
