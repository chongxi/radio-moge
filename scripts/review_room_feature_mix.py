"""Paired saved-only review of real holdouts and unseen empty-room houses."""
import argparse
import json
from pathlib import Path
import time

from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import publish_json
from review_corpus_controls import change, joint_result
from review_corpus_validation import paired_rows
from review_projection_pair_arm import VIEWS, checked_json, metrics_for


def compare_saved(before_refs, directory, log, prefix):
    refs = {k: file_ref(directory / (k+'.json')) for k in ('metrics', 'per_image', 'protocol')}
    before = {k: checked_json(v) for k, v in before_refs.items()}
    now = {k: checked_json(v) for k, v in refs.items()}
    if before['protocol'] != now['protocol']:
        raise ValueError('Paired validation protocol changed')
    paired_rows(before['per_image'], now['per_image'])
    if metrics_for(now['per_image']) != now['metrics'] or set(now['metrics']) != set(before['metrics']):
        raise ValueError('Per-image metrics or metric coverage changed')
    if any(log.get(prefix+k) != v for k, v in now['metrics'].items()):
        raise ValueError('Committed validation log differs from its saved result')
    return dict(files=refs, before=before_refs, images=len(now['per_image']),
        exact_images_protocol_teacher_support=True,
        metrics={k: change(before['metrics'][k], value) for k, value in now['metrics'].items()})


def run(root, step, output):
    plan = checked_json(file_ref(root / 'plan.json'))
    parent = checked_json(plan['source_results'])
    provenance_ref = file_ref(root / 'training/provenance.json')
    provenance = checked_json(provenance_ref)
    old = checked_json(parent['provenance'])
    initial = checked_json(file_ref(root / 'initial-verification.json'))
    metadata = checked_json(file_ref(root / 'metadata-report.json'))
    config = checked_json(plan['config'])
    if (provenance['revisions'] != old['revisions'] or provenance['adapter'] != old['adapter']
            or initial['step'] != parent['step'] or initial['optimizer_states'] != 100
            or initial['cursor'] != 0 or initial['real_stream_start_cursor'] != plan['source_cursor']
            or config['loss'] != dict(spatial=1., summary=1., cosine=1., summary_cosine=1.)
            or any(k in config for k in ('geometry_supervision', 'ordinary_geometry', 'feature_replay'))):
        raise ValueError('Source model, actual initial state or feature-only contract differs')
    digests = dict(provenance['data_digests'])
    if set(digests) != set(old['data_digests']) | {'dataset/procthor-empty', 'dataset/structural-rooms'}:
        raise ValueError('Unexpected validation population change')
    digests.pop('dataset/procthor-empty')
    digests.pop('dataset/structural-rooms')
    if (digests.pop('train') != metadata['mixed_digest']
            or digests != {k:v for k,v in old['data_digests'].items() if k != 'train'}):
        raise ValueError('Original heldout identities changed')
    logs = [json.loads(line) for line in (root / 'training/validation.jsonl').read_text().splitlines()]
    chosen = [row for row in logs if row['step'] == step]
    updates = step - parent['step']
    if len(chosen) != 1 or not 0 < updates <= plan['additional_updates']:
        raise ValueError('A completed validation in the declared budget is required')
    log = chosen[0]
    planned = checked_json(file_ref(root / 'planned-keys.json'))
    actual, synthetic = 0, 0
    with (root / 'training/actual-rgb-inputs.jsonl').open() as file:
        for number, line in enumerate(file):
            if number == updates:
                break
            row = json.loads(line)
            images = row['logical_images']
            if (row['update'] != parent['step']+number+1 or len(images) != 32
                    or [r['key'] for r in images] != planned[number*32:(number+1)*32]
                    or sum(r['dataset'] in ('procthor_empty', 'structural_rooms') for r in images) != 4
                    or any(r['dataset'] == 'coco2017' or 'ordinary_geometry' in r for r in images)):
                raise ValueError('Actual consumed mixture/keys differ from the frozen RGB plan')
            actual += 32
            synthetic += 4
    if actual != updates*32:
        raise ValueError('Missing committed RGB input audit')
    destination = root / 'training' / f'eval-{step:07d}'
    views = {}
    for name in VIEWS:
        path = destination if name == 'original' else destination / name
        prefix = '' if name == 'original' else ('resolution/native-640x400/'
            if name == 'native-640x400' else 'dataset/'+name.removeprefix('dataset-')+'/')
        views[name] = compare_saved(parent['views'][name]['files'], path, log, prefix)
    baseline = checked_json(plan['procthor_baseline'])
    if baseline['validation'] != plan['validation_manifest'] or baseline['source_step'] != parent['step']:
        raise ValueError('Saved official-house baseline differs from the parent or val population')
    views['dataset-procthor-empty'] = compare_saved(plan['procthor_baseline_files'],
        destination / 'dataset-procthor-empty', log, 'dataset/procthor-empty/')
    structural = checked_json(file_ref(root / 'structural-baseline.json'))
    if (structural['validation'] != plan['structural_validation_manifest']
            or structural['source_step'] != parent['step']):
        raise ValueError('New structural baseline differs from the parent or fixed val population')
    refs = {k: file_ref(root / 'structural-baseline' / (k+'.json'))
            for k in ('metrics', 'per_image', 'protocol')}
    views['dataset-structural-rooms'] = compare_saved(refs,
        destination / 'dataset-structural-rooms', log, 'dataset/structural-rooms/')
    names = dict(spatial_cosine=('original', 'macro/spatial/cosine'),
        summary_cosine=('original', 'macro/summary/cosine'),
        lidar_abs_rel=('dataset-scannetpp-lidar', 'gt_student/abs_rel'))
    before = {k: views[v]['metrics'][m]['before'] for k,(v,m) in names.items()}
    current = {k: views[v]['metrics'][m]['final'] for k,(v,m) in names.items()}
    archive = {k: parent['against_archive']['metrics'][k]['before'] for k in names}
    output.mkdir(exist_ok=False)
    report = dict(stage='saved_empty_room_mix_validation_reviewed', step=step, new_updates=updates,
        fixed1000_complete=updates == plan['additional_updates'], parent=plan['source_results'],
        provenance=provenance_ref, plan=file_ref(root / 'plan.json'), views=views,
        against_parent=joint_result(before, current), against_archive=joint_result(archive, current),
        actual_rgb_presentations=actual, actual_synthetic_presentations=synthetic,
        synthetic_fraction=synthetic/actual, actual_planned_keys_matched=True,
        model_inference=False, checkpoint_payload_read=False, optimizer_updates_by_reviewer=0,
        significance_or_causality_established=False, real_world_planarity_improvement_established=False,
        reviewed_at=time.time(), reviewer=file_ref(Path(__file__)))
    publish_json(output / 'report.json', report)
    print(json.dumps(dict(report=file_ref(output / 'report.json'), step=step,
        against_parent=report['against_parent'], against_archive=report['against_archive'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.step, args.output.resolve())
