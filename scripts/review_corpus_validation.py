"""Compare completed validation artifacts without running model inference.

Keep original-resolution, fixed-bucket, LiDAR and DSLR protocols separate.
Every comparison binds exact image order, frozen-teacher results and heldout
digests to the previous reviewed evaluation. This never changes training.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from statistics import mean
import time

from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import publish_json


HELDOUT_KEYS = ('val', 'geometry_val', 'geometry_val_rgb',
                'dataset/scannetpp-dslr', 'dataset/scannetpp-lidar')


def paired_rows(reference, current, identity=('image', 'dataset'), fixed_keys=None):
    def identities(rows):
        return [tuple(row[k] for k in identity) for row in rows]
    old, new = identities(reference), identities(current)
    if not new or len(new) != len(set(new)) or old != new:
        raise ValueError('Validation image identity/order differs or contains duplicates')
    for a, b in zip(reference, current):
        fixed_a = {k: v for k, v in a.items() if k.startswith('gt_teacher/')}
        fixed_b = {k: v for k, v in b.items() if k.startswith('gt_teacher/')}
        if fixed_keys is not None:
            fixed_a = {k: a[k] for k in fixed_keys}
            fixed_b = {k: b[k] for k in fixed_keys}
        if fixed_a != fixed_b:
            raise ValueError('Frozen teacher or fixed GT support differs')
        if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in b.values()):
            raise ValueError('Nonfinite completed validation result')


def extend_metric(history, value, step, previous_step):
    if not math.isfinite(value):
        raise ValueError('Nonfinite validation metric')
    result = {k: v for k, v in history.items()
              if k == 'before' or k.startswith('step')}
    key = f'step{step}'
    if key in result or f'step{previous_step}' not in result:
        raise ValueError('Current step already reviewed or previous metric is missing')
    result[key] = value
    for baseline in ('before', f'step{previous_step}'):
        if baseline in result:
            label = baseline.removeprefix('step')
            result[f'change_from_{label}'] = value - result[baseline]
            result[f'percent_from_{label}'] = (100 * (value / result[baseline] - 1)
                                                if result[baseline] else None)
    return result


def check_aggregates(rows, metrics):
    if metrics['images'] != len(rows):
        raise ValueError('Stored image count differs')
    sources = sorted({r['dataset'] for r in rows})
    for name in ('spatial/cosine', 'summary/cosine', 'teacher_depth/abs_rel'):
        macro = mean(mean(r[name] for r in rows if r['dataset'] == d) for d in sources)
        if not math.isclose(macro, metrics['macro/' + name], rel_tol=0, abs_tol=1e-12):
            raise ValueError('Stored macro metric differs from actual per-image results')
    for name in ('gt_student/abs_rel', 'gt_student/rmse', 'gt_student/delta1'):
        values = [r[name] for r in rows if name in r]
        if values and not math.isclose(mean(values), metrics[name], rel_tol=0, abs_tol=1e-12):
            raise ValueError('Stored measured GT aggregate differs')


def run(previous_path, training, step, output):
    previous_ref = file_ref(previous_path)
    previous = checked_json(previous_ref)
    if previous['stage'] != 'validation_comparison_complete' or previous['step'] >= step:
        raise ValueError('A completed earlier paired review is required')
    prior_step = previous['step']
    current_key, prior_key = f'step{step}', f'step{prior_step}'
    provenance_ref = file_ref(training / 'provenance.json')
    provenance, parent = checked_json(provenance_ref), checked_json(previous['provenance'])
    if (provenance['revisions'] != parent['revisions'] or provenance['adapter'] != parent['adapter']
            or any(provenance['data_digests'][k] != parent['data_digests'][k] for k in HELDOUT_KEYS)):
        raise ValueError('Model revisions, adapter definition or heldout digests differ')
    records = [json.loads(line) for line in (training / 'validation.jsonl').read_text().splitlines()]
    selected = [r for r in records if r['step'] == step]
    if len(selected) != 1:
        raise ValueError('Exactly one completed validation record is required')
    record = selected[0]
    destination = training / f'eval-{step:07d}'
    views = {}
    for name, old in previous['views'].items():
        path = destination if name == 'original' else destination / name
        refs = {n: file_ref(path / (n + '.json')) for n in ('metrics', 'per_image', 'protocol')}
        metrics, rows, protocol = (checked_json(refs[n]) for n in ('metrics', 'per_image', 'protocol'))
        checked_path(old['manifest'])
        if protocol != checked_json(old['protocol_file']) or protocol != old['protocol']:
            raise ValueError('Evaluation preprocessing/protocol differs')
        for ref in (old['before_per_image'], old[prior_key + '_per_image']):
            paired_rows(checked_json(ref), rows)
        before_metrics, parent_metrics = checked_json(old['before']), checked_json(old[prior_key])
        check_aggregates(rows, metrics)
        value = deepcopy(old)
        value.update(protocol_file=refs['protocol'], same_image_order=True,
                     frozen_teacher_gt_unchanged=True, aggregate_recomputed=True)
        value[current_key] = refs['metrics']
        value[current_key + '_per_image'] = refs['per_image']
        value[current_key + '_protocol'] = refs['protocol']
        for key, history in old['metrics'].items():
            if history['before'] != before_metrics[key] or history[prior_key] != parent_metrics[key]:
                raise ValueError('Previous metric history differs from its bound result')
            value['metrics'][key] = extend_metric(history, metrics[key], step, prior_step)
            record_key = key if name == 'original' else (
                'resolution/native-640x400/' + key if name == 'native-640x400' else
                'dataset/' + name.removeprefix('dataset-') + '/' + key)
            if record.get(record_key) != metrics[key]:
                raise ValueError('Completed record differs from saved view metrics: ' + record_key)
        sources = {r['dataset'] for r in rows}
        if sources != set(old['datasets']):
            raise ValueError('Source coverage differs')
        for dataset, previous_dataset in old['datasets'].items():
            subset = [r for r in rows if r['dataset'] == dataset]
            if len(subset) != previous_dataset['images']:
                raise ValueError('Dataset image count differs')
            for key, history in previous_dataset['metrics'].items():
                measured = mean(r[key] for r in subset if key in r)
                value['datasets'][dataset]['metrics'][key] = extend_metric(history, measured, step, prior_step)
        views[name] = value
    old_laser = previous['known_camera_laser']
    laser_refs = {n: file_ref(destination / 'gt-laser-known-camera' / (n + '.json'))
                  for n in ('metrics', 'per_image', 'protocol')}
    laser, laser_rows, laser_protocol = (checked_json(laser_refs[n]) for n in
                                        ('metrics', 'per_image', 'protocol'))
    if laser_protocol != checked_json(old_laser['current']['protocol']):
        raise ValueError('Known-camera laser protocol differs')
    paired_rows(checked_json(old_laser['current']['per_image']), laser_rows,
                ('id', 'scene'), ('gt_valid_fraction', 'gt_plane_fraction'))
    if len(laser_rows) != old_laser['images'] or laser['images'] != len(laser_rows):
        raise ValueError('Known-camera image count differs')
    laser_report = deepcopy(old_laser)
    laser_report[prior_key] = old_laser['current']
    laser_report['current'] = laser_refs
    for key, history in old_laser['metrics'].items():
        if (not math.isclose(mean(row[key] for row in laser_rows), laser[key], rel_tol=0, abs_tol=1e-12)
                or record['gt_laser_known_camera/' + key] != laser[key]):
            raise ValueError('Known-camera aggregate differs from actual per-image results')
        laser_report['metrics'][key] = extend_metric(history, laser[key], step, prior_step)
    output.mkdir(exist_ok=False)
    publish_json(output / 'validation-record.json', record)
    report = dict(stage='validation_comparison_complete', step=step, created_at=time.time(),
        baseline=f'Archived feature step232288; immediate previous review step{prior_step}',
        cosine_definition='Mean within each dataset, followed by equal-weight mean across datasets; identical images and preprocessing.',
        views=views, known_camera_laser=laser_report, prior_comparison=previous_ref,
        provenance=provenance_ref, parent_provenance=previous['provenance'],
        frozen_model_revisions_unchanged=True, heldout_digest_keys_checked=list(HELDOUT_KEYS),
        completed_validation_record=file_ref(output / 'validation-record.json'),
        model_inference_repeated=False, optimizer_or_training_configuration_modified=False,
        interpretation='Task-specific paired changes only. Original and fixed-bucket protocols remain separate; no causal, statistical-significance or overall-improvement claim.',
        reviewer_code=file_ref(Path(__file__)))
    checked_path(previous_ref)
    checked_path(provenance_ref)
    publish_json(output / 'report.json', report)
    print(json.dumps(dict(report=file_ref(output / 'report.json'), step=step,
        original={k: v for k, v in views['original']['metrics'].items()},
        lidar=views['dataset-scannetpp-lidar']['metrics']['gt_student/abs_rel'],
        laser=laser_report['metrics']['abs_rel'])), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', required=True, type=Path)
    parser.add_argument('--training', required=True, type=Path)
    parser.add_argument('--step', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.previous.resolve(), args.training.resolve(), args.step, args.output.resolve())
