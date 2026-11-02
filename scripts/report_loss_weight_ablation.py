"""Matched-checkpoint report for the GT-loss/feature-weight experiment."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from radio_moge.checkpoint import atomic_json
from radio_moge.data import read_manifest
if __package__:
    from .compare_scene_metrics import paired_report
else:
    from compare_scene_metrics import paired_report


def read(path):
    return json.loads(Path(path).read_text())


def tensor_digest(state):
    digest = hashlib.sha256()
    for part in ('adapter', 'statistics'):
        for name, value in sorted(state[part].items()):
            digest.update(f'{part}/{name}:{value.dtype}:{tuple(value.shape)}'.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def collect(directory):
    metrics = read(directory / 'metrics.json')
    laser = read(directory / 'gt-laser-known-camera/metrics.json')
    result = dict(spatial_cos=metrics['macro/spatial/cosine'],
                  summary_cos=metrics['macro/summary/cosine'],
                  nyu_absrel=metrics['gt_student/abs_rel'],
                  nyu_rmse=metrics['gt_student/rmse'], nyu_delta1=metrics['gt_student/delta1'],
                  nyu_images=metrics['gt_student/abs_rel/count'],
                  laser_absrel=laser['abs_rel'], laser_rmse=laser['rmse'],
                  laser_plane_rmse=laser['plane_rmse_m'],
                  mixed_postprocess_failures=metrics['postprocess_failed'] * metrics['images'],
                  views={})
    for name, folder in [('mixed-native400', 'native-640x400'),
                         ('scannet-dslr', 'dataset-scannetpp-dslr'),
                         ('scannet-iphone', 'dataset-scannetpp-lidar')]:
        value = read(directory / folder / 'metrics.json')
        result['views'][name] = {
            'spatial_cos': value.get('macro/spatial/cosine', value['spatial/cosine']),
            'summary_cos': value.get('macro/summary/cosine', value['summary/cosine']),
            'postprocess_failures': value['postprocess_failed'] * value['images'],
        }
        if 'gt_student/abs_rel' in value:
            result['views'][name]['gt_absrel'] = value['gt_student/abs_rel']
    return result


def compare(a, b, records):
    left, right = read(a / 'per_image.json'), read(b / 'per_image.json')
    nyu = [row for row in records if row.get('depth')]
    images = {row['image'] for row in nyu}
    feature = paired_report(left, right, records,
                            metrics=['spatial/cosine', 'summary/cosine'], repeats=4000)
    feature['sign'] = 'candidate minus baseline; positive is better for cosine'
    for metric in feature['metrics'].values():
        for source in metric['sources'].values():
            if source['status'] == 'complete':
                source['scenes_improved'] = sum(value > 0 for value in
                                                source['per_scene_difference'].values())
    depth = paired_report([r for r in left if r['image'] in images],
                          [r for r in right if r['image'] in images], nyu,
                          metrics=['gt_student/abs_rel', 'gt_student/rmse'], repeats=4000)
    laser_left = read(a / 'gt-laser-known-camera/per_image.json')
    laser_right = read(b / 'gt-laser-known-camera/per_image.json')
    laser_records = [{'image': r['id'], 'group': r['scene'], 'dataset': 'laser'} for r in laser_left]
    laser = paired_report([dict(r, image=r['id']) for r in laser_left],
                          [dict(r, image=r['id']) for r in laser_right], laser_records,
                          metrics=['abs_rel', 'rmse', 'plane_rmse_m'], repeats=4000)
    # Known-camera evaluation does not run the unknown-camera postprocessor or
    # emit its failure flag. Missing flags must not be reported as failures.
    laser.pop('postprocess_failures')
    laser['camera_protocol'] = 'Known measured intrinsics; unknown-camera postprocessor not used.'
    return dict(feature=feature, nyu=depth, laser=laser)


def run(root, verify_weights=False):
    plan = read(root / 'plan.json')
    step = plan['steps_per_arm']
    baseline = root / 'custom-f01/train/eval-0000000'
    result = dict(stage='partial', steps_per_arm=step, baseline=collect(baseline), arms={},
                  comparisons={}, limitations=plan['limits'],
                  selection='fixed final update count, NOT each arm best checkpoint')
    paths, provenances, initial_hashes = {}, {}, {}
    for name, arm in plan['arms'].items():
        output = root / name / 'train'
        evaluation = output / f'eval-{step:07d}'
        if not (evaluation / 'gt-laser-known-camera/metrics.json').exists():
            continue
        stop = read(output / 'stop-report.json') if (output / 'stop-report.json').exists() else {}
        if stop.get('step') != step or stop.get('final_evaluation') != 'complete':
            continue
        metrics = collect(evaluation)
        metrics.update(feature_weight=arm['feature_weight'], objective=arm['objective'])
        result['arms'][name] = metrics
        paths[name] = evaluation
        provenances[name] = read(output / 'provenance.json')
        if verify_weights:
            state = torch.load(output / 'initial.pt', map_location='cpu', weights_only=True, mmap=True)
            initial_hashes[name] = tensor_digest(state)
            del state
    if provenances:
        first = next(iter(provenances.values()))
        for provenance in provenances.values():
            assert provenance['data_digests'] == first['data_digests']
            assert provenance['initialization_provenance']['source_step'] == 232288
            assert provenance['execution_revision'] == plan['revision']
        result['same_data_digests'] = True
    if initial_hashes:
        state = torch.load(plan['source'], map_location='cpu', weights_only=True, mmap=True)
        expected = tensor_digest(state)
        del state
        assert all(value == expected for value in initial_hashes.values())
        result['identical_initial_adapter_statistics'] = initial_hashes
        result['source_adapter_statistics_digest'] = expected
    config = read(next(iter(plan['arms'].values()))['path'])
    records = read_manifest(config['data']['val_manifest'])
    pairs = [('custom-f01', 'custom-f1'), ('official-f01', 'official-f1'),
             ('custom-f01', 'official-f01'), ('custom-f1', 'official-f1')]
    for left, right in pairs:
        if left in paths and right in paths:
            result['comparisons'][left + '__vs__' + right] = compare(paths[left], paths[right], records)
    for name, path in paths.items():
        result['comparisons']['baseline__vs__' + name] = compare(baseline, path, records)
    if len(paths) == len(plan['arms']):
        result['stage'] = 'complete'
    atomic_json(result, root / ('report-verified.json' if verify_weights else 'report.json'))
    print(json.dumps({k: result[k] for k in ('stage', 'steps_per_arm', 'baseline', 'arms')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--verify-weights', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.root, args.verify_weights)
