"""Audit completed GT-stage evaluations from small saved artifacts, without inference."""
import argparse
import json
import math
from pathlib import Path
import time

from audit_data_evaluations import expected_manifest
from audit_resolution_evaluations import check_view, digest, load, verify_wandb
from monitor_joint_full_day import write

FEATURE_WEIGHTS = ('spatial', 'summary', 'cosine', 'summary_cosine')
LASER_REQUIRED = {'abs_rel', 'rmse', 'delta1', 'invalid_fraction', 'gt_depth', 'gt_points',
                  'gt_gradient', 'gt_normal', 'gt_plane', 'gt_valid_fraction', 'gt_plane_fraction'}


def audit_laser(rows, metrics, log, expected):
    assert [(r['id'], r['scene']) for r in rows] == [(r['id'], r['scene']) for r in expected]
    assert rows and all(r['split'] == 'val' for r in expected)
    assert len({r['id'] for r in rows}) == len(rows)
    assert LASER_REQUIRED | {'images', 'plane_images'} <= metrics.keys()
    errors = []
    for row, record in zip(rows, expected, strict=True):
        assert LASER_REQUIRED <= row.keys()
        assert ('plane_rmse_m' in row) == bool(record['planes'])
        assert 0 < row['gt_valid_fraction'] <= 1
        assert 0 <= row['gt_plane_fraction'] <= row['gt_valid_fraction']
        assert all(isinstance(v, (int, float)) and math.isfinite(v)
                   for k, v in row.items() if k not in ('id', 'scene'))
    for key, value in metrics.items():
        if key == 'images':
            actual = len(rows)
        elif key == 'plane_images':
            actual = sum('plane_rmse_m' in r for r in rows)
        else:
            values = [r[key] for r in rows if key in r]
            assert values
            actual = math.fsum(values) / len(values)
        assert math.isfinite(value) and math.isclose(value, actual, rel_tol=1e-12, abs_tol=1e-10), key
        assert log['gt_laser_known_camera/' + key] == value, key
        errors.append(abs(value - actual))
    return {'images': len(rows), 'scenes': len({r['scene'] for r in rows}),
            'exact_identity_order_match': True, 'all_per_image_values_finite': True,
            'max_aggregate_difference': max(errors, default=0)}


def audit(root, feature_parent):
    config, plan = load(root / 'config.json'), load(root / 'plan.json')
    assert config.get('geometry_supervision', {}).get('target') == 'ground_truth'
    path = root / 'train/validation.jsonl'
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return []
    assert rows[0]['step'] == 0 and len({r['step'] for r in rows}) == len(rows)
    main = config['data']['val_manifest']
    views = {'original': {'manifest': main, 'bucket': None,
                          'num_tokens': config['evaluation']['num_tokens'], 'prefix': '', 'suffix': ''}}
    for name, view in config['evaluation'].get('resolution_variants', {}).items():
        views[name] = dict(view, manifest=main, prefix=f'resolution/{name}/', suffix=name)
    for name, view in config['evaluation'].get('dataset_variants', {}).items():
        views['dataset-' + name] = dict(view, bucket=None, prefix=f'dataset/{name}/', suffix='dataset-' + name)
    manifests = {v['manifest']: expected_manifest(v['manifest']) for v in views.values()}
    laser_manifest = root / 'laser-val-manifest.jsonl'
    assert digest(laser_manifest) == plan['manifest_sha256']['val']
    laser_expected = [json.loads(line) for line in laser_manifest.read_text().splitlines() if line.strip()]
    completed = []
    for row in rows:
        step = row['step']
        destination = root / f'eval-{step:07d}-data-verification.json'
        if destination.exists():
            assert load(destination)['stage'] == 'passed'
            continue
        folder = root / f'train/eval-{step:07d}'
        required = [folder / view['suffix'] / name for view in views.values()
                    for name in ('metrics.json', 'per_image.json', 'protocol.json')]
        required += [folder / 'gt-laser-known-camera' / name
                     for name in ('metrics.json', 'per_image.json', 'protocol.json')]
        if not all(p.exists() for p in required):
            continue
        proofs = {}
        for name, view in views.items():
            local, expected, sensor_ids = manifests[view['manifest']]
            current = folder / view['suffix']
            baseline = (feature_parent / 'train/eval-0232288' if step == 0
                        else root / 'train/eval-0000000') / view['suffix']
            protocol, metrics = load(current / 'protocol.json'), load(current / 'metrics.json')
            assert protocol['bucket_hw'] == view['bucket'] and protocol['num_tokens'] == view['num_tokens']
            assert protocol['images'] == len(expected)
            proof = check_view(load(current / 'per_image.json'), metrics, row, expected, view['prefix'],
                               load(baseline / 'per_image.json'), expected_sensor_ids=sensor_ids)
            total = math.fsum(metrics['feature/' + k] * config['loss'].get(k, 0) for k in FEATURE_WEIGHTS)
            assert math.isclose(row[view['prefix'] + 'feature/total'], total, abs_tol=1e-12)
            if step == 0:
                parent_metrics = load(baseline / 'metrics.json')
                fields = ['feature/spatial', 'feature/summary'] + [
                    ('macro/' if key in ('cosine', 'norm_ratio') else '') + branch + '/' + key
                    for branch in ('spatial', 'summary')
                    for key in ('cosine', 'norm_ratio', 'abs_log_norm_ratio', 'relative_l2')]
                differences = {key: metrics[key] - parent_metrics[key] for key in fields}
                assert max(abs(v) for v in differences.values()) < 1e-7
                proof['initial_difference_from_archived_step232288'] = differences
            proof.update(protocol=protocol, manifest_sha256=digest(local), weighted_feature_total=total,
                         sha256={n: digest(current / n) for n in ('metrics.json', 'per_image.json', 'protocol.json')})
            proofs[name] = proof
        laser = folder / 'gt-laser-known-camera'
        protocol = load(laser / 'protocol.json')
        assert protocol['camera'] == 'known measured intrinsics' and protocol['scale_alignment'] is False
        proof = audit_laser(load(laser / 'per_image.json'), load(laser / 'metrics.json'), row, laser_expected)
        proof.update(protocol=protocol, manifest_sha256=digest(laser_manifest),
                     sha256={n: digest(laser / n) for n in ('metrics.json', 'per_image.json', 'protocol.json')})
        proofs['gt-laser-known-camera'] = proof
        write(destination, {'stage': 'passed', 'checked_at': time.time(), 'step': step, 'views': proofs,
                            'source_step': 232288, 'initial_stage_step': 0,
                            'scope': 'Four fixed validation views and independent heldout measured laser geometry'})
        completed.append(step)
    return completed


def review(root, step):
    """Compare audited heldout geometry and feature direction/amplitude to stage zero."""
    destination = root / f'eval-{step:07d}-feature-alignment-review.json'
    if destination.exists():
        return load(destination)
    for kind in ('data', 'wandb'):
        assert load(root / f'eval-{step:07d}-{kind}-verification.json')['stage'] == 'passed'
    rows = [json.loads(line) for line in (root / 'train/validation.jsonl').read_text().splitlines()
            if line.strip()]
    steps = sorted(r['step'] for r in rows if r['step'] <= step)
    assert steps[0] == 0 and steps[-1] == step
    previous = steps[-2] if len(steps) > 1 else 0
    def metrics(s, suffix):
        return load(root / f'train/eval-{s:07d}' / suffix / 'metrics.json')
    def compare(current, initial, prev):
        return {k: {'current': v, 'initial': initial[k], 'previous': prev[k],
                    'change_from_initial': v - initial[k], 'change_from_previous': v - prev[k]}
                for k, v in current.items()}
    views = {}
    for name, suffix in [('original', ''), ('native-640x400', 'native-640x400'),
                         ('scannetpp-dslr', 'dataset-scannetpp-dslr'),
                         ('scannetpp-lidar', 'dataset-scannetpp-lidar')]:
        def select(values):
            return {f'{branch}/{key}': values[('macro/' if key in ('cosine', 'norm_ratio') else '')
                                             + branch + '/' + key]
                    for branch in ('spatial', 'summary')
                    for key in ('cosine', 'norm_ratio', 'abs_log_norm_ratio', 'relative_l2')} | {
                        'feature/spatial': values['feature/spatial'],
                        'feature/summary': values['feature/summary']}
        current, initial, prev = [metrics(s, suffix) for s in (step, 0, previous)]
        per_image = load(root / f'train/eval-{step:07d}' / suffix / 'per_image.json')
        quantiles = {}
        for branch in ('spatial', 'summary'):
            for key in ('norm_ratio', 'abs_log_norm_ratio', 'relative_l2'):
                values = sorted(row[branch + '/' + key] for row in per_image)
                def percentile(q):
                    index = (len(values) - 1) * q
                    lo, hi = math.floor(index), math.ceil(index)
                    return values[lo] + (values[hi] - values[lo]) * (index - lo)
                quantiles[branch + '/' + key] = {f'p{int(q * 100)}': percentile(q)
                                               for q in (.1, .5, .9, .95)}
        depth_keys = [k for k in ('gt_student/abs_rel', 'gt_student/rmse') if k in current]
        views[name] = {'metrics': compare(select(current), select(initial), select(prev)),
                       'per_image_summary_distribution': quantiles,
                       'observational_depth': compare({k: current[k] for k in depth_keys}, initial, prev)}
    laser = [metrics(s, 'gt-laser-known-camera') for s in (step, 0, previous)]
    config = load(root / 'config.json')
    proof = {'stage': 'observed', 'checked_at': time.time(), 'step': step, 'initial_step': 0,
             'source_step': 232288, 'previous_step': previous, 'views': views,
             'heldout_laser_geometry': compare(*laser), 'loss_weights': config['loss'],
             'learning_rate': config['train']['lr'], 'near_identity_claimed': False,
             'no_significance_claim': True, 'initial_evaluation_is_baseline_only': step == 0,
             'targets': {'cosine': 1, 'norm_ratio': 1, 'abs_log_norm_ratio': 0},
             'geometry_protocol': 'Known measured camera; metric scale; all valid GT pixels; plane auxiliary.',
             'aggregation': 'Cosine/norm_ratio equal-source macro means; other feature errors and GT metrics equal-image means.'}
    write(destination, proof)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--feature-parent', type=Path, required=True)
    parser.add_argument('--wandb-step', type=int)
    parser.add_argument('--review-step', type=int)
    args = parser.parse_args()
    print(json.dumps({'new_complete_audits': audit(args.root, args.feature_parent)}))
    if args.wandb_step is not None:
        verify_wandb(args.root, args.wandb_step, audit_kind='data')
    if args.review_step is not None:
        result = review(args.root, args.review_step)
        print(json.dumps({'reviewed_step': result['step'], 'previous_step': result['previous_step'],
                          'heldout_laser_geometry': result['heldout_laser_geometry']}))
