"""Independently audit old/native/new-data views and decide automatic continuation."""
import argparse
import json
import math
from pathlib import Path
import posixpath
import time

from audit_resolution_evaluations import check_view, digest, load, verify_wandb
from monitor_joint_full_day import write


def expected_manifest(remote):
    local = Path(remote.replace('/home/buildhub/shared/', '/shared/', 1))
    rows = [json.loads(line) for line in local.read_text().splitlines()]
    expected = [(posixpath.normpath(posixpath.join(posixpath.dirname(remote), r['image'])),
                 r['dataset']) for r in rows]
    assert len(expected) == len(set(expected))
    sensor_ids = [identity for identity, row in zip(expected, rows, strict=True) if 'depth' in row]
    return local, expected, sensor_ids


def audit(root):
    config, plan = load(root/'config.json'), load(root/'plan.json')
    path = root/'train/validation.jsonl'
    if not path.exists():
        return []
    logs = [json.loads(line) for line in path.read_text().splitlines()]
    assert logs[0]['step'] == plan['parent_step']
    assert len({r['step'] for r in logs}) == len(logs)
    main = config['data']['val_manifest']
    views = {'original': dict(manifest=main, bucket=None,
                             num_tokens=config['evaluation']['num_tokens'], prefix='', suffix='')}
    for name, view in config['evaluation'].get('resolution_variants', {}).items():
        views[name] = dict(view, manifest=main, prefix=f'resolution/{name}/', suffix=name)
    for name, view in config['evaluation'].get('dataset_variants', {}).items():
        views['dataset-'+name] = dict(view, bucket=None, prefix=f'dataset/{name}/',
                                     suffix='dataset-'+name)
    manifests = {v['manifest']: expected_manifest(v['manifest']) for v in views.values()}
    completed = []
    for row in logs:
        step = row['step']
        proof_path = root/f'eval-{step:07d}-data-verification.json'
        if proof_path.exists():
            assert load(proof_path)['stage'] == 'passed'
            continue
        proofs = {}
        for name, view in views.items():
            local, expected, sensor_ids = manifests[view['manifest']]
            folder = root/f'train/eval-{step:07d}'/view['suffix']
            baseline = root/f"train/eval-{plan['parent_step']:07d}"/view['suffix']
            protocol = load(folder/'protocol.json')
            assert protocol['bucket_hw'] == view['bucket']
            assert protocol['num_tokens'] == view['num_tokens']
            assert protocol['images'] == len(expected)
            metrics = load(folder/'metrics.json')
            proof = check_view(load(folder/'per_image.json'), metrics, row, expected,
                               view['prefix'], load(baseline/'per_image.json'),
                               expected_sensor_ids=sensor_ids)
            assert proof['sensor_rows'] == len(sensor_ids)
            total = math.fsum(metrics['feature/'+k]*v for k, v in config['loss'].items())
            assert math.isclose(row[view['prefix']+'feature/total'], total, abs_tol=1e-12)
            proof.update(protocol=protocol, manifest_sha256=digest(local),
                         sha256={n: digest(folder/n) for n in
                                 ('metrics.json', 'per_image.json', 'protocol.json')})
            proofs[name] = proof
        write(proof_path, dict(stage='passed', checked_at=time.time(), step=step, views=proofs))
        completed.append(step)
    return completed


def continuation_gate(baseline, latest, policy):
    """Predeclared practical guardrails; not a significance or improvement claim."""
    checks = []
    for metric, rule in policy.items():
        a, b = baseline[metric], latest[metric]
        if not all(math.isfinite(v) for v in [a, b]):
            raise ValueError('Nonfinite validation metric')
        limit = a * rule['max_ratio'] if 'max_ratio' in rule else a - rule['max_drop']
        passed = b <= limit if 'max_ratio' in rule else b >= limit
        checks.append(dict(metric=metric, baseline=a, current=b, limit=limit, passed=passed))
    return dict(stage='passed' if all(r['passed'] for r in checks) else 'needs_inspection',
                checks=checks, scope='Practical continuation guardrails, not proof of improvement')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--wandb-step', type=int)
    args = parser.parse_args()
    print(json.dumps(dict(new_complete_audits=audit(args.root))))
    if args.wandb_step is not None:
        verify_wandb(args.root, args.wandb_step, audit_kind='data')
