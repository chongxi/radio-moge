"""Audit each completed original/native validation from saved small artifacts only."""

import argparse
from collections import Counter
import hashlib
import json
import math
import netrc
from pathlib import Path
import posixpath
import shlex
import subprocess
import time

from monitor_joint_full_day import write
from verify_joint_evaluation_wandb import REMOTE


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_view(rows, metrics, log, expected, prefix, reference):
    assert [(r['image'], r['dataset']) for r in rows] == expected
    counts = Counter(r['dataset'] for r in rows)
    for row in rows:
        assert all(type(v) in (int, float) and math.isfinite(v)
                   for k, v in row.items() if k not in ('image', 'dataset'))
    assert all(r.get('postprocess_failed') == 0 for r in rows)
    errors = []
    for key, value in metrics.items():
        if key == 'images':
            actual = len(rows)
        elif key.endswith('/count'):
            actual = sum(key[:-6] in r for r in rows)
        elif key.startswith('dataset/'):
            _, source, field = key.split('/', 2)
            selected = [r for r in rows if r['dataset'] == source]
            values = [r[field] for r in selected if field in r]
            actual = len(selected) if field == 'images' else math.fsum(values) / len(values)
        elif key.startswith('macro/'):
            field = key[6:]
            groups = [[r[field] for r in rows if r['dataset'] == s and field in r]
                      for s in counts]
            means = [math.fsum(v) / len(v) for v in groups if v]
            actual = math.fsum(means) / len(means)
        else:
            values = [r[key] for r in rows if key in r]
            actual = math.fsum(values) / len(values)
        assert math.isfinite(value) and math.isclose(value, actual, abs_tol=1e-10, rel_tol=1e-12), key
        assert log[prefix + key] == value, key
        errors.append(abs(value - actual))
    gt = [r for r in rows if 'gt_teacher/abs_rel' in r]
    assert all(r['dataset'] == 'nyuv2' for r in gt)
    assert [(r['image'], r['dataset']) for r in reference] == expected
    teacher_diffs = [abs(a[k] - b[k]) for a, b in zip(rows, reference, strict=True)
                     for k in b if k.startswith('gt_teacher/')]
    assert max(teacher_diffs, default=0) < 1e-8
    return dict(images=len(rows), source_counts=dict(counts), nyu_sensor_rows=len(gt),
                exact_identity_order_source_match=True, all_per_image_values_finite=True,
                postprocess_failures=0, max_aggregate_difference=max(errors, default=0),
                max_frozen_teacher_difference=max(teacher_diffs, default=0))


def audit(root):
    config, plan = load(root/'config.json'), load(root/'plan.json')
    remote_manifest = config['data']['val_manifest']
    manifest = Path(remote_manifest.replace('/home/buildhub/shared/', '/shared/', 1))
    expected = [(posixpath.normpath(posixpath.join(posixpath.dirname(remote_manifest), r['image'])),
                 r['dataset']) for r in map(json.loads, manifest.read_text().splitlines())]
    assert len(expected) == len(set(expected)) == 1268
    log_path = root/'train/validation.jsonl'
    if not log_path.exists():
        return []
    logs = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert logs[0]['step'] == plan['parent_step']
    assert len({r['step'] for r in logs}) == len(logs)
    views = {'original': dict(bucket=None, num_tokens=config['evaluation']['num_tokens']),
             **config['evaluation']['resolution_variants']}
    completed = []
    for row in logs:
        step = row['step']
        proof_path = root/f'eval-{step:07d}-resolution-verification.json'
        if proof_path.exists():
            assert load(proof_path)['stage'] == 'passed'
            continue
        proofs = {}
        for name, view in views.items():
            suffix = '' if name == 'original' else name
            folder = root/f'train/eval-{step:07d}'/suffix
            baseline = root/f"train/eval-{plan['parent_step']:07d}"/suffix
            protocol = load(folder/'protocol.json')
            assert protocol['bucket_hw'] == view['bucket']
            assert protocol['num_tokens'] == view['num_tokens'] and protocol['images'] == 1268
            metrics = load(folder/'metrics.json')
            prefix = '' if name == 'original' else f'resolution/{name}/'
            proof = check_view(load(folder/'per_image.json'), metrics, row, expected, prefix,
                               load(baseline/'per_image.json'))
            assert proof['nyu_sensor_rows'] == 116
            total = math.fsum(metrics['feature/'+k]*v for k, v in config['loss'].items())
            assert math.isclose(row[prefix+'feature/total'], total, abs_tol=1e-12)
            proof.update(protocol=protocol, weighted_feature_total=total,
                         sha256={n: digest(folder/n) for n in
                                 ('metrics.json', 'per_image.json', 'protocol.json')})
            proofs[name] = proof
        write(proof_path, dict(stage='passed', checked_at=time.time(), step=step,
                               manifest_sha256=digest(manifest), views=proofs,
                               scope='Same fixed1268; native view compared to its own initial baseline.'))
        completed.append(step)
    return completed


def verify_wandb(root, step):
    audit = load(root/f'eval-{step:07d}-resolution-verification.json')
    assert audit['stage'] == 'passed'
    dest = root/f'eval-{step:07d}-wandb-verification.json'
    if dest.exists():
        assert load(dest)['stage'] == 'passed'
        return
    identity = load(root/'train/wandb-run.json')
    assert identity['mode'] == 'online' and identity['id'] not in ('eadda6e3', 'd382894f')
    rows = [json.loads(line) for line in (root/'train/validation.jsonl').read_text().splitlines()]
    row = next(r for r in rows if r['step'] == step)
    expected = {k: v for k, v in row.items() if k != 'step' and type(v) in (int, float)}
    secret = netrc.netrc().authenticators('api.wandb.ai')[2]
    payload = dict(key=secret, run_path='/'.join(identity[k] for k in ('entity', 'project', 'id')),
                   config=load(root/'config.json'), step=step, expected=expected)
    result = subprocess.run(
        ['ssh', '-F', '/home/hub/.ssh/radio-moge-config', 'radio-moge-gpu',
         shlex.join(['/home/buildhub/radio-moge/.venv/bin/python', '-c', REMOTE])],
        input=json.dumps(payload), text=True, capture_output=True)
    del secret, payload
    if result.returncode:
        raise RuntimeError('Read-only W&B verification failed; credential-bearing output withheld')
    proof = json.loads(result.stdout)
    proof['independent_audit'] = audit
    write(dest, proof)
    print(json.dumps({k: v for k, v in proof.items() if k != 'independent_audit'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--wandb-step', type=int)
    args = p.parse_args()
    print(json.dumps(dict(new_complete_dual_view_audits=audit(args.root))))
    if args.wandb_step is not None:
        verify_wandb(args.root, args.wandb_step)
