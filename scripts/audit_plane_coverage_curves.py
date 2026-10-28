"""Audit newly completed training-probe/validation points; no model inference."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from audit_plane_subset_pilot import audit_eval, fixed_teacher, KEYS


def load(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines(keepends=True)
            if s.endswith('\n')] if path.exists() else []


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.incoming')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def audit(root):
    reference = root.parent / 'teacher-plane-subset-pilot-20261028'
    assert load(reference / 'completion-verification.json')['stage'] == 'passed'
    plan = load(root / 'plan.json')
    expected_steps = plan['evaluation_steps']
    arms = plan.get('arms', ['plane', 'control'])
    treatment = 'joint' if plan.get('profile') == 'joint-cosine-512' else 'plane'
    completed, fresh = {}, []
    for arm in arms:
        config = load(root / f'{arm}-config.json')
        validation = {r['step']: r for r in rows(root / arm / 'validation.jsonl')}
        probes = {r['step']: r for r in rows(root / arm / 'train-probe.jsonl')}
        completed[arm] = {}
        for step in sorted(validation.keys() & probes.keys()):
            assert step in expected_steps or step <= plan['max_updates']
            target = root / f'{arm}-eval-{step:07d}-verification.json'
            if target.exists():
                proof = load(target)
                assert proof['stage'] == 'passed'
                completed[arm][str(step)] = proof['metrics']
                continue
            metrics, checks = {}, {}
            for partition, folder, original, log in (
                ('validation', f'eval-{step:07d}', reference / 'plane/eval-0000000', validation[step]),
                ('train-probe', f'train-probe-{step:07d}',
                 reference / 'assessment/initial/train-probe', probes[step])):
                manifest = rows(root / f'{partition}.jsonl')
                assert manifest == rows(reference / f'{partition}.jsonl')
                identities = [r['image'] for r in manifest]
                path = root / arm / folder
                measured, summary, receipt = audit_eval(path, identities)
                before = load(original / 'per_image.json')
                # Prior completed report pins the immutable teacher/reference rows.
                old_partition = 'validation' if partition == 'validation' else 'train-probe'
                old_proof = load(reference / 'evaluation-verification.json')
                pinned = old_proof['aggregates'][old_partition]['initial']['sha256']['per_image.json']
                assert hashlib.sha256((original / 'per_image.json').read_bytes()).hexdigest() == pinned
                fixed_teacher([before, measured])
                for key in KEYS:
                    assert abs(summary[key] - log[key]) < 1e-12, key
                raw = load(path / 'metrics.json')
                total = sum(raw[f'feature/{k}'] * config['loss'][k]
                            for k in ('spatial', 'summary', 'cosine'))
                assert abs(log['feature/total'] - total) < 1e-12
                if step == 0:
                    for a, b in zip(before, measured, strict=True):
                        for key in KEYS:
                            assert abs(a[key] - b[key]) < 1e-8, key
                metrics[partition], checks[partition] = summary, receipt
            proof = dict(stage='passed',checked_at=time.time(),arm=arm,step=step,
                         fixed_reference_identities_masks_teacher_gt=True,
                         initial_matches_previous_parent=step == 0,
                         metrics=metrics,receipts=checks)
            atomic(target,proof)
            completed[arm][str(step)] = metrics
            fresh.append(str(target))
    paired = {}
    for step in sorted(set(completed[treatment]) & set(completed['control']),key=int):
        paired[step] = {partition: {key: {
            treatment: completed[treatment][step][partition][key],
            'control': completed['control'][step][partition][key],
            'difference': completed[treatment][step][partition][key]-completed['control'][step][partition][key]}
            for key in KEYS} for partition in ('validation','train-probe')}
    report = dict(stage='passed' if sorted(map(int,paired)) == expected_steps else 'in_progress',
                  checked_at=time.time(),expected_steps=expected_steps,
                  verified_steps={arm: sorted(map(int,points)) for arm,points in completed.items()},
                  matched_steps=sorted(map(int,paired)),comparisons=paired,
                  note='Only matched treatment/control steps support the effect comparison; no best-step selection.')
    atomic(root/'curve-verification.json',report)
    print(json.dumps(dict(stage=report['stage'],verified_steps=report['verified_steps'],new_proofs=fresh)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    audit(parser.parse_args().root)
