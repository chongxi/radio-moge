"""Independently audit completed, matched plane/control evaluations without inference."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


KEYS = ['planes/student/normalized_rms', 'planes/student/p95_normalized',
        'planes/student/normal_angle_deg', 'gt_student/abs_rel', 'gt_student/rmse',
        'gt_student/delta1', 'outside_plane/teacher_abs_rel', 'depth_edges/log_gradient_mae',
        'depth_edges/retention_ratio', 'teacher_depth/abs_rel', 'spatial/cosine',
        'spatial/norm_ratio', 'summary/cosine', 'summary/norm_ratio']


def load(path):
    return json.loads(path.read_text())


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def audit_eval(path, expected_images):
    rows, metrics = load(path / 'per_image.json'), load(path / 'metrics.json')
    assert [r['image'] for r in rows] == expected_images
    assert all(r['postprocess_failed'] == r['planes/failed'] == 0 for r in rows)
    assert all(math.isfinite(v) for r in rows for v in r.values() if numeric(v))
    assert any(r['planes/regions'] > 0 for r in rows)
    recomputed = {'images': len(rows)}
    for key in {k for r in rows for k, v in r.items() if numeric(v)}:
        values = [r[key] for r in rows if numeric(r.get(key)) and math.isfinite(r[key])]
        if values:
            recomputed[key] = statistics.mean(values)
            recomputed[key + '/count'] = len(values)
    differences = [abs(metrics[k] - v) for k, v in recomputed.items()]
    assert max(differences) < 1e-12
    return rows, {k: recomputed[k] for k in KEYS}, dict(
        images=len(rows), max_aggregate_error=max(differences),
        sha256={name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                for name in ('metrics.json', 'per_image.json')})


def fixed_teacher(sets):
    reference = sets[0]
    for rows in sets[1:]:
        for a, b in zip(reference, rows, strict=True):
            assert a['image'] == b['image']
            assert a['planes/mask_sha256'] == b['planes/mask_sha256']
            for key in a:
                if key.startswith(('gt_teacher/', 'planes/teacher/')):
                    assert abs(a[key] - b[key]) < 1e-10, key


def run(root):
    status = load(root / 'status.json')
    assert status['stage'] == 'complete' and status['child_pid'] is None
    assert status['child_exit'] == 0 and status['parent_unchanged']
    steps = status['matched_steps']
    assert 0 < steps <= 300
    arm_proofs = {}
    for arm in ('plane', 'control'):
        stop = load(root / arm / 'stop-report.json')
        assert stop['step'] == steps and stop['final_evaluation'] == 'complete'
        rows = [json.loads(s) for s in (root / arm / 'train.jsonl').read_text().splitlines()]
        assert rows[-1]['step'] == steps
        assert all(r['lr'] == r['lr_used'] == 1e-5 for r in rows)
        assert all(math.isfinite(v) for r in rows for v in r.values() if numeric(v))
        assert all(r['update_norm'] > 0 for r in rows if 'update_norm' in r)
        arm_proofs[arm] = dict(logged_rows=len(rows), max_gradient=max(r['gradient_norm'] for r in rows),
                               stop=stop, run=load(root / arm / 'wandb-run.json'))
    ids = load(root / 'initial-adapter-verification.json')
    assert ids['plane'] == ids['control']
    results, proofs = {}, {}
    for partition in ('validation', 'train-probe', 'screenshot'):
        images = [json.loads(s)['image'] for s in (root / f'{partition}.jsonl').read_text().splitlines()]
        paths = ({'initial': root / 'plane/eval-0000000',
                  'control': root / 'control' / f'eval-{steps:07d}',
                  'plane': root / 'plane' / f'eval-{steps:07d}'} if partition == 'validation'
                 else {arm: root / 'assessment' / arm / partition
                       for arm in ('initial', 'control', 'plane')})
        rows, metrics, checks = {}, {}, {}
        for arm, path in paths.items():
            rows[arm], metrics[arm], checks[arm] = audit_eval(path, images)
        fixed_teacher(list(rows.values()))
        if partition == 'validation':
            other, _, check = audit_eval(root / 'control/eval-0000000', images)
            for a, b in zip(rows['initial'], other, strict=True):
                for key in KEYS:
                    assert (key in a) == (key in b)
                    if key in a:
                        assert abs(a[key] - b[key]) < 1e-10
            fixed_teacher([rows['initial'], other])
            checks['control_initial'] = check
        changes = {key: dict(initial=metrics['initial'][key], control=metrics['control'][key],
                            plane=metrics['plane'][key],
                            plane_vs_control_percent=100 * (metrics['plane'][key] /
                                                           metrics['control'][key] - 1),
                            plane_vs_initial_percent=100 * (metrics['plane'][key] /
                                                           metrics['initial'][key] - 1))
                   for key in KEYS}
        key = 'planes/student/normalized_rms'
        wins = sum(p[key] < c[key] for p, c in zip(rows['plane'], rows['control'], strict=True)
                   if key in p)
        results[partition] = dict(images=len(images), flatness_better_than_control=wins,
                                  images_with_planes=sum(key in p for p in rows['plane']),
                                  metrics=changes,
                                  per_image=[dict(image=p['image'],
                                                  initial=i[key], control=c[key], plane=p[key])
                                             for i, c, p in zip(rows['initial'], rows['control'],
                                                                rows['plane'], strict=True)
                                             if key in p])
        proofs[partition] = checks
    report = dict(stage='passed', verified_at=time.time(), steps=steps, arms=arm_proofs,
                  initial_adapter_identical=True, fixed_teacher_masks=True,
                  frozen_teacher_gt_unchanged=True, aggregates=proofs, results=results,
                  limitations=['Single small-subset run; no optimal-weight claim.',
                               'Training-probe, validation and screenshot are reported separately.',
                               'Teacher planes are pseudo-labels, not independent ground truth.'])
    target = root / 'evaluation-verification.json'
    temporary = target.with_suffix('.incoming')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary.replace(target)
    print(json.dumps(dict(stage='passed', steps=steps, results=results), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    run(parser.parse_args().root)
