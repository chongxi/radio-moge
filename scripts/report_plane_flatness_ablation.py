"""Audit intrinsic-flatness diagnostics and report the matched, single-factor pilot."""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import time


def load(path):
    return json.loads(path.read_text())


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.incoming')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def diagnostic_report(root):
    folder = root / 'gradient-diagnostics'
    raw = load(folder / 'gradient-verification.json')
    assert raw['stage'] == 'passed' and raw['optimizer_updates'] == 0
    assert raw['source_files_unchanged'] and raw['model_parameters_unchanged'] and finite(raw)
    assert len(raw['results']) == 20 and len(raw['selection']) == 5
    states = {'initial_old': ('initial', 'teacher_axis'), 'initial_new': ('initial', 'own_plane'),
              'control': ('control', 'teacher_axis'), 'plane': ('plane', 'own_plane')}
    indexed = {(r['arm'], r['id']): r for r in raw['results']}
    assert len(indexed) == 20
    steps = load(root / 'status.json')['matched_steps']
    metric_errors, objective_errors, comparisons, nonflat_grad_differences = [], [], [], []
    terms = ('feature', 'plane_flatness', 'plane_normal', 'plane_anchor')
    for sample in raw['selection']:
        part, identity = sample['partition'], sample['record']['id']
        for name, (arm, mode) in states.items():
            row = indexed[name, identity]
            assert row['flatness_mode'] == mode
            path = (root / ('plane/eval-0000000' if arm == 'initial' else f'{arm}/eval-{steps:07d}')
                    if part == 'validation' else root / f'assessment/{arm}/{part}')
            saved = next(r for r in load(path / 'per_image.json') if r['image'] == sample['record']['image'])
            assert saved['planes/mask_sha256'] == sample['teacher_mask_sha256']
            own_rms = statistics.mean(p['own_plane_normalized_rms'] for p in row['regions'])
            metric_errors.append(abs(own_rms - saved['planes/student/normalized_rms']))
            expected = statistics.mean(p['own_plane_normalized_rms'] ** 2 if mode == 'own_plane'
                                       else p['flatness_loss'] for p in row['regions'])
            objective_errors.append(abs(expected - row['losses']['plane_flatness']))
            for term in terms[1:]:
                assert row['terms'][term]['raw_points_gradient_outside_masks_l2'] == 0
        old, new = indexed['initial_old', identity], indexed['initial_new', identity]
        # Identical checkpoint/forward, only the flatness objective differs.
        for term in ('feature', 'plane_normal', 'plane_anchor'):
            assert abs(old['terms'][term]['weighted_loss'] - new['terms'][term]['weighted_loss']) < 1e-7
            a, b = (x['adapter_gradients']['weighted_gradient_l2'][term] for x in (old, new))
            nonflat_grad_differences.append(dict(id=identity, term=term, old=a, new=b,
                                                difference=b-a, relative_difference=(b-a)/max(a, 1e-20)))
        comparisons.append(dict(id=identity, partition=part, gradients={term: dict(
            old=old['adapter_gradients']['weighted_gradient_l2'][term],
            new=new['adapter_gradients']['weighted_gradient_l2'][term],
            new_to_old=new['adapter_gradients']['weighted_gradient_l2'][term]
                       / max(old['adapter_gradients']['weighted_gradient_l2'][term], 1e-20))
            for term in terms}))
    assert max(metric_errors) < 1e-7 and max(objective_errors) < 1e-7
    ratios = [r['gradients']['plane_flatness']['new_to_old'] for r in comparisons]
    proof = dict(stage='passed', checked_at=time.time(), optimizer_updates=0, images=5, cases=20,
                 fixed_masks_match_evaluations=True, max_geometry_metric_difference=max(metric_errors),
                 max_loss_definition_difference=max(objective_errors),
                 same_initial_nonflatness_forward_losses=True,
                 initial_nonflatness_gradient_repeat_differences=nonflat_grad_differences,
                 initial_gradient_comparisons=comparisons,
                 initial_flatness_gradient_ratio=dict(min=min(ratios), median=statistics.median(ratios), max=max(ratios)),
                 screenshot={name: indexed[name, 'nyuv2_0271'] for name in states},
                 limitations=raw['limitations'] + [
                     'Keeping scalar weights does not preserve gradient magnitudes between objectives.',
                     'Region flatness_loss is the historical fixed-axis decomposition; applied loss uses result.flatness_mode and losses.plane_flatness.'])
    atomic(folder / 'analysis-verification.json', proof)
    rows = ''.join('<tr><td>' + html.escape(r['partition'] + ' / ' + r['id']) + '</td>'
                   + ''.join(f'<td>{r["gradients"]["plane_flatness"][k]:.4f}</td>'
                             for k in ('old', 'new', 'new_to_old')) + '</tr>' for r in comparisons)
    fragment = f'''<section id="flatness-gradients"><h2>同一初始权重上的梯度检查</h2>
<p>5 张预先选定的图，分别检查初始权重的新旧公式和两组最终权重，共 20 次只读诊断；参数更新为 0。
特征、normal、anchor 的定义和初始前向损失保持一致；重复反向计算的数值差异另行记录。下面是已经乘以 20 权重的 flatness 参数梯度范数。</p>
<div style="overflow:auto"><table><tr><th>诊断图</th><th>旧公式梯度 L2</th><th>新公式梯度 L2</th><th>新／旧</th></tr>{rows}</table></div>
<p>这 5 张图上的梯度范数比中位数为 {statistics.median(ratios):.3f}。
相同的损失权重并不意味着相同的梯度强度，因此不能把结果全部归因于方向变化。
这里测量的是单张图、eval 模式下的梯度，不是训练中累积 batch 的 Adam 更新，也不代表所有场景。</p></section>'''
    target = folder / 'result-section.html'
    temporary = target.with_suffix('.incoming')
    temporary.write_text(fragment)
    temporary.replace(target)
    return proof


def report(root):
    plan, audit, curve = (load(root / name) for name in
                          ('plan.json', 'evaluation-verification.json', 'curve-verification.json'))
    assert plan['profile'] == 'flatness-ablation-512'
    assert audit['stage'] == curve['stage'] == 'passed'
    assert audit['steps'] == 300 and curve['matched_steps'] == list(range(0, 301, 50))
    diagnostic = diagnostic_report(root)
    values = audit['results']['validation']['metrics']
    changes = {k: v['plane_vs_control_percent'] for k, v in values.items()}
    wall = diagnostic['screenshot']
    old_wall = wall['control']['regions'][0]['own_plane_normalized_rms']
    new_wall = wall['plane']['regions'][0]['own_plane_normalized_rms']
    initial_wall = wall['initial_new']['regions'][0]['own_plane_normalized_rms']
    wall_change = 100 * (new_wall / old_wall - 1)
    wall_change_from_initial = 100 * (new_wall / initial_wall - 1)
    checks = {
        'validation_flatness_improved': changes['planes/student/normalized_rms'] < 0,
        'validation_GT_absrel_not_worse': changes['gt_student/abs_rel'] <= 0,
        'validation_GT_rmse_not_worse': changes['gt_student/rmse'] <= 0,
        'validation_GT_delta1_not_worse': changes['gt_student/delta1'] >= 0,
        'validation_edges_not_worse': changes['depth_edges/log_gradient_mae'] <= 0,
        'validation_edge_retention_not_lower': changes['depth_edges/retention_ratio'] >= 0,
        'validation_outside_planes_not_worse': changes['outside_plane/teacher_abs_rel'] <= 0,
        'target_screenshot_wall_not_worse': wall_change <= 0,
    }
    prior = root.parent / 'teacher-plane-coverage512-pilot-20261028'
    old_audit = load(prior / 'evaluation-verification.json')
    assert old_audit['stage'] == 'passed'
    reproducibility = {part: {key: value['control'] - old_audit['results'][part]['metrics'][key]['plane']
                             for key, value in result['metrics'].items()}
                       for part, result in audit['results'].items()}
    maximum = max(abs(v) for d in reproducibility.values() for v in d.values())
    repeat_percent = {part: {key: 100 * value / old_audit['results'][part]['metrics'][key]['plane']
                            for key, value in differences.items()}
                      for part, differences in reproducibility.items()}
    post_start = [x['validation'] for step, x in curve['comparisons'].items() if int(step) > 0]
    curve_counts = {key: sum(row[key]['plane'] < row[key]['control'] for row in post_start)
                    for key in ('planes/student/normalized_rms', 'gt_student/abs_rel',
                                'depth_edges/log_gradient_mae', 'outside_plane/teacher_abs_rel')}
    flat, gt, edge = (changes[k] for k in ('planes/student/normalized_rms', 'gt_student/abs_rel',
                                         'depth_edges/log_gradient_mae'))
    conclusion = (f'相对旧 flatness，新公式在验证集的平面残差变化 {flat:+.2f}%，'
                  f'真实深度 AbsRel 变化 {gt:+.2f}%，边缘误差变化 {edge:+.2f}%；'
                  f'截图目标墙面相对旧公式变化 {wall_change:+.2f}%，'
                  f'相对训练前变化 {wall_change_from_initial:+.2f}%。'
                  + ('本次小面板满足预设的联合检查，仍需更大独立验证。' if all(checks.values())
                     else '本次未同时满足“更平且保住深度、边缘与外部结构”的联合要求。'))
    proof = dict(stage='passed', checked_at=time.time(), title_zh='自身平面 flatness：最终匹配对照',
                 conclusion_zh=conclusion, arm_labels=plan['arm_labels'], steps=300,
                 validation_changes_percent=changes,
                 target_wall=dict(initial=initial_wall, old=old_wall, new=new_wall,
                                  change_percent=wall_change, new_vs_initial_percent=wall_change_from_initial),
                 joint_geometry_checks=checks, joint_geometry_success=all(checks.values()),
                 prior_old_objective_reproduction=dict(max_absolute_difference=maximum, differences=reproducibility,
                                                      percent_differences=repeat_percent,
                                                      within_1e_8=maximum < 1e-8),
                 post_start_curve_points=len(post_start), new_lower_error_points=curve_counts,
                 gradient_diagnostic=str(root / 'gradient-diagnostics/analysis-verification.json'),
                 evidence_hashes={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                  for name in ('evaluation-verification.json', 'curve-verification.json',
                                               'gradient-diagnostics/gradient-verification.json')},
                 limitations=['Single small pilot; no statistical-significance or optimal-weight claim.',
                              'Same scalar weight can mean different gradient strength.',
                              'Training-probe, 16 validation scenes and target screenshot are separate.',
                              'This validation panel has been examined in prior pilots; not a fresh final benchmark.',
                              'The target screenshot motivated this change; its outcome is diagnostic, not blind-test evidence.'])
    atomic(root / 'result-report.json', proof)
    print(json.dumps(proof, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    report(parser.parse_args().root)
