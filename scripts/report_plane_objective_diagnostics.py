"""Audit and visualize the saved read-only plane-objective diagnosis."""
import argparse
import base64
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def atomic(path, value):
    temporary = path.with_suffix('.incoming')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def report(base):
    root = base / 'teacher-plane-loss-diagnostics-20261028'
    pilot = base / 'teacher-plane-coverage512-pilot-20261028'
    raw = json.loads((root / 'gradient-verification.json').read_text())
    assert raw['stage'] == 'passed' and raw['optimizer_updates'] == 0
    assert raw['source_files_unchanged'] and raw['model_parameters_unchanged']
    assert finite(raw)
    assert len(raw['results']) == 15 and len(raw['selection']) == 5
    for sample in raw['selection']:
        part = sample['partition']
        folder = pilot / ('plane/eval-0000000' if part == 'validation'
                          else f'assessment/initial/{part}')
        rows = json.loads((folder / 'per_image.json').read_text())
        row = next(x for x in rows if x['image'] == sample['record']['image'])
        assert row['planes/mask_sha256'] == sample['teacher_mask_sha256']
        for arm in ('initial', 'control', 'plane'):
            result = next(x for x in raw['results'] if x['arm'] == arm and x['id'] == sample['record']['id'])
            current = (pilot / ('plane/eval-0000000' if arm == 'initial' else f'{arm}/eval-0000300')
                       if part == 'validation' else pilot / f'assessment/{arm}/{part}')
            saved = next(x for x in json.loads((current / 'per_image.json').read_text())
                         if x['image'] == sample['record']['image'])
            rms = statistics.mean(p['own_plane_normalized_rms'] for p in result['regions'])
            assert abs(rms - saved['planes/student/normalized_rms']) < 1e-7
            for term in ('plane_flatness', 'plane_normal', 'plane_anchor'):
                assert result['terms'][term]['raw_points_gradient_outside_masks_l2'] == 0
    cfg = json.loads((pilot / 'plane-config.json').read_text())
    logs = list(map(json.loads, (pilot / 'plane/train.jsonl').read_text().splitlines()))
    weights = {k: v for k, v in cfg['loss'].items() if v}
    error = max(abs(sum(row[k]*w for k, w in weights.items())-row['total']) for row in logs)
    assert error < 1e-6
    contributions = {k: dict(weight=w, mean_weighted_loss=statistics.mean(row[k]*w for row in logs),
                             mean_fraction=statistics.mean(row[k]*w/row['total'] for row in logs))
                     for k, w in weights.items()}
    pairs = {}
    for key in raw['results'][0]['adapter_gradients']['pairwise_cosine']:
        values = [x['adapter_gradients']['pairwise_cosine'][key] for x in raw['results']]
        assert all(-1.000001 <= x <= 1.000001 for x in values)
        pairs[key] = dict(min=min(values), median=statistics.median(values), max=max(values))
    screenshot = {x['arm']: x for x in raw['results'] if x['partition'] == 'screenshot'}
    mask = raw['selection'][-1]['teacher_planes'][0]
    scale = screenshot['plane']['summary_amplitude_counterfactual']
    scale_error = max(abs(x['own_plane_normalized_rms'] - scale['1.0']['own_plane_normalized_rms'])
                      for x in scale.values())
    assert scale_error < 1e-7
    summary = dict(stage='passed', checked_at=time.time(), optimizer_updates=0, images=5,
                   checkpoint_states=3, fixed_teacher_masks_match_original=True,
                   decoded_geometry_metrics_match_original=True,
                   weighted_total_max_error=error, logged_training_samples=len(logs),
                   logged_gradient_above_clip=sum(x['gradient_norm'] > 1 for x in logs),
                   contributions=contributions, gradient_cosines=pairs,
                   screenshot=screenshot, target_teacher_plane=mask,
                   summary_scaling_shape_max_difference=scale_error,
                   limitations=raw['limitations'], findings=[
                       'Fixed-teacher-normal variance mixes intrinsic curvature and orientation error.',
                       'All point-space gradients of plane losses outside fixed masks are zero; shared parameters still couple outputs.',
                       'In this decoder summary only affects metric scale; fixed spatial output makes shape invariant to summary changes.',
                       'No evidence of strong global opposition among the three plane terms on these five examples.',
                       'Single-image target-shape gradients do not establish why training on other scenes harmed the held-out wall.',
                       'GT depth in the selected wall mask is less planar than the teacher; teacher labels are not ground-truth certification.'])
    atomic(root / 'analysis-verification.json', summary)
    # Plot the exact fixed mask on the image; no smoothing or geometry replacement.
    with np.load(root / 'masks-nyuv2_0271.npz') as data:
        rgb, labels = data['rgb'], data['labels']
    assert hashlib.sha256(labels.tobytes()).hexdigest() == raw['selection'][-1]['teacher_mask_sha256']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout='constrained')
    axes[0].imshow(rgb)
    axes[0].set_title('Input RGB')
    axes[1].imshow(rgb)
    axes[1].imshow(np.ma.masked_where(labels == 0, labels), cmap='tab10', alpha=.5, vmin=1, vmax=10)
    axes[1].set_title('Fixed teacher RANSAC masks')
    axes[2].imshow(rgb)
    axes[2].contour(labels == 1, levels=[.5], colors=['#f46232'], linewidths=1)
    axes[2].imshow(np.ma.masked_where(labels != 1, np.ones_like(labels)), cmap='autumn', alpha=.25)
    axes[2].set_title(f'Target region 1: {mask["pixels"]:,} pixels')
    for ax in axes:
        ax.set_axis_off()
    fig.savefig(root / 'teacher-mask-review.png', dpi=140)
    plt.close(fig)
    picture = base64.b64encode((root / 'teacher-mask-review.png').read_bytes()).decode()
    names = {'feature': '特征损失合计', 'plane_flatness': 'flatness ×20',
             'plane_normal': 'normal ×2', 'plane_anchor': 'anchor ×0.1'}
    targets = []
    for arm, label in [('initial', '训练前'), ('control', '对照'), ('plane', '平面约束组')]:
        x = screenshot[arm]['regions'][0]
        targets.append(f'<tr><td>{label}</td><td>{x["own_plane_normalized_rms"]:.6f}</td>'
                       f'<td>{x["teacher_normal_angle_deg"]:.3f}°</td>'
                       f'<td>{x["orientation_fraction"]:.1%}</td></tr>')
    term_rows = []
    for term, label in names.items():
        x = screenshot['plane']['terms'][term]
        norm = screenshot['plane']['adapter_gradients']['weighted_gradient_l2'][term]
        term_rows.append(f'<tr><td>{label}</td><td>{x["weighted_loss"]:.6f}</td><td>{norm:.3f}</td>'
                         f'<td>{x["module_gradient_l2"]["spatial"]:.3f}</td>'
                         f'<td>{x["module_gradient_l2"]["summary"]:.3g}</td></tr>')
    cos_rows = ''.join(f'<tr><td>{html.escape(k)}</td><td>{v["min"]:.3f}</td>'
                       f'<td>{v["median"]:.3f}</td><td>{v["max"]:.3f}</td></tr>' for k,v in pairs.items())
    scalar_rows = ''.join(f'<tr><td>{k}</td><td>{v["weight"]}</td><td>{v["mean_weighted_loss"]:.6f}</td>'
                          f'<td>{v["mean_fraction"]:.1%}</td></tr>' for k,v in contributions.items())
    scales = ''.join(f'<tr><td>{k}×</td><td>{v["metric_scale_ratio"]:.6f}×</td>'
                    f'<td>{v["own_plane_normalized_rms"]:.9f}</td></tr>' for k,v in scale.items())
    fragment = f'''<section id="loss-diagnostics"><h2>为什么没有稳定变平：损失与梯度检查</h2>
<p class="notice">已完成只读诊断：训练前／对照／平面约束三份权重 × 5 张预先选定的图，共 15 次检查，参数更新为 0。
当前证据指向“形状目标与朝向目标混合、平面外和边缘缺少直接保护”；尚不能把失败归因于训练不足、掩码错误或 adapter 结构。</p>
<h3>1. flatness 不只衡量弯曲</h3><p>当前实现沿教师法向计算方差，精确分解为“自身最佳拟合平面的残差＋朝向偏差”。
截图目标区域的 flatness 数值约 82% 来自朝向偏差；这是损失值分解，不是梯度占比。
即便表面本身完全平，只要方向与教师不同，这一项仍会惩罚它。</p>
<div style="overflow:auto"><table><tr><th>目标墙面／黑板</th><th>自身平面残差 ↓</th><th>朝向差 ↓</th><th>flatness 中朝向部分</th></tr>{''.join(targets)}</table></div>
<h3>2. summary 幅值改变尺度，不能直接改变弯曲</h3><p>固定 spatial 输出时，当前 decoder 的 summary 仅进入 metric scale head。
对本轮模型把 summary 乘 0.8／1.2，只改变整体大小，归一化弯曲不变。训练时共享参数仍可能间接影响 spatial，这与直接改 summary 输出不同。</p>
<table><tr><th>Summary 幅值</th><th>整体尺度变化</th><th>目标区域归一化残差</th></tr>{scales}</table>
<h3>3. 梯度不是“各项严重打架”</h3><p>15 个检查点中，flatness 与 normal 的梯度 cosine 为 0.56～0.87，主要同向；
feature 与 flatness 接近正交，未发现持续强烈反向。不能把墙面退化解释成这三项简单互相抵消。
下表是本轮权重在目标截图上的加权损失和参数梯度范数；不是训练中实际累积的 Adam 更新。</p>
<div style="overflow:auto"><table><tr><th>项</th><th>加权损失值</th><th>参数梯度 L2</th><th>Spatial 模块梯度</th><th>Summary 模块梯度</th></tr>{''.join(term_rows)}</table></div>
<p>目标墙面自身弯曲指标与总梯度的 cosine 为 {screenshot['plane']['target_shape_gradient_cosine']['total']:.3f}：
按当前训练中固定焦距／平移求解结果的梯度定义，在这张图上的无穷小普通梯度下降方向仍倾向让墙变平。
这不等于完整相机求解的导数，也不等于它在其他训练场景上更新后会变平，更不等于 Adam 的实际更新效果。</p>
<h3>4. 固定掩码没有直接平滑整张图，但也没有保护外部区域</h3><p>三项平面损失在掩码外的点空间直接梯度均为 0。
然而网络参数共享，局部监督仍可改变其他像素；本轮 points／gradient／normal 等全图几何权重均为 0，也没有专门的边缘保持项。</p>
<img alt="原始截图、固定教师平面掩码及目标区域边界" style="width:100%;height:auto" src="data:image/png;base64,{picture}">
<p>目标区域含 {mask['pixels']:,} 个像素（画面 {mask['pixels']/labels.size:.1%}），检测器内部置信分数 {mask['confidence']:.3f}。
教师自身平面残差 {mask['teacher_normalized_rms']:.4f}；对应 NYU 深度有效覆盖 {mask['gt_valid_fraction']:.1%}，
按教师内参回投后的 GT 平面残差 {mask['gt_own_plane_rms']:.4f}、法向差 {mask['teacher_gt_normal_angle_deg']:.2f}°。
教师伪标签不能据此视作真值；GT 的噪声、边界、内参等影响仍需区分。</p>
<p>目视检查：编号 1 主要覆盖黑板内侧，避开了前方椅子和右下白色面板；没有看到把这些物体一同压平的明显掩码错误。
这只支持当前截图的区域划分合理，不代表所有训练图的平面标签都可靠。</p>
<details><summary>训练日志贡献与全部梯度 cosine</summary><p>31 个日志采样点均触发梯度裁剪阈值；不等于检查了全部 300 步。
平面三项合计约占损失值 19%，但不能据此推断只占 19% 更新。所有权重状态统一按实验损失做诊断；对照训练时平面损失为 0。</p>
<div style="overflow:auto"><table><tr><th>项</th><th>权重</th><th>加权均值</th><th>平均总损失占比</th></tr>{scalar_rows}</table>
<table><tr><th>梯度对</th><th>最小 cosine</th><th>中位数</th><th>最大</th></tr>{cos_rows}</table></div></details>
<p><b>下一步可检验：</b>把“自身表面弯曲”与“教师朝向”分开约束，先做单因素对照；边缘保持再单独验证。
本页只报告诊断，尚未更改损失或开始新训练。5 张图不能证明所有场景的因果关系。</p></section>'''
    p = root / 'result-section.html'
    t = p.with_suffix('.incoming')
    t.write_text(fragment)
    t.replace(p)
    print(json.dumps(dict(stage='passed', images=5, states=3,
                          summary_scaling_shape_error=scale_error, weighted_total_error=error)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    report(parser.parse_args().base)
