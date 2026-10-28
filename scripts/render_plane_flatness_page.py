"""Publish the isolated flatness ablation at the existing browser URL."""
import argparse
import html
import json
from pathlib import Path
import time

from render_plane_coverage_page import curve, rows


def render(base, output):
    root = base / 'teacher-plane-ownflat512-pilot-20261028'
    source = (base / 'teacher-plane-loss-diagnostics-20261028/results.html').read_text()
    state = json.loads((root / 'status.json').read_text()) if (root / 'status.json').exists() else {'stage': 'preparing'}
    stages = {'preparing': '准备完成', 'starting': '启动中', 'plane': '新 flatness 训练中',
              'control': '旧 flatness 对照训练中', 'assessment': '训练完成，评估独立截图',
              'complete': '训练完成，结果核验与备份中', 'needs_inspection': '异常检查中'}
    if (root / 'completion-verification.json').exists():
        stages['complete'] = '实验完成 · 结果与备份已核验'
    series, cards, table = [], [], []
    titles = {'plane': '新：自身最佳拟合平面', 'control': '旧：教师法线方向残差'}
    for arm in ('plane', 'control'):
        train = rows(root / arm / 'train.jsonl')
        step = train[-1]['step'] if train else 0
        identity = root / arm / 'wandb-run.json'
        link = ''
        if identity.exists():
            rid = json.loads(identity.read_text())['id']
            link = f'<a href="https://forge.coreweave.com/wandb/brainsim/radio-moge/runs/{html.escape(rid)}" target="_blank" rel="noopener">W&amp;B 曲线 ↗</a>'
        cards.append(f'<div class="card"><b>{titles[arm]}</b><p>{step} / 300 次更新</p><progress max="300" value="{step}"></progress>{link}</div>')
        for part, file in [('validation', 'validation.jsonl'), ('train-probe', 'train-probe.jsonl')]:
            values = rows(root / arm / file)
            series.append((arm, part, values))
            for row in values:
                values = [titles[arm], part, row['step']]
                values += [f'{row[k]:.6f}' if k in row else '—' for k in
                           ('planes/student/normalized_rms', 'gt_student/abs_rel',
                            'depth_edges/log_gradient_mae', 'spatial/cosine', 'summary/cosine',
                            'summary/norm_ratio')]
                table.append('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in values) + '</tr>')
    charts = ''.join(f'<div class="card"><h3>{title}</h3>{curve(series, key)}</div>' for key, title in
                     [('planes/student/normalized_rms', '自身平面归一化 RMS ↓'),
                      ('gt_student/abs_rel', '真实深度 AbsRel ↓'),
                      ('depth_edges/log_gradient_mae', '深度边缘误差 ↓'),
                      ('outside_plane/teacher_abs_rel', '平面外教师一致性误差 ↓')])
    stamp = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    result = (root / 'result-section.html').read_text() if (root / 'result-section.html').exists() else ''
    new = f'''<main id="flatness-ablation"><small>单因素实验 · 同一 512 张图 / 256 场景</small>
<h1>把表面弯曲与教师朝向分开</h1><span class="tag">{stages.get(state['stage'], html.escape(state['stage']))}</span>
<button onclick="location.reload()">刷新</button><p class="muted">更新于 {stamp}</p>
<p>两组均使用特征损失 + 20 × flatness + 2 × normal + 0.1 × anchor。唯一训练变量是 flatness 的定义；旧公式对照也保留完整几何监督。</p>
<div class="grid"><div class="card"><b>新公式</b><p>λ₀ / (λ₁ + λ₂ + ε)</p><p>学生点云中心化协方差的最小特征值，按自身平面内尺度归一化。分母保留梯度。</p></div>
<div class="card"><b>旧公式</b><p>mean[(centered points · teacher normal)²] / mean[‖centered points‖²]</p><p>同时包含表面弯曲与朝向偏差。保留旧实现用于对照。</p></div></div>
<p>两组从同一父权重重新初始化 Adam 与随机状态；学习率 1×10⁻⁵、有效 batch 4、最多各 300 步或 30 分钟。每 50 步评估固定的 16 张验证图和 16 张训练诊断图，截图单独评估。最终按匹配步数比较，不按验证结果挑选最好的一步。</p>
<p class="notice">训练诊断不是独立验证。预期目标是未见场景平面更平，同时保住真实深度、边缘和平面外结构；训练损失下降本身不代表成功。</p>
<div class="grid">{''.join(cards)}</div>{result}<section><h2>中间评估</h2>
<p>橙色＝新 flatness；蓝绿色＝旧 flatness。实线＝验证集；虚线＝训练诊断。等待两组完成匹配步数与逐图核验后给出结论。</p>
<div class="grid">{charts}</div><details><summary>几何、cosine 与 summary 幅值记录</summary><div style="overflow:auto"><table><tr><th>组</th><th>数据</th><th>步</th><th>平面 RMS</th><th>GT AbsRel</th><th>边缘误差</th><th>Spatial cos</th><th>Summary cos</th><th>Summary 幅值比</th></tr>{''.join(table)}</table></div></details></section></main>'''
    position = source.index('<main')
    prefix, body = source[:position], source[position:]
    body = body.removesuffix('</html>')
    prefix = prefix.replace('墙面平整度 · 小子集对照实验', '平面 flatness · 单因素对照实验')
    if result and '<section id="geometry">' in body:
        # A single current viewer owns the controls; history keeps its static results.
        start = body.index('<section id="geometry">')
        end = body.index('<section><details>', start)
        body = body[:start] + body[end:]
    page = (prefix + '<body>' + new + '<details style="max-width:1140px;margin:24px auto">'
            '<summary style="padding:20px">展开已完成的 512 图实验、损失诊断和旧结果（历史）</summary>'
            + body + '</details></body></html>')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.incoming')
    temporary.write_text(page)
    temporary.replace(output)
    print(json.dumps(dict(page=str(output), bytes=output.stat().st_size, stage=state['stage'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.base, args.output)
