"""Keep the original browser URL and show the expanded-coverage experiment separately."""
import argparse
import html
import json
from pathlib import Path
import time


COLORS = {'plane': '#b64b27', 'control': '#147c85'}


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines(keepends=True)
            if s.endswith('\n')] if path.exists() else []


def curve(series, key):
    valid = [(arm, part, [(r['step'], r[key]) for r in values if isinstance(r.get(key), (int, float))])
             for arm, part, values in series]
    values = [v for _, _, points in valid for _, v in points]
    if not values:
        return '<p>等待完整评估。</p>'
    lo, hi = min(values), max(values)
    pad = max((hi - lo) * .15, abs(hi) * .01, 1e-6)
    lo, hi = lo - pad, hi + pad
    bits = ['<svg viewBox="0 0 700 260" role="img" aria-label="训练诊断与验证指标曲线">']
    for i in range(5):
        value = lo + (hi - lo) * i / 4
        y = 210 - 180 * i / 4
        bits.append(f'<path d="M75 {y}H675" stroke="#d9e1de"/>'
                    f'<text x="4" y="{y + 4}" fill="#51635f" font-size="13">{value:.5f}</text>')
    for step in range(0, 301, 50):
        x = 75 + step * 2
        bits.append(f'<text x="{x}" y="238" text-anchor="middle" font-size="13">{step}</text>')
    for arm, part, points in valid:
        color = COLORS[arm]
        xy = [(75 + step * 2, 210 - (v - lo) / (hi - lo) * 180) for step, v in points]
        dash = ' stroke-dasharray="7 5"' if part == 'train-probe' else ''
        coords = ' '.join(f'{x:.2f},{y:.2f}' for x, y in xy)
        bits.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
        for (step, value), (x, y) in zip(points, xy):
            label = html.escape(f'{arm} {part} step {step}: {value:.7f}')
            bits.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"><title>{label}</title></circle>')
    return ''.join(bits) + '</svg>'


def render(base, output):
    root = base / 'teacher-plane-coverage512-pilot-20261028'
    prior = base / 'teacher-plane-subset-pilot-20261028/results.html'
    source = prior.read_text()
    state = json.loads((root / 'status.json').read_text()) if (root / 'status.json').exists() else {'stage': 'preparing'}
    labels = {'preparing': '准备与配置检查', 'starting': '启动中', 'plane': '平面约束组运行中',
              'control': '特征损失对照组运行中', 'assessment': '训练结束，正在独立截图评估',
              'complete': '训练已完成，正在核验并备份', 'needs_inspection': '运行异常，正在检查'}
    finished = (root / 'completion-verification.json').exists()
    audited = (root / 'evaluation-verification.json').exists()
    if finished:
        labels['complete'] = '实验完成 · 评估与备份已核验'
    curve_note = ('两组全部 7 个预定评估点已逐图独立核验；最终对比固定使用第 300 步。'
                  if audited else '图中只显示已写完的评估日志；最终结论等待两组匹配步数及独立核验。')
    stamp = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    cards, series = [], []
    for arm, title in [('plane', '特征＋平面约束'), ('control', '特征损失对照')]:
        train = rows(root / arm / 'train.jsonl')
        step = train[-1]['step'] if train else 0
        identity = root / arm / 'wandb-run.json'
        link = ''
        if identity.exists():
            rid = json.loads(identity.read_text())['id']
            link = f'<a href="https://forge.coreweave.com/wandb/brainsim/radio-moge/runs/{html.escape(rid)}" target="_blank" rel="noopener">打开 W&amp;B 曲线 ↗</a>'
        cards.append(f'<div class="card"><b>{title}</b><p>{step} / 300 次更新</p><progress value="{step}" max="300"></progress>{link}</div>')
        for partition, file in [('validation', 'validation.jsonl'), ('train-probe', 'train-probe.jsonl')]:
            series.append((arm, partition, rows(root / arm / file)))
    charts = ''.join('<div class="card"><h3>' + title + '</h3>' + curve(series, key) + '</div>'
                     for key, title in [('planes/student/normalized_rms', '归一化平面残差 ↓'),
                                        ('gt_student/abs_rel', '真实深度 AbsRel ↓')])
    table = []
    for arm, part, values in series:
        for r in values:
            table.append('<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in
                         [arm, part, r['step'], *[f'{r[k]:.6f}' if isinstance(r.get(k), (int, float)) else '—'
                         for k in ('planes/student/normalized_rms', 'gt_student/abs_rel',
                                   'depth_edges/log_gradient_mae', 'spatial/cosine',
                                   'summary/cosine', 'summary/norm_ratio')]]) + '</tr>')
    new = f'''<main id="coverage"><small>新增实验 · 固定结构和损失权重</small><h1>扩大平面监督的场景覆盖</h1>
<span class="tag">{labels.get(state['stage'], html.escape(state['stage']))}</span>
<button onclick="location.reload()">刷新进度</button><p class="muted">页面更新：{stamp}</p>
<p>512 张训练图 / 256 个场景，包含上轮全部 64 张图。保持同一初始权重、adapter、学习率 1×10⁻⁵、有效 batch 4、平面损失权重 20 / 2 / 0.1。</p>
<p>两组各最多 300 步或 30 分钟；每 50 步评估相同的 16 张训练诊断图和 16 张独立验证图。截图继续单独保留。</p>
<p class="notice">本轮约 2.34 遍数据，上轮约 18.75 遍：比较的是相同更新预算下，更广场景覆盖能否改善泛化。训练图与验证图分开报告，不把训练诊断视为独立验证。</p>
<div class="grid">{''.join(cards)}</div><section><h2>中间评估曲线</h2>
<p>橙色：平面约束；蓝绿色：对照。实线：验证集；虚线：训练诊断。{curve_note}</p>
<div class="grid">{charts}</div><details><summary>查看 cosine、summary 幅值和几何指标原始记录</summary>
<div style="overflow:auto"><table><tr><th>组</th><th>数据</th><th>步数</th><th>平面残差</th><th>GT AbsRel</th><th>边缘误差</th><th>Spatial cos</th><th>Summary cos</th><th>Summary 幅值比</th></tr>{''.join(table)}</table></div></details></section></main>'''
    if (root / 'result-section.html').exists():
        new = new.replace('</main>', (root / 'result-section.html').read_text() + '</main>')
    prefix, rest = source.split('<main>', 1)
    old, suffix = rest.rsplit('</main>', 1)
    if (root / 'result-section.html').exists():
        # Only one interactive viewer owns these controls; retain prior metrics/diagnostic images.
        start = old.index('<section id="geometry">')
        end = old.index('<section><details>', start)
        old = old[:start] + old[end:]
        old = old.replace('id="results"', 'id="prior-results"')
    old_title = '小子集结果' if (root / 'result-section.html').exists() else '小子集结果与交互点云'
    page = prefix + new + '<details style="max-width:1140px;margin:20px auto"><summary style="padding:20px">展开上一轮：64 张图的' + old_title + '</summary><main>' + old + '</main></details>' + suffix
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.incoming')
    temporary.write_text(page)
    temporary.replace(output)
    print(json.dumps({'page': str(output), 'bytes': output.stat().st_size, 'stage': state['stage']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.base, args.output)
