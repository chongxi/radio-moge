"""Render the current full-data run while retaining the prior experiment viewer."""

import argparse
from html import escape
import json
from pathlib import Path
import time


def read(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def lines(path):
    if not path.exists():
        return []
    result = []
    for row in path.read_text().splitlines():
        try:
            result.append(json.loads(row))
        except json.JSONDecodeError:
            pass
    return result


def chart(rows, keys, title):
    colors = ['#65c7f7', '#f0b760']
    valid = [r for r in rows if all(k in r for k in keys)]
    if not valid:
        return f'<section><h3>{title}</h3><p>等待首次完整评估，尚无新结果。</p></section>'
    xmax = max(1, max(r['step'] for r in valid))
    values = [r[k] for r in valid for k in keys]
    low, high = min(values) - .01, max(values) + .01
    paths = []
    for key, color in zip(keys, colors):
        points = ' '.join(f"{40+520*r['step']/xmax:.2f},{170-140*(r[key]-low)/(high-low):.2f}"
                          for r in valid)
        paths.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for r in valid:
            x, y = 40+520*r['step']/xmax, 170-140*(r[key]-low)/(high-low)
            paths.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"><title>'
                         f'{key} · step {r["step"]} · {r[key]:.6f}</title></circle>')
    return (f'<section><h3>{title}</h3><svg viewBox="0 0 600 210" role="img" aria-label="{title}">'
            f'<text x="2" y="28">{high:.3f}</text><text x="2" y="180">{low:.3f}</text>'
            '<path d="M40 25V180H565" stroke="#61708a" fill="none"/>' + ''.join(paths) +
            f'<text x="40" y="203">0</text><text x="500" y="203">{xmax} steps</text></svg>' +
            '<p>' + ' · '.join(f'<span style="color:{c}">{escape(k)}</span>'
                               for k,c in zip(keys,colors)) + '</p></section>')


def render(root, page):
    monitor = read(root/'monitor.json', {})
    status = read(root/'status.json', {})
    identity = read(root/'train/wandb-run.json', {})
    train = lines(root/'train/train.jsonl')
    val = lines(root/'train/validation.jsonl')
    last = train[-1] if train else {}
    history = root/'previous-results.html'
    doc = (root/'experiment.md').read_text() if (root/'experiment.md').exists() else ''
    stage = status.get('stage', '准备中，尚未启动')
    perf = {}
    if (root/'train/performance.json').exists():
        perf = json.loads((root/'train/performance.json').read_text())
    batch = perf.get('effective_batch', 16)
    throughput = read(root/'train/throughput-benchmark.json', {})
    utilization = read(root/'performance-utilization-verification.json', {})
    performance_panel = ''
    if throughput.get('stage') == 'passed':
        entries = []
        for row in throughput['paired_rounds']:
            selected = all(row[k] == perf.get(k) for k in ('workers','compute_batch','effective_batch'))
            entries.append(f'<tr><td>{row["effective_batch"]}{" · 已采用" if selected else ""}</td>'
                           f'<td>{row["compute_batch"]}</td><td>{row["workers"]}</td>'
                           f'<td>{row["images_per_second"]:.2f}</td>'
                           f'<td>{row["peak_memory_gib"]:.2f}</td></tr>')
        actual = ''
        if utilization.get('stage') == 'passed':
            gpu = utilization['summary']
            loop_speed = utilization['production_train_loop_images_per_second']
            wall_speed = utilization['production_wall_images_per_second']
            actual = (f'<p><strong>正式续训实测：GPU 平均利用率 {gpu["utilization.gpu"]["mean"]:.1f}%'
                      f'，显存 {gpu["memory.used"]["mean"]/1024:.1f} GiB，平均功耗 '
                      f'{gpu["power.draw"]["mean"]:.0f} W / 700 W。</strong></p>'
                      f'<p>训练循环 {loop_speed:.2f} 图/秒（调整前约 25.35），'
                      f'约 {loop_speed/25.35:.2f} 倍；日志区间含保存等开销的实际速度 '
                      f'{wall_speed:.2f} 图/秒。GPU 连续采样 30 次，期间没有完成新的验证。</p>')
        performance_panel = ('<section id="performance"><h2>H200 吞吐实测与 batch 选择</h2>' + actual +
            '<div class="scroll"><table><thead><tr><th>有效 batch</th><th>计算 batch 上限</th>'
            '<th>预取进程</th><th>短测图/秒 ↑</th><th>峰值分配显存 GiB</th></tr></thead><tbody>' +
            ''.join(entries) + '</tbody></table></div>'
            '<p class="muted">两轮相反顺序；每项 2 步预热、8 步计时。包括在线数据读取、两个冻结编码器、反向传播和 Adam。'
            '不同宽高的图片分开计算，因此实际计算 batch 可低于上限。160 次测试更新全部丢弃，从第 2,735 步恢复权重、Adam、学习率和随机状态。'
            '这是吞吐选型，不代表更大的 batch 已改善 cosine 或泛化。余下训练量与学习率按图片数衔接，原截止时间保持。</p></section>')
    performance_note = (f'性能优化后有效 batch {batch}，计算 batch 上限 {perf["compute_batch"]}，预取进程 {perf["workers"]}；学习率按图片数衔接，原预算不延长。' if perf else '初始有效 batch 16（计算 4 × 累积 4）。')
    if stage == 'training' and not train:
        stage = '正式训练初始化 · 尚未记录优化器更新'
    started = status.get('original_started_at', status.get('started_at'))
    stamp = time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(monitor.get('checked_at', time.time())))
    link = (f'<a href="{escape(identity["url"])}" target="_blank" rel="noopener">打开本次 W&B</a>'
            if identity.get('url') else '新 W&B 尚未初始化')
    cells = [f'<tr><td>{r["step"]}</td><td>{r.get("macro/spatial/cosine",float("nan")):.6f}</td>'
             f'<td>{r.get("macro/summary/cosine",float("nan")):.6f}</td>'
             f'<td>{r.get("macro/summary/norm_ratio",float("nan")):.6f}</td></tr>' for r in val]
    download = read(root.parent/'scannetpp-download-20261028/monitor.json', {})
    download_panel = ''
    if download:
        state = download['state']
        labels = {'waiting_for_training_updates':'等待正式训练更新', 'starting_download':'准备下载',
                  'downloading':'下载中', 'extracting':'校验并解压', 'complete':'下载完成',
                  'needs_inspection':'下载暂停，正在检查'}
        download_stamp = time.strftime('%Y-%m-%d %H:%M UTC',time.gmtime(download['checked_at']))
        download_panel = f'''<section id="scannetpp"><h2>ScanNet++ 下载</h2>
<p><span class="pill">{escape(labels.get(state['stage'],state['stage']))}</span>
 · 已完成传输 {state['verified_download_bytes']/1e9:.2f} GB / {state['completed_files']:,} 个文件</p>
<p>官方默认资产与划分，清单 {state.get('scenes',0):,} 个场景。单进程限速 20 MiB/s，逐文件校验；总字节数尚未核实，不用估算大小显示完成百分比。</p>
<p class="muted">下载开始前已确认正式训练更新。这批新数据尚未加入当前训练。状态更新：{download_stamp}</p></section>'''
    body = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>4×2048 QK norm · 全量训练</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#101723;color:#e6edf7;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1200px;margin:auto;padding:28px 20px}}h1{{font-size:32px;line-height:1.3}}h2{{font-size:23px}}h3{{font-size:17px}}a{{color:#75d0ff}}section,details{{background:#182232;border:1px solid #304057;border-radius:12px;padding:20px;margin:14px 0}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.grid section{{margin:0;min-width:0}}svg{{width:100%;color:#bbc7d8}}svg text{{fill:currentColor;font-size:12px}}.muted{{color:#acbbcf}}.pill{{display:inline-block;padding:4px 12px;background:#26405a;border-radius:20px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #354359}}.scroll{{overflow-x:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.7 system-ui}}iframe{{border:0;width:100%;height:960px;background:white}}@media(max-width:700px){{.grid{{grid-template-columns:1fr}}h1{{font-size:25px}}main{{padding:18px 12px}}section,details{{padding:15px}}}}</style><main>
<p class="muted">RADIO → 冻结 MoGe · 24 小时首阶段</p><h1>4×2048 joint attention + QK norm</h1>
<p><span class="pill">{escape(stage)}</span>　{link}</p><p class="muted">更新：{stamp} · 已记录更新步 {last.get('step',0):,} · 训练池 796,478 图 / 七来源 · 固定评估 1,268 图</p>
{download_panel}
{performance_panel}
<section><h2>本次训练与之前的短筛选</h2><p>从头初始化 seed43；在线 RGB 与数据增强。{performance_note}四层、32头、2D RoPE、head gate、4个 register、SwiGLU、QK norm。预算为原始 86,400 秒截止时间或 1,920,000 次训练图片呈现，先到即停；batch 增大时剩余更新步数相应减少。</p>
<p>四个 RADIO spatial tap 继续逐图清理。按用户新要求，两个原始 summary token 也分别做 8×中位数异常幅值置零，再投影。MoGe 教师目标不清理。summary 清理是新增实验项，之前短筛选的收益不能视为它的效果证据。</p>
<p>目标 spatial / summary cosine 均超过 0.98；尚未达成。cosine、幅值比、真实深度和几何质量分别判断。</p></section>
<div class="grid">{chart(val,['macro/spatial/cosine','macro/summary/cosine'],'验证 cosine · 七来源宏平均')}{chart(val,['macro/spatial/norm_ratio','macro/summary/norm_ratio'],'验证幅值比 · 七来源宏平均')}</div>
<section><h2>完整验证点</h2><div class="scroll"><table><thead><tr><th>步数</th><th>spatial cos</th><th>summary cos</th><th>summary norm ratio</th></tr></thead><tbody>{''.join(cells)}</tbody></table></div><p class="muted">训练 loss / cosine 与完整验证分开记录。初始验证完成前不展示推断出的收益。</p></section>
<details id="experiment"><summary>实验 Markdown：完整结构、清理规则、训练差异</summary><pre>{escape(doc)}</pre></details>
'''
    if started:
        body += '<p class="muted">任务启动：' + time.strftime('%Y-%m-%d %H:%M UTC',time.gmtime(started)) + '</p>'
    if history.exists():
        body += f'<details id="history"><summary>此前已完成的结构筛选与点云对比</summary><iframe title="历史实验" loading="lazy" srcdoc="{escape(history.read_text(),quote=True)}"></iframe></details>'
    body += '</main></html>'
    temp = page.with_suffix('.incoming')
    temp.write_text(body)
    temp.replace(page)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--page',type=Path,default=Path('.kanna/visualizations/teacher-plane-pilot.html'))
    a=p.parse_args()
    render(a.root,a.page)
