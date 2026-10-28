"""Live adapter learning-speed comparison and sourced, staged research agenda."""
import argparse
import html
import json
from pathlib import Path
import time


NAME = 'adapter-joint-cosine512-pilot-20261028'
SOURCES = [
    ('2024 · 2D RoPE', 'https://arxiv.org/abs/2403.13298',
     '视觉实验支持相对位置编码与分辨率外推；我们的 adapter 收益仍需实测。',
     '固定 joint 结构：当前 2D sin/cos vs axial RoPE，再考虑 mixed RoPE。'),
    ('2020／近年广泛采用 · SwiGLU', 'https://arxiv.org/abs/2002.05202',
     'FFN 的乘性门控；不同于 attention 输出门控。', '固定层数和宽度，GELU vs 参数量近似匹配的 SwiGLU。'),
    ('2025 · Gated Attention', 'https://arxiv.org/abs/2505.06708',
     'SDPA 后的 query-dependent sigmoid gate；论文主要证据来自语言模型。',
     '一次只增加 attention gate，观察 summary 是否过度支配 spatial。'),
    ('2024 · Register tokens', 'https://arxiv.org/abs/2309.16588',
     '提供额外计算 token，减少高幅值背景 token 伪影；在 frozen backbone 后加不能保证修复已有伪影。',
     '联合 attention 增加 0 vs 4 个寄存 token，保持输出仍为 spatial + summary。'),
    ('2023 · ViT-Adapter / 多尺度先验', 'https://arxiv.org/abs/2205.08534',
     '密集预测需要局部与多尺度信息；原论文不是我们当前小 adapter 的直接证明。',
     '同一 RADIO taps，局部卷积与多尺度融合 vs 纯全局 attention。'),
    ('2023 · EfficientViT', 'https://hanlab.mit.edu/projects/efficientvit',
     '多尺度线性 attention 适合高分辨率效率；理论 FLOPs 不等于当前 H200 上更快。',
     '只在实测 token 二次复杂度成为瓶颈后比较 linear / window attention。'),
    ('2026 · Spiral RoPE', 'https://arxiv.org/abs/2602.03227',
     '多方向二维旋转位置编码的新工作；本任务尚未验证。',
     '标准 2D RoPE 有收益后再比较斜向关系编码，避免直接展开大组合。'),
    ('2025 · DepthFM', 'https://depthfm.github.io/',
     'Flow matching 已用于单目深度，借用了预训练生成模型先验。',
     '探索 feature-space conditional flow，与相近容量的直接残差预测比较；1 / 2 / 4 次调用。'),
    ('2026 · Marigold V2', 'https://arxiv.org/abs/2609.08084',
     '从预训练 flow-matching DiT 得到单步深度预测；不能把大型生成先验的收益归给 FM 单一因素。',
     '记录推理调用次数、实际延迟与训练成本，不预设生成式一定更准或更慢。'),
]


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines(keepends=True) if s.endswith('\n')] if path.exists() else []


def render(base, output):
    root = base / NAME
    state = json.loads((root / 'status.json').read_text()) if (root / 'status.json').exists() else {'stage': 'preparing'}
    labels = {'control': '原 adapter', 'joint': '新增 joint attention · 2 blocks'}
    data, cards = {}, []
    for arm in ('control', 'joint'):
        train = rows(root / arm / 'train.jsonl')
        step = train[-1]['step'] if train else 0
        identity = root / arm / 'wandb-run.json'
        link = ''
        if identity.exists():
            rid = json.loads(identity.read_text())['id']
            link = f'<a href="https://forge.coreweave.com/wandb/brainsim/radio-moge/runs/{html.escape(rid)}" target="_blank" rel="noopener">W&amp;B ↗</a>'
        cards.append(f'<article><b>{labels[arm]}</b><p>{step} / 600 更新</p><progress value="{step}" max="600"></progress> {link}</article>')
        data[arm] = {part: rows(root / arm / file) for part, file in
                     [('validation', 'validation.jsonl'), ('train-probe', 'train-probe.jsonl')]}
    source_rows = ''.join(f'<tr><td><a href="{url}" target="_blank" rel="noopener">{title}</a></td><td>{evidence}</td><td>{experiment}</td></tr>'
                          for title, url, evidence, experiment in SOURCES)
    history = base / 'teacher-plane-ownflat512-pilot-20261028/results.html'
    old = history.read_text() if history.exists() else ''
    old = ('<iframe title="已完成平面实验" loading="lazy" style="width:100%;height:1000px;border:0" srcdoc="'
           + html.escape(old, quote=True) + '"></iframe>') if old else ''
    payload = json.dumps(data, allow_nan=False).replace('</', '<\\/')
    stamp = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    result = (root / 'result-section.html').read_text() if (root / 'result-section.html').exists() else ''
    speed = (root / 'speed-section.html').read_text() if (root / 'speed-section.html').exists() else ''
    page = '''<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Adapter · 学习速度与深度精度</title>
<style>body{margin:0;background:#f6f5f1;color:#18312f;font:16px system-ui,sans-serif}main,.history{max-width:1160px;margin:30px auto;padding:20px}h1{font-size:clamp(28px,4vw,44px)}p{line-height:1.75}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}article,.panel{background:white;border:1px solid #d9dfd8;border-radius:14px;padding:20px;margin:14px 0}a{color:#006d70}button,select{font:inherit;padding:8px;margin:4px;border-radius:6px;border:1px solid #bbc8c4}progress{width:70%;accent-color:#007d7c}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:13px;text-align:left;border-bottom:1px solid #dde1dc;vertical-align:top}svg{width:100%;height:auto}.muted{color:#5d6a67}.notice{border-left:4px solid #bd7428;padding:12px 18px;background:#fff5e6}.scroll{overflow:auto}details>summary{cursor:pointer;padding:16px}text{font-family:system-ui}#joint-charts article{min-width:0}</style></head><body><main>
<small>研究与实验 · 冻结视觉特征 → MoGe 特征接口 → 深度</small><h1>哪些 adapter 能学得更快、更准？</h1>
<p>当前阶段：<b>__STAGE__</b> · 更新于 __STAMP__ <button onclick="location.reload()">刷新</button></p>
<p>第一轮比较原 adapter 与新增的 spatial／summary 双向联合 attention。原来的 2 个 spatial block 保留；新分支是 2 个宽度 256 的 block。因此这里的“2 层”是新增分支深度，并非整个网络只有 2 层。</p>
<p>两组从同一父模型出发，新分支以零残差初始化，初始输出相同。使用同一 512 张图／256 场景、有效 batch 4、LR 1×10⁻⁵；仅用原特征损失。每组最多 600 次更新或 20 分钟，每 100 步评估。两组串行执行，随后评估独立截图。</p>
<p class="notice">这轮检验已有 adapter 的增量改造，不是从零训练架构排名。joint 组增加了参数，因此若有收益，后续还需相近容量的非交互分支对照。16 张验证图只用于小规模筛选；不能据此宣布最终最优或已经收敛。</p>
<div class="grid">__CARDS__</div>__RESULT____SPEED__<section class="panel"><h2>Cosine 学习曲线</h2>
<p>Spatial 与 summary 分开看，同时保留幅值、真实深度和几何指标。蓝绿＝原 adapter；橙色＝joint。</p>
<label>横轴 <select id="joint-axis"><option value="step">优化步数</option><option value="optimizer_seconds">训练循环秒数（不含评估／加载）</option><option value="elapsed_seconds">总运行秒数（含评估／加载）</option></select></label>
<label>数据 <select id="joint-part"><option value="validation">验证集 · 16 场景</option><option value="train-probe">训练诊断 · 16 张，非独立验证</option></select></label>
<div id="joint-charts" class="grid"></div><p class="muted">首个达到初始 cosine +0.001 的评估点用于比较速度；未达到时明确记为未达到。单次运行的小差异不作显著性结论。分支延迟包含 adapter 与固定 decoder，不含共享 RADIO backbone。</p></section>
<section class="panel"><h2>分阶段实验，而不是一次改完</h2><ol><li>当前：原 adapter vs joint attention，建立学习曲线与耗时基线。</li><li>结构：新增分支 1 / 2 / 4 blocks、相近容量的非交互对照；每次只改一个因素。</li><li>位置与非线性：2D sin/cos vs RoPE；GELU vs SwiGLU；单独测试 attention gate。</li><li>另一条训练目标路线：固定视觉输入、预测目标和容量，直接预测 vs conditional feature flow matching；分别测 1 / 2 / 4 次网络调用。</li></ol>
<p>训练速度、推理速度、最终精度是三个维度。获胜候选还需扩大验证场景并重复训练，不能只按训练 cosine 或最好的一步选择。</p>
<div class="scroll"><table><tr><th>一手文献</th><th>已有证据与边界</th><th>适合我们的实验</th></tr>__SOURCES__</table></div></section>
<section class="panel"><h2>Flow matching 如何接到当前系统？</h2><p>候选方案是在特征空间学习从初始 adapted feature 到目标 MoGe feature 的修正轨迹，以冻结 RADIO 特征和时间 t 为条件。训练时可见目标特征；推理时只从可计算的初始特征出发，绝不能输入教师特征。它与直接残差预测必须用相近容量和相同数据比较，才能分辨迭代过程是否有实际收益。</p><p class="muted">这是实验假设，尚未启动。大模型论文的收益包含生成预训练先验，不能直接迁移成小 adapter 的效果承诺。</p></section>
</main><details class="history"><summary>展开已完成的平面监督实验与交互点云（历史）</summary>__HISTORY__</details>
<script>const jointData=__DATA__;function jointDraw(){const axis=document.getElementById('joint-axis').value,part=document.getElementById('joint-part').value;const keys=[['spatial/cosine','Spatial cosine ↑'],['summary/cosine','Summary cosine ↑'],['gt_student/abs_rel','真实深度 AbsRel ↓'],['summary/norm_ratio','Summary 幅值比（理想 1）']];document.getElementById('joint-charts').innerHTML=keys.map(([key,title])=>{const series=['control','joint'].map(a=>jointData[a][part].filter(r=>Number.isFinite(r[key])&&Number.isFinite(r[axis])));const all=series.flat();if(!all.length)return `<article><b>${title}</b><p>等待完成的评估点</p></article>`;let xmin=0,xmax=Math.max(1,...all.map(r=>r[axis])),lo=Math.min(...all.map(r=>r[key])),hi=Math.max(...all.map(r=>r[key]));let pad=Math.max((hi-lo)*.15,key.includes('cosine')?.0003:.001);lo-=pad;hi+=pad;const x=v=>60+v/xmax*430,y=v=>205-(v-lo)/(hi-lo)*165;let svg='<svg viewBox="0 0 520 245" role="img" aria-label="'+title+'"><path d="M60 35V205H490" fill="none" stroke="#9aaca6"/>';for(let i=0;i<4;i++){let v=lo+(hi-lo)*i/3;svg+=`<text x="54" y="${y(v)+4}" text-anchor="end" font-size="11">${v.toFixed(4)}</text>`;}series.forEach((s,i)=>{const color=['#007d7c','#cf7834'][i];svg+=`<polyline points="${s.map(r=>x(r[axis])+','+y(r[key])).join(' ')}" stroke="${color}" stroke-width="2" fill="none"/>`;s.forEach(r=>svg+=`<circle cx="${x(r[axis])}" cy="${y(r[key])}" r="4" fill="${color}"><title>${i?'joint':'control'} / step ${r.step}: ${r[key].toFixed(6)}; ${axis} ${r[axis].toFixed(1)}</title></circle>`);});svg+=`<text x="60" y="228" font-size="12">0</text><text x="490" y="228" text-anchor="end" font-size="12">${xmax.toFixed(0)} ${axis==='step'?'steps':'s'}</text></svg>`;return `<article><b>${title}</b>${svg}</article>`;}).join('');}document.getElementById('joint-axis').onchange=jointDraw;document.getElementById('joint-part').onchange=jointDraw;jointDraw();</script></body></html>'''
    for key, value in {'STAGE': html.escape(state['stage']), 'STAMP': stamp,
                        'CARDS': ''.join(cards), 'SOURCES': source_rows, 'HISTORY': old, 'DATA': payload,
                        'RESULT': result, 'SPEED': speed}.items():
        page = page.replace('__' + key + '__', value)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.incoming')
    temporary.write_text(page)
    temporary.replace(output)
    print(json.dumps(dict(page=str(output), stage=state['stage'], bytes=output.stat().st_size)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.base, args.output)
