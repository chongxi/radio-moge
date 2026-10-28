"""Show the bounded direct-cosine experiment, preserving actual prior results."""
import argparse
import html
import json
from pathlib import Path
import time

NAME = 'adapter-dual-cosine512-pilot-20261028'


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines(keepends=True)
            if line.endswith('\n')] if path.exists() else []


def render(base, output):
    root = base / NAME
    status = json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {'stage':'preparing'}
    data, cards = {}, []
    labels = {'control':'原损失', 'angular':'显式双 cosine ＋原 MSE'}
    for arm in labels:
        train = rows(root/arm/'train.jsonl')
        step = train[-1]['step'] if train else 0
        wb = root/arm/'wandb-run.json'
        link = ''
        if wb.exists():
            rid = json.loads(wb.read_text())['id']
            link = f'<a href="https://forge.coreweave.com/wandb/brainsim/radio-moge/runs/{html.escape(rid)}" target="_blank" rel="noopener">W&amp;B ↗</a>'
        data[arm] = {part:rows(root/arm/file) for part,file in
                     [('validation','validation.jsonl'),('train-probe','train-probe.jsonl')]}
        values = data[arm]['validation']
        latest = ('<p>验证 Spatial：<b>%.6f</b> · Summary：<b>%.6f</b></p>' %
                  (values[-1]['spatial/cosine'],values[-1]['summary/cosine'])) if values else '<p>等待初始评估</p>'
        cards.append(f'<article><h2>{labels[arm]}</h2><p>{step} / 2,000 更新　{link}</p>'
                     f'<progress max="2000" value="{step}"></progress>{latest}</article>')
    previous = base/'adapter-joint-cosine512-pilot-20261028/result-section.html'
    history = ''
    if previous.exists():
        old = '<!doctype html><meta charset="utf-8"><style>body{font:15px system-ui;padding:16px;color:#18312f}table{width:100%}td,th{padding:8px;text-align:left}.cards{display:flex;gap:20px}</style>' + previous.read_text()
        history = '<iframe title="已完成联合 attention 对照" loading="lazy" style="width:100%;height:1100px;border:0" srcdoc="'+html.escape(old,quote=True)+'"></iframe>'
    result = (root/'result-section.html').read_text() if (root/'result-section.html').exists() else ''
    page = '''<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Spatial ＋ Summary · 目标 cosine 0.98</title>
<style>body{margin:0;background:#f6f5f1;color:#18312f;font:16px system-ui,sans-serif}main,details.history{max-width:1160px;margin:24px auto;padding:20px}h1{font-size:clamp(28px,4vw,44px)}h2{font-size:22px}p{line-height:1.75}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:16px}article,.panel{background:white;border:1px solid #d9dfd8;border-radius:14px;padding:20px;margin:14px 0;min-width:0}a{color:#006d70}button,select{font:inherit;padding:8px;margin:4px;border-radius:6px;border:1px solid #bbc8c4}progress{width:100%;accent-color:#007d7c}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:13px;text-align:left;border-bottom:1px solid #dde1dc;vertical-align:top}svg{width:100%;height:auto}.notice{border-left:4px solid #bd7428;padding:12px 18px;background:#fff5e6}.scroll{overflow:auto}summary{cursor:pointer;padding:16px}.muted{color:#5d6a67}</style></head><body><main>
<small>目标函数实验 · 结构保持一致</small><h1>让 Spatial 与 Summary 都接近 0.98</h1>
<p>当前阶段：<b>__STAGE__</b> · __STAMP__ <button onclick="location.reload()">刷新</button></p>
<p>先检验直接优化目标是否有效：同一个原 adapter、同一父模型、相同数据顺序和全新优化器。保留两项归一化 MSE，同时提高 Spatial cosine 权重并加入 Summary cosine。</p>
<div class="scroll"><table><tr><th>损失项</th><th>原损失</th><th>显式双 cosine</th></tr><tr><td>Spatial 归一化 MSE</td><td>1</td><td>1</td></tr><tr><td>Summary 归一化 MSE</td><td>1</td><td>1</td></tr><tr><td>1 − Spatial cosine</td><td>0.1</td><td>1</td></tr><tr><td>1 − Summary cosine</td><td>0</td><td>1</td></tr></table></div>
<p>MSE 同时约束幅值与方向；cosine 仅约束方向。实际特征未经单位化后直接传给冻结的 MoGe decoder。由于权重不同，两组的加权总 loss 数值不能直接当作优劣比较。</p>
<p>512 张图／256 个训练场景，原有 16 场景验证与 16 张训练诊断分开记录。两组各最多 2,000 步或 20 分钟，每 200 步评估；LR 1×10⁻⁵、有效 batch 4，约 15.63 次数据遍历。串行执行，按相同步数比较终点。</p>
<p class="notice">0.98 是待检验目标，尚未证明能达到。以终点验证集的两个图像均值都超过 0.98 为本轮门槛；中间越线不算终点达标，也不自动提前停止。这个被多次查看的 16 图面板只作筛选，任何收益还需更大、未用于调参的验证集确认。幅值与真实深度必须一起看。</p>
<div class="grid">__CARDS__</div>__RESULT__
<section class="panel"><h2>学习曲线与目标距离</h2><p>蓝绿＝原损失；橙色＝显式双 cosine。两个 cosine 与幅值、真实深度分别记录。</p>
<label>数据 <select id="dual-part"><option value="validation">验证 · 16 场景</option><option value="train-probe">训练诊断 · 16 图</option></select></label>
<label>横轴 <select id="dual-axis"><option value="step">优化步数</option><option value="optimizer_seconds">训练循环秒数</option><option value="elapsed_seconds">总运行秒数</option></select></label>
<label><input id="dual-target" type="checkbox" checked>显示 0.98 目标线</label><div id="dual-charts" class="grid"></div>
<p class="muted">训练诊断分数高不等于泛化。若训练与验证都停滞，再区分优化与表示能力；若训练继续上升而验证停滞，则优先检查泛化。单次小样本实验不能证明最优权重或收敛上限。</p></section>
</main><details class="history"><summary>已完成的联合 attention 对照与实际点云（历史）</summary>__HISTORY__</details>
<script>const data=__DATA__;function draw(){const part=document.getElementById('dual-part').value,axis=document.getElementById('dual-axis').value,target=document.getElementById('dual-target').checked;const keys=[['spatial/cosine','Spatial cosine ↑'],['summary/cosine','Summary cosine ↑'],['spatial/norm_ratio','Spatial 幅值比 · 理想 1'],['summary/norm_ratio','Summary 幅值比 · 理想 1'],['gt_student/abs_rel','真实深度 AbsRel ↓'],['planes/student/normalized_rms','平面残差 ↓']];document.getElementById('dual-charts').innerHTML=keys.map(([key,label])=>{const series=['control','angular'].map(a=>data[a][part].filter(r=>Number.isFinite(r[key])&&Number.isFinite(r[axis]))),all=series.flat();if(!all.length)return `<article><b>${label}</b><p>等待完成的评估</p></article>`;let lo=Math.min(...all.map(r=>r[key])),hi=Math.max(...all.map(r=>r[key]));const angular=key.endsWith('/cosine');if(target&&angular){lo=Math.min(lo,.98);hi=Math.max(hi,.98);}const pad=Math.max((hi-lo)*.12,.0003);lo-=pad;hi+=pad;const max=Math.max(1,...all.map(r=>r[axis])),x=v=>60+v/max*430,y=v=>205-(v-lo)/(hi-lo)*165;let svg=`<svg viewBox="0 0 520 245" role="img" aria-label="${label}"><path d="M60 35V205H490" fill="none" stroke="#9aaca6"/>`;for(let i=0;i<4;i++){const v=lo+(hi-lo)*i/3;svg+=`<text x="54" y="${y(v)+4}" font-size="11" text-anchor="end">${v.toFixed(4)}</text>`;}if(target&&angular)svg+=`<path d="M60 ${y(.98)}H490" stroke="#986730" stroke-dasharray="5 5"/><text x="485" y="${y(.98)-5}" text-anchor="end" font-size="11">目标 0.98</text>`;series.forEach((s,i)=>{const color=['#007d7c','#cf7834'][i];svg+=`<polyline points="${s.map(r=>x(r[axis])+','+y(r[key])).join(' ')}" fill="none" stroke="${color}" stroke-width="2"/>`;for(const r of s)svg+=`<circle cx="${x(r[axis])}" cy="${y(r[key])}" r="4" fill="${color}"><title>${i?'显式双 cosine':'原损失'} / ${r.step} 步：${r[key].toFixed(6)}；${axis} ${r[axis].toFixed(1)}</title></circle>`;});svg+=`<text x="60" y="228" font-size="12">0</text><text x="490" y="228" text-anchor="end" font-size="12">${max.toFixed(0)} ${axis==='step'?'steps':'s'}</text></svg>`;return `<article><b>${label}</b>${svg}</article>`;}).join('');}for(const id of ['dual-part','dual-axis','dual-target'])document.getElementById(id).onchange=draw;draw();</script></body></html>'''
    replacements = dict(STAGE=html.escape(status['stage']), STAMP=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),
                        CARDS=''.join(cards), RESULT=result, HISTORY=history,
                        DATA=json.dumps(data,allow_nan=False).replace('</','<\\/'))
    for key,value in replacements.items():
        page = page.replace('__'+key+'__',value)
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_suffix('.incoming')
    tmp.write_text(page)
    tmp.replace(output)
    print(json.dumps(dict(page=str(output),stage=status['stage'],bytes=output.stat().st_size)))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    render(args.base,args.output)
