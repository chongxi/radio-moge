"""Render width/depth learning curves and keep the previous real geometry viewer."""

import html
import json
import time


def read(path, fallback=None):
    return json.loads(path.read_text()) if path.exists() else fallback


def render(root, destination):
    plan = read(root / "plan.json")
    attention = plan.get("screen_profile") == "attention_tricks"
    wide4096 = plan.get("screen_profile") == "wide4096"
    wide = plan.get("screen_profile") in ("wide2048", "wide4096")
    count = len(plan["candidates"])
    m = read(root / "monitor.json", {})
    s = m.get("status") or {}
    pre = read(root / "preflight-verification.json", {}).get("candidates", {})
    series = {}
    runs = {}
    for c in plan["candidates"]:
        arm = root / c["name"]
        series[c["name"]] = {}
        for part in ("validation", "train_probe"):
            p = arm / f"{part}-nfe1.jsonl"
            series[c["name"]][part] = (
                [json.loads(x) for x in p.read_text().splitlines(keepends=True) if x.endswith("\n")]
                if p.exists()
                else []
            )
        runs[c["name"]] = read(arm / "wandb-run.json", {}).get("id")
    old = root.parent / (
        "adapter-combined-wide4096-screen-20261028"
        if attention
        else
        "adapter-combined-wide2048-screen-20261028"
        if wide4096
        else "adapter-combined-capacity-screen-20261028"
        if wide
        else "adapter-scratch-diverse-screen-20261028"
    )
    prior = read(old / "result-report.json", {})
    oldhtml = (
        '<meta charset="utf-8"><style>body{font:15px system-ui;padding:16px;color:#19342f}.controls{display:flex;flex-wrap:wrap;gap:8px}.controls label{max-width:100%;min-width:0}select{max-width:100%}</style><h2>上一轮：16 架构从零初筛</h2><p>'
        + html.escape(prior.get("conclusion_zh", ""))
        + "</p>"
        + (old / "result-section.html").read_text()
    )
    if wide:
        oldhtml = oldhtml.replace("上一轮：16 架构从零初筛", "上一轮：六种容量、两个初始化 seed")
    if wide4096:
        oldhtml = oldhtml.replace(
            "上一轮：六种容量、两个初始化 seed", "上一轮：宽度 1024 / 1536 / 2048，两个初始化 seed"
        )
    if attention:
        oldhtml = oldhtml.replace("上一轮：16 架构从零初筛", "上一轮：2048 / 4096 宽度对照")
    progress = f"已完成 {len(s.get('completed', []))} / {count}；当前 {s.get('current') or '缓存加载／GPU预检'}，{s.get('step', 0)} 步"
    stage = s.get("stage", "prepared")
    if stage == "complete":
        confirmation = m.get("confirmation") or {}
        progress = f"{count} 次训练已完成；只读深度复测已完成 {len(confirmation.get('completed', []))} / {count}"
        stage = "训练结束，核对复测与备份"
    if read(root / "completion-verification.json", {}).get("stage") == "passed":
        stage = "全部完成：训练、评估与备份"
    document = """<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>组合 Adapter · 参数量扩展</title><style>
body{margin:0;background:#f5f4ef;color:#19342f;font:16px system-ui}main{max-width:1200px;margin:auto;padding:20px}h1{font-size:clamp(28px,4vw,43px)}h2{font-size:23px}p{line-height:1.7}.panel,article{background:white;border:1px solid #d4ddd6;border-radius:12px;padding:18px;margin:16px 0;min-width:0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,330px),1fr));gap:16px}.controls{display:flex;flex-wrap:wrap;gap:12px}.controls>label{min-width:0;max-width:100%}select,button{font:inherit;padding:7px;margin:4px;box-sizing:border-box;max-width:calc(100% - 8px)}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left;white-space:nowrap}svg{width:100%;height:auto}.notice{background:#fff2dd;border-left:4px solid #b47e37;padding:14px}a{color:#00776d}summary{padding:15px;cursor:pointer}</style></head><body><main>
<small>组合设计不变 · 深度 × 宽度 · 两个初始化 seed</small><h1>加大参数量，能否学得更好？</h1>
<p><b>__STAGE__</b> · __TIME__ <button onclick="location.reload()">刷新</button></p><p>__PROGRESS__</p>
<div class="grid"><article><h2>六个规模</h2><p>4 / 8 个 block × 宽度 512 / 768 / 1024。约 17.6M～110.8M 参数。全部保留联合 spatial＋summary attention、2D RoPE、gate、4 个 register 和 SwiGLU。</p></article><article><h2>配对短测</h2><p>每个规模独立随机初始化 seed 42、43；相同数据顺序、学习率和损失。每次 1,000 步，有效 batch 16；0 / 100 / 250 / 500 / 1,000 步记录结果。预算上限每次 10 分钟，整轮 1 小时。</p></article><article><h2>同一数据与目标</h2><p>七来源 1,792 张训练图，224 张筛选验证图。空间和 summary 的 MSE＋cosine 权重均为 1。冻结特征缓存复用；所有 adapter 都从零训练。</p></article></div>
<p class="notice">这一轮检查容量收益，未增加训练数据或步数。两个 seed 只改变初始化；相同 LR 不保证对每个规模都最优。已有的 224 图确认面板会对全部 12 次训练做复测，不能称为新的未见测试集；实际几何检查固定 56 图，其中 32 张 NYU 有 GT。</p>
<section class="panel"><h2>每次训练的最新完整评估</h2><p>不把尚未完成的 seed 当作平均值，也不以训练集指标宣布泛化改善。</p><div class="scroll" id="ranking"></div></section>
<section class="panel"><h2>相同 seed 下的容量对照</h2><div class="controls"><label>Seed <select id="seed"><option>42</option><option>43</option></select></label><label>候选 <select id="shape"></select></label><label>数据 <select id="partition"><option value="validation">筛选验证</option><option value="train_probe">训练诊断 112 图</option></select></label><label>来源 <select id="source"><option value="macro">七来源宏平均</option>__SOURCES__</select></label><label>横轴 <select id="axis"><option value="step">步数</option><option value="optimizer_seconds">训练循环秒数</option><option value="elapsed_seconds">含评估总秒数</option></select></label><label><input id="target" type="checkbox">0.98 目标</label></div><div class="grid" id="charts"></div><p>蓝绿＝同 seed 的 4×512；橙色＝选中规模。时间来自缓存特征训练，不代表端到端全数据收敛时间。</p></section>
<section class="panel"><h2>重复运行的深度与特征检查</h2><div id="confirmation"></div></section>
__PAIRED__
__RESULT__
<details class="history"><summary>上一轮结果与真实点云（已完成）</summary><iframe title="上一轮架构筛选结果" loading="lazy" style="width:100%;height:1100px;border:0" srcdoc="__HISTORY__"></iframe></details>
<script>const series=__SERIES__,specs=__SPECS__,preflight=__PREFLIGHT__,runs=__RUNS__,confirm=__CONFIRM__;
const $=id=>document.getElementById(id),shapes=['4x512','4x768','4x1024','8x512','8x768','8x1024'];$('shape').innerHTML=shapes.map(s=>`<option value="${s}">${s.replace('x',' × ')}</option>`).join('');$('shape').value='4x1024';
function table(){let rows=specs.map(c=>{const r=series[c.name].validation.at(-1);return `<tr><td>${c.blocks}×${c.width}</td><td>${c.seed}</td><td>${preflight[c.name]?(preflight[c.name].parameters/1e6).toFixed(2):'待预检'}</td><td>${r?r.step:'待运行'}</td><td>${r?r['macro/spatial/cosine'].toFixed(6):'—'}</td><td>${r?r['macro/summary/cosine'].toFixed(6):'—'}</td><td>${r?r.optimizer_seconds.toFixed(1):'—'}</td><td>${runs[c.name]?`<a href="https://forge.coreweave.com/wandb/brainsim/radio-moge/runs/${runs[c.name]}" target="_blank">W&B</a>`:'—'}</td></tr>`});$('ranking').innerHTML='<table><tr><th>深度×宽度</th><th>Seed</th><th>参数 M</th><th>步数</th><th>Spatial</th><th>Summary</th><th>训练秒</th><th>记录</th></tr>'+rows.join('')+'</table>';}
function draw(){const part=$('partition').value,axis=$('axis').value,src=$('source').value,prefix=src==='macro'?'macro/':'dataset/'+src+'/',names=['combined_4x512_s'+$('seed').value,'combined_'+$('shape').value+'_s'+$('seed').value];$('charts').innerHTML=[['spatial/cosine','Spatial cosine ↑'],['summary/cosine','Summary cosine ↑'],['spatial/norm_ratio','Spatial 幅值比 · 1 为教师'],['summary/norm_ratio','Summary 幅值比 · 1 为教师']].map(([metric,label])=>{const key=prefix+metric,curves=names.map(n=>(series[n]?.[part]||[]).filter(r=>Number.isFinite(r[key]))),all=curves.flat();if(!all.length)return `<article><b>${label}</b><p>等待完整评估</p></article>`;let lo=Math.min(...all.map(r=>r[key])),hi=Math.max(...all.map(r=>r[key]));if($('target').checked&&metric.endsWith('cosine')){lo=Math.min(lo,.98);hi=Math.max(hi,.98)}const pad=Math.max(.0003,(hi-lo)*.1);lo-=pad;hi+=pad;const max=Math.max(1,...all.map(r=>r[axis])),x=v=>58+430*v/max,y=v=>205-165*(v-lo)/(hi-lo);let svg=`<svg viewBox="0 0 520 245"><path d="M58 35V205H488" fill="none" stroke="#9aaca6"/>`;for(let i=0;i<4;i++){const v=lo+(hi-lo)*i/3;svg+=`<text x="53" y="${y(v)+4}" text-anchor="end" font-size="11">${v.toFixed(4)}</text>`}if($('target').checked&&metric.endsWith('cosine'))svg+=`<path d="M58 ${y(.98)}H488" stroke="#98702f" stroke-dasharray="5 5"/>`;curves.forEach((s,i)=>{const col=['#007d7c','#cf7834'][i];svg+=`<polyline points="${s.map(r=>x(r[axis])+','+y(r[key])).join(' ')}" fill="none" stroke="${col}" stroke-width="2"/>`;s.forEach(r=>svg+=`<circle cx="${x(r[axis])}" cy="${y(r[key])}" r="4" fill="${col}"><title>${names[i]} · ${r.step}步：${r[key].toFixed(6)}</title></circle>`)});svg+=`<text x="58" y="230" font-size="12">0</text><text x="488" y="230" text-anchor="end" font-size="12">${max.toFixed(0)} ${axis==='step'?'steps':'s'}</text></svg>`;return `<article><b>${label}</b>${svg}</article>`}).join('')}
function confirmations(){if(!Object.keys(confirm).length){$('confirmation').innerHTML='<p>全部候选在开始前已列入后续检查。训练完成后更新，不填预测结果。</p>';return}const fmt=v=>Number.isFinite(v)?v.toFixed(6):'—';$('confirmation').innerHTML='<div class="scroll"><table><tr><th>运行</th><th>Spatial</th><th>Summary</th><th>NYU AbsRel↓</th><th>有效 GT 数</th><th>平面残差↓</th></tr>'+Object.entries(confirm).map(([n,r])=>`<tr><td>${n}</td><td>${fmt(r.spatial)}</td><td>${fmt(r.summary)}</td><td>${fmt(r.absrel)}</td><td>${r.gt_count??0}</td><td>${fmt(r.plane)}</td></tr>`).join('')+'</table></div>'}
for(const id of ['seed','shape','partition','source','axis','target'])$(id).onchange=draw;table();draw();confirmations();</script></main></body></html>"""
    if wide:
        document = document.replace("组合 Adapter · 参数量扩展", "组合 Adapter · 宽度 2048")
        document = document.replace("深度 × 宽度", "固定四层 · 宽度 1024→2048")
        document = document.replace("加大参数量，能否学得更好？", "加宽到 2048，收益还会继续吗？")
        document = document.replace("六个规模", "三个宽度").replace(
            "4 / 8 个 block × 宽度 512 / 768 / 1024。约 17.6M～110.8M 参数。",
            "固定 4 个 block，宽度 1024 / 1536 / 2048。约 60.4M / 128.5M / 221.7M 参数。",
        )
        document = document.replace("全部 12 次训练", "全部 6 次训练").replace("4×512", "4×1024")
        document = document.replace(
            "shapes=['4x512','4x768','4x1024','8x512','8x768','8x1024']",
            "shapes=['4x1024','4x1536','4x2048']",
        )
        document = document.replace("$('shape').value='4x1024'", "$('shape').value='4x2048'")
        document = document.replace("'combined_4x512_s'", "'combined_4x1024_s'")
        document = document.replace(
            "上一轮结果与真实点云（已完成）", "上一轮容量结果与真实点云（已评估）"
        )
    if wide4096:
        document = document.replace("宽度 2048", "宽度 4096")
        document = document.replace("宽度 1024→2048", "宽度 2048→4096")
        document = document.replace("加宽到 2048", "加宽到 4096")
        document = document.replace("三个宽度", "两个宽度")
        document = document.replace(
            "宽度 1024 / 1536 / 2048。约 60.4M / 128.5M / 221.7M 参数。",
            "宽度 2048 / 4096。约 221.7M / 846.4M 参数。",
        )
        document = document.replace("全部 6 次训练", "全部 4 次训练").replace("4×1024", "4×2048")
        document = document.replace(
            "shapes=['4x1024','4x1536','4x2048']", "shapes=['4x2048','4x4096']"
        )
        document = document.replace("$('shape').value='4x2048'", "$('shape').value='4x4096'")
        document = document.replace("'combined_4x1024_s'", "'combined_4x2048_s'")
    if attention:
        substitutions = {
            "组合 Adapter · 参数量扩展": "4×2048 · Attention 对照",
            "组合设计不变 · 深度 × 宽度 · 两个初始化 seed": "固定 4×2048 · 五种 attention 设计 · 两个初始化 seed",
            "加大参数量，能否学得更好？": "Head 数、QK norm、summary 读出，哪个有帮助？",
            "六个规模": "五组对照",
            "4 / 8 个 block × 宽度 512 / 768 / 1024。约 17.6M～110.8M 参数。": "32-head 基线；16-head；64-head；32-head＋QK norm；32-head＋summary 专用 attention。约 221.6M～238.6M 参数。",
            "每个规模独立随机初始化": "每个变体独立随机初始化",
            "这一轮检查容量收益，未增加训练数据或步数。": "每组只改变一个设计，未增加训练数据或步数。Summary 读出增加约 7.6% 参数，并非等参数对照；head 数会改变少量 gate 参数及 RoPE 的维度分配。",
            "每个规模都最优": "每个变体都最优",
            "全部 12 次训练": "全部 10 次训练",
            "相同 seed 下的容量对照": "相同 seed 下的单项改动对照",
            "shapes=['4x512','4x768','4x1024','8x512','8x768','8x1024']": "shapes=['baseline','heads16','heads64','qknorm','summary_readout']",
            "$('shape').value='4x1024'": "$('shape').value='summary_readout'",
            "${c.blocks}×${c.width}": "${c.variant}",
            "<th>深度×宽度</th>": "<th>变体 · 固定4×2048</th>",
            "'combined_4x512_s'": "'attention_baseline_s'",
            "'combined_'+$('shape').value": "'attention_'+$('shape').value",
            "4×512": "32-head 基线",
            "橙色＝选中规模": "橙色＝选中变体",
        }
        for before, after in substitutions.items():
            assert before in document, before
            document = document.replace(before, after)
    confirm = {}
    for c in plan["candidates"]:
        p = root / "confirmation" / c["name"]
        f = read(p / "features-nfe1/metrics.json", {})
        g = read(p / "geometry-metrics.json", {})
        if f:
            confirm[c["name"]] = dict(
                spatial=f["macro/spatial/cosine"],
                summary=f["macro/summary/cosine"],
                absrel=g.get("gt_student/abs_rel"),
                gt_count=g.get("gt_student/abs_rel/count"),
                plane=g.get("planes/student/normalized_rms"),
            )
    report = read(root / "capacity-report.json", {})
    paired_html = ""
    if report:
        paired_html = (
            '<section class="panel"><h2>两次初始化的容量结果</h2><p>'
            + html.escape(report["conclusion_zh"])
            + '</p><p>下表为两个 seed 的均值；括号保留两次实际值的范围。差值先在相同 seed 内与 4×512 比较，不把两个 seed 当成充分的统计显著性证据。</p><div class="scroll"><table><tr><th>规模</th><th>参数 M</th><th>Spatial</th><th>Summary</th><th>NYU AbsRel↓</th><th>平面 RMS↓</th><th>训练秒</th></tr>'
        )
        for shape, item in report["paired"].items():
            means, ranges = item["mean"], item["range"]
            paired_html += (
                "<tr><td>" + html.escape(shape) + f"</td><td>{means['parameters'] / 1e6:.2f}</td>"
            )
            for metric in (
                "spatial/cosine",
                "summary/cosine",
                "gt_student/abs_rel",
                "planes/student/normalized_rms",
            ):
                paired_html += f"<td>{means[metric]:.5f}<br><small>({ranges[metric][0]:.5f}～{ranges[metric][1]:.5f})</small></td>"
            paired_html += f"<td>{means['train_seconds']:.1f}</td></tr>"
        paired_html += "</table></div><p>固定复测面板 224 图的特征为七来源宏平均；深度 GT 仅 32 张 NYU，平面与边缘检查为 56 图。该面板之前看过，不能称为新的独立测试。训练时间是缓存特征循环，不是全数据收敛时间。</p></section>"
    if wide:
        paired_html = paired_html.replace("4×512", "4×1024")
    if wide4096:
        paired_html = paired_html.replace("4×1024", "4×2048")
    if attention:
        paired_html = paired_html.replace("两次初始化的容量结果", "两次初始化的 attention 结果")
        paired_html = paired_html.replace("4×512", "32-head 基线").replace("<th>规模</th>", "<th>变体</th>")
        if report and "qknorm" in report["paired"]:
            baseline = report["paired"]["baseline"]["mean"]
            qknorm = report["paired"]["qknorm"]["mean"]
            per_seed = []
            for seed in plan["initialization_seeds"]:
                old = report["confirmation"][f"attention_baseline_s{seed}"]["gt_student/abs_rel"]
                new = report["confirmation"][f"attention_qknorm_s{seed}"]["gt_student/abs_rel"]
                per_seed.append(f"seed {seed}：{new / old - 1:+.1%}")
            paired_html += (
                '<section class="panel"><h2>QK norm 的收益与代价</h2><p>'
                f"相对基线，平均平面 RMS 变化 {qknorm['planes/student/normalized_rms'] / baseline['planes/student/normalized_rms'] - 1:+.1%}；"
                f"NYU AbsRel 变化 {qknorm['gt_student/abs_rel'] / baseline['gt_student/abs_rel'] - 1:+.1%}，"
                + "但两次结果分别为 " + "、".join(per_seed)
                + f"。RMSE 变化 {qknorm['gt_student/rmse'] / baseline['gt_student/rmse'] - 1:+.2%}。误差与残差越低越好。</p><p>"
                f"边缘保留率 {baseline['depth_edges/retention_ratio']:.3f} → {qknorm['depth_edges/retention_ratio']:.3f}；"
                f"summary 幅值比 {baseline['summary/norm_ratio']:.3f} → {qknorm['summary/norm_ratio']:.3f}（目标为 1）。"
                f"训练循环耗时增加 {qknorm['train_seconds'] / baseline['train_seconds'] - 1:.1%}。"
                "cosine 和平面改善仍需与深度准确度、边缘和幅值一起判断。</p></section>"
            )
    values = dict(
        STAGE=html.escape(stage),
        TIME=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        PROGRESS=html.escape(progress),
        SOURCES="".join(
            f'<option value="{s}">{s}</option>' for s in plan["source_counts"]["train"]
        ),
        HISTORY=html.escape(oldhtml, quote=True),
        SERIES=json.dumps(series, allow_nan=False).replace("</", "<\\/"),
        SPECS=json.dumps(plan["candidates"]),
        PREFLIGHT=json.dumps(pre),
        RUNS=json.dumps(runs),
        CONFIRM=json.dumps(confirm, allow_nan=False),
        PAIRED=paired_html,
        RESULT=(root / "result-section.html").read_text()
        if (root / "result-section.html").exists()
        else "",
    )
    for k, v in values.items():
        document = document.replace("__" + k + "__", v)
    t = destination.with_suffix(".incoming")
    t.write_text(document)
    t.replace(destination)
