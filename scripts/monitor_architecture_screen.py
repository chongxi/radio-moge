"""Copy only small screen evidence and refresh the existing browser page; never launch GPU work."""

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import shlex
import subprocess
import time


NAME = "adapter-scratch-diverse-screen-20261028"


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".incoming")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def render(root, destination):
    plan = json.loads((root / "plan.json").read_text())
    monitor = json.loads((root / "monitor.json").read_text())
    status = monitor.get("status") or {}
    cache = monitor.get("cache") or {}
    series = {}
    for spec in plan["candidates"]:
        for nfe in [1, 2, 4] if spec["family"] == "conditional_flow" else [1]:
            name = spec["name"] + f"/nfe{nfe}"
            series[name] = {}
            for part in ("validation", "train_probe"):
                p = root / spec["name"] / f"{part}-nfe{nfe}.jsonl"
                series[name][part] = (
                    [
                        json.loads(line)
                        for line in p.read_text().splitlines(keepends=True)
                        if line.endswith("\n")
                    ]
                    if p.exists()
                    else []
                )
    specs = "".join(
        "<tr><td>"
        + html.escape(c["name"])
        + "</td><td>"
        + html.escape(json.dumps({k: v for k, v in c.items() if k != "name"}, ensure_ascii=False))
        + "</td></tr>"
        for c in plan["candidates"]
    )
    old = root.parent / "adapter-dual-cosine512-pilot-20261028/result-section.html"
    history = ""
    if old.exists():
        content = (
            '<!doctype html><meta charset="utf-8"><style>body{font:15px system-ui;color:#18312f;padding:15px}table{width:100%}td,th{padding:7px}.grid{display:flex;gap:15px;flex-wrap:wrap}</style>'
            + old.read_text()
        )
        history = (
            '<iframe title="上一轮双 cosine 结果" loading="lazy" style="width:100%;height:1100px;border:0" srcdoc="'
            + html.escape(content, quote=True)
            + '"></iframe>'
        )
    stage = (monitor.get("queue") or {}).get("stage", "preparing")
    progress = f"缓存 {cache.get('images', 0)} / 2,240 图；已结束 {len(status.get('completed', []))} / 16 候选"
    if status.get("current"):
        progress += f"；当前 {status['current']}：{status.get('step', 0)} 步"
    confirmation_html = ""
    shortlist_path = root / "shortlist.json"
    if shortlist_path.exists():
        selected = json.loads(shortlist_path.read_text())["selected"]
        confirm = monitor.get("confirmation") or {}
        rows = []
        for item in selected:
            arm = root / "confirmation" / item["name"]
            feature_path = arm / f"features-nfe{item['nfe']}" / "metrics.json"
            geometry_path = arm / "geometry-metrics.json"
            feature = json.loads(feature_path.read_text()) if feature_path.exists() else {}
            geometry = json.loads(geometry_path.read_text()) if geometry_path.exists() else {}

            def number(value):
                return f"{value:.6f}" if isinstance(value, (int, float)) else "待完成"

            rows.append(
                "<tr><td>"
                + html.escape(item["name"])
                + f" / NFE {item['nfe']}</td>"
                + f"<td>{item['spatial']:.6f} / {item['summary']:.6f}</td>"
                + "<td>"
                + number(feature.get("macro/spatial/cosine"))
                + " / "
                + number(feature.get("macro/summary/cosine"))
                + "</td><td>"
                + number(geometry.get("gt_student/abs_rel"))
                + " / n="
                + str(geometry.get("gt_student/abs_rel/count", 0))
                + "</td></tr>"
            )
        confirmation_html = (
            '<section class="panel"><h2>名单已冻结 · 预留确认集</h2><p>阶段：'
            + html.escape(confirm.get("stage", "等待确认"))
        )
        confirmation_html += (
            '。确认阶段只做推理，零参数更新。特征指标为 224 图、七来源宏平均；几何检查固定 56 图，其中真实深度指标仅来自 32 张 NYU。</p><div class="scroll"><table><tr><th>入围结构</th><th>筛选 Spatial / Summary</th><th>确认 Spatial / Summary</th><th>NYU GT AbsRel / 有效计数</th></tr>'
            + "".join(rows)
            + '</table></div><p class="muted">确认集未参与本轮名单选择。GT 计数不足时必须同时查看失败记录，不能把缺失图像当成改善；尚未完成的结果不会填预测值。</p></section>'
        )
    template = """<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>从零训练 · Adapter 架构筛选</title><style>
body{margin:0;background:#f5f4ef;color:#19342f;font:16px system-ui,sans-serif}main,details.history{max-width:1200px;margin:24px auto;padding:20px}h1{font-size:clamp(28px,4vw,44px)}h2{font-size:22px}p{line-height:1.75}article,.panel{border:1px solid #d4ddd6;border-radius:14px;background:white;padding:20px;margin:16px 0;min-width:0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:16px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:11px;border-bottom:1px solid #ddd;vertical-align:top}select,button{font:inherit;padding:8px;max-width:100%;margin:4px;border-radius:6px;border:1px solid #afbeb8}svg{width:100%;height:auto}.notice{background:#fff2dd;border-left:4px solid #bf7927;padding:14px 18px}.muted{color:#596d65}a{color:#00756e}summary{cursor:pointer;padding:12px}.controls{display:flex;gap:12px;flex-wrap:wrap}</style></head><body><main>
<small>七来源数据 · 随机初始化 · 16 个候选</small><h1>谁的 Spatial 与 Summary 学得更快？</h1>
<p><b>__STAGE__</b> · __STAMP__ <button onclick="location.reload()">刷新</button></p><p>__PROGRESS__</p>
<p>这轮从零训练 adapter，不继承旧 adapter 或 Adam 状态。RADIO 与 MoGe 保持冻结，同一批缓存特征供全部候选使用。先筛选，再用未参与本轮排名的确认场景与实际深度输出检查入围者。</p>
<div class="grid"><article><b>更广的数据覆盖</b><p>ARKitScenes、Hypersim、NYU、COCO、TartanAir、Structured3D、IRS 各 256 张训练图：共 1,792 张。筛选验证与确认集各 224 张、来源均衡，路径／场景组／已知内容哈希互斥。</p></article><article><b>共同短跑预算</b><p>最多 1,000 步／候选，每个候选最多 10 分钟。0／100／250／500／1,000 步评估。有效 batch 16；LR 3e-4，前 100 步 warmup。保留两项 MSE，并直接约束两个 cosine。</p></article><article><b>分别看两个指标</b><p>Spatial 与 Summary 的来源宏平均分别展示，并记录幅值。按相同步数、相同训练时间和总运行时间查看。模型参数量、flow 的推理调用次数单独列出。</p></article></div>
<p class="notice">当前是固定视图的架构初筛，不是全数据收敛结论。时间上限导致的提前停止会显示实际步数。单一 seed／共同学习率可能影响排名；先看 500→1,000 步排序是否稳定。COCO 场景独立性未知；TartanAir 与 IRS 各验证分区仅有一个环境／家族，不能夸大泛化证据。</p>
<section class="panel"><h2>最新完成的验证点 · 不是最终胜者</h2><p class="muted">排序分数＝min（Spatial 来源宏平均，Summary 来源宏平均）。两项原值与步数同时保留；不同预算不能直接下胜负结论。</p><div class="scroll" id="ranking"></div></section>
<section class="panel"><h2>配对学习曲线</h2><div class="controls"><label>候选 <select id="candidate"></select></label><label>对照 <select id="baseline"></select></label><label>数据 <select id="partition"><option value="validation">筛选验证</option><option value="train_probe">训练诊断 112 图</option></select></label><label>来源 <select id="source"><option value="macro">七来源宏平均</option>__SOURCES__</select></label><label>横轴 <select id="axis"><option value="step">优化步数</option><option value="optimizer_seconds">训练循环秒数</option><option value="elapsed_seconds">含评估总秒数</option></select></label><label><input type="checkbox" id="target">显示 0.98 目标</label></div><div class="grid" id="charts"></div><p>蓝绿＝对照；橙色＝候选。缓存的一次性准备成本另列，不伪装成每个模型的训练加速。Flow matching 训练的速度场损失不与直接回归总 loss 混排；图中只比较实际无教师推理后的特征。</p></section>
<details><summary>完整候选与控制变量</summary><div class="scroll"><table><tr><th>候选</th><th>定义</th></tr>__SPECS__</table></div><p>legacy 的残差 LayerScale 1e-5 与 0.01 为单独对照；其余联合结构默认 0.01。深度／宽度变化同时改变容量和计算量。RoPE、gate、register、SwiGLU、local 各有单因素版本，combined 是组合探索。</p></details>
<details><summary>研究依据与可验证假设</summary><p><a href="https://arxiv.org/abs/2403.13298">2D RoPE</a>、<a href="https://arxiv.org/abs/2309.16588">register tokens</a>、<a href="https://arxiv.org/abs/2505.06708">attention 输出 gating</a>、<a href="https://arxiv.org/abs/2002.05202">SwiGLU</a> 为组件参考，并非照搬论文模型或预设收益。<a href="https://arxiv.org/abs/2403.13788">DepthFM</a> 提供深度任务中的 flow 先例；本轮是没有生成式预训练先验的特征空间条件 flow，不等同于复现 DepthFM。</p></details>
</main><details class="history"><summary>上一轮：加强双 cosine 的结果与真实点云</summary>__HISTORY__</details>
<script>const series=__DATA__,specs=__SPEC_DATA__,preflight=__PREFLIGHT__;const $=id=>document.getElementById(id),names=Object.keys(series);for(const id of ['candidate','baseline'])$(id).innerHTML=names.map(n=>`<option value="${n}">${n}</option>`).join('');$('candidate').value='joint_4x512/nfe1';$('baseline').value='legacy_2x1280/nfe1';
function draw(){const part=$('partition').value,source=$('source').value,axis=$('axis').value,prefix=source==='macro'?'macro/':'dataset/'+source+'/',chosen=[$('baseline').value,$('candidate').value];$('charts').innerHTML=[['spatial/cosine','Spatial cosine ↑'],['summary/cosine','Summary cosine ↑'],['spatial/norm_ratio','Spatial 幅值比 · 1 为教师'],['summary/norm_ratio','Summary 幅值比 · 1 为教师']].map(([key,label])=>{key=prefix+key;const curves=chosen.map(n=>series[n][part].filter(r=>Number.isFinite(r[key])&&Number.isFinite(r[axis]))),all=curves.flat();if(!all.length)return `<article><b>${label}</b><p>等待这两个候选的评估数据</p></article>`;let lo=Math.min(...all.map(r=>r[key])),hi=Math.max(...all.map(r=>r[key]));if($('target').checked&&key.endsWith('cosine')){lo=Math.min(lo,.98);hi=Math.max(hi,.98);}const pad=Math.max(.0003,(hi-lo)*.1);lo-=pad;hi+=pad;const max=Math.max(1,...all.map(r=>r[axis])),x=v=>58+v/max*430,y=v=>205-(v-lo)/(hi-lo)*165;let svg=`<svg viewBox="0 0 520 245" aria-label="${label}"><path d="M58 35V205H488" fill="none" stroke="#9aaca6"/>`;for(let i=0;i<4;i++){const v=lo+(hi-lo)*i/3;svg+=`<text x="53" y="${y(v)+4}" font-size="11" text-anchor="end">${v.toFixed(4)}</text>`;}if($('target').checked&&key.endsWith('cosine'))svg+=`<path d="M58 ${y(.98)}H488" stroke="#98702f" stroke-dasharray="5 5"/>`;curves.forEach((s,i)=>{const color=['#007d7c','#cf7834'][i];svg+=`<polyline points="${s.map(r=>x(r[axis])+','+y(r[key])).join(' ')}" fill="none" stroke="${color}" stroke-width="2"/>`;for(const r of s)svg+=`<circle cx="${x(r[axis])}" cy="${y(r[key])}" r="4" fill="${color}"><title>${chosen[i]} / ${r.step} 步：${r[key].toFixed(6)}；${axis}=${r[axis].toFixed(2)}</title></circle>`;});svg+=`<text x="58" y="230" font-size="12">0</text><text x="488" y="230" text-anchor="end" font-size="12">${max.toFixed(0)} ${axis==='step'?'steps':'s'}</text></svg>`;return `<article><b>${label}</b>${svg}</article>`;}).join('');}
const latest=names.map(n=>({n,r:series[n].validation.at(-1)})).filter(x=>x.r).sort((a,b)=>Math.min(b.r['macro/spatial/cosine'],b.r['macro/summary/cosine'])-Math.min(a.r['macro/spatial/cosine'],a.r['macro/summary/cosine']));$('ranking').innerHTML=latest.length?'<table><tr><th>候选／调用次数</th><th>实际步数</th><th>参数 M</th><th>Spatial</th><th>Summary</th><th>训练秒</th><th>推理 ms/图</th></tr>'+latest.map(({n,r})=>`<tr><td>${n}</td><td>${r.step}</td><td>${preflight[n.split('/')[0]]?(preflight[n.split('/')[0]].parameters/1e6).toFixed(2):'—'}</td><td>${r['macro/spatial/cosine'].toFixed(6)}</td><td>${r['macro/summary/cosine'].toFixed(6)}</td><td>${r.optimizer_seconds.toFixed(1)}</td><td>${r.inference_ms_per_image.toFixed(2)}</td></tr>`).join('')+'</table>':'<p>缓存／预检进行中。此处只展示已完成的真实评估，不填预测值。</p>';for(const id of ['candidate','baseline','partition','source','axis','target'])$(id).onchange=draw;draw();</script></body></html>"""
    anchor = '<section class="panel"><h2>配对学习曲线</h2>'
    template = template.replace(anchor, confirmation_html + anchor)
    preflight_path = root / "preflight-verification.json"
    preflight = (
        json.loads(preflight_path.read_text()).get("candidates", {})
        if preflight_path.exists()
        else {}
    )
    replacements = dict(
        STAGE=html.escape(stage),
        STAMP=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        PROGRESS=html.escape(progress),
        SPECS=specs,
        HISTORY=history,
        SOURCES="".join(
            f'<option value="{s}">{s}</option>' for s in plan["source_counts"]["train"]
        ),
        DATA=json.dumps(series, allow_nan=False).replace("</", "<\\/"),
        SPEC_DATA=json.dumps(plan["candidates"]),
        PREFLIGHT=json.dumps(preflight),
    )
    for key, value in replacements.items():
        template = template.replace("__" + key + "__", value)
    temporary = destination.with_suffix(".incoming")
    temporary.write_text(template)
    temporary.replace(destination)


def monitor(base, remote_base):
    root = base / NAME
    known = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in (".json", ".jsonl") and p.stat().st_size < 8_000_000:
            known[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    code = """from pathlib import Path
import sys,json,hashlib,base64,subprocess,time
r=Path(ROOT);known=json.load(sys.stdin);plan=json.loads((r/'plan.json').read_text())
paths=list(r.glob('*.json'))+list(r.glob('*.jsonl'))
paths+=list((r/'confirmation').rglob('*.json'))
for c in plan['candidates']:
 d=r/c['name'];paths+=list(d.glob('*.json'))+list(d.glob('*.jsonl'))+list(d.glob('validation-*/*.json'))+list(d.glob('train_probe-*/*.json'))
files={}
for p in paths:
 assert p.stat().st_size<8_000_000
 data=p.read_bytes();relative=str(p.relative_to(r));digest=hashlib.sha256(data).hexdigest()
 if known.get(relative)!=digest:files[relative]=dict(data=base64.b64encode(data).decode(),sha256=digest)
def read(n):
 p=r/n;return json.loads(p.read_text()) if p.exists() else None
q=read('queue-status.json');s=read('status.json');c=read('cache-status.json');confirmation=read('confirmation-status.json');pids={}
for v in [q,s,c,confirmation]:
 if v:
  for key in ['pid','child_pid']:
   pid=v.get(key)
   if pid:
    p=Path('/proc')/str(pid)/'stat';pids[str(pid)]=p.read_text().split(') ')[1].split()[0] if p.exists() else 'absent'
logs={}
for n in ['wrapper-console.log','cache-console.log','screen-console.log','confirmation-console.log']:
 p=r/n
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-5000));logs[n]=f.read().decode(errors='replace')
print(json.dumps(dict(observed_at=time.time(),queue=q,status=s,cache=c,confirmation=confirmation,processes=pids,console_tails=logs,
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True),files=files)))
""".replace("ROOT", repr(str(remote_base / NAME)))
    p = subprocess.run(
        [
            "ssh",
            "-F",
            "/home/hub/.ssh/radio-moge-config",
            "radio-moge-gpu",
            "/opt/conda/bin/python3 -c " + shlex.quote(code),
        ],
        input=json.dumps(known),
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(p.stdout)
    copied = {}
    for relative, item in result.pop("files").items():
        data = base64.b64decode(item["data"])
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".incoming")
        temporary.write_bytes(data)
        temporary.replace(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        copied[relative] = item["sha256"]
    result["copied_hashes"] = copied
    atomic(root / "monitor.json", result)
    handoff = json.loads((base / "handoff.json").read_text())
    active = handoff["active_adapter_architecture_screen"]
    active.update(
        stage=(result["queue"] or {}).get("stage"),
        last_observed_at=result["observed_at"],
        current=result["status"],
        cache=result["cache"],
        confirmation=result["confirmation"],
    )
    handoff["updated_at"] = time.time()
    atomic(base / "handoff.json", handoff)
    render(root, Path("/home/hub/nano-test/.kanna/visualizations/teacher-plane-pilot.html"))
    print(
        json.dumps(
            {k: result[k] for k in ("observed_at", "queue", "status", "cache", "processes", "gpu")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--remote-base", type=Path, required=True)
    a = p.parse_args()
    monitor(a.base, a.remote_base)
