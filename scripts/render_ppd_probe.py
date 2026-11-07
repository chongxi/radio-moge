"""Render saved PPD probe arrays as a self-contained, offline comparison page."""

import argparse
import base64
import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


LABELS = {
    "moge_native": "MoGe · 原生特征",
    "moge_adapter": "MoGe · 我们的特征",
    "ppd_native": "PPD · 原生特征",
    "ppd_adapter": "PPD · 我们的特征",
    "ppd_zero": "PPD · 语义置零",
}


def jpeg(array):
    output = io.BytesIO()
    Image.fromarray(array).save(output, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode()


def colored(value, low, high, valid=None, error=False):
    value = ((value - low) / (high - low) * 255).clip(0, 255).astype(np.uint8)
    rgb = cv2.applyColorMap(value, cv2.COLORMAP_INFERNO if error else cv2.COLORMAP_TURBO)[..., ::-1]
    if valid is not None:
        rgb[~valid] = (20, 26, 37)
    return jpeg(rgb)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.base / "report.json").read_text())
    controls = json.loads((args.controls / "report.json").read_text())
    cases = []
    for result in report["results"]:
        with np.load(args.base / (result["name"] + ".npz")) as arrays:
            target, valid = arrays["target"], arrays["valid"]
            low, high = np.percentile(target[valid], [2, 98])
            maps = {key: arrays["aligned_" + key].copy() for key in LABELS if key != "ppd_zero"}
            pictures = {"rgb": jpeg(cv2.resize(arrays["rgb"], (640, 480), interpolation=cv2.INTER_AREA)),
                        "target": colored(target, low, high, valid)}
        scores = dict(result["metrics"])
        if result["index"] < 12 and result["resolution_hw"] == [768, 1024]:
            with np.load(args.controls / (result["name"] + ".npz")) as arrays:
                maps["ppd_zero"] = arrays["aligned_ppd_zero"].copy()
            scores["ppd_zero"] = controls["results"][result["index"]]["metrics"]["ppd_zero"]
        for key, value in maps.items():
            pictures[key] = {"depth": colored(value, low, high),
                             "error": colored(np.abs(value - target) / np.maximum(target, 1e-6), 0, .20, valid, True)}
        cases.append({**result, "metrics": scores, "pictures": pictures,
                      "color_min": float(low), "color_max": float(high)})
    data = {"cases": cases, "aggregates": report["aggregates"], "controls": controls["aggregate"], "labels": LABELS}
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".new")
    temporary.write_text(html)
    os.replace(temporary, args.output)
    print(json.dumps({"html": str(args.output), "bytes": args.output.stat().st_size, "cases": len(cases)}))


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>拟合特征接入 PPD · 实验对照</title>
<style>
:root{color-scheme:dark;font:15px/1.55 system-ui,sans-serif;background:#10151e;color:#e8edf4}*{box-sizing:border-box}body{margin:0;padding:24px;max-width:1500px;margin:auto}h1{font-size:25px;margin:0 0 8px}p{margin:8px 0;color:#afbbce}strong{color:#ebf3ff}button,select{font:inherit;background:#202b3e;color:#edf4ff;border:1px solid #45516a;padding:7px 10px;border-radius:7px}button{cursor:pointer}button:hover{background:#344662}.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.toolbar{margin:20px 0}.panel{background:#182130;border:1px solid #2b384e;border-radius:12px;padding:16px;margin-top:16px}.main{display:grid;grid-template-columns:1fr 1fr;gap:14px}figure{margin:0}figure img{width:100%;display:block;border-radius:6px}figcaption{padding:8px 0;font-weight:600}.reference{display:grid;grid-template-columns:1fr 1fr 2fr;gap:14px;margin-top:18px}.reference img{max-height:220px;object-fit:contain;background:#10151e}.note{font-size:13px;color:#a7b6cc}.pill{background:#163f3b;color:#8cebd0;border-radius:20px;padding:3px 10px;font-size:12px}.metric{color:#a3efe4;font-size:15px;margin-left:8px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}th,td{padding:7px 10px;border-bottom:1px solid #2b384e;text-align:right}th:first-child,td:first-child{text-align:left}.selected{background:#1c3940}.strip{height:12px;background:linear-gradient(90deg,#30123b,#4669dd,#1bcfd4,#6dfd59,#dfdf24,#ff7e1d,#c82702,#7a0403);border-radius:4px}.range{display:flex;justify-content:space-between;font-size:12px;color:#a7b6cc}input[type=range]{width:100%;accent-color:#76e4cf}.overlay{position:relative;overflow:hidden;aspect-ratio:4/3}.overlay img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}.overlay #over{clip-path:inset(0 0 0 50%)}.overlay #bar{position:absolute;top:0;bottom:0;left:50%;width:2px;background:white;pointer-events:none}.hidden{display:none!important}.spread{justify-content:space-between}a{color:#90d5ff}@media(max-width:750px){body{padding:14px}.main{grid-template-columns:1fr}.reference{grid-template-columns:1fr 1fr}.reference>div{grid-column:1/-1}h1{font-size:21px}th,td{padding:6px 5px}}
</style>
<h1>我们的拟合特征，能接入 PPD 吗？</h1>
<p><strong>可以接入。</strong>下面是固定 16 张图、两种分辨率的完整对照。所有模型冻结，PPD 采样 4 步；同一场景的两条 PPD 路线共享噪声。</p>
<p class="note">12 张 NYU 图有深度真值；4 张空房图仅作视觉检查。所有深度图使用共同色标。NYU 指标经过逐图真值尺度对齐，是相对深度结构比较，不是米制精度或完整公开基准。</p>
<div class="panel"><div class="row spread"><strong>12 个 NYU 场景 · 平均结果</strong><span class="pill">未训练 / 未更新权重</span></div><div id="summary"></div><p class="note" id="seednote"></p></div>
<div class="row toolbar"><button id="prev">←</button><select id="scene"></select><button id="next">→</button><select id="resolution"><option value="768x1024">1024 × 768</option><option value="480x640">640 × 480</option></select><select id="mode"><option value="depth">对齐后的深度</option><option value="error">相对误差图</option></select><button id="compare">切换滑动叠加</button></div>
<div class="row"><label>左侧 <select id="left"></select></label><label>右侧 <select id="right"></select></label><span class="note" id="feature"></span></div>
<div class="panel"><div class="main" id="side"><figure><figcaption id="captionL"></figcaption><img id="imgL" alt="左侧路线深度图"></figure><figure><figcaption id="captionR"></figcaption><img id="imgR" alt="右侧路线深度图"></figure></div>
<div id="sliderBox" class="hidden"><p id="sliderLabel" class="note"></p><div class="overlay"><img id="under" alt="左侧路线"><img id="over" alt="右侧路线"><span id="bar"></span></div><input type="range" min="0" max="100" value="50" id="slider"></div>
<p class="note" id="legend"></p><div class="strip" id="colors"></div><div class="range"><span id="minimum"></span><span id="maximum"></span></div>
<div class="reference"><figure><figcaption>输入 RGB</figcaption><img id="rgb" alt="输入图片"></figure><figure><figcaption id="targetTitle"></figcaption><img id="target" alt="对齐参考"></figure><div><strong>当前场景</strong><div id="metrics"></div><p class="note" id="sceneNote"></p></div></div></div>
<div class="panel note">实验固定使用 Muon 最终模型 step 272646。PPD 代码版本 427a86f，MoGe2 版本权重。高分辨率零语义对照只在 12 张 NYU 图上运行；语义置零属于分布外输入，用来检查模型是否依赖特征，不代表重新训练的 RGB-only 模型。</div>
<script>
const D=__DATA__;const $=id=>document.getElementById(id);let overlay=false;
const initial=D.cases.filter(x=>x.resolution_hw[0]===768);initial.forEach(c=>{const o=document.createElement('option');o.value=c.index;o.textContent=String(c.index+1).padStart(2,'0')+' · '+c.group.replace('nyuv2:','').replace('structured3d:scene:','空房 scene ');$('scene').append(o)});
function table(metrics,includeGradient=false){let h='<table><thead><tr><th>路线</th><th>AbsRel ↓</th><th>δ₁ ↑</th>'+(includeGradient?'<th>梯度误差 ↓</th>':'')+'</tr></thead><tbody>';for(const [k,v] of Object.entries(metrics)){if(!D.labels[k])continue;h+='<tr class="'+(k==='ppd_adapter'?'selected':'')+'"><td>'+D.labels[k]+'</td><td>'+(100*v.abs_rel).toFixed(2)+'%</td><td>'+(100*v.delta1).toFixed(2)+'%</td>'+(includeGradient?'<td>'+v.log1p_gradient_mae.toFixed(4)+'</td>':'')+'</tr>'}return h+'</tbody></table>'}
function update(){const resolution=$('resolution').value,c=D.cases.find(x=>x.index===+$('scene').value&&x.resolution_hw.join('x')===resolution),mode=$('mode').value;
for(const [id,fallback] of [['left','ppd_native'],['right','ppd_adapter']]){const previous=$(id).value||fallback;$(id).innerHTML='';for(const [k,label]of Object.entries(D.labels)){if(!c.pictures[k])continue;const o=document.createElement('option');o.value=k;o.textContent=label;$(id).append(o)}$(id).value=c.pictures[previous]?previous:fallback}
const l=$('left').value,r=$('right').value;function caption(k){return D.labels[k]+(c.metrics[k]?' · AbsRel '+(100*c.metrics[k].abs_rel).toFixed(2)+'%':'')}
$('imgL').src=c.pictures[l][mode];$('imgR').src=c.pictures[r][mode];$('under').src=$('imgL').src;$('over').src=$('imgR').src;$('captionL').textContent=caption(l);$('captionR').textContent=caption(r);$('sliderLabel').textContent='左：'+caption(l)+' ｜ 右：'+caption(r);$('rgb').src=c.pictures.rgb;$('target').src=c.pictures.target;$('targetTitle').textContent=c.has_ground_truth?'NYU 深度真值':'MoGe 原生参考（非真值）';$('feature').textContent='空间特征 cos '+c.feature_metrics.mean_cosine.toFixed(4);
$('metrics').innerHTML=c.has_ground_truth?table(c.metrics):'<p class="note">此空房图没有配套真值。图像经 MoGe 参考深度对齐，不能用于证明物理平整度改善。</p>';
$('sceneNote').textContent=c.group+' · 噪声种子 '+c.noise_seed;$('summary').innerHTML=table(D.aggregates[resolution],true);
const a=D.controls;$('seednote').textContent=resolution==='768x1024'?'第二份噪声：原生 PPD '+(100*a.ppd_native_seed2.abs_rel).toFixed(2)+'%，我们的特征 → PPD '+(100*a.ppd_adapter_seed2.abs_rel).toFixed(2)+'%。语义置零：'+(100*a.ppd_zero.abs_rel).toFixed(2)+'%。':'当前显示的是适配器训练分辨率。PPD 的标准输入面积更接近 1024 × 768。';
$('legend').textContent=mode==='depth'?'同一场景所有路线共用色标：蓝色近，红色远。显示范围按参考深度 2–98 百分位截取。':(c.has_ground_truth?'相对真值的逐像素误差；越亮误差越大。无效真值像素显示为深灰。':'相对原生 MoGe 参考的差异；这不是与真值的误差。');$('minimum').textContent=mode==='depth'?c.color_min.toFixed(2)+' m（对齐后）':'0%';$('maximum').textContent=mode==='depth'?c.color_max.toFixed(2)+' m（对齐后）':'20% 及以上';$('colors').style.background=mode==='depth'?'':'linear-gradient(90deg,#000004,#57106e,#bc3754,#f98e09,#fcffa4)';}
for(const id of ['scene','resolution','mode','left','right'])$(id).addEventListener('change',update);$('prev').onclick=()=>{$('scene').value=(+$('scene').value+15)%16;update()};$('next').onclick=()=>{$('scene').value=(+$('scene').value+1)%16;update()};$('compare').onclick=()=>{overlay=!overlay;$('side').classList.toggle('hidden',overlay);$('sliderBox').classList.toggle('hidden',!overlay)};$('slider').oninput=()=>{const x=$('slider').value;$('over').style.clipPath='inset(0 0 0 '+x+'%)';$('bar').style.left=x+'%'};update();
</script></html>'''


if __name__ == "__main__":
    main()
