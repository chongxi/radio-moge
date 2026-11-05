"""Self-contained read-only viewer for a hash-bound completed mask review."""
import argparse
import base64
import json
from pathlib import Path

import numpy as np

from probe_scannetpp_v2_source_pilot import bound_path, read
from radio_moge.corpus_admission import file_ref


def run(root, review, output):
    ref = file_ref(review)
    report = read(ref, root)
    if (report['stage'] != 'complete_saved_dense_mask_verified_visual_evidence_rendered_not_reviewed'
            or not report['saved_masks_exactly_reconstructed'] or not report['source_decisions_recomputed']):
        raise ValueError('Completed all-grid audit required')
    images = {k: 'data:image/png;base64,'+base64.b64encode(bound_path(v, root).read_bytes()).decode()
              for k, v in report['panels'].items() if k != 'page'}
    with np.load(bound_path(report['pixel_inspection'], root)) as a:
        depth = base64.b64encode(a['depth'].astype('<f4').tobytes()).decode()
        category = np.zeros(a['depth'].shape, np.uint8)
        category[a['numerical_support']] = 1
        category[a['conditional_support']] = 2
        flags = base64.b64encode(category.tobytes()).decode()
        reasons = base64.b64encode(a['scan_reason_bits'].tobytes()).decode()
    payload = dict(report=report, report_sha256=ref['sha256'], images=images,
                   depth=depth, category=flags, reasons=reasons)
    html = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ScanNet++ 整图监督区域检查</title><style>
body{margin:0;background:#101820;color:#eef3f8;font:15px/1.6 system-ui,sans-serif}main{max-width:1450px;margin:auto;padding:24px}h1{font-size:25px;margin:0 0 8px}p{color:#c7d4df;margin:8px 0}.stats{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}.stats div{background:#1d2c3a;padding:12px 18px;border-radius:8px}.stats strong{display:block;font-size:24px;color:white}.note{padding:12px 16px;background:#233647;border-left:4px solid #57b8ff;margin:16px 0}.views{display:grid;grid-template-columns:1fr 1fr;gap:16px}.panel{background:#1a2734;border-radius:8px;padding:10px;min-width:0}img,canvas{display:block;width:100%;height:auto;background:#000}.caption{margin:0 0 8px;font-weight:600}select,button{font:inherit;color:#fff;background:#263e52;border:1px solid #607b90;border-radius:5px;padding:8px;cursor:pointer}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:14px 0}.legend span{display:inline-block;margin:8px 18px 8px 0}.dot{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:6px}.green{background:#27cf8b}.orange{background:#efa342}.gray{background:#545d6b}.small{font-size:13px;color:#aebfce}.probe{padding:10px;background:#172635;min-height:54px;font-variant-numeric:tabular-nums}details{margin-top:18px}summary{cursor:pointer}.wide .views{grid-template-columns:1fr}.wide canvas{max-width:1280px;image-rendering:pixelated}.wide img{max-width:1280px}select:focus-visible,button:focus-visible{outline:3px solid #fff}@media(max-width:750px){main{padding:12px}.views{grid-template-columns:1fr}h1{font-size:21px}}
</style><main id="main"><h1>ScanNet++：整张图中，哪些像素可作为候选监督？</h1>
<p id="identity"></p><div class="stats"><div>全图保留比例<strong id="fraction"></strong></div><div>保留像素 / 全图像素<strong id="points"></strong></div><div>整图最低覆盖门槛<strong id="floor"></strong></div></div>
<div class="note">这是固定两台扫描仪共同支持的<strong>候选区域</strong>。橙色表示证据不足或不一致，不能直接等同于错误深度。没有填洞、补深度或启动训练。</div>
<div class="legend"><span><i class="dot green"></i>两台均通过，保留</span><span><i class="dot orange"></i>有数值深度，但来源证据未全部通过</span><span><i class="dot gray"></i>原本无有效数值深度</span></div>
<div class="controls"><label for="layer">右侧显示：</label><select id="layer"><option value="rgb_overlay">RGB 与分类 mask 叠加</option><option value="support_categories">分类 mask</option><option value="numerical_depth">过滤前的数值候选深度</option><option value="retained_depth">保留区域的深度</option><option value="scanner0_mask">扫描仪 0 支持区域</option><option value="scanner1_mask">扫描仪 1 支持区域</option></select><button id="zoom">放大查看</button></div>
<div class="views"><div class="panel"><p class="caption">去畸变透视 DSLR RGB</p><img id="rgb" alt="实际透视 DSLR RGB，显示尺寸缩放"></div><div class="panel"><p class="caption" id="layerName"></p><canvas id="view" aria-label="候选深度和条件支持区域，可移动指针检查像素"></canvas></div></div>
<div class="probe" id="probe" aria-live="polite">在右图上移动指针或点击，查看该像素的深度和两台扫描仪未通过的条件。</div>
<p class="small">两幅深度图共用固定对数色尺 0.05–20 m，蓝近红远；黑色表示该层没有有效深度。绿色不代表独立物理真值，1 cm 是工程一致性过滤预算。</p>
<details><summary>检查范围与判定说明</summary><p id="scope"></p><p>来源核对要求：原数值有效、两台固定扫描仪都可见同一目标表面、有限采样的网格对应条件通过、完整九像素方向和量程证据齐全、每个残差均不超过固定预算。原有排除记录继续有效。</p><p>扫描和网格来自同一次采集，未标定完整位姿不确定性。此次只有一张开发图，不代表整个数据集的质量，也不是训练准入结论。</p><p class="small" id="proof"></p></details></main>
<script>const data=PAYLOAD;const $=id=>document.getElementById(id),r=data.report,[h,w]=r.grid,n=h*w;
const bytes=s=>Uint8Array.from(atob(s),x=>x.charCodeAt(0));const depths=new DataView(bytes(data.depth).buffer),category=bytes(data.category),reasons=bytes(data.reasons);
const labels=['网格表面对应未确定','精确目标可见性未通过','九像素原始证据不完整','量程残差超出固定预算'];
$('identity').textContent=`样本 ${r.corpus_ordinal} · 场景 ${r.scene} · 完整 ${w} × ${h} 网格`;
$('fraction').textContent=(r.coverage.fraction*100).toFixed(2)+'%';$('points').textContent=r.coverage.points.toLocaleString()+' / '+n.toLocaleString();$('floor').textContent=r.coverage.conditional_dense_coverage_sufficient?'通过 ≥5% 且 ≥32 点':'未通过';
$('scope').textContent=`核对 ${r.target_slots_verified.toLocaleString()} 个像素目标、${r.station_slots_verified.toLocaleString()} 个扫描仪观测。未通过部分仍保留在分母中；只对显示的 RGB 副本缩放。`;
$('proof').textContent='保存结果复核 SHA256：'+data.report_sha256;$('rgb').src=data.images.rgb;
const canvas=$('view'),ctx=canvas.getContext('2d');canvas.width=w;canvas.height=h;let version=0;
function show(){const id=++version,key=$('layer').value,img=new Image();$('layerName').textContent=$('layer').selectedOptions[0].textContent;img.onload=()=>{if(id===version)ctx.drawImage(img,0,0,w,h)};img.src=data.images[key]}
function probe(e){const b=canvas.getBoundingClientRect(),x=Math.min(w-1,Math.max(0,Math.floor((e.clientX-b.left)*w/b.width))),y=Math.min(h-1,Math.max(0,Math.floor((e.clientY-b.top)*h/b.height))),i=y*w+x,c=category[i];let s=`像素 (x=${x}, y=${y}) · ${['无有效数值深度','数值候选有深度，来源证据未全部通过','两台来源条件均通过'][c]}`;if(c)s+=` · 深度 ${depths.getFloat32(i*4,true).toFixed(4)} m`;for(let scan=0;scan<2;scan++){const bits=reasons[scan*n+i];s+=` ｜扫描仪 ${scan}：`+(c===0?'未进入来源检验':(bits?labels.filter((_,k)=>bits&(1<<k)).join('、'):'条件通过'))}$('probe').textContent=s}
canvas.addEventListener('pointermove',probe);canvas.addEventListener('click',probe);$('layer').addEventListener('change',show);$('zoom').onclick=()=>{const yes=$('main').classList.toggle('wide');$('zoom').textContent=yes?'并排查看':'放大查看'};show();
</script></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html.replace('PAYLOAD', json.dumps(payload, ensure_ascii=False).replace('</', '<\\/')))
    print(json.dumps(dict(path=str(output), bytes=output.stat().st_size, report=ref)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.review.resolve(), args.output.resolve())
