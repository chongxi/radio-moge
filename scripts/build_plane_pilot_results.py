"""Build a self-contained four-way viewer from verified saved geometry, without inference."""

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path

import numpy as np


NAMES = {'initial': '本次训练前 · 83,494 步', 'control': '对照 · 只用特征损失',
         'plane': '实验 · 特征＋平面约束', 'teacher': 'MoGe 教师'}


def fit(points):
    center = points.mean(0)
    shifted = points - center
    cov = shifted.T @ shifted / len(points)
    values, vectors = np.linalg.eigh(cov)
    normal = vectors[:, 0]
    extent = np.sqrt(np.trace(cov))
    residual = np.abs(shifted @ normal) / max(extent, 1e-12)
    return center, normal, extent, float(np.sqrt(np.mean(residual ** 2)))


def encode(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


def build(root):
    audit = json.loads((root / 'evaluation-verification.json').read_text())
    assert audit['stage'] == 'passed'
    steps = audit['steps']
    verified = json.loads((root / 'final-review/prepared-dataset-backup-status.json').read_text())
    checked = {i['destination']: i['sha256'] for i in verified['verified_checkpoints']}
    def manifest(partition):
        return [json.loads(s) for s in (root / f'{partition}.jsonl').read_text().splitlines()]
    examples = [('screenshot', manifest('screenshot')[0], '独立截图 · 未参与训练'),
                ('validation', manifest('validation')[0], '验证样本 1 · 未参与训练'),
                ('validation', manifest('validation')[1], '验证样本 2 · 未参与训练'),
                ('train-probe', manifest('train-probe')[0], '训练样本 · 参与了训练')]
    samples, hashes = [], {}
    for partition, row, title in examples:
        paths = ({'initial': root / 'plane/eval-0000000',
                  'control': root / 'control' / f'eval-{steps:07d}',
                  'plane': root / 'plane' / f'eval-{steps:07d}'} if partition == 'validation'
                 else {arm: root / 'assessment' / arm / partition
                       for arm in ('initial', 'control', 'plane')})
        arrays, sample_metrics = {}, {}
        reference = None
        for arm, path in paths.items():
            file = path / f"geometry-{row['id']}.npz"
            assert str(file) in checked, f'Geometry copy not verified yet: {file}'
            # The serial copy already read back SHA-256; retain that receipt instead of rehashing.
            hashes[str(file.relative_to(root))] = checked[str(file)]
            with np.load(file) as data:
                values = {k: data[k] for k in data.files}
            if reference is None:
                reference = values
            else:
                for key in ('labels', 'rgb', 'teacher_points'):
                    np.testing.assert_array_equal(values[key], reference[key])
            arrays[arm] = values['student_points']
            sample_metrics[arm] = next(r for r in json.loads((path / 'per_image.json').read_text())
                                       if r['image'] == row['image'])
        arrays['teacher'] = reference['teacher_points']
        labels, rgb = reference['labels'], reference['rgb']
        h, w = labels.shape
        stride = 4
        clouds, plane_metrics = {}, {}
        for arm, points in arrays.items():
            residuals = np.zeros((h, w), dtype=np.float32)
            plane_metrics[arm] = {}
            for label in range(1, int(labels.max()) + 1):
                mask = labels == label
                center, normal, extent, rms = fit(points[mask].astype(np.float64))
                residuals[mask] = np.abs((points[mask] - center) @ normal) / extent
                plane_metrics[arm][str(label)] = dict(rms=rms, pixels=int(mask.sum()),
                                                     extent_m=float(extent))
            packed = np.concatenate((points, residuals[..., None]), axis=-1)[::stride, ::stride]
            clouds[arm] = encode(packed.astype('<f4'))
        samples.append(dict(id=row['id'], title=title, scene=row.get('scene'),
                            width=(w + stride - 1) // stride, height=(h + stride - 1) // stride,
                            colors=encode(rgb[::stride, ::stride].astype('uint8')),
                            labels=encode(labels[::stride, ::stride].astype('uint8')),
                            clouds=clouds, planes=plane_metrics,
                            scores={arm: {k: value.get(k) for k in (
                                'planes/student/normalized_rms', 'gt_student/abs_rel',
                                'spatial/cosine', 'summary/cosine', 'summary/norm_ratio')}
                                    for arm, value in sample_metrics.items()}))
    cards = []
    for partition, title in [('train-probe', '训练图 · 16 张诊断样本'),
                             ('validation', '独立验证 · 16 个场景'),
                             ('screenshot', '你的会议室截图 · 独立样本')]:
        data = audit['results'][partition]
        flatness = data['metrics']['planes/student/normalized_rms']
        gt = data['metrics']['gt_student/abs_rel']
        cards.append(f'<div class="card"><b>{title}</b><p>平面残差相对对照：'
                     f'<strong>{flatness["plane_vs_control_percent"]:+.2f}%</strong><br>'
                     f'真实深度 AbsRel 相对对照：{gt["plane_vs_control_percent"]:+.2f}%</p>'
                     f'<small>{data["flatness_better_than_control"]}/{data["images_with_planes"]}'
                     ' 张图的平均平面残差低于对照。两个误差指标均越低越好。</small></div>')
    rows = []
    for key, title in [('planes/student/normalized_rms', '平面 RMS / 区域尺度 ↓'),
                       ('planes/student/normal_angle_deg', '平面法向误差（度）↓'),
                       ('gt_student/abs_rel', '真实深度 AbsRel ↓'),
                       ('gt_student/rmse', '真实深度 RMSE ↓'),
                       ('depth_edges/log_gradient_mae', '边缘 log-depth 梯度误差 ↓'),
                       ('outside_plane/teacher_abs_rel', '平面外相对教师深度误差 ↓'),
                       ('spatial/cosine', 'Spatial cosine ↑'),
                       ('summary/cosine', 'Summary cosine ↑'),
                       ('summary/norm_ratio', 'Summary 幅值比（教师＝1）')]:
        m = audit['results']['validation']['metrics'][key]
        rows.append(f'<tr><td>{title}</td><td>{m["initial"]:.6f}</td>'
                    f'<td>{m["control"]:.6f}</td><td>{m["plane"]:.6f}</td></tr>')
    data_json = json.dumps(samples, separators=(',', ':'), allow_nan=False).replace('</', '<\\/')
    fragment = '''<section id="results"><h2>结果：训练图能变平，未见稳定泛化</h2>
<p class="notice"><b>这次小实验验证了“平面约束能驱动形状变化”，但没有验证它能改善新场景。</b>
训练样本平整度明显改善；独立验证集平均平整度不优于原损失对照，且深度精度下降。
因此不把这个版本直接替换成正式模型，也不把训练集收益当成泛化收益。</p>
<div class="grid">CARDS</div><p>两组均完成 300 次更新；同一初始权重、数据顺序、学习率与优化器设置。
所有对比使用最终权重，教师平面区域完全一致。平面残差按区域尺度归一化，缩小点云不能获得优势。</p>
<p>300 次更新 × 有效 batch 4 ÷ 64 张训练图，约相当于 18.75 遍数据。
训练图已明显受益，但仅有训练前与第 300 步完整评估，还不能判断继续训练能否改善验证表现。
验证集有 11/16 张图的平面残差下降，但少数退化抵消了平均收益；不能据此断言完全没有泛化，也不能把原因确定为训练不足。</p>
<details><summary>独立验证集详细指标（16 张，逐图等权平均）</summary><div style="overflow:auto">
<table><thead><tr><th>指标</th><th>训练前</th><th>特征损失对照</th><th>加入平面约束</th></tr></thead>
<tbody>ROWS</tbody></table></div></details></section>
<section id="geometry"><h2>真实输出 · 四路同步点云</h2>
<p>默认显示你提供的会议室；也可切换两个预先选定的验证样本和一个训练样本。
这里显示原始模型输出，未作平面拟合后替换、缩放对齐或几何修补。</p>
<div class="controls"><label>样本 <select id="sample"></select></label>
<label>区域 <select id="patch"></select></label>
<label>显示 <select id="mode"><option value="cloud">3D 点云</option><option value="residual">原图坐标的平面残差</option></select></label>
<label>颜色 <select id="color"><option value="residual">平面残差热力图</option><option value="rgb">原始颜色</option><option value="labels">教师平面编号</option></select></label>
<button id="front">正视</button><button id="oblique">斜视</button><button id="reset">重置</button></div>
<p class="muted">拖动任一点云同步旋转，滚轮缩放；Shift＋拖动平移。四幅图采用同一个观察相机和米制坐标。
残差热力图从蓝（接近平面）到红（偏离区域尺度 ≥3%）；每个模型在同一教师区域内单独拟合最佳平面，仅用于测量。</p>
<div class="cloud-grid">PANELS</div><div id="plane-scores" style="overflow:auto"></div>
<p id="sample-caption" class="muted"></p></section>
<style>.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center}.controls select{padding:8px;border:1px solid #a6b6b2;border-radius:7px;max-width:100%}
.cloud-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:20px}.cloud-panel{border:1px solid #ccd7d3;border-radius:12px;overflow:hidden}
.cloud-panel h3{font-size:16px;margin:12px 16px}.cloud-panel canvas{width:100%;height:320px;display:block;background:#102128;touch-action:none;cursor:grab}
@media(max-width:700px){.cloud-grid{grid-template-columns:1fr}.cloud-panel canvas{height:300px}}
</style><script id="geometry-data" type="application/json">DATA</script>
<script>
(()=>{'use strict';
const samples=JSON.parse(document.getElementById('geometry-data').textContent),names=['initial','control','plane','teacher'];
const titles=['训练前','特征损失对照','加入平面约束','MoGe 教师'],palette=['#78858b','#eb6d44','#30c6be','#e7c84b','#9891e3','#78bc55','#cf72bb'];
const $=id=>document.getElementById(id),canvases=names.map(n=>$('cloud-'+n));let current,clouds,colors,labels,yaw=.45,pitch=.05,zoom=1,pan=[0,0],center=[0,0,0],span=1,queued=false;
function bytes(s){return Uint8Array.from(atob(s),c=>c.charCodeAt(0));}
samples.forEach((s,i)=>$('sample').add(new Option(s.title+' · '+s.id,i)));
function heat(v){const t=Math.max(0,Math.min(1,v/.03));return `rgb(${Math.round(40+215*t)},${Math.round(170-105*t)},${Math.round(225-180*t)})`;}
function focus(){let pts=clouds.teacher,selected=+$('patch').value,lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
 for(let i=0;i<labels.length;i++){if(selected&&labels[i]!==selected)continue;if(!Number.isFinite(pts[i*4+2])||pts[i*4+2]<=0)continue;for(let j=0;j<3;j++){lo[j]=Math.min(lo[j],pts[i*4+j]);hi[j]=Math.max(hi[j],pts[i*4+j]);}}
 center=lo.map((v,i)=>(v+hi[i])/2);span=Math.max(...lo.map((v,i)=>hi[i]-v),.1);zoom=1;pan=[0,0];schedule();scores();}
function load(){current=samples[+$('sample').value];clouds={};for(const n of names)clouds[n]=new Float32Array(bytes(current.clouds[n]).buffer);colors=bytes(current.colors);labels=bytes(current.labels);
 $('patch').replaceChildren(new Option('全画面',0));for(const k of Object.keys(current.planes.teacher))$('patch').add(new Option('教师平面 '+k,k));$('patch').value='1';
 $('sample-caption').textContent=current.title+'；场景 '+current.scene+'。显示按固定像素网格抽样的原始点，指标使用完整分辨率。';focus();}
function scores(){const selected=$('patch').value;let line='<table><tr><th>当前教师区域</th>'+titles.map(t=>'<th>'+t+'</th>').join('')+'</tr>';
 const vals=names.map(n=>selected==='0'?Object.values(current.planes[n]).reduce((a,p)=>a+p.rms,0)/Object.keys(current.planes[n]).length:current.planes[n][selected]?.rms);
 line+='<tr><td>归一化平面 RMS ↓</td>'+vals.map(v=>'<td>'+Number(v).toFixed(6)+'</td>').join('')+'</tr></table>';$('plane-scores').innerHTML=line;}
function schedule(){if(!queued){queued=true;requestAnimationFrame(()=>{queued=false;draw();});}}
function draw(){const selected=+$('patch').value,mode=$('mode').value,color=$('color').value,cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
 canvases.forEach((canvas,k)=>{const W=canvas.width,H=canvas.height,ctx=canvas.getContext('2d'),pts=clouds[names[k]],scale=Math.min(W,H)*1.2*zoom/span;
 ctx.fillStyle='#102128';ctx.fillRect(0,0,W,H);const order=[];
 for(let i=0;i<labels.length;i++){if(selected&&labels[i]!==selected)continue;let x=pts[4*i],y=pts[4*i+1],z=pts[4*i+2];if(!Number.isFinite(x+y+z)||z<=0)continue;
 if(mode==='residual'){order.push([i,(i%current.width+.5)/current.width*W,(Math.floor(i/current.width)+.5)/current.height*H,0]);continue;}
 x-=center[0];y-=center[1];z-=center[2];const rx=cy*x+sy*z,rz=-sy*x+cy*z,ry=cp*y-sp*rz,depth=sp*y+cp*rz;
 order.push([i,W/2+rx*scale+pan[0],H/2+ry*scale+pan[1],depth]);}
 order.sort((a,b)=>b[3]-a[3]);for(const [i,x,y] of order){if(x<0||x>W||y<0||y>H)continue;
 ctx.fillStyle=mode==='residual'||color==='residual'?(labels[i]?heat(pts[4*i+3]):'#52616a'):color==='labels'?palette[labels[i]%palette.length]:`rgb(${colors[3*i]},${colors[3*i+1]},${colors[3*i+2]})`;
 const size=mode==='residual'?Math.ceil(W/current.width)+1:3.6;ctx.fillRect(x-size/2,y-size/2,size,size);}
 ctx.fillStyle='#afc9cb';ctx.font='12px system-ui';ctx.fillText(order.length.toLocaleString()+' points · 同步视角',12,H-12);
 });}
 for(const canvas of canvases){let drag=null;canvas.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);});
 canvas.addEventListener('pointermove',e=>{if(!drag)return;let dx=e.clientX-drag[0],dy=e.clientY-drag[1];drag=[e.clientX,e.clientY];if(e.shiftKey){pan[0]+=dx*canvas.width/canvas.clientWidth;pan[1]+=dy*canvas.height/canvas.clientHeight;}else{yaw+=dx*.006;pitch=Math.max(-1.45,Math.min(1.45,pitch+dy*.006));}schedule();});
 canvas.addEventListener('pointerup',()=>drag=null);canvas.addEventListener('pointercancel',()=>drag=null);canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.15,Math.min(10,zoom*Math.exp(-e.deltaY*.001)));schedule();},{passive:false});}
 $('sample').onchange=load;$('patch').onchange=focus;$('color').onchange=schedule;$('mode').onchange=schedule;
 $('front').onclick=()=>{yaw=0;pitch=0;schedule();};$('oblique').onclick=()=>{yaw=.65;pitch=.12;schedule();};$('reset').onclick=()=>{yaw=.45;pitch=.05;focus();};load();
})();</script>'''
    panels = ''.join(f'<div class="cloud-panel"><h3>{html.escape(title)}</h3>'
                     f'<canvas id="cloud-{name}" width="720" height="440" '
                     f'aria-label="{html.escape(title)}交互点云"></canvas></div>'
                     for name, title in NAMES.items())
    fragment = fragment.replace('CARDS', ''.join(cards)).replace('ROWS', ''.join(rows))
    fragment = fragment.replace('PANELS', panels).replace('DATA', data_json)
    target = root / 'result-section.html'
    temporary = target.with_suffix('.incoming')
    temporary.write_text(fragment)
    temporary.replace(target)
    proof = dict(stage='passed', samples=[s['id'] for s in samples], fixed_teacher_arrays=True,
                 raw_geometry_no_alignment=True, point_stride=4, input_hashes=hashes,
                 output_sha256=hashlib.sha256(fragment.encode()).hexdigest())
    target = root / 'visualization-verification.json'
    temporary = target.with_suffix('.incoming')
    temporary.write_text(json.dumps(proof, indent=2))
    temporary.replace(target)
    print(json.dumps(dict(bytes=len(fragment.encode()), samples=proof['samples'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    build(parser.parse_args().root)
