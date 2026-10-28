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
    # Match planarity_metrics: in-plane extent, excluding normal-direction variance.
    extent = np.sqrt(np.maximum(values[1:], 0).sum())
    residual = np.abs(shifted @ normal) / max(extent, 1e-12)
    return center, normal, extent, float(np.sqrt(np.mean(residual ** 2)))


def encode(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


def build(root):
    audit = json.loads((root / 'evaluation-verification.json').read_text())
    assert audit['stage'] == 'passed'
    steps = audit['steps']
    plan = json.loads((root / 'plan.json').read_text())
    names = dict(NAMES)
    ablation = plan.get('profile') == 'flatness-ablation-512'
    joint = plan.get('profile') == 'joint-cosine-512'
    dual = plan.get('profile') == 'dual-cosine-512'
    treatment = 'angular' if dual else 'joint' if joint else 'plane'
    if joint:
        names = {'initial': names['initial'], 'control': '原 adapter · 特征损失',
                 'joint': '新增联合 attention · 特征损失', 'teacher': names['teacher']}
    if dual:
        names = {'initial': names['initial'], 'control': '原损失 · 同一 adapter',
                 'angular': '加强双 cosine · 保留 MSE', 'teacher': names['teacher']}
    if ablation:
        names.update(control='旧 flatness · 保留 normal＋anchor',
                     plane='新 flatness · 保留 normal＋anchor')
    report_path = root / 'result-report.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    validation = audit['results']['validation']
    passes = steps * plan['effective_batch'] / plan['train_images']
    conclusion = html.escape(report.get('conclusion_zh', '请结合平整度、真实深度与边缘指标判断效果。'))
    title = html.escape(report.get('title_zh', '最终匹配步数的评估结果'))
    interval = plan['evaluation_steps'][1] if plan.get('evaluation_steps') else None
    cadence = (f'每 {interval} 步均已记录训练诊断与独立验证，曲线显示全部预定评估点。'
               if plan.get('evaluation_steps') else '本轮仅记录训练前与最终评估。')
    intro = (f'<section id="results"><h2>{title}</h2><p class="notice">{conclusion}</p>'
             '<div class="grid">CARDS</div>'
             f'<p>两组均完成 {steps} 次更新；同一初始输出、数据顺序、学习率与优化器设置。'
             '所有对比使用最终权重，教师平面区域完全一致。平面残差按区域尺度归一化，缩小点云不能获得优势。</p>'
             f'<p>{steps} 次更新 × 有效 batch {plan["effective_batch"]} ÷ '
             f'{plan["train_images"]} 张训练图，约相当于 {passes:.2f} 遍数据。{cadence}'
             f'验证集有 {validation["flatness_better_than_control"]}/{validation["images_with_planes"]}'
             ' 张图的平均平面残差低于对照。固定的 16 张训练诊断图不代表整个训练集；'
             '单次小规模实验不等于标准基准或收敛证明。</p>')
    old_comparison = report.get('comparison_old64', {})
    if ablation:
        wall = report['target_wall']
        intro += ('<p>这是已在前几轮查看过的小验证面板；目标截图还用于提出本次修改，'
                  '因此结果用于诊断，不是新的盲测基准。教师一致性误差与真实深度精度分开解读。</p>'
                  '<h3>你关心的墙面：单独查看区域 1</h3>'
                  '<table><tr><th>自身平面归一化 RMS ↓</th><th>训练前</th><th>旧 flatness</th>'
                  f'<th>新 flatness</th></tr><tr><td>墙面／黑板区域 1</td><td>{wall["initial"]:.6f}</td>'
                  f'<td>{wall["old"]:.6f}</td><td>{wall["new"]:.6f}</td></tr></table>'
                  f'<p>相对旧公式变化 {wall["change_percent"]:+.2f}%；相对训练前变化 '
                  f'{wall["new_vs_initial_percent"]:+.2f}%。这与整张截图所有平面的平均值不同。</p>')
        counts = report.get('new_lower_error_points', {})
        intro += (f'<p>在 {report.get("post_start_curve_points", 0)} 个训练后预定评估点中，'
                  f'新公式有 {counts.get("planes/student/normalized_rms", 0)} 个点的验证平均平面残差更低，'
                  f'{counts.get("gt_student/abs_rel", 0)} 个点的验证 GT AbsRel 更低。'
                  '这说明趋势，但不等同于独立重复实验或统计显著性。</p>')
        repeats = report['prior_old_objective_reproduction']['percent_differences']
        repeat_rows = ''.join(f'<tr><td>{label}</td><td>{repeats["validation"][key]:+.3f}%</td></tr>'
                              for key, label in [('planes/student/normalized_rms', '验证平面残差'),
                                                 ('gt_student/abs_rel', '验证 GT AbsRel'),
                                                 ('depth_edges/log_gradient_mae', '验证边缘误差')])
        intro += ('<details><summary>旧公式重跑的数值波动</summary><p>本次旧公式对照与上一轮相同配置的平面组'
                  '并非逐位复现。下表是这两次旧公式运行之间的变化，不能据一次重复估计完整方差；'
                  '很小的差异不应当作稳健结论。</p><table><tr><th>指标</th><th>重跑相对上次</th></tr>'
                  + repeat_rows + '</table></details>')
    if joint:
        intro += ('<p>两组均只用特征损失；几何指标仅用于评估。联合分支约多 263 万参数，'
                  '因此这是结构与额外容量共同变化的初筛。验证面板和截图已多次查看，'
                  '不能视为新的盲测。特征 cosine 与真实深度精度分开解读。</p>')
    if dual:
        intro += ('<p>两组结构与参数量相同，仅改变方向损失：Spatial cosine 权重 0.1→1，'
                  'Summary cosine 权重 0→1，两项归一化 MSE 保持各 1。几何仅用于评估，'
                  '未施加平面损失。0.98 门槛看最终验证图像均值；训练诊断达标不等于泛化。'
                  '该验证面板和截图已多次查看，仍需更广泛的独立验证。</p>')
    if old_comparison:
        comparison_rows = []
        for key, label in [('planes/student/normalized_rms', '验证平面残差 ↓'),
                           ('gt_student/abs_rel', '验证 GT AbsRel ↓'),
                           ('depth_edges/log_gradient_mae', '验证边缘误差 ↓')]:
            item = old_comparison['validation'][key]
            comparison_rows.append('<tr><td>' + label + '</td>' + ''.join(
                f'<td>{item[group][arm]:.6f}</td>'
                for group in ('old64', 'coverage512') for arm in ('control', 'plane')) + '</tr>')
        intro += ('<details><summary>与上一轮 64 张图比较：相同验证面板、相同 300 步</summary>'
                  '<p>覆盖扩大 8 倍，每张图的重复训练次数相应减少；这是相同更新预算下的比较，'
                  '不能区分充分训练后的最终上限。</p><div style="overflow:auto"><table><tr>'
                  '<th>指标</th><th>64 图对照</th><th>64 图平面</th><th>512 图对照</th><th>512 图平面</th>'
                  '</tr>' + ''.join(comparison_rows) + '</table></div></details>')
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
        paths = ({'initial': root / ('control' if joint or dual else 'plane') / 'eval-0000000',
                  'control': root / 'control' / f'eval-{steps:07d}',
                  treatment: root / treatment / f'eval-{steps:07d}'} if partition == 'validation'
                 else {arm: root / 'assessment' / arm / partition
                       for arm in ('initial', 'control', treatment)})
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
        primary = (f'Spatial cosine 差：{data["metrics"]["spatial/cosine"][treatment+"_minus_control"]:+.6f}<br>'
                   f'Summary cosine 差：{data["metrics"]["summary/cosine"][treatment+"_minus_control"]:+.6f}<br>'
                   if joint or dual else '')
        cards.append(f'<div class="card"><b>{title}</b><p>{primary}平面残差相对对照：'
                     f'<strong>{flatness[treatment+"_vs_control_percent"]:+.2f}%</strong><br>'
                     f'真实深度 AbsRel 相对对照：{gt[treatment+"_vs_control_percent"]:+.2f}%</p>'
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
                       ('spatial/norm_ratio', 'Spatial 幅值比（教师＝1）'),
                       ('summary/norm_ratio', 'Summary 幅值比（教师＝1）')]:
        m = audit['results']['validation']['metrics'][key]
        rows.append(f'<tr><td>{title}</td><td>{m["initial"]:.6f}</td>'
                    f'<td>{m["control"]:.6f}</td><td>{m[treatment]:.6f}</td></tr>')
    data_json = json.dumps(samples, separators=(',', ':'), allow_nan=False).replace('</', '<\\/')
    fragment = intro + '''
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
    if ablation:
        fragment = fragment.replace('<th>特征损失对照</th><th>加入平面约束</th>',
                                    '<th>旧 flatness</th><th>新 flatness</th>')
        fragment = fragment.replace("['训练前','特征损失对照','加入平面约束','MoGe 教师']",
                                    "['训练前','旧 flatness','新 flatness','MoGe 教师']")
    if joint:
        fragment = fragment.replace('<th>特征损失对照</th><th>加入平面约束</th>',
                                    '<th>原 adapter</th><th>新增联合 attention</th>')
        fragment = fragment.replace("names=['initial','control','plane','teacher']",
                                    "names=['initial','control','joint','teacher']")
        fragment = fragment.replace("['训练前','特征损失对照','加入平面约束','MoGe 教师']",
                                    "['训练前','原 adapter','联合 attention','MoGe 教师']")
    if dual:
        fragment = fragment.replace('<th>特征损失对照</th><th>加入平面约束</th>',
                                    '<th>原损失</th><th>加强双 cosine</th>')
        fragment = fragment.replace("names=['initial','control','plane','teacher']",
                                    "names=['initial','control','angular','teacher']")
        fragment = fragment.replace("['训练前','特征损失对照','加入平面约束','MoGe 教师']",
                                    "['训练前','原损失','加强双 cosine','MoGe 教师']")
    panels = ''.join(f'<div class="cloud-panel"><h3>{html.escape(title)}</h3>'
                     f'<canvas id="cloud-{name}" width="720" height="440" '
                     f'aria-label="{html.escape(title)}交互点云"></canvas></div>'
                     for name, title in names.items())
    fragment = fragment.replace('CARDS', ''.join(cards)).replace('ROWS', ''.join(rows))
    fragment = fragment.replace('PANELS', panels).replace('DATA', data_json)
    target = root / 'result-section.html'
    temporary = target.with_suffix('.incoming')
    temporary.write_text(fragment)
    temporary.replace(target)
    proof = dict(stage='passed', samples=[s['id'] for s in samples], fixed_teacher_arrays=True,
                 raw_geometry_no_alignment=True, point_stride=4, input_hashes=hashes,
                 arm_labels=names, normalization='RMS / sqrt(lambda1 + lambda2)',
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
