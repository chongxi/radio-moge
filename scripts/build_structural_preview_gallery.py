"""Publish a self-contained comparison of actual, hash-checked RGB previews."""
import argparse
import base64
import hashlib
import json
from pathlib import Path


def build(report_path, output):
    report = json.loads(report_path.read_text())
    if report['completed'] != 6 or report['training_admission'] is not False:
        raise ValueError('Expected the bounded, unadmitted six-image preview')
    records = []
    for row in report['records']:
        path = report_path.parent / Path(row['image']['path']).name
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != row['image']['sha256']:
            raise ValueError('Preview image changed')
        records.append(dict(scene=row['scene'], view=row['view'],
            src='data:image/png;base64,' + base64.b64encode(content).decode(),
            sha256=row['image']['sha256']))
    template = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>墙面、立柱与横梁 · 实际渲染样例</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f5f3ef;color:#20282a;font:15px/1.5 system-ui,sans-serif}
main{max-width:1450px;margin:auto;padding:24px}header{display:flex;justify-content:space-between;align-items:start;gap:20px}
h1{font-size:25px;margin:0 0 7px}p{margin:0;color:#596466}.badge{color:#386350;background:#e4eee6;padding:6px 11px;border-radius:20px;white-space:nowrap}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0}.chips span{background:white;border:1px solid #dcded7;border-radius:6px;padding:5px 10px}
.controls{display:flex;align-items:center;justify-content:space-between;gap:14px;margin:18px 0}
select,button{font:inherit;background:white;color:#20282a;border:1px solid #bac4bf;border-radius:7px;padding:8px 12px;cursor:pointer}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}figure{margin:0;background:white;border:1px solid #d8ded8;border-radius:12px;overflow:hidden}
figcaption{padding:14px 16px;border-bottom:1px solid #d8ded8}h2{font-size:18px;margin:0 0 4px}figcaption p{font-size:13px}
img{width:100%;display:block;aspect-ratio:4/3;object-fit:contain;background:#e9e9e5}
.note{margin-top:18px;padding:14px 17px;background:#e8ece8;border-radius:9px;color:#405149}
footer{font-size:12px;color:#68736e;margin-top:15px}button:focus,select:focus{outline:2px solid #568773}
@media(max-width:780px){main{padding:15px}.grid{grid-template-columns:1fr}header{display:block}.badge{display:inline-block;margin-top:12px}h1{font-size:22px}}
</style><main><header><div><h1>空房间 · 墙面、柱与梁</h1><p>3 个场景，6 张实际模拟器 RGB。可以切换场景，比较两种视角。</p></div>
<span class="badge">样例已逐张查看</span></header>
<div class="chips"><span>无窗户／无室外视野</span><span>目标墙距相机 12 m</span><span>方柱＋圆柱</span><span>横梁＋水平圆柱</span></div>
<div class="controls"><label>场景 <select id="scene"><option value="0">01 · 基础柱梁</option><option value="1">02 · 增加后排方柱</option><option value="2">03 · 后排圆柱与纵向梁</option></select></label><button id="next">下一个场景 →</button></div>
<div class="grid"><figure><figcaption><h2>左：正对远墙 · 普通相机</h2><p>32° 竖直视场，保留真实相机透视；能看到柱脚、地面和顶面。</p></figcaption><img id="perspective" alt="正对远墙的普通相机实际 RGB"></figure>
<figure><figcaption><h2>右：正交投影 · 无透视</h2><p>平行视线，没有近大远小；此列只用于明确你想要的视角。</p></figcaption><img id="orthographic" alt="无透视正交相机实际 RGB"></figure></div>
<div class="note">墙、柱、梁都是场景中的三维结构；没有窗户、家具或可见人物。圆柱用于曲面和边界变化，平面主体是墙面、方柱侧面与横梁。</div>
<footer>这批样例尚未加入训练，也未启动新的批量任务。后续训练仍采用 RGB 对齐 MoGe-v2 的特征损失；不添加几何 loss。固定训练清单保持不变。</footer>
</main><script>const rows=__ROWS__;
function show(){const n=Number(document.getElementById('scene').value);for(const [id,view] of [['perspective','frontal_far_perspective'],['orthographic','orthographic_comparison_only']]){document.getElementById(id).src=rows.find(x=>x.scene===n&&x.view===view).src;}}
document.getElementById('scene').addEventListener('change',show);document.getElementById('next').addEventListener('click',()=>{const s=document.getElementById('scene');s.value=(Number(s.value)+1)%3;show();});show();</script></html>'''
    html = template.replace('__ROWS__', json.dumps(records, ensure_ascii=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.incoming')
    temporary.write_text(html)
    temporary.replace(output)
    return dict(images=len(records), page=str(output),
                sha256=hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(build(args.report, args.output)))
