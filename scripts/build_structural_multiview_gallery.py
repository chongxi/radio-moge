"""Show actual closed-room multiview RGB, keeping projection labels explicit."""

import argparse
import base64
import hashlib
import json
from pathlib import Path


def build(batch, output):
    report = json.loads((batch / "report.json").read_text())
    manifest = batch / "rgb-candidates.jsonl"
    if (
        not report["all_requested_scenes_closed"]
        or report["completed"] != 32
        or hashlib.sha256(manifest.read_bytes()).hexdigest() != report["manifest"]["sha256"]
    ):
        raise ValueError("Require the closed, verified32-image preflight")
    rows = []
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        image = batch / "rgb" / Path(row["image"]).name
        content = image.read_bytes()
        if hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError("Image identity differs")
        rows.append(
            dict(
                scene=row["scene_ordinal"],
                slot=row["slot"],
                projection=row["projection"],
                src="data:image/png;base64," + base64.b64encode(content).decode(),
                yaw=(row["requested_camera"]["rotation"]["y"] + 180) % 360 - 180,
                wall_distance=row["wall_distance_m"],
            )
        )
    template = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>空房间与柱梁 · 普通相机多角度</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f3ed;color:#23322e;font:15px/1.5 system-ui,sans-serif}
main{max-width:1100px;margin:auto;padding:24px}h1{font-size:25px;margin:0 0 7px}p{margin:0;color:#52655e}
.badges{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0}.badges span{background:#e4ece4;padding:6px 11px;border-radius:8px}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:18px 0}select,button{font:inherit;padding:8px 12px;border:1px solid #a6b5a9;border-radius:7px;background:white;color:#23322e;cursor:pointer}
.views{display:flex;gap:8px;flex-wrap:wrap}.views button[aria-pressed=true]{background:#295b49;color:white;border-color:#295b49}
figure{margin:18px 0 0;border-radius:12px;overflow:hidden;border:1px solid #cbd5ca;background:white}
figcaption{padding:12px 16px;display:flex;justify-content:space-between;gap:15px;flex-wrap:wrap}
img{display:block;width:100%;max-height:650px;object-fit:contain;background:#e3e6df;aspect-ratio:4/3}
footer{margin-top:14px;color:#607269;font-size:13px}.active-note{font-size:13px;color:#4e6a5e}
@media(max-width:700px){main{padding:14px}h1{font-size:21px}img{max-height:none}}
</style><main><h1>空房间 · 多角度看墙面、柱与梁</h1><p>普通相机为主，保留真实透视。可切换场景和左右机位。</p>
<div class="badges"><span>量产：90% 普通相机</span><span>10% 正交补充</span><span>无窗户／无室外</span><span>圆柱＋方柱＋横梁</span></div>
<div class="controls"><label>场景 <select id="scene"></select></label><button id="next">下一个场景 →</button><span class="active-note">8 个场景／32 张实际预检图已查看</span></div>
<div class="views" id="views"><button data-slot="0">正面远景</button><button data-slot="1">左侧机位</button><button data-slot="2">右侧机位</button><button data-slot="3">正交补充</button></div>
<figure><figcaption><strong id="caption"></strong><span id="detail"></span></figcaption><img id="rgb" alt="模拟器实际渲染的无窗空房间与柱梁"></figure>
<footer>正式生成还包含高低机位和不同距离。目标为10,000张训练图＋320张独立验证图，按完整场景分开。仅生成RGB，不添加几何标签或几何loss；这些预检图尚未进入训练。</footer>
</main><script>const rows=__ROWS__;const names=['正面远景','左侧机位','右侧机位','正交补充'];let slot=0;
const select=document.getElementById('scene');for(let i=0;i<8;i++){const o=document.createElement('option');o.value=i;o.textContent='场景 '+String(i+1).padStart(2,'0');select.appendChild(o);}
function show(){const scene=Number(select.value),r=rows.find(x=>x.scene===scene&&x.slot===slot);document.getElementById('rgb').src=r.src;document.getElementById('caption').textContent='场景 '+(scene+1)+' · '+names[slot];document.getElementById('detail').textContent=r.projection==='perspective'?'普通相机 · 水平转角 '+r.yaw.toFixed(1)+'°':'正交投影 · 平行视线';document.querySelectorAll('[data-slot]').forEach(b=>b.setAttribute('aria-pressed',Number(b.dataset.slot)===slot));}
select.addEventListener('change',show);document.getElementById('next').addEventListener('click',()=>{select.value=(Number(select.value)+1)%8;show();});document.querySelectorAll('[data-slot]').forEach(b=>b.addEventListener('click',()=>{slot=Number(b.dataset.slot);show();}));show();</script></html>"""
    output.parent.mkdir(exist_ok=True, parents=True)
    temp = output.with_name(output.name + ".incoming")
    temp.write_text(template.replace("__ROWS__", json.dumps(rows, ensure_ascii=False)))
    temp.replace(output)
    return dict(
        images=len(rows), output=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest()
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(build(args.batch, args.output)))
