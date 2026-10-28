"""Publish a self-contained progress/result page; no sibling artifacts exposed."""

import argparse
import base64
import html
import json
from pathlib import Path
import time


def render(base, output):
    diagnostic = base / 'teacher-plane-diagnostic-20261028-nyu0271'
    pilot = base / 'teacher-plane-subset-pilot-20261028'
    status_path = pilot / 'status.json'
    state = json.loads(status_path.read_text()) if status_path.exists() else {'stage': 'preparing'}
    labels = {'preparing': '准备中', 'starting': '启动中', 'plane': '平面约束组训练中',
              'control': '原损失对照组训练中', 'assessment': '训练结束，正在评估',
              'complete': '训练和评估已结束，结果核验中', 'needs_inspection': '运行异常，正在检查'}
    status = labels.get(state['stage'], state['stage'])
    progress = []
    for arm, title in [('plane', '特征损失＋平面约束'), ('control', '原有特征损失')]:
        rows_path = pilot / arm / 'train.jsonl'
        rows = [json.loads(s) for s in rows_path.read_text().splitlines()] if rows_path.exists() else []
        step = rows[-1]['step'] if rows else 0
        progress.append(f'<div class="card"><b>{title}</b><p>{step} / 300 次更新</p>'
                        f'<progress value="{step}" max="300"></progress></div>')
    report = json.loads((diagnostic / 'report.json').read_text())
    image = base64.b64encode((diagnostic / 'plane-diagnostic.png').read_bytes()).decode()
    overlay = base64.b64encode((diagnostic / 'plane-overlay.png').read_bytes()).decode()
    stamp = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    page = f'''<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>墙面平整度 · 小子集对照实验</title>
<style>body{{margin:0;background:#f5f5f1;color:#19272a;font:16px/1.7 system-ui,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:32px 20px}}h1{{line-height:1.25;font-size:32px}}
.tag{{background:#e0efe9;padding:7px 13px;border-radius:20px;display:inline-block}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}}
.card,section{{background:white;border:1px solid #dbe1df;border-radius:14px;padding:20px;margin:20px 0}}
img{{width:100%;height:auto;display:block}}small,.muted{{color:#647173}}a{{color:#146556}}
progress{{width:100%;height:12px;accent-color:#187963}}button{{padding:8px 18px;border:1px solid #8ba59c;
border-radius:8px;background:white;cursor:pointer}}table{{border-collapse:collapse;width:100%}}
td,th{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}.notice{{border-left:4px solid #c48a36;padding-left:14px}}
</style><main><small>RADIO → MoGe · 几何验证</small><h1>能否把墙面训平？</h1>
<span class="tag">{html.escape(status)}</span> <button onclick="location.reload()">刷新进度</button>
<p class="muted">页面更新：{stamp}</p>
<p>64 张训练图 / 32 个场景；16 张验证图 / 16 个独立场景。两组从同一份第 83,494 步权重开始，
学习率 1×10⁻⁵，有效 batch 4，最多各 300 次更新。会议室截图单独保留，不用于训练或调参。</p>
<div class="grid">{''.join(progress)}</div>
<section><h2>训练后的结果</h2><p class="notice">本次小实验的对照结果尚未完成。
下面展示的是训练前已有模型的问题诊断，不能据此判断新损失是否有效。</p>
<p>将比较：固定教师区域内的平面残差、法向角度、真实深度误差、区域外深度、物体边缘，
以及 spatial / summary 的 cosine similarity 和幅值。</p></section>
<section><h2>问题诊断：你截图中的会议室</h2><p>现有第 83,494 步模型与 MoGe 教师对比。
教师几何中识别到 {report['plane_count']} 个大平面，覆盖 {report['coverage_fraction']:.1%} 的画面。
黑板区域的归一化平面残差为教师的 <b>4.40 倍</b>；该指标除以区域尺度，单纯缩小点云不能改善它。</p>
<p><a href="https://buildhub-nano-test--p7867.pub.astera.work/#image=nyuv2_0271&mode=cloud" target="_blank" rel="noopener">打开原始交互点云：同一会议室 ↗</a></p>
<img alt="固定教师平面区域与现有模型的残差" src="data:image/png;base64,{image}">
<details><summary>查看教师选中的平面区域</summary><img alt="教师平面分区" src="data:image/png;base64,{overlay}"></details>
</section><p class="muted">这是小规模可行性测试。教师平面是伪标签；验证集结果与单张截图将分开报告。
当前不对平面以外的物体施加平滑损失。</p></main></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.incoming')
    temporary.write_text(page)
    temporary.replace(output)
    print(json.dumps(dict(page=str(output), bytes=output.stat().st_size, stage=state['stage'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.base, args.output)
