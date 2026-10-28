"""Publish measured speed tradeoffs without treating small differences as robust gains."""
import argparse
import json
from pathlib import Path
import time


def atomic(path, text):
    temporary=path.with_suffix('.incoming')
    temporary.write_text(text)
    temporary.replace(path)


def report(root):
    audit=json.loads((root/'evaluation-verification.json').read_text())
    assert audit['stage']=='passed'
    arms=audit['arms']
    metrics=audit['results']['validation']['metrics']
    elapsed={a:x['stop']['elapsed_seconds'] for a,x in arms.items()}
    training={a:x['optimizer_seconds'] for a,x in arms.items()}
    parameter_change=100*(arms['joint']['parameters']/arms['control']['parameters']-1)
    training_change=100*(training['joint']/training['control']-1)
    conclusion=(f'本次 600 步短跑没有显示明确的学习提速或深度精度优势。联合分支的验证 spatial cosine '
                f'仅比原结构高 {metrics["spatial/cosine"]["joint_minus_control"]:.6f}，summary cosine 高 '
                f'{metrics["summary/cosine"]["joint_minus_control"]:.6f}；真实深度 AbsRel '
                f'变化 {metrics["gt_student/abs_rel"]["joint_vs_control_percent"]:+.3f}%（越低越好）。'
                '这只是已有 adapter 后接零初始化小残差分支的初筛，不能否定其他联合 attention 设计。')
    data=dict(stage='passed',checked_at=time.time(),title_zh='联合 attention：本轮收益很小，尚无明确提速',
              conclusion_zh=conclusion,parameters={a:x['parameters'] for a,x in arms.items()},
              parameter_increase_percent=parameter_change,optimizer_seconds=training,
              optimizer_seconds_change_percent=training_change,end_to_end_seconds=elapsed,
              validation=metrics,first_observed_crossings={k:v['first_observed_crossing'] for k,v in audit['speed']['validation'].items()},
              limits=['Single seed and repeatedly viewed 16-image validation panel; small differences are not a robust effect.',
                      'More trainable parameters in the joint arm; interaction and capacity are not isolated.',
                      'Adapter+decoder branch latency is an in-evaluation observation, not a dedicated warmed microbenchmark.',
                      'Both arms improve some metrics versus initial; compare joint against fresh control to attribute effects.',
                      'Zero residual initialization and 600-step warm start do not establish from-scratch convergence speed.'])
    atomic(root/'result-report.json',json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False))
    rows=[]
    for key,label in [('spatial/cosine','Spatial cosine'),('summary/cosine','Summary cosine')]:
        targets=audit['speed']['validation'][key]['first_observed_crossing']
        cells=[]
        for arm in ('control','joint'):
            value=targets[arm]
            cells.append('未达到' if value is None else
                         f'{value["step"]} 步 / {value["optimizer_seconds"]:.2f} 秒训练 / {value["elapsed_seconds"]:.2f} 秒总耗时')
        rows.append('<tr><td>'+label+' +0.001</td>'+''.join('<td>'+x+'</td>' for x in cells)+'</tr>')
    body=f'''<section class="panel" id="joint-speed-results"><h2>速度与额外成本</h2>
<p>首次达标评估点的步数相同：spatial 为 400 步，summary 为 500 步。训练循环耗时略增；
没有证据表明本轮联合分支更快达到相同特征相似度。首次达标不表示以后一直达标，summary 曲线存在波动。</p>
<div class="scroll"><table><tr><th>固定目标</th><th>原 adapter</th><th>联合 attention</th></tr>{''.join(rows)}
<tr><td>600 步训练循环（不含加载／评估）</td><td>{training['control']:.2f} 秒</td><td>{training['joint']:.2f} 秒（{training_change:+.2f}%）</td></tr>
<tr><td>每组总运行时间</td><td>{elapsed['control']:.2f} 秒</td><td>{elapsed['joint']:.2f} 秒</td></tr>
<tr><td>可训练参数</td><td>{arms['control']['parameters']:,}</td><td>{arms['joint']['parameters']:,}（{parameter_change:+.2f}%）</td></tr>
</table></div><p>同训练时间区间的 cosine 提升曲线用线性插值作描述性积分，不代表统计显著性；
验证 spatial 平均增益仅多 {audit['speed']['validation']['spatial/cosine']['common_interval_average_cosine_gain']['optimizer_seconds']['joint_minus_control']:.7f}，
summary 多 {audit['speed']['validation']['summary/cosine']['common_interval_average_cosine_gain']['optimizer_seconds']['joint_minus_control']:.7f}。</p>
<p class="muted">推理分支单次测量包含 adapter 与固定 decoder，排除共享 RADIO；首次测量有明显预热影响，不能据此给出稳定延迟结论。
本轮是 4.69 遍小子集的 warm-start 初筛，不是从零训练至收敛的排名。后续独立结构对照需要相近容量和重复运行。</p></section>'''
    atomic(root/'speed-section.html',body)
    print(json.dumps(dict(conclusion=conclusion,training_seconds_change_percent=training_change,parameters_change_percent=parameter_change),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    report(parser.parse_args().root)
