"""Report the measured cosine threshold, train/validation gap and depth tradeoffs."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def atomic(path, value):
    tmp=path.with_suffix('.incoming')
    tmp.write_text(value)
    tmp.replace(path)


def report(root):
    audit=json.loads((root/'evaluation-verification.json').read_text())
    plan=json.loads((root/'plan.json').read_text())
    assert audit['stage']=='passed' and plan['profile']=='dual-cosine-512'
    assert audit['full_architectures_identical']
    target=plan['target_cosine']
    metrics=audit['results']['validation']['metrics']
    reached=audit['endpoint_target_reached']['validation']['angular']
    conclusion=(f'本次 {audit["steps"]} 步后，加强双 cosine 的验证 Spatial 为 '
                f'{metrics["spatial/cosine"]["angular"]:.6f}，Summary 为 '
                f'{metrics["summary/cosine"]["angular"]:.6f}；'
                f'两个验证均值同时超过 {target} 的门槛：{"达到" if reached else "未达到"}。'
                f'相对原损失，Spatial cosine 变化 '
                f'{metrics["spatial/cosine"]["angular_minus_control"]:+.6f}，Summary 变化 '
                f'{metrics["summary/cosine"]["angular_minus_control"]:+.6f}；'
                f'真实深度 AbsRel 变化 {metrics["gt_student/abs_rel"]["angular_vs_control_percent"]:+.2f}%（越低越好）。'
                '这是单次小面板初筛，不能作为泛化或收敛上限的证明。')
    crossings={}
    for part,file in [('validation','validation.jsonl'),('train-probe','train-probe.jsonl')]:
        crossings[part]={}
        for arm in plan['arms']:
            points=[json.loads(x) for x in (root/arm/file).read_text().splitlines()]
            crossings[part][arm]={}
            for which,keys in [('spatial',['spatial/cosine']),('summary',['summary/cosine']),
                               ('both',['spatial/cosine','summary/cosine'])]:
                first=next((x for x in points if all(x[k]>target for k in keys)),None)
                crossings[part][arm][which]=({k:first[k] for k in ('step','optimizer_seconds','elapsed_seconds')}
                                             if first else None)
    distributions={}
    for part in ('validation','train-probe','screenshot'):
        distributions[part]={}
        for arm in plan['arms']:
            folder=(root/arm/f'eval-{audit["steps"]:07d}' if part=='validation'
                    else root/'assessment'/arm/part)
            payload=(folder/'per_image.json').read_bytes()
            assert hashlib.sha256(payload).hexdigest()==audit['aggregates'][part][arm]['sha256']['per_image.json']
            images=json.loads(payload)
            distributions[part][arm]={
                'images':len(images),
                'images_both_above_target':sum(all(x[k]>target for k in ('spatial/cosine','summary/cosine')) for x in images),
                **{key:dict(min=min(x[key] for x in images),max=max(x[key] for x in images),
                            images_above_target=sum(x[key]>target for x in images))
                   for key in ('spatial/cosine','summary/cosine')},
            }
    data=dict(stage='passed',checked_at=time.time(),title_zh='加强双 cosine：终点结果与 0.98 目标',
              conclusion_zh=conclusion,steps=audit['steps'],target_cosine=target,
              endpoint_target_reached=audit['endpoint_target_reached'],
              validation=metrics,per_partition_image_distribution=distributions,
              first_observed_target_crossings=crossings,
              parameters={a:x['parameters'] for a,x in audit['arms'].items()},
              training_seconds={a:x['optimizer_seconds'] for a,x in audit['arms'].items()},
              total_seconds={a:x['stop']['elapsed_seconds'] for a,x in audit['arms'].items()},
              limits=plan['limitations'])
    assert len(set(data['parameters'].values()))==1
    atomic(root/'result-report.json',json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False))
    lines=[]
    for part,label in [('validation','验证'),('train-probe','训练诊断')]:
        for which,metric in [('spatial','Spatial'),('summary','Summary'),('both','两者同时')]:
            cells=[]
            for arm in plan['arms']:
                v=crossings[part][arm][which]
                cells.append('未达到' if v is None else f'{v["step"]} 步 · {v["optimizer_seconds"]:.2f} 秒训练')
            lines.append(f'<tr><td>{label} · {metric} &gt; {target}</td><td>{cells[0]}</td><td>{cells[1]}</td></tr>')
    body='<section class="panel" id="dual-target-results"><h2>首次越过 0.98 的评估点</h2><div class="scroll"><table><tr><th>指标</th><th>原损失</th><th>加强双 cosine</th></tr>'+''.join(lines)+'</table></div><p>首次越线只作诊断；终点是否达标另行记录。训练诊断只覆盖固定 16 张训练图，不代表全部 512 张图。加权总 loss 因权重不同，不能横向比较。</p></section>'
    atomic(root/'target-section.html',body)
    print(json.dumps(dict(conclusion=conclusion,endpoint_target_reached=data['endpoint_target_reached']),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    report(p.parse_args().root)
