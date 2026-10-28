"""Render verified actual finalist geometry using the existing synchronized viewer."""

import argparse
import ast
import base64
import hashlib
import html
import json
from pathlib import Path

import numpy as np


LABELS = {
    "legacy_2x1280": "原 attention · 1,000 步",
    "joint_combined": "RoPE＋gate＋register＋SwiGLU",
    "fusion_mlp": "Fusion MLP · 1,000 步",
    "joint_4x1024": "联合 Transformer · 4×1024",
    "teacher": "MoGe 教师",
}


def encode(value):
    return base64.b64encode(np.ascontiguousarray(value).tobytes()).decode()


def build(root, capacity=False):
    assert json.loads((root / "confirmation-verification.json").read_text())["stage"] == "passed"
    selected = json.loads((root / "shortlist.json").read_text())["selected"]
    names = [item["name"] for item in selected] + ["teacher"]
    labels_by_name = dict(LABELS)
    if capacity:
        specs = json.loads((root / "plan.json").read_text())["candidates"]
        labels_by_name.update(
            {s["name"]: f"{s['blocks']}×{s['width']} · seed {s['seed']}" for s in specs}
        )
    receipts = json.loads((root / "final-review/prepared-dataset-backup-status.json").read_text())
    verified = {item["destination"]: item["sha256"] for item in receipts["verified_checkpoints"]}
    records = [json.loads(s) for s in (root / "confirmation.jsonl").read_text().splitlines()]
    ids = [
        next(i for i, row in enumerate(records) if row["dataset"] == source)
        for source in ("nyuv2", "arkitscenes", "structured3d", "coco2017")
    ]
    hashes, samples = {}, []
    for index in ids:
        arrays, reference = {}, None
        for item in selected:
            p = root / "confirmation" / item["name"] / f"geometry-{index:03d}.npz"
            assert str(p) in verified, f"Wait for verified serial geometry copy: {p}"
            hashes[str(p.relative_to(root))] = verified[str(p)]
            with np.load(p) as archive:
                values = {k: archive[k] for k in archive.files}
            if reference is None:
                reference = values
            else:
                for key in ("labels", "rgb", "teacher_points"):
                    np.testing.assert_array_equal(values[key], reference[key])
            arrays[item["name"]] = values["student_points"]
        arrays["teacher"] = reference["teacher_points"]
        labels, rgb = reference["labels"], reference["rgb"]
        clouds, planes = {}, {}
        for name, points in arrays.items():
            residual = np.zeros(labels.shape, dtype=np.float32)
            planes[name] = {}
            for label in range(1, int(labels.max()) + 1):
                mask = labels == label
                p = points[mask].astype(np.float64)
                score = None
                if np.isfinite(p).all() and (p[:, 2] > 0).all():
                    centered = p - p.mean(0)
                    values, vectors = np.linalg.eigh(centered.T @ centered / len(p))
                    extent = float(np.sqrt(np.maximum(values[1:], 0).sum()))
                    if extent > 1e-8:
                        distances = np.abs(centered @ vectors[:, 0]) / extent
                        residual[mask] = distances
                        score = float(np.sqrt(np.mean(distances**2)))
                if score is None:
                    residual[mask] = np.nan
                planes[name][str(label)] = dict(rms=score, pixels=int(mask.sum()))
            packed = np.concatenate((points, residual[..., None]), -1)[::4, ::4]
            clouds[name] = encode(packed.astype("<f4"))
        samples.append(
            dict(
                id=records[index]["screen_id"],
                title=("复测面板 · " if capacity else "确认集 · ") + records[index]["dataset"],
                scene=records[index]["group"],
                width=160,
                height=120,
                colors=encode(rgb[::4, ::4].astype(np.uint8)),
                labels=encode(labels[::4, ::4]),
                clouds=clouds,
                planes=planes,
            )
        )
    source = Path(__file__).with_name("build_plane_pilot_results.py").read_text()
    constants = [
        n.value
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and '<section id="geometry">' in n.value
        and "const samples=" in n.value
    ]
    assert len(constants) == 1
    fragment = constants[0][constants[0].index('<section id="geometry">') :]
    fragment = fragment.replace("真实输出 · 四路同步点云", "本轮实际输出 · 入围结构同步点云")
    fragment = fragment.replace(
        "默认显示你提供的会议室；也可切换两个预先选定的验证样本和一个训练样本。",
        "四个样本在推理前按来源和固定顺序选定，全部来自本轮预留确认集；每个 adapter 都是从零训练后的 1,000 步权重。",
    )
    fragment = fragment.replace("四幅图", "所有面板")
    fragment = fragment.replace(
        "names=['initial','control','plane','teacher']", "names=" + json.dumps(names)
    )
    fragment = fragment.replace(
        "['训练前','特征损失对照','加入平面约束','MoGe 教师']",
        json.dumps([labels_by_name[n] for n in names], ensure_ascii=False),
    )
    fragment = fragment.replace(
        "$('patch').value='1';",
        "$('patch').value=Object.keys(current.planes.teacher).length?Object.keys(current.planes.teacher)[0]:'0';",
    )
    fragment = fragment.replace(
        "Number(v).toFixed(6)", "(Number.isFinite(v)?v.toFixed(6):'无有效平面')"
    )
    fragment = fragment.replace(
        "function heat(v){", "function heat(v){if(!Number.isFinite(v))return '#ff00ff';"
    )
    panel_labels = {n: labels_by_name[n] for n in names}
    if capacity:
        panel_labels = {
            "baseline": "4×512 · 同 seed 基线",
            "candidate": "选中规模",
            "teacher": "MoGe 教师",
        }
        fragment = fragment.replace(
            "本轮实际输出 · 入围结构同步点云", "本轮实际输出 · 容量对照同步点云"
        )
        fragment = fragment.replace("全部来自本轮预留确认集", "全部来自之前看过的固定复测面板")
        fragment = fragment.replace(
            "names=" + json.dumps(names), "names=['baseline','candidate','teacher']"
        )
        fragment = fragment.replace(
            json.dumps([labels_by_name[n] for n in names], ensure_ascii=False),
            json.dumps(list(panel_labels.values()), ensure_ascii=False),
        )
        controls = '<label>初始化 seed <select id="geom-seed"><option>42</option><option>43</option></select></label><label>对比规模 <select id="geom-shape">'
        controls += (
            "".join(
                f'<option value="{d}x{w}">{d} × {w}</option>'
                for d in (4, 8)
                for w in (512, 768, 1024)
            )
            + "</select></label>"
        )
        fragment = fragment.replace(
            '<div class="controls"><label>样本', '<div class="controls">' + controls + "<label>样本"
        )
        old_load = "function load(){current=samples[+$('sample').value];clouds={};for(const n of names)clouds[n]=new Float32Array(bytes(current.clouds[n]).buffer);colors=bytes(current.colors);labels=bytes(current.labels);"
        new_load = """function load(){const raw=samples[+$('sample').value],seed=$('geom-seed').value,shape=$('geom-shape').value,wanted=['combined_4x512_s'+seed,'combined_'+shape+'_s'+seed,'teacher'];
 current={...raw,planes:Object.fromEntries(names.map((n,i)=>[n,raw.planes[wanted[i]]]))};clouds={};for(let i=0;i<names.length;i++)clouds[names[i]]=new Float32Array(bytes(raw.clouds[wanted[i]]).buffer);colors=bytes(raw.colors);labels=bytes(raw.labels);
 titles[0]='4×512 · seed '+seed;titles[1]=shape.replace('x','×')+' · seed '+seed;names.forEach((n,i)=>$('cloud-'+n).previousElementSibling.textContent=titles[i]);"""
        assert old_load in fragment
        fragment = fragment.replace(old_load, new_load)
        fragment = fragment.replace(
            "$('sample').onchange=load;",
            "$('geom-shape').value='4x1024';$('geom-seed').onchange=load;$('geom-shape').onchange=load;$('sample').onchange=load;",
        )
    panels = "".join(
        f'<div class="cloud-panel"><h3>{html.escape(label)}</h3><canvas id="cloud-{n}" width="720" height="440" aria-label="{html.escape(label)}交互点云"></canvas></div>'
        for n, label in panel_labels.items()
    )
    fragment = fragment.replace("PANELS", panels).replace(
        "DATA", json.dumps(samples, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    )
    output = root / "result-section.html"
    temp = output.with_suffix(".incoming")
    temp.write_text(fragment)
    temp.replace(output)
    proof = dict(
        stage="passed",
        samples=[s["id"] for s in samples],
        arms=names,
        visible_panels=list(panel_labels),
        raw_geometry_no_alignment=True,
        fixed_teacher_arrays=True,
        input_sha256_receipts=hashes,
        point_stride=4,
        normalization="own-plane RMS / in-plane extent",
        output_sha256=hashlib.sha256(fragment.encode()).hexdigest(),
    )
    output = root / "visualization-verification.json"
    temp = output.with_suffix(".incoming")
    temp.write_text(json.dumps(proof, indent=2))
    temp.replace(output)
    print(
        json.dumps(
            dict(
                stage="passed",
                samples=len(samples),
                panels=len(panel_labels),
                bytes=len(fragment.encode()),
            )
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--capacity", action="store_true")
    args = p.parse_args()
    build(args.root, args.capacity)
