"""Export the validation-selected spatial cosine checkpoint and MoGe-2 on fixed scenes."""

import argparse
import base64
import gc
import gzip
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from radio_moge.checkpoint import atomic_json
from radio_moge.data import evaluation_image, read_manifest
from radio_moge.decoder import postprocess
from radio_moge.metrics import depth_metrics
from radio_moge.pointcloud import verify_geometry
from radio_moge.system import build_system


def check_membership(config, records):
    """Stream metadata; do not stat every training image or load the whole dataset."""
    keys = ("id", "image", "group", "sha256")
    selected = {key: {r[key] for r in records if r.get(key)} for key in keys}
    checks = []
    for role in ("train_manifest", "val_manifest"):
        path = Path(config["data"][role])
        overlap = {key: 0 for key in keys}
        count = 0
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                count += 1
                # abspath is intentional: avoid hundreds of thousands of NFS resolves.
                import os

                row["image"] = os.path.abspath(path.parent / row["image"])
                for key in keys:
                    overlap[key] += int(bool(row.get(key)) and row[key] in selected[key])
        if any(overlap.values()):
            raise ValueError(f"Selected scenes overlap {role}: {overlap}")
        checks.append(dict(role=role, records=count, overlap=overlap))
    return checks


def native_center_crop(image, depth, size):
    """Keep original pixels and matching sensor rays; size is width, height."""
    width, height = size
    h, w = image.shape[-2:]
    if min(width, height) <= 0 or width % 16 or height % 16:
        raise ValueError("Native size must use positive multiples of 16")
    if width > w or height > h:
        raise ValueError("Native comparison requires a crop, never upsampling or stretching")
    if depth is None or tuple(depth.shape) != (h, w):
        raise ValueError("A matching sensor depth map is required")
    left, top = (w - width) // 2, (h - height) // 2
    box = [left, top, left + width, top + height]
    return (image[..., top:top + height, left:left + width].contiguous(),
            depth[top:top + height, left:left + width].contiguous(),
            dict(source_wh=[w, h], crop_xyxy=box, output_wh=[width, height],
                 interpolation="none", policy="center crop RGB and sensor depth identically"))


@torch.inference_mode()
def export(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "data.json").exists():
        raise FileExistsError("Export already exists; do not rerun inference")
    ids = json.loads(args.selection.read_text())["ids"]
    indexed = {r["id"]: r for r in read_manifest(args.manifest)}
    records = [indexed[sid] for sid in ids]
    assert len(records) == len(set(ids)) == 20
    assert len({r["group"] for r in records}) == 20
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    assert state["step"] == 91367
    config = state["config"]
    assert config["adapter"]["spec"]["summary_cleaning"] == "per-token-zero-channel"
    assert config["evaluation"]["num_tokens"] == 1200
    native_size = args.native_size
    evaluation = dict(config["evaluation"])
    if native_size:
        evaluation.update(num_tokens=None, native_size_wh=list(native_size),
                          spatial_tokens=(native_size[0] // 16) * (native_size[1] // 16))
    report = dict(
        status="initializing", started_at=time.time(), checkpoint_step=state["step"],
        checkpoint=str(args.checkpoint), selection_metric="macro/spatial/cosine",
        selection_value=0.9599220078619738,
        labels=["Cosine 最佳 · 第 91,367 步", "MoGe-2 · 原始模型"],
        names=["student", "teacher"], selected_ids=ids,
        model_config=config["models"], evaluation=evaluation,
        training_evaluation=config["evaluation"], backend="pytorch",
        revisions=state["revisions"], samples=[],
        selection_policy="Previously inspected fixed20 NYUv2 scenes; selected before inference, no selection on these outputs.",
        coordinates="OpenCV x right, y down, z forward, meters; browser flips y and z for Three.js.",
        projection="Own recovered intrinsics and masks; no scale alignment or registration.",
    )
    if native_size:
        width, height = native_size
        report["view"] = dict(
            title=f"{width}×{height} · Our adapter × MoGe-2",
            subtitle="同一中心裁剪 · 原生 RADIO 网格 · 同步三维视角",
            badge_label="选权重时的验证 spatial cosine",
            protocol=(f"同一组 20 张 640×480 图像中心裁剪为 {width}×{height}；RGB 和传感器深度同步裁剪，"
                      f"无拉伸或补边。RADIO 实际输入 {width}×{height}，特征网格 "
                      f"{width // 16}×{height // 16}（{evaluation['spatial_tokens']} 个空间 token）；"
                      "MoGe-2 使用相同 token 网格及其原始 DINO 预处理，解码输出均为裁剪尺寸。"
                      "两侧均为 PyTorch 推理。默认各自恢复内参与有效掩码，不做尺度对齐或点云配准。"
                      "共同相机模式仅改变显示投影，深度色标对两侧相同。"),
            caveat="本页指标仅对应这 20 张中心裁剪图。右上角 cosine 来自原验证协议，并非本次 640×400 分数；不能与旧整图分数直接归因比较。",
        )
    atomic_json({k: v for k, v in report.items() if k != "samples"}, args.output / "status.json")
    report["split_checks"] = check_membership(config, records)
    print("Fixed20 scene membership checks passed; loading frozen models", flush=True)
    system, adapter = build_system(config, "cuda")
    adapter.load_state_dict(state["adapter"], strict=True)
    adapter.eval().requires_grad_(False)
    del state
    gc.collect()
    for i, record in enumerate(records):
        image, gt = evaluation_image(record, config["evaluation"]["max_side"])
        preprocessing = None
        if native_size:
            image, gt, preprocessing = native_center_crop(image, gt, native_size)
        rgb = (image[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        image = image.cuda()
        h, w = image.shape[-2:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features, target = system.encode(image, adapter.indices, num_tokens=evaluation["num_tokens"])
            encoded = adapter(features)
        grid = list(features.spatial.shape[-2:])
        assert list(target.spatial.shape[-2:]) == list(encoded.spatial.shape[-2:]) == grid
        if native_size:
            assert (w, h) == tuple(native_size) and grid == [h // 16, w // 16]
            assert all(list(t.shape[-2:]) == grid for t in features.intermediates)
        predictions = [
            postprocess(system.decoder(x, (h, w)), image, config["models"].get("moge_source"))
            for x in (encoded, target)
        ]
        depths = [p["depth"][0].cpu().numpy() for p in predictions]
        geometry = [verify_geometry(depth, pred) for depth, pred in zip(depths, predictions)]
        masks = [g[1] for g in geometry]
        for depth, mask in zip(depths, masks):
            assert mask.any() and np.isfinite(depth[mask]).all() and (depth[mask] > 0).all()
        metrics = {}
        for name, depth in zip(report["names"], depths):
            tensor = torch.from_numpy(depth)
            metrics[name] = dict(
                sensor=depth_metrics(tensor, gt, gt > 0),
                teacher=depth_metrics(tensor, torch.from_numpy(depths[1]), torch.from_numpy(masks[1])),
            )
        assert all(np.isfinite(value) for branch in metrics.values()
                   for ref in branch.values() for value in ref.values())
        stride = 2
        blob = b"".join(np.asarray(d[::stride, ::stride], dtype="<f4").tobytes() for d in depths)
        blob += b"".join(np.asarray(m[::stride, ::stride], dtype=np.uint8).tobytes() for m in masks)
        color = io.BytesIO()
        Image.fromarray(rgb).save(color, format="PNG")
        valid_depths = np.concatenate([d[m] for d, m in zip(depths, masks)])
        sample = dict(
            id=record["id"], scene=record["scene"], scene_type=record.get("scene_type", "indoor"),
            width=w, height=h, stride=stride,
            range=np.quantile(valid_depths, [.02, .98]).tolist(),
            cameras=[g[0] for g in geometry], full_valid_points=[int(m.sum()) for m in masks],
            projection_error_m=max(g[2] for g in geometry), metrics=metrics,
            preprocessing=preprocessing, feature_grid_hw=grid,
            rgb="data:image/png;base64," + base64.b64encode(color.getvalue()).decode(),
            payload=base64.b64encode(gzip.compress(blob, compresslevel=6, mtime=0)).decode(),
        )
        # Full-resolution arrays are retained separately; only display sampling is reduced.
        np.savez_compressed(args.output / f"{record['id']}.npz", rgb=rgb,
                            student_depth=depths[0], teacher_depth=depths[1],
                            student_mask=masks[0], teacher_mask=masks[1],
                            cameras=np.asarray(sample["cameras"]), sensor_depth=gt.numpy())
        report["samples"].append(sample)
        atomic_json(dict(status="exporting", completed=i + 1, total=len(records),
                         sample_id=record["id"], updated_at=time.time()), args.output / "status.json")
        print(f"Exported {i + 1}/20: {record['id']}", flush=True)
        del features, target, encoded, predictions
    report["summary"] = {
        name: {ref: {metric: float(np.mean([s["metrics"][name][ref][metric] for s in report["samples"]]))
                     for metric in ("abs_rel", "rmse", "delta1")}
               for ref in ("sensor", "teacher")}
        for name in report["names"]
    }
    report.update(status="complete", finished_at=time.time())
    atomic_json(report, args.output / "data.json")
    atomic_json(dict(status="complete", completed=20, total=20, finished_at=report["finished_at"],
                     checkpoint_step=91367, max_projection_error_m=max(s["projection_error_m"] for s in report["samples"]),
                     split_checks=report["split_checks"], summary=report["summary"], evaluation=evaluation,
                     backend="pytorch", feature_grids_hw=sorted({tuple(s["feature_grid_hw"]) for s in report["samples"]})),
                args.output / "status.json")
    print(json.dumps(dict(status="complete", images=20, summary=report["summary"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "manifest", "selection", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--native-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
                        help="Center-crop original pixels and disable fixed-token resizing")
    export(parser.parse_args())
