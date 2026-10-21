"""Update the fixed twenty-scene comparison from verified, already exported predictions."""

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import torch

from radio_moge.checkpoint import atomic_json
from radio_moge.config import ROOT
from radio_moge.metrics import depth_metrics


def read_export(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    geometry = json.loads((directory / "geometry/manifest.json").read_text())
    assert (
        geometry["depth_manifest_sha256"]
        == hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
    )
    assert geometry["checkpoint_sha256"] == manifest["run"]["checkpoint_sha256"]
    assert [s["id"] for s in manifest["samples"]] == [s["id"] for s in geometry["samples"]]
    return manifest, {s["id"]: (s, g) for s, g in zip(manifest["samples"], geometry["samples"])}


def planes(directory, sample):
    return (
        np.frombuffer(gzip.decompress((directory / sample["depth"]).read_bytes()), "<f4")
        .reshape(4, sample["height"], sample["width"])
        .copy()
    )


def build(args):
    torch.set_num_threads(2)
    if args.output.exists():
        raise FileExistsError(args.output)
    old, old_samples = read_export(args.baseline)
    current, current_samples = read_export(args.candidate)
    previous = json.loads(args.previous_data.read_text())
    ids = [s["id"] for s in previous["samples"]]
    assert len(ids) == len(set(ids)) == 20
    assert ids == [s["id"] for s in current["samples"]]
    assert previous["adapter_sha256"] == current["run"]["initial_adapter_sha256"]
    samples = []
    for prior in previous["samples"]:
        sid = prior["id"]
        a, ga = old_samples[sid]
        b, gb = current_samples[sid]
        assert (a["width"], a["height"]) == (b["width"], b["height"])
        left, right = planes(args.baseline, a), planes(args.candidate, b)
        np.testing.assert_allclose(left[2], right[2], rtol=1e-5, atol=1e-5, equal_nan=True)
        np.testing.assert_array_equal(left[3], right[3])
        stride = prior["stride"]
        count = right[0, ::stride, ::stride].size
        prior_planes = np.frombuffer(
            gzip.decompress(base64.b64decode(prior["payload"])), "<f4", count=3 * count
        ).reshape(3, -1)
        np.testing.assert_allclose(
            right[0, ::stride, ::stride].ravel(),
            prior_planes[1],
            rtol=1e-5,
            atol=1e-5,
            equal_nan=True,
        )
        depths = [left[1], right[1], right[2]]
        masks = [
            np.asarray(Image.open(args.baseline / ga["after"]["mask"])) > 0,
            np.asarray(Image.open(args.candidate / gb["after"]["mask"])) > 0,
            np.asarray(Image.open(args.candidate / gb["teacher"]["mask"])) > 0,
        ]
        np.testing.assert_array_equal(
            masks[2], np.asarray(Image.open(args.baseline / ga["teacher"]["mask"])) > 0
        )
        gt, teacher = torch.from_numpy(right[3]), torch.from_numpy(right[2])
        metrics = {}
        for name, depth in zip(("old", "new", "teacher"), depths):
            tensor = torch.from_numpy(depth)
            metrics[name] = {
                "sensor": depth_metrics(tensor, gt, gt > 0),
                "teacher": depth_metrics(tensor, teacher, torch.from_numpy(masks[2])),
            }
        for key in ("abs_rel", "rmse", "delta1"):
            np.testing.assert_allclose(
                metrics["new"]["sensor"][key], b["metrics"]["after"][key], rtol=1e-5, atol=1e-7
            )
            np.testing.assert_allclose(
                metrics["old"]["sensor"][key],
                prior["metrics"]["old"]["sensor"][key],
                rtol=1e-5,
                atol=1e-7,
            )
        blob = b"".join(np.asarray(d[::stride, ::stride], dtype="<f4").tobytes() for d in depths)
        blob += b"".join(np.asarray(m[::stride, ::stride], dtype=np.uint8).tobytes() for m in masks)
        samples.append(
            {
                **prior,
                "metrics": metrics,
                "cameras": [
                    ga["after"]["intrinsics"],
                    gb["after"]["intrinsics"],
                    gb["teacher"]["intrinsics"],
                ],
                "payload": base64.b64encode(gzip.compress(blob, compresslevel=6, mtime=0)).decode(),
                "projection_error_m": gb["after"]["projection_max_error_m"],
            }
        )
    summary = {
        name: {
            ref: {
                metric: float(np.mean([s["metrics"][name][ref][metric] for s in samples]))
                for metric in ("abs_rel", "rmse", "delta1")
            }
            for ref in ("sensor", "teacher")
        }
        for name in ("old", "new", "teacher")
    }
    report = {
        "status": "complete",
        "samples": samples,
        "summary": summary,
        "checkpoint_step": current["run"]["steps"],
        "adapter_sha256": current["run"]["checkpoint_sha256"],
        "initial_adapter_sha256": current["run"]["initial_adapter_sha256"],
        "continuation_objective": args.continuation_objective,
        "geometry_initial_sensor": current["summary"]["before"],
        "verification": "Exact fixed20 order; full-resolution teacher and sensor parity with old export; stage-start depth matches prior feature-best snapshot; sensor scores reproduced.",
        "limits": "Previously inspected20 NYU scenes, not a fresh blind test. No selection on these20. Own intrinsics and metric depth, no alignment.",
        "inputs": {
            "baseline": str(args.baseline),
            "candidate": str(args.candidate),
            "previous_data": str(args.previous_data),
        },
    }
    args.output.mkdir(parents=True)
    atomic_json(report, args.output / "data.json")
    bundle = args.output / "viewer.bundle.js"
    subprocess.run(
        [
            "bun",
            "build",
            str(ROOT / "web/heldout-twenty.js"),
            "--target=browser",
            "--minify",
            "--outfile=" + str(bundle),
        ],
        check=True,
    )
    html = (ROOT / "web/heldout-twenty.html").read_text()
    old_caption = (
        "新模型：特征阶段最佳第 29,000 步。本次展示采用几何第 3,000 步验证时保留的最佳权重。"
    )
    assert old_caption in html
    if args.continuation_objective == "features-only":
        caption = (
            "新模型：30,000 步特征训练后，从最佳权重继续 10,000 步仅特征训练；"
            f"验证集选中续训第 {current['run']['steps']:,} 步。"
        )
    else:
        caption = (
            "新模型：30,000 步特征训练 + 10,000 步几何训练完成；"
            f"验证集选中几何第 {current['run']['steps']:,} 步。"
        )
    html = html.replace(old_caption, caption)
    html = html.replace(
        "模型在查看这批结果之前已固定。",
        "模型由独立验证集选择。这 20 张图此前已经查看过，是固定可视化审计，非全新盲测。",
    )
    html = html.replace("__DATA__", json.dumps(report, separators=(",", ":")).replace("</", "<\\/"))
    html = html.replace("__BUNDLE__", bundle.read_text().replace("</script", "<\\/script"))
    temporary = args.output / "index.html.tmp"
    temporary.write_text(html)
    temporary.replace(args.output / "index.html")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "summary": summary,
                "geometry_initial_sensor": report["geometry_initial_sensor"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "previous-data", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument(
        "--continuation-objective",
        choices=["features-only", "features-and-geometry"],
        default="features-and-geometry",
    )
    build(parser.parse_args())
