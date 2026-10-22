"""Compare independent architectures on the same twenty exported test scenes."""

import argparse
import base64
import gzip
import hashlib
import html
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import torch

from build_updated_twenty import planes, read_export
from radio_moge.checkpoint import atomic_json
from radio_moge.config import ROOT
from radio_moge.metrics import depth_metrics


def build(args):
    torch.set_num_threads(2)
    if args.output.exists():
        raise FileExistsError(args.output)
    baseline, a_samples = read_export(args.baseline)
    candidate, b_samples = read_export(args.candidate)
    fixed = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    ids = [row["id"] for row in fixed]
    assert len(ids) == len(set(ids)) == 20
    assert ids == list(a_samples) == list(b_samples)
    labels = [args.baseline_label, args.candidate_label, "MoGe-v2 教师"]
    samples = []
    for sid in ids:
        a, ga = a_samples[sid]
        b, gb = b_samples[sid]
        assert not a["failures"] and not b["failures"]
        assert (a["width"], a["height"]) == (b["width"], b["height"])
        left, right = planes(args.baseline, a), planes(args.candidate, b)
        np.testing.assert_allclose(left[2], right[2], rtol=1e-5, atol=1e-5, equal_nan=True)
        np.testing.assert_array_equal(left[3], right[3])
        rgb_a = np.asarray(Image.open(args.baseline / a["rgb"]))
        rgb_b = np.asarray(Image.open(args.candidate / b["rgb"]))
        np.testing.assert_array_equal(rgb_a, rgb_b)
        assert rgb_b.shape == (b["height"], b["width"], 3)
        np.testing.assert_allclose(
            ga["teacher"]["intrinsics"], gb["teacher"]["intrinsics"], atol=1e-6
        )
        depths = [left[1], right[1], right[2]]
        cameras = [ga["after"], gb["after"], gb["teacher"]]
        masks = [
            np.asarray(Image.open(directory / camera["mask"])) > 0
            for directory, camera in zip([args.baseline, args.candidate, args.candidate], cameras)
        ]
        np.testing.assert_array_equal(
            masks[2], np.asarray(Image.open(args.baseline / ga["teacher"]["mask"])) > 0
        )
        for camera in cameras:
            k = np.asarray(camera["intrinsics"])
            assert k.shape == (3, 3) and np.isfinite(k).all() and k[0, 0] > 0 and k[1, 1] > 0
            assert camera["projection_max_error_m"] < 1e-4
        gt = torch.from_numpy(right[3])
        metrics = {}
        for name, depth, exported in zip(
            ("old", "new", "teacher"),
            depths,
            [a["metrics"]["after"], b["metrics"]["after"], b["metrics"]["teacher"]],
        ):
            tensor = torch.from_numpy(depth)
            metrics[name] = {
                "sensor": depth_metrics(tensor, gt, gt > 0),
                "teacher": depth_metrics(
                    tensor, torch.from_numpy(right[2]), torch.from_numpy(masks[2])
                ),
            }
            for key in ("abs_rel", "rmse", "delta1"):
                np.testing.assert_allclose(
                    metrics[name]["sensor"][key], exported[key], rtol=1e-5, atol=1e-7
                )
        stride = 4
        blob = b"".join(np.asarray(d[::stride, ::stride], dtype="<f4").tobytes() for d in depths)
        blob += b"".join(np.asarray(m[::stride, ::stride], dtype=np.uint8).tobytes() for m in masks)
        samples.append(
            dict(
                id=sid,
                scene=b["scene"],
                scene_type=b["scene_type"],
                width=b["width"],
                height=b["height"],
                stride=stride,
                range=b["range"],
                metrics=metrics,
                rgb="data:image/jpeg;base64,"
                + base64.b64encode((args.candidate / b["rgb"]).read_bytes()).decode(),
                cameras=[c["intrinsics"] for c in cameras],
                projection_error_m=max(c["projection_max_error_m"] for c in cameras),
                payload=base64.b64encode(gzip.compress(blob, compresslevel=6, mtime=0)).decode(),
            )
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
    report = dict(
        status="complete",
        labels=labels,
        samples=samples,
        summary=summary,
        models={"baseline": baseline["run"], "candidate": candidate["run"]},
        initialization_policy="Independent architecture initializations retained separately; no equality assumed.",
        verification="Exact fixed20 order; full-resolution teacher, sensor, RGB and teacher-camera parity; own-camera projection verified; all displayed scores reproduced from full-resolution depth.",
        limits="Previously seen qualitative audit, not a fresh blind test. Models selected by separate884 validation. Unequal training budgets; not a matched-step architecture effect estimate.",
        inputs={k: str(getattr(args, k)) for k in ("baseline", "candidate", "manifest")},
        source_manifest_sha256={
            name: hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
            for name, directory in [("baseline", args.baseline), ("candidate", args.candidate)]
        },
    )
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
    page = (ROOT / "web/architecture-twenty.html").read_text()
    for index, label in enumerate(labels):
        page = page.replace(f"__LABEL{index}__", html.escape(label))
    page = page.replace("__DATA__", json.dumps(report, separators=(",", ":")).replace("</", "<\\/"))
    page = page.replace("__BUNDLE__", bundle.read_text().replace("</script", "<\\/script"))
    tmp = args.output / "index.html.tmp"
    tmp.write_text(page)
    tmp.replace(args.output / "index.html")
    print(json.dumps({"output": str(args.output), "summary": summary}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    build(parser.parse_args())
