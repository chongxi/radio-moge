"""Deployment-path depth inference test.

One RADIO pass (``Perception``: a stand-in for the existing perception module)
produces the final backbone map, summary and the intermediate maps the head
needs. The exported ``GeometryHead`` (adapter + frozen MoGe-2 decoder) turns
them into metric depth, points, intrinsics, mask and normals. No MoGe encoder
is constructed.

Inputs are center-cropped and resized to the training bucket whose aspect
ratio is closest, exactly as in evaluation. Three input modes:

* ``--images``: image files or directories (no ground truth).
* ``--manifest``: JSONL rows with ``image`` and optional ``depth``/``depth_scale``.
* ``--reference``: a directory of saved ``*.npz`` (``rgb``, ``sensor_depth``,
  ``student_depth``, ``student_mask``) from an earlier training-path export;
  the same RGB is re-run and the depth difference is reported.
"""

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import evaluation_image, oriented_rgb, read_manifest
from .decoder import postprocess
from .export import GeometryHead
from .metrics import depth_metrics
from .upstream import extract_radio, load_radio, radio_input

DEFAULT_BUCKETS = [(480, 640), (640, 480), (400, 640)]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class Perception:
    """Stand-in for the perception module: ONE RADIO forward per image.

    A real integration replaces this with the existing RADIO call, as long as it
    returns normalized backbone features, all summary tokens and the requested
    intermediate maps from that same pass.
    """

    def __init__(self, models_config, device):
        self.radio = load_radio(models_config, device)

    def __call__(self, image, indices):
        return extract_radio(self.radio, radio_input(image, self.radio, None), indices)


def choose_bucket(width, height, buckets):
    aspect = math.log(width / height)
    return min(buckets, key=lambda hw: abs(math.log(hw[1] / hw[0]) - aspect))


def parse_bucket(text):
    h, w = (int(v) for v in text.lower().split("x"))
    return (h, w)


def image_rows(paths):
    files = []
    for path in map(Path, paths):
        if path.is_dir():
            files += sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        else:
            files.append(path)
    return [{"image": str(p)} for p in files]


def samples(args, buckets):
    """Yield (name, image[1,3,H,W] in 0..1, gt depth or None, reference or None, bucket)."""
    if args.reference:
        for path in sorted(Path(args.reference).glob("*.npz"))[: args.limit]:
            saved = np.load(path)
            rgb = saved["rgb"]
            image = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div_(255)
            reference = dict(depth=saved["student_depth"], mask=saved["student_mask"])
            yield path.stem, image, torch.from_numpy(saved["sensor_depth"]), reference, \
                tuple(rgb.shape[:2])
        return
    rows = read_manifest(args.manifest) if args.manifest else image_rows(args.images)
    for index, row in enumerate(rows[: args.limit]):
        bucket = args.bucket or choose_bucket(*oriented_rgb(row).size, buckets)
        image, gt = evaluation_image(row, bucket=bucket)
        yield f"{index:04d}_{Path(row['image']).stem}", image, gt, None, bucket


def turbo(x):
    """Google's polynomial approximation of the Turbo colormap (no matplotlib needed)."""
    r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (
        -152.94239396 + x * 59.28637943))))
    g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (
        4.27729857 + x * 2.82956604))))
    b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (
        -89.90310912 + x * 27.34824973))))
    return np.stack((r, g, b), -1)


INVALID_GRAY = 96


def colorize(depth, valid, lo, hi):
    """Log-scaled Turbo color in [lo, hi] (blue near, red far); invalid pixels gray."""
    x = np.clip((np.log(np.maximum(depth, 1e-6)) - math.log(lo)) / math.log(hi / lo), 0, 1)
    rgb = (np.clip(turbo(x), 0, 1) * 255).astype(np.uint8)
    rgb[~valid] = INVALID_GRAY
    return rgb


def ratio_error(predicted, gt, valid, limit=1.5):
    """Signed log(pred/gt): blue = too near, white = exact, red = too far (saturates at limit)."""
    x = np.clip(np.log(np.maximum(predicted, 1e-6) / np.where(valid, gt, 1)) / math.log(limit),
                -1, 1)[..., None]
    white, near, far = np.ones(3), np.array([0.15, 0.35, 1.0]), np.array([1.0, 0.2, 0.15])
    rgb = np.where(x < 0, white + (near - white) * -x, white + (far - white) * x)
    rgb = (rgb * 255).astype(np.uint8)
    rgb[~valid] = INVALID_GRAY
    return rgb


def visualization(rgb, depth, mask, gt, path):
    finite = mask & np.isfinite(depth) & (depth > 0)
    ref = gt[gt > 0] if gt is not None and (gt > 0).any() else depth[finite]
    lo, hi = np.quantile(ref, [0.02, 0.98]) if ref.size else (0.1, 10.0)
    lo, hi = max(float(lo), 1e-3), max(float(hi), float(lo) * 1.01)
    panels = [rgb, colorize(depth, finite, lo, hi)]
    if gt is not None:
        valid = gt > 0
        panels.append(colorize(gt, valid, lo, hi))
        panels.append(ratio_error(depth, gt, valid))
    Image.fromarray(np.concatenate(panels, axis=1)).save(path, quality=90)


def mean(rows, key):
    values = [r[key] for r in rows if key in r and math.isfinite(r[key])]
    return float(np.mean(values)) if values else None


@torch.inference_mode()
def run(args):
    device = torch.device(args.device)
    output = Path(args.output)
    (output / "depth").mkdir(parents=True, exist_ok=True)
    (output / "vis").mkdir(exist_ok=True)
    head = GeometryHead.load(args.head, device=device)
    info = head.artifact_info
    buckets = head.buckets or DEFAULT_BUCKETS
    perception = Perception(info["models"], device)
    use_bf16 = device.type == "cuda" and (head.precision or "bf16") == "bf16"
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 else nullcontext

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    rows = []
    with open(output / "results.jsonl", "w") as log:
        for name, image, gt, reference, bucket in samples(args, buckets):
            image = image.to(device)
            sync()
            t0 = time.perf_counter()
            with autocast():
                features = perception(image, head.indices)
            sync()
            t1 = time.perf_counter()
            with autocast():
                raw = head(features, image.shape[-2:])
            sync()
            t2 = time.perf_counter()
            geometry = postprocess(raw, image, head.moge_source)
            sync()
            t3 = time.perf_counter()
            depth = geometry["depth"][0].float().cpu().numpy()
            mask = geometry["mask"][0].cpu().numpy().astype(bool)
            intrinsics = geometry["intrinsics"][0].float().cpu().numpy()
            row = dict(name=name, height=image.shape[-2], width=image.shape[-1],
                       bucket_hw=list(bucket), grid_hw=list(features.spatial.shape[-2:]),
                       radio_ms=(t1 - t0) * 1e3, head_ms=(t2 - t1) * 1e3,
                       postprocess_ms=(t3 - t2) * 1e3, valid_fraction=float(mask.mean()),
                       median_depth_m=float(np.median(depth[mask])) if mask.any() else None,
                       fx_norm=float(intrinsics[0, 0]), fy_norm=float(intrinsics[1, 1]))
            gt_np = None
            if gt is not None:
                gt_np = gt.numpy()
                scores = depth_metrics(torch.from_numpy(depth), gt, gt > 0)
                row.update({f"gt/{k}": v for k, v in scores.items()})
            if reference is not None:
                both = mask & reference["mask"]
                diff = np.abs(depth - reference["depth"])[both]
                row["ref/max_abs_diff_m"] = float(diff.max()) if diff.size else None
                row["ref/mean_abs_diff_m"] = float(diff.mean()) if diff.size else None
                row["ref/mask_agreement"] = float((mask == reference["mask"]).mean())
                ref_scores = depth_metrics(torch.from_numpy(reference["depth"]), gt, gt > 0)
                row.update({f"ref_gt/{k}": v for k, v in ref_scores.items()})
            np.savez_compressed(output / "depth" / f"{name}.npz", depth=depth.astype(np.float32),
                                mask=mask, intrinsics=intrinsics)
            rgb = (image[0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
            visualization(rgb, depth, mask, gt_np, output / "vis" / f"{name}.jpg")
            log.write(json.dumps(row) + "\n")
            log.flush()
            rows.append(row)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in row.items()}), flush=True)

    keys = sorted({k for r in rows for k in r if isinstance(r[k], float)})
    timed = rows[1:] if len(rows) > 1 else rows  # first image includes CUDA warm-up
    summary = dict(
        images=len(rows),
        head=str(args.head),
        training_step=info.get("training_step"),
        indices=list(head.indices),
        buckets=[list(b) for b in buckets],
        bf16=use_bf16,
        device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        mean={k: mean(rows, k) for k in keys if not k.endswith("_ms")},
        mean_ms_excluding_first={k: mean(timed, k) for k in keys if k.endswith("_ms")},
    )
    if any("ref/max_abs_diff_m" in r for r in rows):
        summary["ref_worst_max_abs_diff_m"] = max(r["ref/max_abs_diff_m"] for r in rows
                                                  if r.get("ref/max_abs_diff_m") is not None)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--head", required=True, help="geometry-head.pt from radio_moge.export")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", nargs="+", help="image files or directories")
    source.add_argument("--manifest", help="JSONL with image (and optional depth) paths")
    source.add_argument("--reference", help="directory of earlier *.npz exports to reproduce")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bucket", type=parse_bucket, default=None,
                        help="force HxW (height x width), e.g. 400x640")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
