"""Decoded, metric-depth comparisons of initial/trained adapters and MoGe-2.

All branches use the same frozen decoder and upstream focal/metric postprocessing.
Legacy runs without initial.pt can reconstruct initialization, but must first
reproduce the logged first optimization window's feature losses on training data.
"""

import argparse
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from accelerate.utils import set_seed
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from .checkpoint import atomic_json, atomic_save
from .data import check_split, evaluation_image, manifest_digest, read_manifest
from .decoder import postprocess
from .losses import FeatureStatistics, weighted_loss
from .metrics import depth_metrics
from .system import build_system
from .train import make_loader


def autocast(config, device):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.device(device).type == "cuda" and config["train"]["precision"] == "bf16"
        else nullcontext()
    )


def select_examples(records, count):
    """Pick first frame of distinct scene types, then distinct scenes, in manifest order.

    Selection is made before inference and never uses errors or visual quality.
    """
    selected, seen_types, seen_groups = [], set(), set()
    for i, row in enumerate(records):
        kind = row.get("scene_type", row.get("group", row["image"]))
        group = row.get("group", row["image"])
        if kind not in seen_types and group not in seen_groups:
            selected.append(i)
            seen_types.add(kind)
            seen_groups.add(group)
        if len(selected) == count:
            return selected
    for i, row in enumerate(records):
        group = row.get("group", row["image"])
        if group not in seen_groups:
            selected.append(i)
            seen_groups.add(group)
        if len(selected) == count:
            break
    return selected


def fingerprint(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def verify_initialization(system, adapter, state, train_log, device):
    """Compare all feature losses before the first optimizer update; never fit weights."""
    config = state["config"]
    if any(config["loss"].get(k, 0) for k in ("points", "mask", "scale", "gradient", "normal")):
        raise ValueError("Legacy reconstruction is supported only for feature-only training")
    records = read_manifest(config["data"]["train_manifest"])
    if manifest_digest(records) != state["data_digests"]["train"]:
        raise ValueError("Training manifest/files changed; cannot verify original initialization")
    expected = json.loads(Path(train_log).read_text().splitlines()[0])
    if expected["step"] != 1:
        raise ValueError("Verification requires a logged first optimizer step")
    stats = FeatureStatistics(adapter.config["target_dim"], adapter.config["target_summary_dim"])
    stats.load_state_dict(state["statistics"])
    stats.to(device)
    count = config["train"]["accumulation"]
    observed = {}
    for image in make_loader(records, config, 0, count):
        with autocast(config, device):
            features, target = system.encode(image.to(device), adapter.indices)
            predicted = adapter(features)
        losses = stats.loss(predicted, target)
        losses["total"] = weighted_loss(losses, config["loss"])
        for name, value in losses.items():
            observed[name] = observed.get(name, 0.0) + value.item() / count
    differences = {k: abs(v - expected[k]) for k, v in observed.items()}
    # Allow only small numeric differences, not an approximately similar random model.
    passed = all(np.isclose(v, expected[k], rtol=1e-5, atol=1e-6) for k, v in observed.items())
    report = dict(
        verified=bool(passed), method="seeded build_system, first training-window loss replay",
        seed=config["train"]["seed"], expected={k: expected[k] for k in observed},
        observed=observed, absolute_difference=differences, rtol=1e-5, atol=1e-6,
        adapter_sha256=fingerprint(adapter.state_dict()), torch=str(torch.__version__),
    )
    if not passed:
        raise ValueError(f"Initialization replay did not match; refusing 'before' label: {report}")
    return report


def color_depth(depth, lo, hi):
    """A shared metric range, never independent per-model normalization."""
    valid = np.isfinite(depth) & (depth > 0)
    scaled = np.nan_to_num((depth - lo) / max(hi - lo, 1e-6), nan=0, posinf=1, neginf=0)
    value = (scaled.clip(0, 1) * 255).astype(np.uint8)
    rgb = cv2.cvtColor(cv2.applyColorMap(value, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
    rgb[~valid] = [255, 0, 255]
    return rgb


def comparison_panel(image, depths, path, caption, step, width=480):
    teacher = depths["teacher"]
    valid = np.isfinite(teacher) & (teacher > 0)
    if not valid.any():
        raise ValueError("Teacher has no finite positive depths")
    lo, hi = np.quantile(teacher[valid], [0.02, 0.98]).tolist()
    hi = max(hi, lo + 1e-3)
    rgb = (image[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    h = round(rgb.shape[0] * width / rgb.shape[1])
    canvas = Image.new("RGB", (width * 4, h + 132), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=22)
    small = ImageFont.load_default(size=18)
    draw.text((12, 10), caption, fill="black", font=font)
    labels = ["RGB", "Before: initial adapter", f"After: {step} steps", "MoGe-2 teacher"]
    arrays = [rgb] + [color_depth(depths[key], lo, hi) for key in ("before", "after", "teacher")]
    for i, (label, array) in enumerate(zip(labels, arrays)):
        draw.text((i * width + 12, 44), label, fill="black", font=font)
        panel = Image.fromarray(array).resize((width, h), Image.Resampling.NEAREST)
        canvas.paste(panel, (i * width, 76))
    ramp = np.linspace(max(lo, 1e-8), hi, width - 120, dtype=np.float32)[None].repeat(14, 0)
    canvas.paste(Image.fromarray(color_depth(ramp, lo, hi)), (12, h + 94))
    draw.text((width - 100, h + 90), f"{lo:.2f}-{hi:.2f} m", font=small, fill="black")
    draw.text((width + 80, h + 90),
              "Same metric scale; dark = near, yellow = far; clipped at teacher 2-98% range. "
              "Magenta = invalid depth.", font=small, fill="black")
    canvas.save(path)
    return {"minimum_m": lo, "maximum_m": hi, "range_source": "teacher percentiles 2,98"}


@torch.no_grad()
def compare(system, before, after, records, config, output, step, count):
    device = next(after.parameters()).device
    before.eval()
    after.eval()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selected = select_examples(records, count)
    atomic_json({"selection": "first distinct scene types, then distinct scenes in manifest order",
                 "indices": selected, "records": [records[i] for i in selected]}, output / "selection.json")
    rows, pictures = [], []
    for index in selected:
        record = records[index]
        image, gt = evaluation_image(record, config["evaluation"]["max_side"])
        image = image.to(device)
        with autocast(config, device):
            features, target_features = system.encode(
                image, after.indices, num_tokens=config["evaluation"]["num_tokens"]
            )
            encodings = dict(before=before(features), after=after(features), teacher=target_features)
        row = {"index": index, "id": record.get("id", str(index)), "scene": record.get("scene")}
        arrays = {}
        for name, encoded in encodings.items():
            raw = system.decoder(encoded, image.shape[-2:])
            # Save raw point-z separately; never silently present it as final metric depth.
            arrays[f"{name}_raw_point_z"] = raw["points"][0, ..., 2].cpu().numpy()
            row[f"{name}/predicted_mask_fraction"] = (raw["mask"] > 0.5).float().mean().item()
            try:
                prediction = postprocess(raw, image, config["models"].get("moge_source"))
                depth = prediction["depth"]
                arrays[name] = depth[0].cpu().numpy()
                if gt is not None:
                    metrics = depth_metrics(depth, gt[None].to(device), gt[None].to(device) > 0)
                    row.update({f"{name}/gt_{k}": v for k, v in metrics.items()})
                row[f"{name}/postprocess_failed"] = False
            except (ValueError, np.linalg.LinAlgError) as error:
                if name == "teacher":
                    raise
                arrays[name] = np.full(image.shape[-2:], np.nan, dtype=np.float32)
                row[f"{name}/postprocess_failed"] = True
                row[f"{name}/postprocess_error"] = str(error)
        if gt is not None:
            arrays["ground_truth"] = gt.numpy()
        filename = f"{index:04d}.png"
        row["display_range"] = comparison_panel(
            image, arrays, output / filename, f"Held-out image {index}: {row['scene']}", step
        )
        row["visualization"] = filename
        row = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()}
        np.savez_compressed(output / f"{index:04d}-depths.npz", **arrays)
        rows.append(row)
        with Image.open(output / filename) as picture:
            pictures.append(picture.resize((1280, round(picture.height * 1280 / picture.width))))
        print(json.dumps(row, allow_nan=False), flush=True)
    sheet = Image.new("RGB", (1280, sum(p.height for p in pictures)), "white")
    top = 0
    for picture in pictures:
        sheet.paste(picture, (0, top))
        top += picture.height
    sheet.save(output / "overview.png")
    atomic_json({"images": rows, "note": "Diagnostic subset, not new full-test benchmark scores. "
                 "No scale alignment. Predicted validity masks do not hide depth values. "
                 "Invalid/nonpositive depth is magenta. All branches use the frozen MoGe decoder "
                 "and official focal recovery/metric-scale postprocessing."}, output / "results.json")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    baseline = parser.add_mutually_exclusive_group(required=True)
    baseline.add_argument("--initial", help="Actual stage-start initial.pt checkpoint")
    baseline.add_argument("--reconstruct-from-log", help="Legacy first-step train.jsonl verification")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = Path(args.output)
    if args.count < 1:
        parser.error("--count must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Comparison output must be new/empty: {output}")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    set_seed(config["train"]["seed"])
    system, before = build_system(config, args.device)
    before.eval()
    if args.initial:
        initial = torch.load(args.initial, map_location="cpu", weights_only=True)
        if (initial["step"] != 0 or initial["adapter_config"] != state["adapter_config"]
                or initial["config"]["models"] != config["models"]):
            raise ValueError("Initial checkpoint does not match this run's models/architecture")
        before.load_state_dict(initial["adapter"])
        provenance = {"method": "saved stage-start checkpoint", "checkpoint": args.initial,
                      "adapter_sha256": fingerprint(before.state_dict())}
    else:
        provenance = verify_initialization(system, before, state, args.reconstruct_from_log, args.device)
    after = deepcopy(before)
    after.load_state_dict(state["adapter"])
    provenance.update(trained_checkpoint=args.checkpoint, trained_step=state["step"],
                      revisions=state["revisions"], trained_sha256=fingerprint(after.state_dict()))
    atomic_json(provenance, output / "baseline-provenance.json")
    atomic_save({"adapter": before.state_dict(), "adapter_config": before.config,
                 "provenance": provenance}, output / "initial-adapter.pt")
    records = read_manifest(args.manifest)
    check_split(read_manifest(config["data"]["train_manifest"]), records)
    compare(system, before, after, records, config, output, state["step"], args.count)


if __name__ == "__main__":
    main()
