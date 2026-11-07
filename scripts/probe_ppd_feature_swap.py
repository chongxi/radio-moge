"""Frozen, paired transfer probe: RADIO adapter spatial features -> official PPD.

This is a small internal validation experiment, not the published NYU benchmark.
All methods see the same crop; PPD branches share an identical noise tensor.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from radio_moge.adapters import EncoderFeatures, build_adapter
from radio_moge.data import evaluation_image, read_manifest
from radio_moge.decoder import MoGeDecoder, postprocess
from radio_moge.upstream import extract_radio, load_moge, load_radio, radio_input


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publish(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".new")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def select_scenes(rows, count, dataset, empty_only=False):
    selected, groups = [], set()
    for row in sorted(rows, key=lambda x: x["sha256"]):
        if row["dataset"] != dataset or row["group"] in groups:
            continue
        if empty_only and row.get("configuration") != "empty":
            continue
        if dataset == "nyuv2" and "depth" not in row:
            continue
        selected.append(row)
        groups.add(row["group"])
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} distinct {dataset} scenes")
    return selected


def scalar_metrics(prediction, target, mask):
    p, g = prediction[mask].astype(np.float64), target[mask].astype(np.float64)
    ratio = np.maximum(p / g, g / p)
    lp, lg = np.log1p(prediction), np.log1p(target)
    gradient_errors = []
    for axis in (0, 1):
        a, b = [slice(None)] * 2, [slice(None)] * 2
        a[axis], b[axis] = slice(1, None), slice(None, -1)
        a, b = tuple(a), tuple(b)
        valid = mask[a] & mask[b]
        gradient_errors.append(np.abs((lp[a] - lp[b]) - (lg[a] - lg[b]))[valid])
    return {
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
        "log1p_gradient_mae": float(np.concatenate(gradient_errors).mean()),
        "pixels": int(mask.sum()),
    }


@torch.inference_mode()
def sample(dit, image, semantics, initial_noise, timesteps, sampler):
    latent = initial_noise.clone()
    conditioning = image - 0.5
    for timestep in timesteps:
        velocity = dit(
            x=torch.cat((latent, conditioning), dim=1),
            semantics=semantics,
            timestep=timestep,
        )
        latent = sampler.step(pred=velocity, x_t=latent, t=timestep)
    return latent + 0.5


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--ppd-source", type=Path, required=True)
    parser.add_argument("--ppd-checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if (args.root / "plan.json").exists():
        raise FileExistsError("Use a new output directory; this probe never resumes or overwrites")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    sys.path.insert(0, str(args.ppd_source.resolve()))
    from ppd.models.dit import DiT
    from ppd.moge.model.modules import DINOv2Encoder
    from ppd.utils.diffusion.schedule import LinearSchedule
    from ppd.utils.diffusion.timesteps import Timesteps
    from ppd.utils.diffusion.sampler import EulerSampler
    from ppd.utils.align_depth_func import recover_metric_depth_ransac

    rows = read_manifest(args.manifest)
    selected = select_scenes(rows, 12, "nyuv2") + select_scenes(rows, 4, "structured3d", True)
    # Freeze identities before any model inference or inspection of results.
    selected_refs = []
    for index, row in enumerate(selected):
        actual_hash = digest(row["image"])
        if actual_hash != row["sha256"]:
            raise ValueError(f"Image bytes changed: {row['image']}")
        selected_refs.append({"index": index, "row": row, "image_sha256": actual_hash,
                              "depth_sha256": digest(row["depth"]) if "depth" in row else None})
    plan = {
        "purpose": "Frozen PPD MoGe2 spatial feature substitution; no training",
        "checkpoint": {"path": str(args.checkpoint), "sha256": digest(args.checkpoint)},
        "ppd_checkpoint": {"path": str(args.ppd_checkpoint), "sha256": digest(args.ppd_checkpoint)},
        "ppd_commit": subprocess.check_output(["git", "-C", str(args.ppd_source), "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": digest(__file__),
        "manifest": {"path": str(args.manifest), "sha256": digest(args.manifest)},
        "selection": "First 12 NYU and 4 empty Structured3D distinct groups ordered by encoded-image SHA256",
        "selected": selected_refs,
        "resolutions_hw": [[480, 640], [768, 1024]],
        "sampling_steps": args.steps, "seed": args.seed,
        "preprocess": "Identical centered 4:3 crop, PIL bilinear RGB resize for all branches; no augmentation",
        "noise": "CPU normal generator seeded seed+image_index, one shared tensor for both PPD branches",
        "precision": "FP32 parameters with BF16 autocast; MoGe neck/heads FP32",
        "metric": "NYU valid 0.1<depth<10m, all predictions resized to 480x640 before evaluation",
        "alignment": "Official PPD per-image RANSAC affine prediction -> log1p(GT); MoGe input log1p(metric depth). RNG reset equally per branch.",
        "metric_limit": "GT-assisted relative-structure evaluation, not metric-depth accuracy or the published benchmark protocol",
        "synthetic_visual_alignment": "Same RANSAC procedure using native MoGe depth as reference; no ground-truth claim",
        "no_optimizer_or_training": True,
    }
    publish(args.root / "plan.json", plan)
    print(json.dumps({"event": "plan", "images": len(selected), "resolutions": plan["resolutions_hw"]}), flush=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if checkpoint["step"] != 272646:
        raise ValueError("Expected the fixed final Muon checkpoint")
    models = dict(checkpoint["config"]["models"], moge_source=str(args.vendor / "MoGe"), radio_source=str(args.vendor / "RADIO"))
    moge, model_config = load_moge(models, device)
    decoder = MoGeDecoder(moge)
    # Use PPD's actual encoder implementation for its native semantic baseline.
    ppd_encoder = DINOv2Encoder(**model_config["encoder"])
    ppd_encoder.load_state_dict(moge.encoder.state_dict(), strict=True)
    ppd_encoder.enable_pytorch_native_sdpa()
    ppd_encoder.eval().requires_grad_(False).to(device)
    radio = load_radio(models, device)
    adapter = build_adapter(**checkpoint["adapter_config"])
    adapter.load_state_dict(checkpoint["adapter"], strict=True)
    adapter.eval().requires_grad_(False).to(device)
    del checkpoint
    gc.collect()
    state = torch.load(args.ppd_checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not all(key.startswith("dit.") for key in state):
        raise ValueError("Unexpected released PPD checkpoint contents")
    # No random CPU allocation of the 500M-parameter diffusion model.
    with torch.device("meta"):
        dit = DiT()
    dit.load_state_dict({key[4:]: value for key, value in state.items()}, strict=True, assign=True)
    dit.eval().requires_grad_(False).to(device)
    del state
    gc.collect()
    timesteps = Timesteps(T=1000, steps=args.steps, device=device)
    sampler = EulerSampler(LinearSchedule(T=1000), timesteps, prediction_type="velocity")
    publish(args.root / "loaded.json", {
        "step": 272646, "dit_parameters": sum(x.numel() for x in dit.parameters()),
        "all_state_dicts_strict": True, "adapter_config": adapter.config,
        "moge_model_config": model_config, "timesteps": timesteps.timesteps.cpu().tolist(),
        "torch": torch.__version__, "device": torch.cuda.get_device_name(),
    })
    print(json.dumps({"event": "loaded", "allocated_gib": torch.cuda.memory_allocated() / 2**30}), flush=True)

    results, parity = [], []
    for height, width in plan["resolutions_hw"]:
        for index, row in enumerate(selected):
            start = time.monotonic()
            name = f"{index:02d}-{row['dataset']}-{height}x{width}"
            image, _ = evaluation_image(row, bucket=(height, width))
            _, target_tensor = evaluation_image(row, bucket=(480, 640))
            image = image.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                spatial, summary = ppd_encoder(image, height // 16, width // 16, return_class_token=True)
                native = EncoderFeatures(spatial.float(), summary.float())
                if index == 0:
                    reference, reference_summary = moge.encoder(image, height // 16, width // 16, return_class_token=True)
                    agreement = F.cosine_similarity(reference.float(), spatial.float(), dim=1)
                    entry = {"height": height, "width": width,
                             "spatial_cosine": float(agreement.mean()),
                             "spatial_max_abs": float((reference.float() - spatial.float()).abs().max()),
                             "summary_max_abs": float((reference_summary.float() - summary.float()).abs().max()),
                             "exact": torch.equal(reference, spatial) and torch.equal(reference_summary, summary)}
                    parity.append(entry)
                    if entry["spatial_cosine"] < 0.9999:
                        raise ValueError(f"Teacher boundary mismatch: {entry}")
                features = extract_radio(radio, radio_input(image, radio), adapter.indices)
                fitted = adapter(features)
                if fitted.spatial.shape != native.spatial.shape:
                    raise ValueError("Spatial grids must match without feature resizing")
                similarity = F.cosine_similarity(fitted.spatial.float(), native.spatial.float(), dim=1)
                feature_metrics = {"mean_cosine": float(similarity.mean()), "p10_cosine": float(torch.quantile(similarity.flatten(), 0.1)),
                                   "raw_relative_rmse": float((fitted.spatial.float() - native.spatial).square().mean().sqrt() / native.spatial.square().mean().sqrt())}
                generator = torch.Generator(device="cpu").manual_seed(args.seed + index)
                noise = torch.randn((1, 1, height, width), generator=generator).to(device)
                outputs = {}
                for key, encoded in (("ppd_native", native), ("ppd_adapter", fitted)):
                    semantics = encoded.spatial.flatten(2).transpose(1, 2)
                    outputs[key] = sample(dit, image, semantics, noise, timesteps, sampler)[0, 0].float().cpu().numpy()
                for key, encoded in (("moge_native", native), ("moge_adapter", fitted)):
                    raw = decoder(encoded, (height, width))
                    depth = postprocess(raw, image, models["moge_source"])["depth"][0]
                    outputs[key] = depth.float().cpu().numpy()
            if not all(np.isfinite(x).all() for x in outputs.values()):
                raise ValueError(f"Non-finite model output: {name}")
            output_small = {k: cv2.resize(v, (640, 480), interpolation=cv2.INTER_LINEAR) for k, v in outputs.items()}
            target = target_tensor.numpy() if target_tensor is not None else output_small["moge_native"].copy()
            mask = np.isfinite(target) & (target > 0.1)
            if target_tensor is not None:
                mask &= target < 10.0
            aligned, scores = {}, {}
            for key, prediction in output_small.items():
                relative = prediction if key.startswith("ppd_") else np.log1p(np.maximum(prediction, 1e-6))
                np.random.seed(args.seed + index)
                aligned[key] = recover_metric_depth_ransac(relative, target, mask)
                if not np.isfinite(aligned[key]).all():
                    raise ValueError(f"Non-finite alignment: {name}/{key}")
                if target_tensor is not None:
                    scores[key] = scalar_metrics(aligned[key], target, mask)
            result = {
                "name": name, "index": index, "dataset": row["dataset"], "group": row["group"],
                "resolution_hw": [height, width], "noise_seed": args.seed + index,
                "feature_metrics": feature_metrics, "metrics": scores,
                "has_ground_truth": target_tensor is not None,
                "ppd_raw_difference_rmse": float(np.sqrt(np.mean((outputs["ppd_adapter"] - outputs["ppd_native"]) ** 2))),
                "seconds": time.monotonic() - start,
            }
            temporary = args.root / (name + ".npz.new")
            with temporary.open("xb") as stream:
                np.savez_compressed(stream,
                    rgb=(image[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8),
                    target=target, valid=mask, **outputs,
                    **{f"aligned_{k}": value for k, value in aligned.items()})
            os.replace(temporary, args.root / (name + ".npz"))
            publish(args.root / (name + ".json"), result)
            results.append(result)
            print(json.dumps({"event": "image", **result}), flush=True)
            del raw, depth, fitted, features, native, spatial, summary, noise, image

    aggregates = {}
    for resolution in plan["resolutions_hw"]:
        eligible = [r for r in results if r["resolution_hw"] == resolution and r["has_ground_truth"]]
        aggregate = {method: {metric: float(np.mean([r["metrics"][method][metric] for r in eligible]))
                              for metric in ("abs_rel", "rmse", "delta1", "log1p_gradient_mae")}
                     for method in ("moge_native", "moge_adapter", "ppd_native", "ppd_adapter")}
        aggregate["images"] = len(eligible)
        aggregate["ppd_adapter_better_than_moge_adapter_absrel"] = sum(r["metrics"]["ppd_adapter"]["abs_rel"] < r["metrics"]["moge_adapter"]["abs_rel"] for r in eligible)
        aggregate["ppd_adapter_better_than_ppd_native_absrel"] = sum(r["metrics"]["ppd_adapter"]["abs_rel"] < r["metrics"]["ppd_native"]["abs_rel"] for r in eligible)
        aggregates[f"{resolution[0]}x{resolution[1]}"] = aggregate
    publish(args.root / "report.json", {
        "completed": True, "plan_sha256": digest(args.root / "plan.json"),
        "teacher_boundary_parity": parity, "results": results, "aggregates": aggregates,
        "images": len(selected), "paired_cases": len(results), "optimizer_steps": 0,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    })
    print(json.dumps({"event": "complete", "aggregates": aggregates}), flush=True)


if __name__ == "__main__":
    main()
