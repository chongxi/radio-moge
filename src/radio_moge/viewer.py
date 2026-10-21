"""Export frozen, decoded predictions for an interactive held-out-only viewer."""

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np
from PIL import Image
import torch

from .checkpoint import atomic_json
from .compare import autocast, fingerprint
from .config import ROOT
from .data import check_split, evaluation_image, read_manifest
from .decoder import postprocess
from .metrics import depth_metrics
from .pointcloud import verify_geometry
from .system import build_system

CHANNELS = ["before", "after", "teacher", "ground_truth"]


def write_depths(path, arrays):
    """Lossless little-endian float32 planes; browser receives decoded HTTP gzip."""
    shape = arrays[0].shape
    if any(a.shape != shape for a in arrays):
        raise ValueError("Depth planes must have identical shapes")
    with Path(path).open("wb") as file:
        with gzip.GzipFile(fileobj=file, mode="wb", compresslevel=3, mtime=0) as compressed:
            for array in arrays:
                compressed.write(np.asarray(array, dtype="<f4").tobytes())


def summarize(samples):
    result = {}
    for name in CHANNELS[:3]:
        result[name] = {}
        for metric in ("abs_rel", "rmse", "delta1", "invalid_fraction"):
            values = [s["metrics"][name][metric] for s in samples if metric in s["metrics"][name]]
            result[name][metric] = float(np.mean(values)) if values else None
            result[name][metric + "_count"] = len(values)
        result[name]["failures"] = sum(bool(s["failures"].get(name)) for s in samples)
    return result


def initial_provenance(state, initial):
    """Accept a matching stage-start checkpoint or the legacy verified artifact."""
    if "provenance" in initial:
        provenance = initial["provenance"]
    else:
        matches = all(initial.get(k) == state.get(k) for k in (
            "adapter_config", "config", "model_config", "data_digests", "revisions"
        ))
        if not matches or initial.get("step") != 0 or initial.get("cursor") != 0:
            raise ValueError("Initial checkpoint must be step zero from the same training stage")
        provenance = {
            "method": "saved stage-start checkpoint",
            "adapter_sha256": fingerprint(initial["adapter"]),
            "trained_sha256": fingerprint(state["adapter"]),
            "revisions": state["revisions"],
        }
    trusted = (provenance.get("verified")
               or provenance.get("method") == "saved stage-start checkpoint")
    if (not trusted
            or initial["adapter_config"] != state["adapter_config"]
            or provenance["trained_sha256"] != fingerprint(state["adapter"])
            or provenance["adapter_sha256"] != fingerprint(initial["adapter"])
            or provenance["revisions"] != state["revisions"]):
        raise ValueError("Initial adapter provenance does not match the trained checkpoint")
    return provenance


@torch.no_grad()
def export_viewer(checkpoint, initial_adapter, manifest, output, original_metrics=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    initial = torch.load(initial_adapter, map_location="cpu", weights_only=True)
    provenance = initial_provenance(state, initial)
    config = state["config"]
    records = read_manifest(manifest)
    train = read_manifest(config["data"]["train_manifest"])
    check_split(train, records)
    val_path = config["data"].get("val_manifest")
    validation = read_manifest(val_path) if val_path else []
    check_split(train, validation)
    check_split(validation, records)
    system, after = build_system(config, "cuda")
    after.load_state_dict(state["adapter"])
    after.eval()
    before = deepcopy(after)
    before.load_state_dict(initial["adapter"])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    assets = staging / "assets"
    assets.mkdir()
    geometry_dir = staging / "geometry"
    geometry_dir.mkdir()
    geometry_samples = []
    samples = []
    original = json.loads(Path(original_metrics).read_text()) if original_metrics else None
    if original is not None and [r["image"] for r in original] != [r["image"] for r in records]:
        raise ValueError("Original evaluation image order differs from held-out manifest")
    max_difference = 0.0
    for index, record in enumerate(records):
        image, gt = evaluation_image(record, config["evaluation"]["max_side"])
        image = image.cuda()
        height, width = image.shape[-2:]
        with autocast(config, "cuda"):
            features, target = system.encode(image, after.indices, config["evaluation"]["num_tokens"])
            encodings = {"before": before(features), "after": after(features), "teacher": target}
        depths, metrics, failures, cameras = {}, {}, {}, {}
        for name, encoded in encodings.items():
            raw = system.decoder(encoded, (height, width))
            try:
                pred = postprocess(raw, image, config["models"].get("moge_source"))
                depth = pred["depth"]
                if name in {"after", "teacher"}:
                    k, mask, error = verify_geometry(depth[0].cpu().numpy(), pred)
                    filename = f"{index:04d}-{name}-mask.png"
                    Image.fromarray(mask.astype("uint8") * 255).save(geometry_dir / filename)
                    cameras[name] = dict(intrinsics=k, mask=f"geometry/{filename}",
                                         valid_points=int(mask.sum()), projection_max_error_m=error)
            except (ValueError, np.linalg.LinAlgError) as error:
                failures[name] = str(error)
                depth = torch.full((1, height, width), float("nan"), device=image.device)
            depths[name] = depth[0].cpu().numpy()
            metrics[name] = (
                depth_metrics(depth, gt[None].cuda(), gt[None].cuda() > 0) if gt is not None else {}
            )
            if original is not None and name in {"after", "teacher"}:
                old_name = "student" if name == "after" else "teacher"
                for key, value in metrics[name].items():
                    previous = original[index].get(f"gt_{old_name}/{key}")
                    if previous is not None:
                        max_difference = max(max_difference, abs(value - previous))
        depths["ground_truth"] = gt.numpy() if gt is not None else np.full((height, width), np.nan)
        valid = np.isfinite(depths["teacher"]) & (depths["teacher"] > 0)
        bounds = np.quantile(depths["teacher"][valid], [0.02, 0.98]).tolist() if valid.any() else [0, 10]
        rgb = Image.fromarray((image[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8"))
        stem = f"{index:04d}"
        rgb.save(assets / f"{stem}.jpg", quality=94)
        rgb.thumbnail((160, 120))
        rgb.save(assets / f"{stem}-thumb.jpg", quality=80)
        write_depths(assets / f"{stem}.depth.gz", [depths[k] for k in CHANNELS])
        samples.append(dict(
            index=index, id=record.get("id", stem), scene=record.get("scene", record.get("group", stem)),
            scene_type=record.get("scene_type", "unspecified"), width=width, height=height,
            rgb=f"assets/{stem}.jpg", thumbnail=f"assets/{stem}-thumb.jpg",
            depth=f"assets/{stem}.depth.gz", range=bounds, metrics=metrics, failures=failures,
        ))
        geometry_samples.append(dict(id=record.get("id", stem), **cameras))
        if (index + 1) % 20 == 0 or index == 0:
            print(f"Exported {index + 1}/{len(records)} held-out frames", flush=True)
    if original is not None and max_difference > 1e-5:
        raise ValueError(f"Decoded scores changed from original evaluation: {max_difference}")
    manifest_data = dict(
        schema=1, channels=CHANNELS, dtype="float32-le", samples=samples, summary=summarize(samples),
        colormap=cv2.cvtColor(cv2.applyColorMap(np.arange(256, dtype="uint8")[:, None],
                                              cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)[:, 0].tolist(),
        run=dict(name=config.get("tracking", {}).get("name", "NYUv2 sanity"),
                 steps=state["step"], training_budget=config["train"]["steps"],
                 checkpoint=Path(checkpoint).name, train_images=len(train),
                 validation_images=len(validation),
                 test_images=len(records), test_scenes=len({r.get("group") for r in records}),
                 student=("C-RADIOv4-H + intermediate fusion"
                          if state["adapter_config"].get("kind") == "fusion"
                          else "C-RADIOv4-H + MLP"),
                 teacher=config["models"]["moge_repo"],
                 baseline=("Seeded reconstruction; all four initial feature losses matched exactly"
                           if provenance.get("verified") else "Saved stage-start checkpoint"),
                 checkpoint_sha256=fingerprint(state["adapter"]), revisions=state["revisions"],
                 initial_adapter_sha256=provenance["adapter_sha256"],
                 original_evaluation_max_difference=max_difference if original is not None else None,
                 protocol=f"Scene-disjoint custom split: {len(train)} training images, "
                          f"{len(validation)} validation images, {len(records)} test images. "
                          + ("Validation teacher-depth error selects the best checkpoint. "
                             if validation else "Fixed training budget; no validation selection. ")
                          + "Full-frame measured rawDepths; no "
                          "scale alignment. Scores average per-image errors over valid measured "
                          "depth. Teacher pretraining may include NYUv2. Not the official benchmark."),
    )
    atomic_json(manifest_data, staging / "manifest.json")
    if all("after" in s and "teacher" in s for s in geometry_samples):
        atomic_json(dict(schema=1, samples=geometry_samples,
                         checkpoint_sha256=manifest_data["run"]["checkpoint_sha256"],
                         depth_manifest_sha256=hashlib.sha256(
                             (staging / "manifest.json").read_bytes()).hexdigest(),
                         coordinates="OpenCV: x right, y down, z forward; meters",
                         projection="Normalized intrinsics, pixel-center sampling",
                         policy="Each model's own recovered intrinsics; no alignment."),
                    geometry_dir / "manifest.json")
    for source in (ROOT / "web").iterdir():
        if source.is_dir():
            shutil.copytree(source, staging / source.name)
        elif source.suffix in {".html", ".js", ".mjs", ".css", ".svg"}:
            shutil.copy2(source, staging / source.name)
    os.rename(staging, output)
    print(json.dumps({"output": str(output), "images": len(samples), "summary": summarize(samples),
                      "original_evaluation_max_difference": max_difference}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--initial-adapter", "--initial", required=True,
                        help="Actual initial.pt from this stage, or verified initial-adapter.pt")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--original-metrics", help="Original per_image.json; require unchanged scores")
    args = parser.parse_args()
    export_viewer(args.checkpoint, args.initial_adapter, args.manifest, args.output, args.original_metrics)


if __name__ == "__main__":
    main()
