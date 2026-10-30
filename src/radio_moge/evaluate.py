import argparse
import hashlib
from contextlib import nullcontext
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from .checkpoint import atomic_json
from .data import evaluation_image, read_manifest
from .decoder import postprocess
from .feature_diagnostics import feature_metrics
from .losses import FeatureStatistics, geometry_losses
from .metrics import aggregate, dataset_metrics, depth_metrics, fov_error, mask_iou
from .planes import PlaneConfig, detect_planes, planarity_metrics, preservation_metrics
from .system import build_system


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def visualization(image, teacher, student, path):
    rgb = (image[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    t, p = teacher["depth"][0].cpu().numpy(), student["depth"][0].cpu().numpy()
    mask = teacher["mask"][0].cpu().numpy() & np.isfinite(t) & (t > 0)
    if not mask.any():
        return
    lo, hi = np.quantile(t[mask], [0.02, 0.98])

    def panel(array, valid):
        value = np.nan_to_num((array - lo) / max(hi - lo, 1e-6), nan=0, posinf=1, neginf=0)
        value = (value.clip(0, 1) * 255).astype(np.uint8)
        out = np.repeat(value[..., None], 3, -1)
        out[~valid] = [255, 0, 255]
        return out

    error = np.nan_to_num(np.abs(p - t) / np.maximum(t, 1e-6), nan=1, posinf=1)
    error_rgb = np.zeros_like(rgb)
    error_rgb[..., 0] = (error.clip(0, 1) * 255).astype(np.uint8)
    error_rgb[~mask] = 0
    result = np.concatenate((rgb, panel(t, mask), panel(p, np.isfinite(p) & (p > 0)), error_rgb), 1)
    Image.fromarray(result).save(path)


@torch.no_grad()
def evaluate(system, adapter, statistics, records, config, output=None, limit=None):
    device = next(adapter.parameters()).device
    was_training = adapter.training
    adapter.eval()
    settings = config["evaluation"]
    inspect_planes = settings.get("planes", False) or any(
        config.get("loss", {}).get(k, 0)
        for k in ("plane_flatness", "plane_normal", "plane_anchor"))
    plane_config = PlaneConfig(**config.get("planes", {})) if inspect_planes else None
    rows = []
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)

    def autocast():
        return (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and config["train"]["precision"] == "bf16"
            else nullcontext()
        )

    try:
        for index, record in enumerate(records[:limit] if limit else records):
            image, gt = evaluation_image(record, settings["max_side"], settings.get("bucket"))
            image = image.to(device)
            with autocast():
                features, target_features = system.encode(
                    image, adapter.indices, num_tokens=settings["num_tokens"]
                )
                if settings.get("bucket") is not None and settings["num_tokens"] is None:
                    expected = tuple(d // system.radio.patch_size for d in settings["bucket"])
                    if tuple(features.spatial.shape[-2:]) != expected:
                        raise ValueError("Native validation feature grid differs from its bucket")
            # Added branch time excludes the shared RADIO backbone and teacher.
            synchronize(device)
            start = time.perf_counter()
            with autocast():
                predicted_features = adapter(features)
            predicted = system.decoder(predicted_features, image.shape[-2:])
            synchronize(device)
            branch_ms = (time.perf_counter() - start) * 1000
            target = system.decoder(target_features, image.shape[-2:])
            row = {"image": record["image"], "branch_ms": branch_ms}
            if "dataset" in record:
                row["dataset"] = record["dataset"]
            # Keep existing feature losses, and expose direction and magnitude separately.
            row.update(feature_metrics(predicted_features, target_features, statistics))
            row.update(
                {f"raw/{k}": v.item() for k, v in geometry_losses(predicted, target).items()}
            )
            row["mask_iou"] = mask_iou(predicted["mask"] > 0.5, target["mask"] > 0.5)
            row["scale_log_error"] = (
                (predicted["metric_scale"].log() - target["metric_scale"].log()).abs().mean().item()
            )
            try:
                synchronize(device)
                start = time.perf_counter()
                student = postprocess(predicted, image, config["models"].get("moge_source"))
                synchronize(device)
                row["postprocess_ms"] = (time.perf_counter() - start) * 1000
                teacher = postprocess(target, image, config["models"].get("moge_source"))
                valid = teacher["mask"] & torch.isfinite(teacher["depth"]) & (teacher["depth"] > 0)
                if not valid.any():
                    raise ValueError("Teacher has no valid positive-depth pixels")
                row.update(
                    {
                        f"teacher_depth/{k}": v
                        for k, v in depth_metrics(student["depth"], teacher["depth"], valid).items()
                    }
                )
                row["fov_error_deg"] = fov_error(student["intrinsics"], teacher["intrinsics"])
                if not np.isfinite(row["fov_error_deg"]):
                    raise ValueError("Focal recovery produced nonfinite intrinsics")
                if inspect_planes:
                    tpoints = teacher["points"][0].cpu().numpy()
                    patches = detect_planes(tpoints, valid[0].cpu().numpy(), plane_config)
                    row['planes/regions'] = len(patches)
                    row['planes/coverage'] = sum(p.mask.sum() for p in patches) / valid[0].numel()
                    labels = np.zeros(tpoints.shape[:2], dtype=np.uint8)
                    for number, patch in enumerate(patches, 1):
                        labels[patch.mask] = number
                    row['planes/mask_sha256'] = hashlib.sha256(labels.tobytes()).hexdigest()
                    try:
                        for name, pts in (('teacher', tpoints),
                                          ('student', student['points'][0].cpu().numpy())):
                            row.update({f'planes/{name}/{k}': v
                                        for k, v in planarity_metrics(pts, patches).items()})
                        row['planes/failed'] = 0
                        row.update(preservation_metrics(
                            student['depth'][0].cpu().numpy(), teacher['depth'][0].cpu().numpy(),
                            valid[0].cpu().numpy(), patches))
                    except (ValueError, np.linalg.LinAlgError):
                        row['planes/failed'] = 1
                    if output is not None and record.get('id') in settings.get('geometry_ids', []):
                        destination = output / f"geometry-{record['id']}.npz"
                        temporary = destination.with_suffix('.incoming')
                        with temporary.open('wb') as file:
                            np.savez_compressed(
                                file, rgb=(image[0].permute(1, 2, 0).cpu().numpy() * 255)
                                .clip(0, 255).astype('uint8'), labels=labels,
                                teacher_points=tpoints,
                                student_points=student['points'][0].cpu().numpy(),
                                teacher_depth=teacher['depth'][0].cpu().numpy(),
                                student_depth=student['depth'][0].cpu().numpy(),
                                teacher_intrinsics=teacher['intrinsics'][0].cpu().numpy(),
                                student_intrinsics=student['intrinsics'][0].cpu().numpy())
                        temporary.replace(destination)
                if gt is not None:
                    gt = gt.to(device).unsqueeze(0)
                    for name, prediction in (("student", student), ("teacher", teacher)):
                        row.update(
                            {
                                f"gt_{name}/{k}": v
                                for k, v in depth_metrics(prediction["depth"], gt, gt > 0).items()
                            }
                        )
                if output is not None and index < settings["visualizations"]:
                    visualization(image, teacher, student, output / f"{index:04d}.png")
                row["postprocess_failed"] = 0
            except (ValueError, np.linalg.LinAlgError) as error:
                # A random adapter can have unsolvable focal geometry. Keep failures visible.
                row["postprocess_failed"] = 1
                row["postprocess_error"] = str(error)
            for key, value in list(row.items()):
                if isinstance(value, float) and not np.isfinite(value):
                    row[key] = None
            rows.append(row)
        summary = aggregate(rows)
        summary.update(dataset_metrics(rows))
        if output is not None:
            atomic_json(dict(bucket_hw=settings.get('bucket'), num_tokens=settings['num_tokens'],
                             max_side=settings['max_side'], images=len(rows),
                             preprocessing='center aspect crop then resize' if settings.get('bucket')
                             else 'aspect-preserving longest-side resize'), output / 'protocol.json')
            atomic_json(rows, output / "per_image.json")
            atomic_json(summary, output / "metrics.json")
        return summary
    finally:
        adapter.train(was_training)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate against MoGe-2 and optional metric depth"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-tokens", type=int)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    if args.num_tokens:
        config["evaluation"]["num_tokens"] = args.num_tokens
    system, adapter = build_system(config, args.device)
    adapter.load_state_dict(state["adapter"])
    stats = FeatureStatistics(adapter.config["target_dim"], adapter.config["target_summary_dim"])
    stats.load_state_dict(state["statistics"])
    result = evaluate(
        system,
        adapter,
        stats.to(args.device),
        read_manifest(args.manifest),
        config,
        args.output,
        args.limit,
    )
    print(result)


if __name__ == "__main__":
    main()
