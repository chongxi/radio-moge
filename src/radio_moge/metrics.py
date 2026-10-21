import math

import torch


def depth_metrics(predicted, target, valid):
    """Metric-scale scores: no scale/shift alignment hiding scale errors.

    Invalid predictions count as zero depth (and are separately reported), so
    predicting an empty validity mask cannot improve these scores.
    """
    valid = valid & torch.isfinite(target) & (target > 0)
    if not valid.any():
        return {}
    p, t = predicted[valid].float(), target[valid].float()
    bad = ~torch.isfinite(p) | (p <= 0)
    p = torch.where(bad, 0.0, p)
    ratio = torch.maximum(p.clamp_min(1e-8) / t, t / p.clamp_min(1e-8))
    return {
        "abs_rel": ((p - t).abs() / t).mean().item(),
        "rmse": (p - t).square().mean().sqrt().item(),
        "delta1": (ratio < 1.25).float().mean().item(),
        "invalid_fraction": bad.float().mean().item(),
    }


def mask_iou(predicted, target):
    intersection = (predicted & target).sum().item()
    union = (predicted | target).sum().item()
    return intersection / union if union else 1.0


def fov_error(predicted_intrinsics, target_intrinsics):
    p = 2 * torch.atan(0.5 / predicted_intrinsics[..., 0, 0])
    t = 2 * torch.atan(0.5 / target_intrinsics[..., 0, 0])
    return ((p - t).abs().mean() * 180 / math.pi).item()


def aggregate(rows):
    keys = {
        key
        for row in rows
        for key, value in row.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    result = {}
    for key in sorted(keys):
        values = [row[key] for row in rows if key in row and math.isfinite(row[key])]
        if values:
            result[key] = sum(values) / len(values)
            result[f"{key}/count"] = len(values)
    result["images"] = len(rows)
    return result
