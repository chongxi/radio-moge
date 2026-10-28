"""Teacher-only plane discovery and differentiable, masked student geometry losses.

No semantic wall labels are assumed. Masks come solely from teacher geometry;
student outliers cannot remove their own supervision. RANSAC runs on bounded CPU
samples with a private seed and never participates in autograd.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PlaneConfig:
    max_planes: int = 6
    iterations: int = 256
    max_samples: int = 6000
    sample_stride: int = 4
    min_area_ratio: float = 0.02
    distance_ratio: float = 0.004
    max_rms_ratio: float = 0.0025
    min_extent_ratio: float = 0.04
    normal_degrees: float = 15.0
    depth_edge_ratio: float = 0.025
    erosion: int = 2
    seed: int = 42

    def __post_init__(self):
        if min(self.max_planes, self.iterations, self.sample_stride) < 1 or self.max_samples < 3:
            raise ValueError("Plane search budgets must be positive; max_samples must be >= 3")
        for name in ("min_area_ratio", "distance_ratio", "max_rms_ratio",
                     "min_extent_ratio", "depth_edge_ratio"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"Invalid plane threshold: {name}")
        if not 0 < self.normal_degrees < 90 or self.erosion < 0:
            raise ValueError("Invalid normal threshold or erosion")


@dataclass
class PlanePatch:
    mask: np.ndarray
    normal: np.ndarray
    offset: float
    rms: float
    extent: float
    confidence: float

    def metadata(self):
        return {"normal": self.normal.tolist(), "offset": self.offset,
                "rms": self.rms, "extent": self.extent, "confidence": self.confidence,
                "pixels": int(self.mask.sum())}


def point_normals_numpy(points, valid, edge_ratio):
    p = np.where(valid[..., None], points, 0.0)
    dx, dy = p[1:-1, 2:] - p[1:-1, :-2], p[2:, 1:-1] - p[:-2, 1:-1]
    n = np.cross(dx, dy)
    length = np.linalg.norm(n, axis=-1)
    safe = (valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2]
            & valid[2:, 1:-1] & valid[:-2, 1:-1] & (length > 1e-12))
    z = p[1:-1, 1:-1, 2]
    for neighbor in (p[1:-1, 2:, 2], p[1:-1, :-2, 2], p[2:, 1:-1, 2], p[:-2, 1:-1, 2]):
        safe &= np.abs(z - neighbor) <= edge_ratio * np.maximum(z, 1e-8)
    normals = np.zeros_like(p)
    normals[1:-1, 1:-1] = n / np.maximum(length[..., None], 1e-12)
    mask = np.zeros(valid.shape, bool)
    mask[1:-1, 1:-1] = safe
    return normals, mask


def fit_plane(points):
    center = points.mean(0)
    centered = points - center
    values, vectors = np.linalg.eigh(centered.T @ centered / len(points))
    normal = vectors[:, 0]
    if normal @ center > 0:
        normal = -normal
    return normal, -float(normal @ center), np.maximum(values, 0)


def detect_planes(points, valid=None, config=None):
    """Return connected, eroded plane patches in an organized H,W,3 camera point map."""
    cfg = config or PlaneConfig()
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 3 or p.shape[-1] != 3 or min(p.shape[:2]) < 3:
        raise ValueError("Expected organized H,W,3 point map with H,W >= 3")
    good = np.isfinite(p).all(-1) & (p[..., 2] > 0)
    if valid is not None:
        if np.shape(valid) != p.shape[:2]:
            raise ValueError("Point mask shape mismatch")
        good &= valid
    if good.sum() < 3:
        return []
    scale = float(np.median(p[..., 2][good]))
    threshold = cfg.distance_ratio * scale
    normals, available = point_normals_numpy(p, good, cfg.depth_edge_ratio)
    grid = np.zeros(good.shape, bool)
    grid[::cfg.sample_stride, ::cfg.sample_stride] = True
    min_pixels = max(32, int(np.ceil(good.size * cfg.min_area_ratio)))
    cosine = np.cos(np.deg2rad(cfg.normal_degrees))
    rng = np.random.default_rng(cfg.seed)
    patches = []
    # Extra attempts allow a rejected/fragmented candidate without blocking other planes.
    for _ in range(cfg.max_planes * 3):
        ids = np.flatnonzero(available & grid)
        if len(ids) < 3 or available.sum() < min_pixels or len(patches) >= cfg.max_planes:
            break
        if len(ids) > cfg.max_samples:
            ids = rng.choice(ids, cfg.max_samples, replace=False)
        samples, sample_normals = p.reshape(-1, 3)[ids], normals.reshape(-1, 3)[ids]
        triples = samples[rng.integers(len(samples), size=(cfg.iterations, 3))]
        candidate = np.cross(triples[:, 1] - triples[:, 0], triples[:, 2] - triples[:, 0])
        lengths = np.linalg.norm(candidate, axis=1)
        keep = lengths > scale * scale * 1e-8
        candidate = candidate[keep] / lengths[keep, None]
        offsets = -(candidate * triples[keep, 0]).sum(1)
        if not len(candidate):
            break
        best_score, best = 0, None
        for start in range(0, len(candidate), 32):
            ns, ds = candidate[start:start + 32], offsets[start:start + 32]
            support = ((np.abs(samples @ ns.T + ds) < threshold)
                       & (np.abs(sample_normals @ ns.T) > cosine))
            scores = support.sum(0)
            j = int(scores.argmax())
            if scores[j] > best_score:
                best_score, best = int(scores[j]), support[:, j]
        if best is None or best_score < 3:
            break
        n, d, _ = fit_plane(samples[best])
        candidate_mask = (available & (np.abs(p @ n + d) < threshold)
                          & (np.abs(normals @ n) > cosine))
        # Remove only the candidate's supported pixels, not its whole bounding rectangle.
        available[candidate_mask] = False
        labels, _ = ndimage.label(candidate_mask)
        counts = np.bincount(labels.ravel())
        for label_id in np.flatnonzero(counts >= min_pixels):
            if label_id == 0:
                continue
            mask = labels == label_id
            n, d, values = fit_plane(p[mask])
            mask &= (np.abs(p @ n + d) < threshold) & (np.abs(normals @ n) > cosine)
            if cfg.erosion:
                mask = ndimage.binary_erosion(mask, iterations=cfg.erosion)
            # Erosion can split a patch; never join separated surfaces again.
            parts, _ = ndimage.label(mask)
            part_counts = np.bincount(parts.ravel())
            for part in np.flatnonzero(part_counts >= min_pixels):
                if part == 0:
                    continue
                interior = parts == part
                n, d, values = fit_plane(p[interior])
                rms = float(np.sqrt(values[0]))
                extent = float(np.sqrt(values.sum()))
                if (rms > cfg.max_rms_ratio * scale
                        or np.sqrt(values[1]) < cfg.min_extent_ratio * scale):
                    continue
                residual_ok = np.abs(p @ n + d) < threshold
                normal_ok = np.abs(normals @ n) > cosine
                if np.mean((residual_ok & normal_ok)[interior]) < 0.95:
                    continue
                confidence = float(np.clip(1 - rms / threshold, 0, 1))
                patches.append(PlanePatch(interior, n, d, rms, extent, confidence))
                if len(patches) >= cfg.max_planes:
                    break
            if len(patches) >= cfg.max_planes:
                break
    return patches


def point_normals_torch(points):
    dx = points[:, 1:-1, 2:] - points[:, 1:-1, :-2]
    dy = points[:, 2:, 1:-1] - points[:, :-2, 1:-1]
    return F.normalize(torch.linalg.cross(dx, dy, dim=-1), dim=-1, eps=1e-12)


def planarity_metrics(points, patches):
    """Scale-invariant own-plane flatness and teacher-normal pose, on fixed masks.

    Unsupported images have counts only, not artificial zero error values.
    """
    rows = []
    for patch in patches:
        p = np.asarray(points, dtype=np.float64)[patch.mask]
        if not np.isfinite(p).all() or (p[:, 2] <= 0).any():
            raise ValueError("Invalid student support inside teacher plane")
        n, d, values = fit_plane(p)
        extent = float(np.sqrt(values[1:].sum()))
        if extent < 1e-8:
            raise ValueError("Collapsed plane support")
        distances = np.abs(p @ n + d)
        rows.append({"normalized_rms": float(np.sqrt(np.mean(distances ** 2)) / extent),
                     "p95_normalized": float(np.quantile(distances, .95) / extent),
                     "normal_angle_deg": float(np.rad2deg(np.arccos(
                         np.clip(abs(n @ patch.normal), 0, 1))))})
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {}


def plane_losses(student, teacher, batches):
    """Losses on matching camera-space maps; mean per patch, then per supported image.

    Flatness uses student extent WITH its gradient, so global scaling does not
    lower it. The anchor must be retained to rule out collapsed/wrong-position
    solutions. Normals come from student points, never an independent normal head.
    """
    if student.shape != teacher.shape or student.ndim != 4 or student.shape[-1] != 3:
        raise ValueError("Student and teacher must be matching B,H,W,3 maps")
    if len(batches) != len(student):
        raise ValueError("Plane batch count mismatch")
    zero = torch.nan_to_num(student).sum() * 0
    result = {k: [] for k in ("plane_flatness", "plane_normal", "plane_anchor")}
    for b, patches in enumerate(batches):
        image_losses = {k: [] for k in result}
        for patch in patches:
            mask = torch.as_tensor(patch.mask, device=student.device, dtype=torch.bool)
            if mask.shape != student.shape[1:3] or not mask.any():
                raise ValueError("Invalid teacher plane mask")
            p, t = student[b][mask].float(), teacher[b][mask].detach().float()
            if not torch.isfinite(p).all() or not torch.isfinite(t).all():
                raise FloatingPointError("Nonfinite point inside teacher plane mask")
            n = torch.as_tensor(patch.normal, device=p.device, dtype=p.dtype)
            centered = p - p.mean(0)
            extent_sq = centered.square().sum(-1).mean().clamp_min(1e-12)
            image_losses["plane_flatness"].append((centered @ n).square().mean() / extent_sq)
            # Use only stencils completely inside this teacher-selected patch.
            inner = (mask[1:-1, 1:-1] & mask[1:-1, 2:] & mask[1:-1, :-2]
                     & mask[2:, 1:-1] & mask[:-2, 1:-1])
            if inner.any():
                safe = torch.where(mask[..., None], student[b], 0).unsqueeze(0)
                normals = point_normals_torch(safe)[0][inner]
                image_losses["plane_normal"].append((1 - (normals @ n).abs().clamp_max(1)).mean())
            image_losses["plane_anchor"].append(F.smooth_l1_loss(
                p / max(patch.extent, 1e-6), t / max(patch.extent, 1e-6), beta=0.05))
        for key, values in image_losses.items():
            if values:
                result[key].append(torch.stack(values).mean())
    return {k: torch.stack(v).mean() if v else zero for k, v in result.items()}


def plane_config_dict():
    return asdict(PlaneConfig())


def preservation_metrics(student_depth, teacher_depth, valid, patches):
    """Fixed teacher support: outside-plane accuracy and scale-invariant edge fidelity."""
    p, t = np.asarray(student_depth), np.asarray(teacher_depth)
    valid = np.asarray(valid, bool) & np.isfinite(t) & (t > 0)
    if not np.isfinite(p[valid]).all() or (p[valid] <= 0).any():
        raise ValueError("Invalid student depth on fixed teacher support")
    planes = np.zeros_like(valid)
    for patch in patches:
        planes |= patch.mask
    outside = valid & ~planes
    result = {}
    if outside.any():
        result['outside_plane/teacher_abs_rel'] = float(np.mean(
            np.abs(p[outside] - t[outside]) / t[outside]))
    errors, magnitudes, reference = [], [], []
    for axis in (0, 1):
        a, b = [slice(None)] * 2, [slice(None)] * 2
        a[axis], b[axis] = slice(1, None), slice(None, -1)
        a, b = tuple(a), tuple(b)
        support = valid[a] & valid[b]
        td = np.log(np.maximum(t[a], 1e-8)) - np.log(np.maximum(t[b], 1e-8))
        pd = np.log(np.maximum(p[a], 1e-8)) - np.log(np.maximum(p[b], 1e-8))
        edge = support & (np.abs(td) > .025)
        errors.extend(np.abs(pd[edge] - td[edge]))
        magnitudes.extend(np.abs(pd[edge]))
        reference.extend(np.abs(td[edge]))
    if errors:
        result['depth_edges/log_gradient_mae'] = float(np.mean(errors))
        result['depth_edges/retention_ratio'] = float(np.sum(magnitudes) / np.sum(reference))
    return result
