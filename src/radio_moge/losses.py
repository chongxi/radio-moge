import torch
import torch.nn.functional as F
from torch import nn


class FeatureStatistics(nn.Module):
    """Fixed per-channel teacher RMS scales, fitted on training data only.

    Normalizing the LOSS leaves predicted absolute values intact for the decoder.
    RMS rather than variance also handles almost-constant teacher channels.
    """

    def __init__(self, spatial_dim, summary_dim):
        super().__init__()
        self.register_buffer("spatial_scale", torch.ones(spatial_dim))
        self.register_buffer("summary_scale", torch.ones(summary_dim))

    @torch.no_grad()
    def fit(self, features):
        spatial_sq = torch.zeros_like(self.spatial_scale, dtype=torch.float64)
        summary_sq = torch.zeros_like(self.summary_scale, dtype=torch.float64)
        spatial_count = summary_count = 0
        for value in features:
            if not (torch.isfinite(value.spatial).all() and torch.isfinite(value.summary).all()):
                raise FloatingPointError("Nonfinite teacher features during calibration")
            spatial_sq += value.spatial.double().square().sum((0, 2, 3))
            summary_sq += value.summary.double().square().sum(0)
            spatial_count += (
                value.spatial.shape[0] * value.spatial.shape[2] * value.spatial.shape[3]
            )
            summary_count += value.summary.shape[0]
        if not summary_count:
            raise ValueError("Feature calibration needs at least one batch")
        self.spatial_scale.copy_((spatial_sq / spatial_count).sqrt().clamp_min(1e-3))
        self.summary_scale.copy_((summary_sq / summary_count).sqrt().clamp_min(1e-3))

    def loss(self, predicted, target):
        spatial, summary = predicted.spatial.float(), predicted.summary.float()
        if spatial.shape != target.spatial.shape or summary.shape != target.summary.shape:
            raise ValueError("Teacher/student encoder feature shapes differ")
        return {
            "spatial": (
                (spatial - target.spatial.float()) / self.spatial_scale[None, :, None, None]
            )
            .square()
            .mean(),
            "summary": ((summary - target.summary.float()) / self.summary_scale[None])
            .square()
            .mean(),
            "cosine": 1 - F.cosine_similarity(spatial, target.spatial.float(), dim=1).mean(),
        }


def masked_mean(value, mask):
    # where (rather than multiplication) prevents ignored NaN/Inf from contaminating reductions.
    mask = torch.broadcast_to(mask, value.shape)
    return torch.where(mask, value, 0.0).sum() / mask.sum().clamp_min(1)


def geometry_losses(predicted, target):
    p, t = predicted["points"].float(), target["points"].float()
    # Invalid teacher values must be removed BEFORE computing differences/backward.
    valid = torch.isfinite(t).all(-1) & (target["mask"] > 0.5)
    t = torch.where(valid[..., None], t, 0.0)
    p = torch.where(valid[..., None], p, 0.0)
    amplitude = masked_mean(t.norm(dim=-1), valid).detach().clamp_min(1e-3)
    point_error = F.smooth_l1_loss(p / amplitude, t / amplitude, reduction="none")
    result = {"points": masked_mean(point_error, valid[..., None])}
    result["mask"] = F.binary_cross_entropy(
        predicted["mask"].float().clamp(1e-6, 1 - 1e-6), target["mask"].float()
    )
    result["scale"] = F.smooth_l1_loss(
        predicted["metric_scale"].float().clamp_min(1e-8).log(),
        target["metric_scale"].float().clamp_min(1e-8).log(),
    )
    dx = ((p[:, :, 1:] - p[:, :, :-1]) - (t[:, :, 1:] - t[:, :, :-1])) / amplitude
    dy = ((p[:, 1:] - p[:, :-1]) - (t[:, 1:] - t[:, :-1])) / amplitude
    result["gradient"] = (
        masked_mean(dx.abs(), (valid[:, :, 1:] & valid[:, :, :-1])[..., None])
        + masked_mean(dy.abs(), (valid[:, 1:] & valid[:, :-1])[..., None])
    ) / 2
    if "normal" in target and "normal" in predicted:
        nvalid = valid & torch.isfinite(target["normal"]).all(-1)
        tn = torch.where(nvalid[..., None], target["normal"].float(), 0.0)
        pn = torch.where(nvalid[..., None], predicted["normal"].float(), 0.0)
        result["normal"] = masked_mean(1 - F.cosine_similarity(pn, tn, dim=-1), nvalid)
    return result


def weighted_loss(losses, weights):
    active = [value * weights.get(key, 0) for key, value in losses.items() if weights.get(key, 0)]
    if not active:
        raise ValueError("At least one available loss must have nonzero weight")
    return sum(active)
