from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass
class RadioFeatures:
    spatial: torch.Tensor  # B,C,H,W; RADIO's output normalizer may be Identity
    summary: torch.Tensor  # B,Ds, concatenated backbone summary tokens
    intermediates: tuple[torch.Tensor, ...] = ()


@dataclass
class EncoderFeatures:
    spatial: torch.Tensor
    summary: torch.Tensor


def zero_dominant_channels(features: torch.Tensor, threshold: float = 8.0) -> torch.Tensor:
    """toromi_slam's rule, independently per image, without mutating shared features.

    A whole spatial channel is zeroed when its mean absolute activation exceeds
    threshold times the median across channels. This is not a token-norm filter.
    """
    if features.ndim != 4:
        raise ValueError("Channel cleaning expects B,C,H,W spatial features")
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Channel threshold must be finite and positive")
    magnitude = features.float().abs().mean(dim=(-2, -1))
    hot = magnitude > threshold * magnitude.median(dim=1, keepdim=True).values
    return features.masked_fill(hot[:, :, None, None], 0)


class GeometryAdapter(nn.Module):
    """Maps features from ONE RADIO pass to MoGe's fused encoder interface."""

    def __init__(
        self,
        kind="mlp",
        spatial_dim=1280,
        summary_dim=2560,
        target_dim=1024,
        target_summary_dim=1024,
        hidden_dim=2048,
        fusion_dim=512,
        indices=(7, 15, 23, 31),
        channel_cleaning="none",
        channel_threshold=8.0,
    ):
        super().__init__()
        if channel_cleaning not in {"none", "zero-channel"}:
            raise ValueError(f"Unknown channel cleaning: {channel_cleaning}")
        if not math.isfinite(channel_threshold) or channel_threshold <= 0:
            raise ValueError("Channel threshold must be finite and positive")
        self.channel_cleaning = channel_cleaning
        self.channel_threshold = float(channel_threshold)
        self.config = dict(
            kind=kind,
            spatial_dim=spatial_dim,
            summary_dim=summary_dim,
            target_dim=target_dim,
            target_summary_dim=target_summary_dim,
            hidden_dim=hidden_dim,
            fusion_dim=fusion_dim,
            indices=list(indices),
        )
        # Preserve the exact legacy config when cleaning is disabled. Existing
        # checkpoint resume/init checks and exported heads keep their behavior.
        if channel_cleaning != "none" or channel_threshold != 8.0:
            self.config.update(
                channel_cleaning=channel_cleaning, channel_threshold=self.channel_threshold
            )
        self.kind = kind
        self.indices = tuple(indices) if kind == "fusion" else ()
        if kind == "linear":
            self.spatial = nn.Conv2d(spatial_dim, target_dim, 1)
            self.summary = nn.Linear(summary_dim, target_summary_dim)
        elif kind in {"mlp", "fusion"}:
            if kind == "fusion":
                if not indices:
                    raise ValueError("Fusion requires intermediate block indices")
                self.projections = nn.ModuleList(
                    [nn.Conv2d(spatial_dim, fusion_dim, 1) for _ in indices]
                )
                input_dim = fusion_dim
            else:
                input_dim = spatial_dim
            self.spatial = nn.Sequential(
                nn.Conv2d(input_dim, hidden_dim, 1), nn.GELU(), nn.Conv2d(hidden_dim, target_dim, 1)
            )
            self.summary = nn.Sequential(
                nn.Linear(summary_dim, target_summary_dim),
                nn.GELU(),
                nn.Linear(target_summary_dim, target_summary_dim),
            )
        else:
            raise ValueError(f"Unknown adapter: {kind}")

    def _clean(self, features):
        return (
            zero_dominant_channels(features, self.channel_threshold)
            if self.channel_cleaning == "zero-channel"
            else features
        )

    def forward(self, features: RadioFeatures) -> EncoderFeatures:
        if features.spatial.ndim != 4 or features.summary.ndim != 2:
            raise ValueError("Expected spatial B,C,H,W and summary B,Ds")
        if features.spatial.shape[1] != self.config["spatial_dim"]:
            raise ValueError("RADIO spatial width differs from adapter configuration")
        if features.summary.shape[1] != self.config["summary_dim"]:
            raise ValueError("RADIO summary width differs from adapter configuration")
        x = features.spatial.float()
        if self.kind == "fusion":
            if len(features.intermediates) != len(self.indices):
                raise ValueError(
                    "Missing intermediate features: use one forward_intermediates pass"
                )
            if any(v.shape != x.shape for v in features.intermediates):
                raise ValueError("Intermediate and final spatial shapes must match")
            x = sum(p(self._clean(v.float())) for p, v in zip(self.projections, features.intermediates))
            x = x / len(self.projections)
        else:
            x = self._clean(x)
        # No output normalization: the frozen decoder needs teacher magnitudes.
        return EncoderFeatures(self.spatial(x), self.summary(features.summary.float()))
