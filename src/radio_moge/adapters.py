from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RadioFeatures:
    spatial: torch.Tensor  # B,C,H,W, final features normalized by RADIO
    summary: torch.Tensor  # B,Ds, concatenated backbone summary tokens
    intermediates: tuple[torch.Tensor, ...] = ()


@dataclass
class EncoderFeatures:
    spatial: torch.Tensor
    summary: torch.Tensor


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
    ):
        super().__init__()
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
            x = sum(p(v.float()) for p, v in zip(self.projections, features.intermediates))
            x = x / len(self.projections)
        # No output normalization: the frozen decoder needs teacher magnitudes.
        return EncoderFeatures(self.spatial(x), self.summary(features.summary.float()))
