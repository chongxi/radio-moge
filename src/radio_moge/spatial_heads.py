"""Spatial adapters inspired by DPT reassembly and RADIO's attention adaptor.

These predict the frozen MoGe encoder interface, not depth directly. They are
new architectures, not implementations or pretrained weights of those heads.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self, width, heads, expansion=4):
        super().__init__()
        if width % heads:
            raise ValueError("Attention width must be divisible by head count")
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * expansion), nn.GELU(), nn.Linear(width * expansion, width)
        )
        self.scale1 = nn.Parameter(torch.full((width,), 1e-5))
        self.scale2 = nn.Parameter(torch.full((width,), 1e-5))

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = (
            self.qkv(self.norm1(x))
            .reshape(b, n, 3, self.heads, c // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        value = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c)
        x = x + self.scale1 * self.projection(value)
        return x + self.scale2 * self.ffn(self.norm2(x))


class SpatialSummary(nn.Module):
    """Let the global scale/summary branch read the adapted spatial features."""

    def __init__(self, spatial_dim, summary_dim, target_dim, width=512, heads=8):
        super().__init__()
        self.query = nn.Linear(summary_dim, width)
        self.memory = nn.Conv2d(spatial_dim, width, 1)
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.output = nn.Linear(width, target_dim)
        self.skip = nn.Linear(summary_dim, target_dim)

    def forward(self, spatial, summary):
        q = self.query(summary).unsqueeze(1)
        memory = self.memory_norm(self.memory(spatial).flatten(2).transpose(1, 2))
        pooled, _ = self.attention(self.query_norm(q), memory, memory, need_weights=False)
        return self.skip(summary) + self.output((q + pooled).squeeze(1))


def position_encoding(h, w, width, device, dtype):
    if width % 4:
        raise ValueError("2-D positional encoding requires a width divisible by four")
    frequency = torch.exp(
        torch.arange(width // 4, device=device, dtype=torch.float32)
        * (-math.log(10000) / max(1, width // 4 - 1))
    )
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing="ij",
    )
    x, y = x.flatten()[:, None] * frequency, y.flatten()[:, None] * frequency
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=-1).to(dtype)[None]


class AttentionSpatialHead(nn.Module):
    def __init__(self, input_dim, target_dim, taps, width, blocks, heads):
        super().__init__()
        if width % 4 or blocks < 1:
            raise ValueError("Attention needs positive blocks and width divisible by four")
        self.projections = nn.ModuleList([nn.Conv2d(input_dim, width, 1) for _ in range(taps)])
        self.tap_logits = nn.Parameter(torch.zeros(taps))
        self.blocks = nn.Sequential(*[AttentionBlock(width, heads) for _ in range(blocks)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, target_dim))

    def forward(self, features):
        weights = self.tap_logits.softmax(0)
        x = sum(
            weight * project(value)
            for weight, project, value in zip(weights, self.projections, features)
        )
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.blocks(tokens + position_encoding(h, w, c, x.device, x.dtype))
        spatial = self.output(tokens).transpose(1, 2).reshape(b, -1, h, w)
        return spatial, tokens.transpose(1, 2).reshape(b, c, h, w)


class ResidualSpatialBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
        )

    def forward(self, x):
        return x + self.layers(x)


class DPTSpatialHead(nn.Module):
    """Reassemble four ViT taps at 4x, 2x, 1x, 1/2x, then fuse coarse to fine."""

    def __init__(self, input_dim, target_dim, taps, width):
        super().__init__()
        if taps != 4:
            raise ValueError("DPT adapter requires exactly four intermediate taps")
        self.projections = nn.ModuleList([nn.Conv2d(input_dim, width, 1) for _ in range(taps)])
        self.lateral = nn.ModuleList([ResidualSpatialBlock(width) for _ in range(taps)])
        self.refine = nn.ModuleList([ResidualSpatialBlock(width) for _ in range(taps)])
        self.output = nn.Conv2d(width, target_dim, 1)

    def forward(self, features):
        h, w = features[0].shape[-2:]
        sizes = [
            (h * 4, w * 4),
            (h * 2, w * 2),
            (h, w),
            (max(1, (h + 1) // 2), max(1, (w + 1) // 2)),
        ]
        x = None
        for i in range(3, -1, -1):
            lateral = F.interpolate(
                self.projections[i](features[i]), sizes[i], mode="bilinear", align_corners=False
            )
            lateral = self.lateral[i](lateral)
            x = (
                lateral
                if x is None
                else lateral + F.interpolate(x, sizes[i], mode="bilinear", align_corners=False)
            )
            x = self.refine[i](x)
        # Antialiased resizing has limited low-precision backend support.
        x = F.interpolate(x.float(), (h, w), mode="bilinear", align_corners=False, antialias=True)
        return self.output(x), x
