"""Identity-initialized feature corrections for matched warm-start adapter trials.

These are experimental residual extensions, not pretrained implementations of
named architectures. No teacher features, depth or plane labels enter forward().
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from .spatial_heads import position_encoding


REFINEMENTS = ('none', 'pointwise', 'multiscale', 'joint_attention')


class SwiGLUFFN(nn.Module):
    def __init__(self, width):
        super().__init__()
        hidden = math.ceil((8 * width / 3) / 8) * 8
        self.input = nn.Linear(width, hidden * 2)
        self.output = nn.Linear(hidden, width)

    def forward(self, x):
        value, gate = self.input(x).chunk(2, dim=-1)
        return self.output(value * F.silu(gate))


class RefinementBlock(nn.Module):
    def __init__(self, mode, width, heads, activation='gelu'):
        super().__init__()
        self.mode = mode
        self.norm = nn.LayerNorm(width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = (nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                  nn.Linear(4 * width, width)) if activation == 'gelu'
                    else SwiGLUFFN(width))
        if mode == 'joint_attention':
            self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        elif mode == 'multiscale':
            self.local = nn.Conv2d(width, width, 3, padding=1, groups=width)
            self.mix = nn.Conv2d(width, width, 1)
            self.scale_logits = nn.Parameter(torch.zeros(3))
            self.pool_summary = nn.Linear(width, width)

    def forward(self, tokens, height, width):
        value = self.norm(tokens)
        if self.mode == 'joint_attention':
            tokens = tokens + self.attention(value, value, value, need_weights=False)[0]
        elif self.mode == 'multiscale':
            spatial = value[:, 1:].transpose(1, 2).reshape(value.shape[0], -1, height, width)
            scales = [self.local(spatial)]
            for factor in (2, 4):
                coarse = F.adaptive_avg_pool2d(spatial, (max(1, height // factor),
                                                        max(1, width // factor)))
                scales.append(F.interpolate(coarse, (height, width), mode='bilinear',
                                            align_corners=False))
            mixed = self.mix(sum(w * x for w, x in zip(self.scale_logits.softmax(0), scales)))
            summary = self.pool_summary(mixed.mean((2, 3))).unsqueeze(1)
            tokens = tokens + torch.cat((summary, mixed.flatten(2).transpose(1, 2)), 1)
        else:
            # The pointwise control adds nonlinear capacity without token mixing.
            tokens = tokens + value
        return tokens + self.ffn(self.ffn_norm(tokens))


class FeatureRefinement(nn.Module):
    def __init__(self, mode, spatial_dim, summary_dim, width=256, blocks=2, heads=8,
                 activation='gelu'):
        super().__init__()
        if mode not in REFINEMENTS[1:]:
            raise ValueError(f'Unknown feature refinement: {mode}')
        if width < 4 or width % 4 or width % heads or blocks < 1:
            raise ValueError('Refinement width must divide heads/four; blocks must be positive')
        if activation not in ('gelu', 'swiglu'):
            raise ValueError('Refinement activation must be gelu or swiglu')
        self.mode = mode
        self.spatial_in = nn.Conv2d(spatial_dim, width, 1)
        self.summary_in = nn.Linear(summary_dim, width)
        self.blocks = nn.ModuleList([RefinementBlock(mode, width, heads, activation)
                                     for _ in range(blocks)])
        self.spatial_out = nn.Conv2d(width, spatial_dim, 1)
        self.summary_out = nn.Linear(width, summary_dim)
        for output in (self.spatial_out, self.summary_out):
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def forward(self, spatial, summary):
        value = self.spatial_in(spatial)
        b, c, h, w = value.shape
        tokens = value.flatten(2).transpose(1, 2)
        if self.mode == 'joint_attention':
            tokens = tokens + position_encoding(h, w, c, value.device, value.dtype)
        tokens = torch.cat((self.summary_in(summary).unsqueeze(1), tokens), 1)
        for block in self.blocks:
            tokens = block(tokens, h, w)
        correction = self.spatial_out(tokens[:, 1:].transpose(1, 2).reshape(b, c, h, w))
        return spatial + correction, summary + self.summary_out(tokens[:, 0])
