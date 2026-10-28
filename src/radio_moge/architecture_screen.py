"""Scratch adapter candidates. These are task-specific hypotheses, not paper replicas."""

import math

import torch
from torch import nn
import torch.nn.functional as F

from .adapters import EncoderFeatures, GeometryAdapter, zero_dominant_channels
from .spatial_heads import AttentionBlock, position_encoding


def candidates():
    base = dict(
        family="joint",
        width=512,
        blocks=4,
        heads=8,
        position="absolute",
        gate=False,
        registers=0,
        activation="gelu",
        local=False,
        layer_scale=0.01,
    )
    rows = [
        dict(name="legacy_2x1280", family="legacy", layer_scale=1e-5),
        dict(name="legacy_2x1280_ls001", family="legacy", layer_scale=0.01),
        dict(name="fusion_mlp", family="fusion"),
    ]
    for name, changes in [
        ("joint_4x512", {}),
        ("joint_2x512", dict(blocks=2)),
        ("joint_8x512", dict(blocks=8)),
        ("joint_4x768", dict(width=768, heads=12)),
        ("joint_4x1024", dict(width=1024, heads=16)),
        ("joint_rope", dict(position="rope")),
        ("joint_gate", dict(gate=True)),
        ("joint_register4", dict(registers=4)),
        ("joint_swiglu", dict(activation="swiglu")),
        ("joint_local", dict(local=True)),
        ("joint_combined", dict(position="rope", gate=True, registers=4, activation="swiglu")),
        ("conditional_direct", dict(family="conditional_direct")),
        ("conditional_flow", dict(family="conditional_flow")),
    ]:
        rows.append(dict(base, **{"name": name, **changes}))
    return rows


def rope2d(x, h, w, global_tokens):
    """Axial RoPE on spatial Q/K only; summary/register tokens use identity rotation."""
    d = x.shape[-1]
    if d % 4 or x.shape[-2] != h * w + global_tokens:
        raise ValueError("2D RoPE requires head width divisible by four and exact grid")
    freq = 10000 ** (-torch.arange(0, d // 2, 2, device=x.device).float() / (d // 2))
    yy, xx = torch.meshgrid(
        torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij"
    )
    angles = torch.cat((xx.flatten()[:, None] * freq, yy.flatten()[:, None] * freq), -1)
    if global_tokens:
        angles = F.pad(angles, (0, 0, 0, global_tokens))
    angles = angles.repeat_interleave(2, -1).to(x.dtype)
    pair = x.reshape(*x.shape[:-1], -1, 2)
    rotated = torch.stack((-pair[..., 1], pair[..., 0]), -1).flatten(-2)
    return x * angles.cos() + rotated * angles.sin()


class ScreenBlock(nn.Module):
    def __init__(self, width, heads, position, gate, activation, layer_scale, qk_norm=False):
        super().__init__()
        if width % heads or width // heads % 4:
            raise ValueError("Head dimension must be divisible by four")
        self.heads, self.position, self.activation = heads, position, activation
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.projection = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.gate = nn.Linear(width, heads) if gate else None
        hidden = 4 * width if activation == "gelu" else 8 * round(width / 3)
        self.fc1 = nn.Linear(width, hidden * (2 if activation == "swiglu" else 1))
        self.fc2 = nn.Linear(hidden, width)
        self.scale1 = nn.Parameter(torch.full((width,), layer_scale))
        self.scale2 = nn.Parameter(torch.full((width,), layer_scale))
        # Normalize each projected head before RoPE; output feature amplitudes remain free.
        self.q_norm = nn.LayerNorm(width // heads) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(width // heads) if qk_norm else nn.Identity()

    def forward(self, x, h, w, globals_count):
        b, n, c = x.shape
        norm = self.norm1(x)
        q, k, v = (
            self.qkv(norm)
            .reshape(b, n, 3, self.heads, c // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        q, k = self.q_norm(q), self.k_norm(k)
        if self.position == "rope":
            q, k = rope2d(q, h, w, globals_count), rope2d(k, h, w, globals_count)
        attn = F.scaled_dot_product_attention(q, k, v)
        if self.gate is not None:
            attn = attn * self.gate(norm).sigmoid().transpose(1, 2).unsqueeze(-1)
        x = x + self.scale1 * self.projection(attn.transpose(1, 2).reshape(b, n, c))
        value = self.fc1(self.norm2(x))
        if self.activation == "swiglu":
            value, gate = value.chunk(2, -1)
            value = value * F.silu(gate)
        else:
            value = F.gelu(value)
        return x + self.scale2 * self.fc2(value)


class SummaryAttentionReadout(nn.Module):
    """One extra attention-only residual update to summary, with all tokens as memory.

    The spatial and register tokens are not updated. This is a CaiT-inspired
    readout, not a reproduction of the full CaiT architecture. No extra FFN.
    """

    def __init__(self, width, heads, layer_scale):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.gate = nn.Linear(width, heads)
        self.scale = nn.Parameter(torch.full((width,), layer_scale))

    def forward(self, tokens, h, w, globals_count):
        b, n, c = tokens.shape
        norm = self.norm(tokens)
        q, k, v = self.qkv(norm).reshape(b, n, 3, self.heads, c // self.heads).permute(
            2, 0, 3, 1, 4
        ).unbind(0)
        summary_index = h * w
        q = q[:, :, summary_index : summary_index + 1]
        k = rope2d(k, h, w, globals_count)
        update = F.scaled_dot_product_attention(q, k, v)
        gate = self.gate(norm[:, summary_index]).sigmoid()[:, :, None, None]
        update = (update * gate).transpose(1, 2).reshape(b, c)
        return tokens[:, summary_index] + self.scale * self.projection(update)


class JointScreenAdapter(nn.Module):
    def __init__(
        self,
        spec,
        spatial_dim=1280,
        summary_dim=2560,
        target_dim=1024,
        target_summary_dim=1024,
        taps=4,
    ):
        super().__init__()
        self.spec, self.target_dim, self.target_summary_dim = spec, target_dim, target_summary_dim
        width = spec["width"]
        self.input = nn.ModuleList(nn.Conv2d(spatial_dim, width, 1) for _ in range(taps))
        self.tap_logits = nn.Parameter(torch.zeros(taps))
        self.summary_input = nn.Linear(summary_dim, width)
        self.registers = nn.Parameter(torch.randn(spec["registers"], width) * 0.02)
        self.local = nn.Conv2d(width, width, 3, padding=1, groups=width) if spec["local"] else None
        self.blocks = nn.ModuleList(
            ScreenBlock(
                width,
                spec["heads"],
                spec["position"],
                spec["gate"],
                spec["activation"],
                spec["layer_scale"],
                spec.get("qk_norm", False),
            )
            for _ in range(spec["blocks"])
        )
        self.spatial_output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, target_dim))
        self.summary_output = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, target_summary_dim)
        )
        self.conditional = spec["family"].startswith("conditional_")
        if self.conditional:
            self.state_spatial = nn.Conv2d(target_dim, width, 1)
            self.state_summary = nn.Linear(target_summary_dim, width)
            self.time = nn.Sequential(nn.Linear(64, width), nn.SiLU(), nn.Linear(width, width))
        # Construct last, preserving the seed-identical shared backbone/output initialization.
        self.summary_readout = (
            SummaryAttentionReadout(width, spec["heads"], spec["layer_scale"])
            if spec.get("summary_readout", False)
            else None
        )

    def forward(self, features, state=None, t=None):
        if len(features.intermediates) != len(self.input):
            raise ValueError("Exact same RADIO taps required")
        x = sum(
            p(zero_dominant_channels(f.float())) * weight
            for p, f, weight in zip(self.input, features.intermediates, self.tap_logits.softmax(0))
        )
        summary = self.summary_input(features.summary.float())
        b, c, h, w = x.shape
        if self.local is not None:
            x = x + 0.01 * self.local(x)
        if self.conditional:
            if state is None or t is None:
                raise ValueError("Conditional model needs state and time")
            freq = torch.exp(torch.arange(32, device=x.device).float() * (-math.log(10000) / 31))
            phase = t.float()[:, None] * 1000 * freq
            time = self.time(torch.cat((phase.sin(), phase.cos()), -1))
            x = x + self.state_spatial(state.spatial.float()) + time[:, :, None, None]
            summary = summary + self.state_summary(state.summary.float()) + time
        tokens = x.flatten(2).transpose(1, 2)
        if self.spec["position"] == "absolute":
            tokens = tokens + position_encoding(h, w, c, x.device, x.dtype)
        tokens = torch.cat((tokens, summary[:, None], self.registers[None].expand(b, -1, -1)), 1)
        for block in self.blocks:
            tokens = block(tokens, h, w, 1 + len(self.registers))
        summary = (
            self.summary_readout(tokens, h, w, 1 + len(self.registers))
            if self.summary_readout is not None
            else tokens[:, h * w]
        )
        return EncoderFeatures(
            self.spatial_output(tokens[:, : h * w]).transpose(1, 2).reshape(b, -1, h, w),
            self.summary_output(summary),
        )


def make_candidate(spec, dimensions=None):
    dims = dimensions or dict(
        spatial_dim=1280, summary_dim=2560, target_dim=1024, target_summary_dim=1024
    )
    if spec["family"] in ("legacy", "fusion"):
        model = GeometryAdapter(
            kind="attention" if spec["family"] == "legacy" else "fusion",
            channel_cleaning="zero-channel",
            **dims,
        )
        if spec["family"] == "legacy":
            for block in model.modules():
                if isinstance(block, AttentionBlock):
                    with torch.no_grad():
                        block.scale1.fill_(spec["layer_scale"])
                        block.scale2.fill_(spec["layer_scale"])
        return model
    return JointScreenAdapter(spec, **dims)


def normalized(value, stats):
    return EncoderFeatures(
        value.spatial.float() / stats.spatial_scale[None, :, None, None],
        value.summary.float() / stats.summary_scale[None],
    )


def unnormalized(value, stats):
    return EncoderFeatures(
        value.spatial.float() * stats.spatial_scale[None, :, None, None],
        value.summary.float() * stats.summary_scale[None],
    )


def flow_training(model, features, target, stats, noise, time):
    """Linear conditional flow path; teacher endpoint used only in training."""
    endpoint = normalized(target, stats)
    t_spatial, t_summary = time[:, None, None, None], time[:, None]
    state = EncoderFeatures(
        noise.spatial * (1 - t_spatial) + endpoint.spatial * t_spatial,
        noise.summary * (1 - t_summary) + endpoint.summary * t_summary,
    )
    velocity = model(features, state, time)
    losses = dict(
        spatial=F.mse_loss(velocity.spatial.float(), endpoint.spatial - noise.spatial),
        summary=F.mse_loss(velocity.summary.float(), endpoint.summary - noise.summary),
    )
    prediction = unnormalized(
        EncoderFeatures(
            state.spatial + (1 - t_spatial) * velocity.spatial,
            state.summary + (1 - t_summary) * velocity.summary,
        ),
        stats,
    )
    direction = stats.loss(prediction, target)
    return {**losses, "cosine": direction["cosine"], "summary_cosine": direction["summary_cosine"]}


def conditional_predict(model, features, stats, noise, evaluations=1):
    """Teacher-free Euler integration; identical fixed noise across evaluated models."""
    if evaluations not in (1, 2, 4):
        raise ValueError("Screen only supports 1/2/4 function evaluations")
    state = EncoderFeatures(noise.spatial.clone(), noise.summary.clone())
    for i in range(evaluations):
        t = torch.full((len(state.summary),), i / evaluations, device=state.summary.device)
        velocity = model(features, state, t)
        state = EncoderFeatures(
            state.spatial + velocity.spatial / evaluations,
            state.summary + velocity.summary / evaluations,
        )
    return unnormalized(state, stats)
