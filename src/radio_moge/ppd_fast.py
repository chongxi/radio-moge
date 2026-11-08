# SPDX-License-Identifier: Apache-2.0
# Adapted from gangweix/pixel-perfect-depth, commit
# 427a86f5882aa3f7233c1b94fbdccce87f1c8313 (ppd/models/dit.py and rope.py).
# See third_party/licenses/pixel-perfect-depth-APACHE-2.0.txt.
# Modifications: cache fixed-grid/timestep operations and add CUDA graph replay.
"""Fixed-grid, frozen-weight execution of the released PPD sampling operations.

Caches position-dependent RoPE tensors and timestep embeddings, and supplies the
known next timestep without dynamic nonzero/index operations. No weights or
arithmetic expressions are changed. Callers must validate numerical parity for
their device, precision, input shape and checkpoint before using this path.
"""

import torch
from torch import nn
import torch.nn.functional as F


class CachedGridRoPE(nn.Module):
    def __init__(self, original, positions, head_dim, height, width, dtype):
        super().__init__()
        self.tokens = height * width
        self.head_dim = head_dim
        cos, sin = original._compute_frequency_components(head_dim // 2, max(height, width), positions.device, dtype)
        for name, table, axis in (("cy", cos, 0), ("sy", sin, 0), ("cx", cos, 1), ("sx", sin, 1)):
            self.register_buffer(name, F.embedding(positions[..., axis], table)[:, None], persistent=False)

    @staticmethod
    def rotate(x):
        a, b = x.chunk(2, dim=-1)
        return torch.cat((-b, a), -1)

    def forward(self, tokens, positions=None):
        assert tokens.shape[0] == 1 and tokens.shape[-2:] == (self.tokens, self.head_dim)
        vertical, horizontal = tokens.chunk(2, dim=-1)
        vertical = vertical * self.cy + self.rotate(vertical) * self.sy
        horizontal = horizontal * self.cx + self.rotate(horizontal) * self.sx
        return torch.cat((vertical, horizontal), -1)


class StaticPPDSampler(nn.Module):
    def __init__(self, dit, height=480, width=640, steps=4, dtype=torch.bfloat16):
        super().__init__()
        from ppd.utils.diffusion.schedule import LinearSchedule
        from ppd.utils.diffusion.timesteps import Timesteps
        from ppd.utils.diffusion.sampler import EulerSampler

        assert height % 16 == width % 16 == 0 and steps in (1, 2, 4)
        self.dit, self.height, self.width, self.steps = dit, height, width, steps
        device = next(dit.parameters()).device
        timesteps = Timesteps(T=1000, steps=steps, device=device)
        self.sampler = EulerSampler(LinearSchedule(T=1000), timesteps, prediction_type="velocity")
        low = dit.position_getter(1, height // 16, width // 16, device=device)
        high = dit.position_getter(1, height // 8, width // 8, device=device)
        self.register_buffer("position_low", low, persistent=False)
        self.register_buffer("position_high", high, persistent=False)
        original = dit.rope
        attention = dit.blocks[0].attn
        # CUDA autocast runs PPD's Q/K LayerNorm in FP32 even when the linear
        # output is BF16. Cache the trigonometric tables at that actual dtype.
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            q = torch.zeros(1, attention.num_heads, 1, attention.head_dim, dtype=dtype, device=device)
            rope_dtype = attention.q_norm(q).dtype
        rope_low = CachedGridRoPE(original, low, attention.head_dim, height // 16, width // 16, rope_dtype)
        rope_high = CachedGridRoPE(original, high, attention.head_dim, height // 8, width // 8, rope_dtype)
        # Each captured graph must retain the buffers it referenced even if a
        # second sampler shares this DiT and installs equivalent RoPE modules.
        self.rope_low, self.rope_high = rope_low, rope_high
        for i, block in enumerate(dit.blocks):
            block.attn.rope = rope_low if i < len(dit.blocks) // 2 else rope_high
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            for i, t in enumerate(timesteps):
                next_t = timesteps[i + 1] if i + 1 < steps else torch.full_like(t, -1)
                self.register_buffer(f"t_{i}", t.clone(), persistent=False)
                self.register_buffer(f"next_t_{i}", next_t.clone(), persistent=False)
                self.register_buffer(f"embedding_{i}", dit.t_embedder(t[None]).clone(), persistent=False)
        self.eval().requires_grad_(False)

    def velocity(self, x, semantics, embedding):
        dit, height, width = self.dit, self.height, self.width
        x = dit.x_embedder(x)
        batch, _, dim = x.shape
        for i, block in enumerate(dit.blocks):
            x = block(x, embedding, self.position_low if i < dit.depth // 2 else self.position_high)
            if i == dit.depth // 2 - 1:
                semantics = F.normalize(semantics, dim=-1)
                x = dit.proj_fusion(torch.cat([x, semantics], dim=-1))
                patch = dit.patch_size * 2
                x = x.reshape(batch, height // patch, width // patch, 2, 2, dim)
                x = torch.einsum("nhwpqc->nchpwq", x)
                x = x.reshape(batch, dim, height // patch * 2, width // patch * 2)
                x = x.flatten(2).transpose(1, 2)
        return dit.unpatchify(dit.final_layer(x, embedding), height=height, width=width)

    def forward(self, image, semantics, noise):
        assert image.shape == (1, 3, self.height, self.width)
        assert semantics.shape == (1, self.height // 16 * (self.width // 16), 1024)
        assert noise.shape == (1, 1, self.height, self.width)
        latent, conditioning = noise.clone(), image - .5
        for i in range(self.steps):
            velocity = self.velocity(torch.cat((latent, conditioning), 1), semantics, getattr(self, f"embedding_{i}"))
            latent = self.sampler.step_to(velocity, latent, getattr(self, f"t_{i}"), getattr(self, f"next_t_{i}"))
        return latent + .5


class PPDGraph:
    """Single-request, fixed-shape CUDA graph with owned input/output buffers.

    The returned tensor is overwritten by the next replay; clone/copy it before
    another invocation if it must persist. This object is not thread safe.
    """

    def __init__(self, sampler, image, semantics, noise, dtype=torch.bfloat16):
        self.inputs = [x.clone() for x in (image, semantics, noise)]
        self.sampler = sampler
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(3):
                with torch.autocast("cuda", dtype=dtype):
                    self.output = sampler(*self.inputs)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            self.output = sampler(*self.inputs)
        self.graph.replay()
        torch.cuda.synchronize()

    def __call__(self, image, semantics, noise):
        for static, dynamic in zip(self.inputs, (image, semantics, noise)):
            assert static.shape == dynamic.shape and static.dtype == dynamic.dtype
            static.copy_(dynamic)
        self.graph.replay()
        return self.output
