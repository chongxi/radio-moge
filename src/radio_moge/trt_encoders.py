"""Experimental, tensor-only export boundaries for the two frozen encoders.

TensorRT is imported only when explicitly compiling. The ordinary trainer and
its precision/preprocessing policy are unchanged.
"""

from contextlib import nullcontext

import torch
from torch import nn
import torch.nn.functional as F

from .upstream import extract_radio


class RadioTensorOutputs(nn.Module):
    """Input has already passed through the unchanged radio_input resize."""

    def __init__(self, radio, indices):
        super().__init__()
        if any(p.requires_grad for p in radio.parameters()):
            raise ValueError('TensorRT export is restricted to frozen encoders')
        self.radio = radio
        self.indices = tuple(indices)

    def forward(self, image):
        amp = (torch.autocast('cuda', dtype=torch.bfloat16)
               if image.device.type == 'cuda' else nullcontext())
        with amp:
            features = extract_radio(self.radio, image, self.indices)
        return (features.spatial, features.summary, *features.intermediates)


class MoGeTensorOutputs(nn.Module):
    """Keep upstream antialiased resize, normalization and output projections."""

    def __init__(self, encoder, grid):
        super().__init__()
        if any(p.requires_grad for p in encoder.parameters()):
            raise ValueError('TensorRT export is restricted to frozen encoders')
        self.encoder = encoder
        self.grid = tuple(grid)

    def forward(self, image):
        amp = (torch.autocast('cuda', dtype=torch.bfloat16)
               if image.device.type == 'cuda' else nullcontext())
        with amp:
            spatial, summary = self.encoder(image, *self.grid, return_class_token=True)
        return spatial.float(), summary.float()


@torch.no_grad()
def compare_outputs(reference, candidate, names, relative_l2_limit=.01, cosine_limit=.999):
    """Measure every required output; thresholds are a prototype gate, not an eval."""
    if not len(reference) == len(candidate) == len(names):
        raise ValueError('Encoder output count changed')
    result = []
    for name, expected, actual in zip(names, reference, candidate, strict=True):
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise ValueError(f'Encoder output shape/dtype changed: {name}')
        finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
        if not finite:
            result.append(dict(name=name, finite=False, passed=False))
            continue
        a, b = expected.float(), actual.float()
        difference = a - b
        relative_l2 = (difference.square().sum() / a.square().sum().clamp_min(1e-30)).sqrt()
        # Each spatial token is one vector; summary outputs are one vector/image.
        cosine = F.cosine_similarity(a, b, dim=1).mean()
        row = dict(name=name, shape=list(a.shape), dtype=str(expected.dtype), finite=True,
                   relative_l2=float(relative_l2), mean_cosine=float(cosine),
                   max_abs=float(difference.abs().max()))
        row['passed'] = row['relative_l2'] <= relative_l2_limit and row['mean_cosine'] >= cosine_limit
        result.append(row)
    return dict(passed=all(r['passed'] for r in result), outputs=result,
                thresholds=dict(relative_l2=relative_l2_limit, mean_cosine=cosine_limit))


def compile_encoder(module, shape, cache_directory, min_batch=4, opt_batch=64, max_batch=128):
    """Fixed spatial bucket, dynamic batch; preserve mixed precision explicitly."""
    import torch_tensorrt

    if len(shape) != 3 or not 1 <= min_batch <= opt_batch <= max_batch:
        raise ValueError('Invalid NCHW shape/profile')
    return torch_tensorrt.compile(
        module.eval(), ir='dynamo',
        inputs=[torch_tensorrt.Input(min_shape=(min_batch, *shape),
                                    opt_shape=(opt_batch, *shape),
                                    max_shape=(max_batch, *shape), dtype=torch.float32)],
        enabled_precisions={torch.float32, torch.bfloat16}, use_explicit_typing=True,
        disable_tf32=True, min_block_size=5, optimization_level=1,
        workspace_size=4 * 1024**3, max_aux_streams=0,
        cache_built_engines=True, reuse_cached_engines=True,
        engine_cache_dir=str(cache_directory), engine_cache_size=16 * 1024**3,
        timing_cache_path=str(cache_directory / 'timing.bin'),
        # The teacher's antialiased resize must not silently become plain bilinear.
        torch_executed_ops={torch.ops.aten._upsample_bilinear2d_aa.default},
    )
