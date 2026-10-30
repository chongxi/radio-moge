"""Experimental, tensor-only export boundaries for the two frozen encoders.

TensorRT is imported only when explicitly compiling. The ordinary trainer and
its precision/preprocessing policy are unchanged.
"""

from contextlib import contextmanager, nullcontext

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


@contextmanager
def explicit_output_casts():
    """Work around the 2.8 output setter's invalid strongly typed dtype override.

    Torch-TensorRT 2.8's TRTInterpreter.output assigns ITensor.dtype even when
    the inferred type differs in a strongly typed network. TensorRT requires
    an actual Cast layer instead. Keep the exported output contract, record
    every inserted cast, and restore the interpreter after this serial probe.
    No installed dependency files are changed.
    """
    import tensorrt as trt
    import torch_tensorrt
    from torch_tensorrt.dynamo.conversion._TRTInterpreter import TRTInterpreter

    if torch_tensorrt.__version__.split('+')[0] != '2.8.0':
        raise RuntimeError('Recheck the explicit output cast workaround for this Torch-TensorRT version')
    original = TRTInterpreter.output
    casts = []

    def output(interpreter, target, args, kwargs):
        if interpreter.output_dtypes is not None:
            values = args[0]
            values = list(values) if isinstance(values, (list, tuple)) else [values]
            if len(values) != len(interpreter.output_dtypes):
                raise RuntimeError('TensorRT output dtype count differs from the exported graph')
            for i, (value, desired) in enumerate(zip(values, interpreter.output_dtypes, strict=True)):
                wanted = desired.to(trt.DataType, use_default=True)
                if isinstance(value, trt.ITensor) and value.dtype != wanted:
                    casts.append(dict(name=value.name, source=str(value.dtype), target=str(wanted)))
                    values[i] = interpreter.ctx.net.add_cast(value, wanted).get_output(0)
            args = (tuple(values),)
        return original(interpreter, target, args, kwargs)

    TRTInterpreter.output = output
    try:
        yield casts
    finally:
        TRTInterpreter.output = original


def compile_encoder(module, shape, cache_directory, min_batch=4, opt_batch=64, max_batch=128):
    """Fixed spatial bucket, dynamic batch; preserve mixed precision explicitly."""
    import torch_tensorrt

    if len(shape) != 3 or not 1 <= min_batch <= opt_batch <= max_batch:
        raise ValueError('Invalid NCHW shape/profile')
    with explicit_output_casts() as casts:
        compiled = torch_tensorrt.compile(
            module.eval(), ir='dynamo',
            inputs=[torch_tensorrt.Input(min_shape=(min_batch, *shape),
                                        opt_shape=(opt_batch, *shape),
                                        max_shape=(max_batch, *shape), dtype=torch.float32)],
            # Explicit typing follows the exported autocast nodes. Torch-TensorRT 2.8
            # forbids enabled_precisions together with this option.
            use_explicit_typing=True,
            disable_tf32=True, min_block_size=5, optimization_level=1,
            # The recorded dynamic-batch tactics require 5.3-5.6 GiB. A 4 GiB
            # workspace rejects them before any inference can be measured.
            workspace_size=16 * 1024**3, max_aux_streams=0,
            cache_built_engines=True, reuse_cached_engines=True,
            engine_cache_dir=str(cache_directory), engine_cache_size=16 * 1024**3,
            timing_cache_path=str(cache_directory / 'timing.bin'),
            # The teacher's antialiased resize must not silently become plain bilinear.
            torch_executed_ops={torch.ops.aten._upsample_bilinear2d_aa.default},
        )
    compiled.explicit_output_casts = casts
    return compiled
