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


def fp32_attention_decompositions(decompositions):
    """Keep low-precision Q/K/V values, but do attention's internal math in FP32."""
    result = dict(decompositions)

    def promote(function):
        def attention(query, key, value, *args, **kwargs):
            output = function(query.float(), key.float(), value.float(), *args, **kwargs)
            if isinstance(output, tuple):
                return (output[0].to(query.dtype), *output[1:])
            return output.to(query.dtype)
        return attention

    for op, function in decompositions.items():
        if 'scaled_dot_product' in str(op):
            result[op] = promote(function)
    return result


def materialize_reduction_dtypes(module):
    """Make sum(dtype=...) explicit before reduction, not after low-precision sum.

    The 2.8 TensorRT sum converter ignores aten's dtype keyword. Casting only
    the output cannot recover precision lost in the reduction.
    """
    changed = []
    targets = {torch.ops.aten.sum.default: 'dtype', torch.ops.aten.sum.dim_IntList: 'dtype',
               torch.ops.prims.sum.default: 'output_dtype'}
    for node in list(module.graph.nodes):
        keyword = targets.get(node.target) if node.op == 'call_function' else None
        if keyword is None or node.kwargs.get(keyword) is None:
            continue
        dtype = node.kwargs[keyword]
        source = node.args[0]
        with module.graph.inserting_before(node):
            cast = module.graph.call_function(torch.ops.aten._to_copy.default,
                                               args=(source,), kwargs={'dtype': dtype})
            cast.meta = dict(source.meta)
            cast.meta['val'] = source.meta['val'].to(dtype)
            if 'tensor_meta' in cast.meta:
                cast.meta['tensor_meta'] = cast.meta['tensor_meta']._replace(dtype=dtype)
        node.args = (cast, *node.args[1:])
        node.kwargs = {k: v for k, v in node.kwargs.items() if k != keyword}
        changed.append(dict(name=node.name, input_dtype=str(source.meta['val'].dtype),
                            reduction_dtype=str(dtype)))
    module.graph.lint()
    module.recompile()
    return changed


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
    from torch_tensorrt.dynamo._compiler import (
        CompilationSettings, compile_module, get_decompositions, post_lowering, pre_export_lowering,
    )
    from torch_tensorrt.dynamo._engine_cache import DiskEngineCache

    if len(shape) != 3 or not 1 <= min_batch <= opt_batch <= max_batch:
        raise ValueError('Invalid NCHW shape/profile')
    if torch_tensorrt.__version__.split('+')[0] != '2.8.0':
        raise RuntimeError('Recheck the precision lowering for this Torch-TensorRT version')
    inputs = [torch_tensorrt.Input(min_shape=(min_batch, *shape), opt_shape=(opt_batch, *shape),
                                  max_shape=(max_batch, *shape), dtype=torch.float32)]
    settings = CompilationSettings(
        use_explicit_typing=True, disable_tf32=True, min_block_size=5, optimization_level=1,
        workspace_size=16 * 1024**3, max_aux_streams=0,
        cache_built_engines=True, reuse_cached_engines=True,
        timing_cache_path=str(cache_directory / 'timing.bin'),
        torch_executed_ops={torch.ops.aten._upsample_bilinear2d_aa.default},
    )
    # Match the 2.8 Dynamo compile pipeline, changing only the attention
    # decomposition and making reduction dtype semantics explicit.
    program = torch_tensorrt.dynamo.trace(module.eval(), inputs=inputs)
    program = pre_export_lowering(program, settings)
    program = program.run_decompositions(fp32_attention_decompositions(get_decompositions(False)))
    lowered = post_lowering(program.module(), settings)
    reductions = materialize_reduction_dtypes(lowered)
    cache = DiskEngineCache(str(cache_directory), 16 * 1024**3)
    with explicit_output_casts() as casts:
        compiled = compile_module(lowered, inputs, {}, settings, cache)
    compiled.explicit_output_casts = casts
    compiled.explicit_reduction_dtypes = reductions
    return compiled
