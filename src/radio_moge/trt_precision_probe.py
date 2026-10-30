"""Locate encoder drift at export, decomposition, or TensorRT conversion.

This diagnostic performs no TensorRT builds, timings, or training updates.
"""

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import time
import traceback

import torch

from .checkpoint import atomic_json
from .trt_encoder_probe import real_inputs
from .trt_encoders import MoGeTensorOutputs, RadioTensorOutputs, compare_outputs
from .upstream import load_moge, load_radio, radio_input


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


@torch.no_grad()
def main():
    import torch_tensorrt
    from torch_tensorrt.dynamo._compiler import (
        CompilationSettings, get_decompositions, post_lowering, pre_export_lowering,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference-report', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=360)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 600:
        raise ValueError('Precision diagnostic must remain bounded')
    destination = args.root / 'probe.json'
    if destination.exists():
        raise FileExistsError('Diagnostic already exists')
    report = dict(stage='loading', started_at=time.time(), rows=[], production_updates=0,
                  purpose='Separate export/decomposition drift from TensorRT conversion drift',
                  timings_performed=False, engine_builds_performed=False)
    atomic_json(report, destination)
    deadline = time.monotonic() + args.max_seconds
    try:
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        reference = json.loads(args.reference_report.read_text())
        config = reference['config']
        radio = load_radio(config['models'], 'cuda')
        moge, _ = load_moge(config['models'], 'cuda')
        encoder = moge.encoder
        del moge
        inputs = real_inputs(config, reference['source_cursor'], images_per_bucket=4)
        bucket, image = next(iter(inputs.items()))
        radio_image = radio_input(image, radio)
        radio_module = RadioTensorOutputs(radio, config['adapter']['indices']).eval()
        grid = tuple(radio_module(radio_image)[0].shape[-2:])
        report.update(stage='checking', bucket=list(bucket), batch=len(image),
                      source_cursor=reference['source_cursor'], revisions=reference['revisions'])
        for name, eager, value, names in (
            ('radio', radio_module, radio_image,
             ['spatial', 'summary', *[f'layer{i}' for i in config['adapter']['indices']]]),
            ('moge', MoGeTensorOutputs(encoder, grid).eval(), image, ['spatial', 'summary']),
        ):
            row = dict(component=name, stage='exporting', variants=[])
            report['rows'].append(row)
            atomic_json(report, destination)
            expected = eager(value)
            profile = [torch_tensorrt.Input(min_shape=(4, *value.shape[1:]),
                                           opt_shape=(64, *value.shape[1:]),
                                           max_shape=(128, *value.shape[1:]), dtype=torch.float32)]
            program = torch_tensorrt.dynamo.trace(eager, inputs=profile)
            exported = program.module()
            row['exported'] = compare_outputs(expected, exported(value), names)
            row['exported_ops'] = dict(Counter(str(n.target) for n in exported.graph.nodes
                                              if n.op == 'call_function'))
            settings = CompilationSettings(use_explicit_typing=True, disable_tf32=True)
            program = pre_export_lowering(program, settings)
            for label in ('default_decompositions', 'fp32_attention_decompositions'):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Diagnostic deadline reached')
                decompositions = get_decompositions(False)
                if label == 'fp32_attention_decompositions':
                    decompositions = fp32_attention_decompositions(decompositions)
                lowered = post_lowering(program.run_decompositions(decompositions).module(), settings)
                comparison = compare_outputs(expected, lowered(value), names)
                row['variants'].append(dict(name=label, comparison=comparison,
                                            ops=dict(Counter(str(n.target) for n in lowered.graph.nodes
                                                             if n.op == 'call_function'))))
                atomic_json(report, destination)
                del lowered
                gc.collect()
                torch.cuda.empty_cache()
            row['stage'] = 'complete'
            del expected, exported, program
            atomic_json(report, destination)
        report.update(stage='complete', finished_at=time.time())
        atomic_json(report, destination)
    except BaseException as error:
        report.update(stage='failed', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc(), finished_at=time.time())
        atomic_json(report, destination)
        raise


if __name__ == '__main__':
    main()
