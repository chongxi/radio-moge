"""Bounded TensorRT feasibility, feature parity and frozen-encoder timing probe."""

import argparse
from collections import defaultdict
from copy import deepcopy
import gc
import importlib.metadata
import json
from pathlib import Path
import time
import traceback

import torch

from .batch_benchmark import read_manifest_metadata
from .checkpoint import atomic_json
from .train import make_loader
from .trt_encoders import MoGeTensorOutputs, RadioTensorOutputs, compare_outputs, compile_encoder
from .upstream import load_moge, load_radio, radio_input


def real_inputs(config, cursor, images_per_bucket=64):
    cfg = deepcopy(config)
    cfg['data'].update(workers=4, compact_transport=True)
    records = read_manifest_metadata(cfg['data']['train_manifest'])
    loader = make_loader(records, cfg, cursor, 128)
    iterator = iter(loader)
    collected, counts = defaultdict(list), defaultdict(int)
    try:
        for image in iterator:
            bucket = tuple(image.shape[-2:])
            if counts[bucket] < images_per_bucket:
                collected[bucket].append(image)
                counts[bucket] += len(image)
            if all(counts[tuple(b)] >= images_per_bucket for b in cfg['data']['buckets']):
                break
    finally:
        iterator._shutdown_workers()
    if any(counts[tuple(b)] < images_per_bucket for b in cfg['data']['buckets']):
        raise RuntimeError('Not enough real images for both spatial buckets')
    return {k: torch.cat(v)[:images_per_bucket].cuda() for k, v in collected.items()}


@torch.no_grad()
def measure(module, image, deadline, repeats=6):
    for _ in range(2):
        output = module(image)
        del output
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        if time.monotonic() >= deadline:
            raise TimeoutError('Probe deadline reached')
        start = time.perf_counter()
        output = module(image)
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
        del output
    return dict(seconds=samples, mean_ms=sum(samples) * 1000 / repeats,
                images_per_second=len(image) * repeats / sum(samples))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference-report', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=2400)
    parser.add_argument('--reuse-engine-cache-from', type=Path)
    parser.add_argument('--precision-policy', choices=('fp32_attention', 'torch_sensitive'),
                        default='fp32_attention')
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 3600:
        raise ValueError('Probe must remain bounded')
    if args.reuse_engine_cache_from is not None:
        previous = json.loads((args.reuse_engine_cache_from / 'controller.json').read_text())
        if (previous.get('stage') not in ('finished', 'failed')
                or previous.get('child_pid') is not None or 'returncode' not in previous):
            raise RuntimeError('Engine cache source attempt has not finished')
    report_path = args.root / 'probe.json'
    if report_path.exists():
        raise FileExistsError('Probe already started; inspect existing results')
    reference = json.loads(args.reference_report.read_text())
    config = reference['config']
    report = dict(stage='loading', started_at=time.time(), rows=[],
                  production_updates=0, source_step=reference['source_step'],
                  source_cursor=reference['source_cursor'], revisions=reference['revisions'],
                  policy='Frozen encoders only; no adapter/Adam/training checkpoint loads or writes. '
                         'Real augmented images remain on GPU before timing. '
                         'BF16 eager reference and original antialiased preprocessing retained.',
                  versions={n: importlib.metadata.version(n) for n in
                            ('torch', 'torch-tensorrt', 'tensorrt')},
                  input_profiles=dict(min_batch=4, opt_batch=64, max_batch=128,
                                      buckets=config['data']['buckets']),
                  cold_compilation_excluded_from_timings=True, precision_policy=args.precision_policy,
                  readonly_cache_source=str(args.reuse_engine_cache_from)
                  if args.reuse_engine_cache_from is not None else None)
    atomic_json(report, report_path)
    deadline = time.monotonic() + args.max_seconds
    try:
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.85)
        torch.backends.cuda.matmul.allow_tf32 = False
        radio = load_radio(config['models'], 'cuda')
        moge, _ = load_moge(config['models'], 'cuda')
        encoder = moge.encoder
        del moge
        inputs = real_inputs(config, reference['source_cursor'])
        report.update(stage='compiling', inputs_preloaded=True,
                      input_images={str(k): len(v) for k, v in inputs.items()},
                      gpu=torch.cuda.get_device_name())
        atomic_json(report, report_path)
        for bucket, image in inputs.items():
            with torch.no_grad():
                radio_image = radio_input(image, radio)
                radio_module = RadioTensorOutputs(radio, config['adapter']['indices']).eval()
                features = radio_module(radio_image[:4])
                grid = tuple(features[0].shape[-2:])
                del features
            components = [
                ('radio', radio_module, radio_image,
                 ['spatial', 'summary', *[f'layer{i}' for i in config['adapter']['indices']]]),
                ('moge', MoGeTensorOutputs(encoder, grid).eval(), image, ['spatial', 'summary']),
            ]
            for name, eager, value, names in components:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Probe deadline reached')
                cache = args.root / 'engines' / f'{name}-{bucket[0]}x{bucket[1]}'
                previous_cache = (args.reuse_engine_cache_from / 'engines' / cache.name
                                  if args.reuse_engine_cache_from is not None else None)
                cache.mkdir(parents=True, exist_ok=True)
                row = dict(component=name, bucket=list(bucket), stage='compiling',
                           started_at=time.time(), grid=list(grid), outputs=names, comparisons=[], timings=[])
                report['rows'].append(row)
                atomic_json(report, report_path)
                compiled = None
                try:
                    start = time.perf_counter()
                    with torch.no_grad():
                        compiled = compile_encoder(eager, tuple(value.shape[1:]), cache,
                                                   precision_policy=args.precision_policy,
                                                   example_input=value[:4], output_names=names,
                                                   readonly_cache_directory=previous_cache)
                    torch.cuda.synchronize()
                    row['compile_seconds'] = time.perf_counter() - start
                    row['workspace_limit_bytes'] = 16 * 1024**3
                    row['explicit_output_casts'] = compiled.explicit_output_casts
                    row['explicit_reduction_dtypes'] = compiled.explicit_reduction_dtypes
                    row['precision_policy'] = args.precision_policy
                    row['lowered_feature_gate'] = compiled.lowered_feature_gate
                    row['preserved_torch_ops'] = compiled.preserved_torch_ops
                    row['runtime_backend'] = compiled.runtime_backend
                    row['retained_shape_input_bindings'] = compiled.retained_shape_input_bindings
                    row['cache_reuse'] = compiled.cache_reuse
                    row['compiled_modules'] = [dict(name=n, type=type(m).__name__)
                                               for n, m in compiled.named_modules()]
                    row['tensor_rt_partitions'] = sum('tensorrt' in type(m).__module__.lower()
                                                    and hasattr(m, 'engine')
                                                    for m in compiled.modules())
                    row['compiled_graph'] = str(compiled.graph)
                    row['fallback_graphs'] = {n: str(m.graph) for n, m in compiled.named_modules()
                                              if n and isinstance(m, torch.fx.GraphModule)}
                    if not row['tensor_rt_partitions']:
                        raise RuntimeError('No TensorRT engine partitions found; refusing to label fallback as TRT')
                    # A hybrid graph is reported as hybrid, never as full TRT conversion.
                    row['stage'] = 'checking_outputs'
                    atomic_json(report, report_path)
                    for batch in (4, 16, 64):
                        with torch.no_grad():
                            expected = eager(value[:batch])
                            actual = compiled(value[:batch])
                            comparison = compare_outputs(expected, actual, names)
                            del expected, actual
                        row['comparisons'].append(dict(batch=batch, **comparison))
                        atomic_json(report, report_path)
                    if not all(c['passed'] for c in row['comparisons']):
                        row['stage'] = 'feature_gate_failed'
                    else:
                        row['stage'] = 'timing'
                        atomic_json(report, report_path)
                        for batch in (16, 64):
                            for order in (('eager', 'trt'), ('trt', 'eager')):
                                for mode in order:
                                    result = measure(eager if mode == 'eager' else compiled,
                                                     value[:batch], deadline)
                                    row['timings'].append(dict(batch=batch, mode=mode, **result))
                        row['stage'] = 'passed'
                except Exception as error:
                    row.update(stage='failed', error=f'{type(error).__name__}: {error}',
                               traceback=traceback.format_exc())
                finally:
                    row['finished_at'] = time.time()
                    atomic_json(report, report_path)
                    del compiled
                    gc.collect()
                    torch.cuda.empty_cache()
        report.update(stage='passed' if all(r['stage'] == 'passed' for r in report['rows'])
                      else 'completed_with_failures', finished_at=time.time())
        atomic_json(report, report_path)
    except BaseException as error:
        report.update(stage='failed', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc(), finished_at=time.time())
        atomic_json(report, report_path)
        raise


if __name__ == '__main__':
    main()
