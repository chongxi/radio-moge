"""Isolated prepared-pipeline batch sweep. Never resumes or saves training weights."""

import argparse
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
import gc
import json
import math
import os
from pathlib import Path
import time

import torch

from .checkpoint import atomic_json, restore_rng
from .losses import FEATURE_LOSS_KEYS, FeatureStatistics, weighted_loss
from .prefetch import PreparedWindows


def validate_plan(plan, logical):
    profiles = plan['profiles']
    if not 1 <= len(profiles) <= 8 or plan['rounds'] != 2:
        raise ValueError('Expected 1–8 profiles and two counterbalanced rounds')
    if not 0 < plan['max_seconds'] <= 1200:
        raise ValueError('Benchmark must have a deadline of at most 1200 seconds')
    if not 0 < plan['warmup_images'] <= 512 or not 256 <= plan['measured_images'] <= 4096:
        raise ValueError('Unbounded image budget')
    if len({(p['effective_batch'], p['compute_batch'], p['workers'], p.get('input_mode', 'prepared'))
            for p in profiles}) != len(profiles):
        raise ValueError('Duplicate profiles')
    for p in profiles:
        effective, compute = p['effective_batch'], p['compute_batch']
        if not (logical <= compute <= effective <= 256 and p['workers'] in range(5)):
            raise ValueError('Invalid batch or worker limit')
        if effective % logical or compute % logical:
            raise ValueError('Logical batches must divide effective and compute batches')
        if any(plan[key] % effective for key in ('warmup_images', 'measured_images')):
            raise ValueError('All profiles must consume identical warmup and measured image counts')
        if p.get('input_mode', 'prepared') not in ('prepared', 'resident_gpu'):
            raise ValueError('Unknown input timing mode')
        if p.get('input_mode') == 'resident_gpu' and plan['warmup_images'] + plan['measured_images'] > 1536:
            raise ValueError('Resident control caches at most 1536 actual images')


def reset_state(adapter, statistics, optimizer, state):
    optimizer.zero_grad(set_to_none=True)
    adapter.load_state_dict(state['adapter'])
    statistics.load_state_dict(state['statistics'])
    optimizer.load_state_dict(deepcopy(state['optimizer']))
    restore_rng(state['rng'])


def run_profile(system, adapter, statistics, optimizer, accelerator, records, config,
                state, profile, plan, deadline):
    """Real decode/augmentation, frozen encoders, backward, clipping and Adam are timed."""
    from .train import make_loader

    cuda = accelerator.device.type == 'cuda'
    amp = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if cuda else nullcontext
    synchronize = torch.cuda.synchronize if cuda else lambda: None
    cfg = deepcopy(config)
    cfg['data'].update(workers=profile['workers'], compact_transport=True)
    logical = cfg['train']['batch_size']
    effective = profile['effective_batch']
    accumulation = effective // logical
    warmup = plan['warmup_images'] // effective
    count = warmup + plan['measured_images'] // effective
    loader = make_loader(records, cfg, state['cursor'], count * accumulation)
    stream = PreparedWindows(loader, accumulation, profile['compute_batch'], count,
                             pin=cuda, measure=True)
    steps, histogram = [], Counter()
    resident = profile.get('input_mode') == 'resident_gpu'
    if resident and not cuda:
        stream.close()
        raise ValueError('Resident GPU control requires CUDA')
    cached, cache_seconds, cache_bytes = [], 0.0, 0
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    adapter.train()

    def mark():
        if cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return None

    try:
        if resident:
            cache_start = time.perf_counter()
            for _ in range(count):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Resident input preparation deadline reached')
                tensors = [image.to(accelerator.device, non_blocking=True) for image in next(stream)]
                cache_bytes += sum(t.numel() * t.element_size() for t in tensors)
                cached.append(tensors)
            synchronize()
            cache_seconds = time.perf_counter() - cache_start
            stream.close()
            torch.cuda.reset_peak_memory_stats()
        for index in range(count):
            if time.monotonic() >= deadline:
                raise TimeoutError('Batch benchmark deadline reached')
            synchronize()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            fetch_start = time.perf_counter()
            images = cached[index] if resident else next(stream)
            wait = time.perf_counter() - fetch_start
            preparation = stream.preparation_timings[index]
            assert sum(len(x) for x in images) == effective
            values, timings = [], []
            for cpu_image in images:
                begin = mark()
                image = cpu_image.to(accelerator.device, non_blocking=True)
                transfer = mark()
                with amp(), torch.no_grad():
                    features, target = system.encode(image, adapter.indices)
                encoded = mark()
                with amp():
                    prediction = adapter(features)
                forwarded = mark()
                losses = statistics.loss(prediction, target)
                loss = weighted_loss(losses, cfg['loss'])
                accelerator.backward(loss * (len(cpu_image) / effective))
                backward = mark()
                values.append(loss.detach() * (len(cpu_image) / effective))
                timings.append([begin, transfer, encoded, forwarded, backward])
                if index >= warmup:
                    histogram[f'{len(cpu_image)}x{image.shape[-2]}x{image.shape[-1]}'] += 1
                del image, features, target, prediction, losses, loss
            clipped = mark()
            norm = accelerator.clip_grad_norm_(adapter.parameters(), cfg['train']['clip_grad'])
            loss_value, norm_value = torch.stack([torch.stack(values).sum(), norm]).float().cpu().tolist()
            if not all(math.isfinite(v) for v in (loss_value, norm_value)) or norm_value <= 0:
                raise FloatingPointError('Nonfinite loss or nonpositive gradient norm; update cancelled')
            optimizer.step()
            updated = mark()
            synchronize()
            elapsed = time.perf_counter() - start
            if index >= warmup:
                gpu = {}
                if cuda:
                    for i, key in enumerate(('h2d_ms', 'frozen_encoders_ms',
                                             'adapter_forward_ms', 'loss_backward_ms')):
                        gpu[key] = sum(events[i].elapsed_time(events[i + 1]) for events in timings)
                    gpu['clip_check_adam_ms'] = clipped.elapsed_time(updated)
                steps.append(dict(seconds=elapsed, fetch_seconds=wait, loss=loss_value,
                                  gradient_norm=norm_value, **preparation, **gpu))
            del images, values, timings, norm
    finally:
        stream.close()
        del stream, loader
        gc.collect()
    seconds = sum(s['seconds'] for s in steps)
    return dict(stage='passed', **profile, steps=steps,
                measured_images=plan['measured_images'], warmup_images=plan['warmup_images'],
                images_per_second=plan['measured_images'] / seconds,
                actual_compute_batches=dict(histogram),
                mean_fetch_seconds=sum(s['fetch_seconds'] for s in steps) / len(steps),
                resident_setup_seconds=cache_seconds, resident_input_bytes=cache_bytes,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30 if cuda else 0,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30 if cuda else 0,
                disposable_updates=count,
                end_cursor=state['cursor'] + (plan['warmup_images'] + plan['measured_images']) // logical)


def aggregate(rows, profiles):
    results = []
    for p in profiles:
        matched = [r for r in rows if all(r[k] == p[k] for k in p)]
        if len(matched) != 2 or any(r['stage'] != 'passed' for r in matched):
            continue
        elapsed = sum(s['seconds'] for r in matched for s in r['steps'])
        results.append(dict(**p, images_per_second=sum(r['measured_images'] for r in matched) / elapsed,
                            paired_images_per_second=[r['images_per_second'] for r in matched],
                            peak_allocated_gib=max(r['peak_allocated_gib'] for r in matched)))
    return results


def read_manifest_metadata(path):
    """Use the already verified manifest without re-statting 796k files on NFS.

    This is a throughput probe, not a dataset integrity audit. Actual sampled
    images are opened by the production loader and fail normally if unavailable.
    """
    path = Path(path)
    records = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                for key in ('image', 'depth'):
                    if key in row:
                        row[key] = os.path.abspath(path.parent / row[key])
                records.append(row)
    return records


def identity(path):
    stat = path.stat()
    return dict(device=stat.st_dev, inode=stat.st_ino, size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def main():
    from accelerate import Accelerator
    from .system import build_system

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(stage='initializing', started_at=time.time(), checkpoint=str(args.checkpoint),
                  source_identity=identity(args.checkpoint), rows=[], production_optimizer_updates=0,
                  policy='Disposable benchmark only; no evaluation, W&B or checkpoint writes. '
                         'Changing effective batch changes optimization; timings do not establish accuracy.',
                  timing_policy='Fetch is consumer queue wait; preparation wall time includes worker '
                                'wait/normalization/grouping/pinning but excludes queue backpressure. '
                                'Preparation thread CPU excludes DataLoader worker decode CPU. '
                                'CUDA event stages include stream gaps and are not kernel-only traces.',
                  data_pipeline='prepared', compact_transport=True)
    path = args.output / 'benchmark.json'
    atomic_json(report, path)
    plan = json.loads(args.plan.read_text())
    deadline = time.monotonic() + plan['max_seconds']
    optimizer = adapter = statistics = state = None
    try:
        torch.set_num_threads(1)
        accelerator = Accelerator(mixed_precision='bf16')
        if accelerator.device.type != 'cuda' or accelerator.num_processes != 1:
            raise RuntimeError('Benchmark requires one CUDA process')
        # Leave at least 15% of VRAM free. OOM excludes a profile rather than restarting a job.
        torch.cuda.set_per_process_memory_fraction(.85)
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
        config = state['config']
        validate_plan(plan, config['train']['batch_size'])
        if set(config['loss']) - set(FEATURE_LOSS_KEYS) or config['train']['precision'] != 'bf16':
            raise ValueError('Only the existing BF16 feature-only training path is supported')
        records = read_manifest_metadata(config['data']['train_manifest'])
        report.update(plan=plan, config=config, source_step=state['step'], source_cursor=state['cursor'],
                      manifest_records=len(records), revisions=state['revisions'],
                      torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
                      total_memory_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
                      tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                      cudnn_benchmark=torch.backends.cudnn.benchmark)
        atomic_json(report, path)
        system, adapter = build_system(config, accelerator.device)
        if adapter.config != state['adapter_config'] or system.model_config != state['model_config']:
            raise ValueError('Model architecture must match the source checkpoint')
        assert all(not p.requires_grad for m in (system.radio, system.encoder, system.decoder)
                   for p in m.parameters())
        statistics = FeatureStatistics(adapter.config['target_dim'], adapter.config['target_summary_dim'])
        statistics.to(accelerator.device)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=config['train']['lr'],
                                      weight_decay=config['train']['weight_decay'])
        report.update(stage='running', loaded_at=time.time())
        failed = set()
        for round_number in range(2):
            profiles = plan['profiles'] if round_number == 0 else list(reversed(plan['profiles']))
            for profile in profiles:
                key = (*[profile[k] for k in ('effective_batch', 'compute_batch', 'workers')],
                       profile.get('input_mode', 'prepared'))
                if key in failed:
                    continue
                reset_state(adapter, statistics, optimizer, state)
                report['current_profile'] = dict(round=round_number, **profile)
                atomic_json(report, path)
                try:
                    row = run_profile(system, adapter, statistics, optimizer, accelerator,
                                      records, config, state, profile, plan, deadline)
                except torch.OutOfMemoryError:
                    row = dict(stage='oom', **profile)
                    failed.add(key)
                    optimizer.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                row['round'] = round_number
                report['rows'].append(row)
                atomic_json(report, path)
                print(json.dumps({k: v for k, v in row.items() if k not in ('steps', 'actual_compute_batches')}), flush=True)
        reset_state(adapter, statistics, optimizer, state)
        assert all(torch.equal(v.detach().cpu(), state['adapter'][k]) for k, v in adapter.state_dict().items())
        assert identity(args.checkpoint) == report['source_identity']
        assert all(p.grad is None for m in (system.radio, system.encoder, system.decoder) for p in m.parameters())
        report.update(stage='passed', finished_at=time.time(), current_profile=None,
                      paired_results=aggregate(report['rows'], plan['profiles']),
                      adapter_restored_exactly=True, optimizer_rng_restored=True,
                      frozen_models_no_gradients=True, source_identity_unchanged=True)
        atomic_json(report, path)
    except BaseException as error:
        report.update(stage='failed', finished_at=time.time(), error=f'{type(error).__name__}: {error}')
        atomic_json(report, path)
        raise


if __name__ == '__main__':
    main()
