"""Compare CPU preparation pipelines at an identical batch, model and sample stream."""

from copy import deepcopy
import gc
import math
import time

import torch

from .checkpoint import atomic_json, restore_rng
from .losses import weighted_loss
from .prefetch import windows


def probe(system, adapter, raw_adapter, statistics, optimizer, scheduler, accelerator,
          amp, records, config, state, performance, output):
    from .train import make_loader

    path = output / 'data-pipeline-benchmark.json'
    if path.exists():
        raise FileExistsError('Existing data pipeline probe; inspect before any retry')
    original = deepcopy(state['performance'])
    assert performance['effective_batch'] == original['effective_batch'] == 32
    assert performance['compute_batch'] == original['compute_batch'] == 16
    assert performance['workers'] == original['workers'] == 4
    report = dict(stage='running', started_at=time.time(), rows=[], disposable_updates=0,
                  anchor_step=state['step'], anchor_cursor=state['cursor'],
                  effective_batch=32, compute_batch=16, workers=4,
                  warmup_steps=6, measured_steps=30,
                  metric='end-to-end images/sec; fixed sample stream and batch',
                  caveat='fetch time includes waiting and CPU concatenation; not decode throughput')
    atomic_json(report, path)
    logical = config['train']['batch_size']
    accumulation = original['effective_batch'] // logical
    cfg = deepcopy(config)
    cfg['data']['workers'] = original['workers']
    original_scheduler = deepcopy(scheduler.state_dict())

    def reset():
        optimizer.zero_grad(set_to_none=True)
        raw_adapter.load_state_dict(state['adapter'])
        statistics.load_state_dict(state['statistics'])
        optimizer.load_state_dict(deepcopy(state['optimizer']))
        scheduler.load_state_dict(deepcopy(original_scheduler))
        restore_rng(state['rng'])

    # Exact real augmented pixels and compute-batch order, without retaining GPU tensors.
    reference = None
    for prepared in (False, True):
        loader = make_loader(records, cfg, state['cursor'], 4 * accumulation)
        stream = windows(loader, accumulation, 16, 4, prepared=prepared,
                         pin=accelerator.device.type == 'cuda')
        current = []
        try:
            for batch in stream:
                current.append([x.clone() for x in batch])
        finally:
            stream.close()
        if reference is None:
            reference = current
        else:
            assert len(reference) == len(current)
            for before, after in zip(reference, current, strict=True):
                assert len(before) == len(after)
                for a, b in zip(before, after, strict=True):
                    assert torch.equal(a, b), 'Prepared input pixels or ordering differ'
        del stream, loader
    del reference, current
    report['input_equality'] = dict(stage='passed', images=128, updates=4,
                                  exact_augmented_pixels_and_compute_order=True)
    atomic_json(report, path)

    for round_number, prepared in [(0, False), (0, True), (1, True), (1, False)]:
        reset()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        count = report['warmup_steps'] + report['measured_steps']
        loader = make_loader(records, cfg, state['cursor'], count * accumulation)
        stream = windows(loader, accumulation, 16, count, prepared=prepared,
                         pin=accelerator.device.type == 'cuda')
        seconds, waits = [], []
        result = dict(round=round_number, prepared=prepared, stage='running')
        report['current_profile'] = result
        atomic_json(report, path)
        try:
            adapter.train()
            for index in range(count):
                torch.cuda.synchronize()
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                fetch_start = time.perf_counter()
                images = list(next(stream))
                fetch_seconds = time.perf_counter() - fetch_start
                values = []
                for cpu_image in images:
                    image = cpu_image.to(accelerator.device, non_blocking=True)
                    with amp(), torch.no_grad():
                        features, target = system.encode(image, raw_adapter.indices)
                    with amp():
                        predicted = adapter(features)
                    loss = weighted_loss(statistics.loss(predicted, target), config['loss'])
                    weight = len(cpu_image) / 32
                    accelerator.backward(loss * weight)
                    values.append(loss.detach() * weight)
                total = torch.stack(values).sum()
                norm = accelerator.clip_grad_norm_(adapter.parameters(), config['train']['clip_grad'])
                pair = torch.stack([total, norm]).float().cpu().tolist()
                if not all(math.isfinite(x) for x in pair) or pair[1] <= 0:
                    raise FloatingPointError('Invalid disposable data-pipeline update')
                optimizer.step()
                report['disposable_updates'] += 1
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if index >= report['warmup_steps']:
                    seconds.append(elapsed)
                    waits.append(fetch_seconds)
                del images, image, features, target, predicted, loss, total, norm, values
            result.update(stage='passed', seconds=seconds, fetch_seconds=waits,
                          images_per_second=32 * len(seconds) / sum(seconds),
                          peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30)
        finally:
            stream.close()
            del stream, loader
            gc.collect()
        report['rows'].append(result)
        atomic_json(report, path)
        if time.time() - report['started_at'] > 900:
            raise TimeoutError('Data pipeline probe deadline; inspect, do not retry')
    aggregate = []
    for prepared in (False, True):
        rows = [r for r in report['rows'] if r['prepared'] == prepared]
        assert len(rows) == 2 and all(r['stage'] == 'passed' for r in rows)
        elapsed = sum(sum(r['seconds']) for r in rows)
        aggregate.append(dict(prepared=prepared, images_per_second=32 * 60 / elapsed))
    # Adopt only with >3% aggregate improvement and a gain in each paired round.
    selected = (aggregate[1]['images_per_second'] > 1.03 * aggregate[0]['images_per_second']
                and all(next(r for r in report['rows'] if r['round'] == i and r['prepared'])[
                    'images_per_second'] > next(r for r in report['rows']
                    if r['round'] == i and not r['prepared'])['images_per_second'] for i in (0, 1)))
    reset()
    torch.cuda.empty_cache()
    for key, value in raw_adapter.state_dict().items():
        assert torch.equal(value.detach().cpu(), state['adapter'][key]), 'Restore failed'
    result = dict(original, data_pipeline='prepared' if selected else 'legacy')
    result.pop('data_autotune', None)
    report.update(stage='passed', finished_at=time.time(), paired_rounds=aggregate,
                  selected=result, adapter_restored_exactly=True,
                  optimizer_scheduler_rng_restored=True, production_optimizer_updates=0,
                  original_schedule_and_batch_preserved=True)
    atomic_json(report, path)
    return result
