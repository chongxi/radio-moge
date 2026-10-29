"""Bounded, disposable throughput tuning; preserve the logical image stream on resume."""

from copy import deepcopy
import gc
import math
import time

import torch

from .checkpoint import atomic_json, restore_rng
from .losses import weighted_loss


def validate_performance(perf, config):
    if perf.get('authorized') is not True or not math.isfinite(perf['original_started_at']):
        raise ValueError('Reviewed performance authorization and original start are required')
    if config['train'].get('lr_schedule', 'cosine') != 'cosine':
        raise ValueError('Performance transition currently supports cosine only')
    if set(config['loss']) - {'spatial', 'summary', 'cosine', 'summary_cosine'}:
        raise ValueError('Performance grouping is restricted to feature-only objectives')
    profiles = perf.get('profiles', []) if perf.get('autotune') else [perf]
    if not profiles or len(profiles) > 10:
        raise ValueError('Expected 1–10 bounded throughput profiles')
    if perf.get('data_pipeline', 'legacy') not in ('legacy', 'prepared'):
        raise ValueError('Unsupported data preparation pipeline')
    logical = config['train']['batch_size']
    for p in profiles:
        if p['workers'] not in range(5):
            raise ValueError('At most four data workers')
        if (p['effective_batch'] not in (16, 32, 64) or p['compute_batch'] < logical
                or p['compute_batch'] > p['effective_batch']
                or p['compute_batch'] % logical or p['effective_batch'] % logical):
            raise ValueError('Unsupported reviewed batching profile')
    if perf.get('autotune'):
        if not 1 <= perf['warmup_steps'] <= 4 or not 4 <= perf['measured_steps'] <= 20:
            raise ValueError('Throughput probe must remain short and bounded')


def grouped_window(iterator, logical_batches, compute_batch):
    """Keep exact sampler/augmentation keys; combine same-shape tensors within one update.

    Different shapes stay separate. The logical stream and consumed-image count do not change;
    within-window execution order can change, and floating-point reductions need not be bitwise.
    """
    buckets = {}
    for _ in range(logical_batches):
        image = next(iterator)
        buckets.setdefault(tuple(image.shape[1:]), []).append(image)
    for images in buckets.values():
        size = len(images[0])
        if any(len(x) != size for x in images) or compute_batch % size:
            raise ValueError('Logical batches must divide the compute batch')
        count = compute_batch // size
        for start in range(0, len(images), count):
            selected = images[start:start + count]
            yield selected[0] if len(selected) == 1 else torch.cat(selected, dim=0)


def schedule_position(step, perf):
    return perf['anchor_schedule_position'] + (step - perf['anchor_step']) * perf['schedule_ratio']


def interval_due(step, interval, perf):
    return math.floor(schedule_position(step, perf) / interval) > math.floor(
        schedule_position(step - 1, perf) / interval)


def transition_limits(limits, settings, perf):
    result = deepcopy(limits)
    remaining_images = (settings['steps'] - perf['anchor_schedule_position']) * perf['base_effective_batch']
    endpoint = perf['anchor_step'] + int(remaining_images // perf['effective_batch'])
    result['steps'] = min(result['steps'], endpoint)
    result['performance_transition'] = dict(
        anchor_step=perf['anchor_step'], anchor_cursor=perf['anchor_cursor'],
        effective_batch=perf['effective_batch'], base_effective_batch=perf['base_effective_batch'],
        image_budget_preserved=True, wall_clock_budget_preserved=True,
        original_started_at=perf['original_started_at'])
    return result


def choose_profile(rows):
    passed = [r for r in rows if r['stage'] == 'passed']
    if not passed:
        raise RuntimeError('No finite throughput profile passed; no resume')
    # Maximize measured end-to-end images/sec. Within 3%, prefer the smaller effective batch.
    fastest = max(r['images_per_second'] for r in passed)
    near = [r for r in passed if r['images_per_second'] >= .97 * fastest]
    return min(near, key=lambda r: (r['effective_batch'], r['peak_memory_gib'], -r['images_per_second']))


def autotune(system, adapter, raw_adapter, statistics, optimizer, scheduler, accelerator,
             amp, records, config, state, perf, output):
    from .train import make_loader

    report_path = output / 'throughput-benchmark.json'
    if report_path.exists():
        raise FileExistsError('Throughput benchmark already exists; inspect before any retry')
    report = dict(stage='running', started_at=time.time(), anchor_step=state['step'],
                  anchor_cursor=state['cursor'], rows=[], disposable_updates=0,
                  metric='end-to-end images/sec including data loading, encoders, backward and Adam',
                  caveat='Throughput probe is not a generalization comparison; no probe weights retained')
    atomic_json(report, report_path)
    logical = config['train']['batch_size']
    original_scheduler = deepcopy(scheduler.state_dict())

    def reset():
        optimizer.zero_grad(set_to_none=True)
        raw_adapter.load_state_dict(state['adapter'])
        statistics.load_state_dict(state['statistics'])
        optimizer.load_state_dict(deepcopy(state['optimizer']))
        scheduler.load_state_dict(deepcopy(original_scheduler))
        restore_rng(state['rng'])

    ordered = [(r, p) for r in range(2) for p in (perf['profiles'] if r == 0 else list(reversed(perf['profiles'])))]
    for round_number, profile in ordered:
        reset()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cfg = deepcopy(config)
        cfg['data']['workers'] = profile['workers']
        accumulation = profile['effective_batch'] // logical
        count = perf['warmup_steps'] + perf['measured_steps']
        loader = make_loader(records, cfg, state['cursor'], count * accumulation)
        iterator = iter(loader)
        seconds, losses_seen, gradients, waits = [], [], [], []
        result = dict(profile, stage='running', round=round_number)
        report['current_profile'] = profile
        atomic_json(report, report_path)
        try:
            adapter.train()
            for index in range(count):
                torch.cuda.synchronize()
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                begin_data = time.perf_counter()
                images = list(grouped_window(iterator, accumulation, profile['compute_batch']))
                waits.append(time.perf_counter() - begin_data)
                values = []
                for cpu_image in images:
                    image = cpu_image.to(accelerator.device, non_blocking=True)
                    with amp(), torch.no_grad():
                        features, target = system.encode(image, raw_adapter.indices)
                    with amp():
                        predicted = adapter(features)
                    loss = weighted_loss(statistics.loss(predicted, target), config['loss'])
                    weight = len(cpu_image) / profile['effective_batch']
                    accelerator.backward(loss * weight)
                    values.append(loss.detach() * weight)
                total = torch.stack(values).sum()
                norm = accelerator.clip_grad_norm_(adapter.parameters(), config['train']['clip_grad'])
                pair = torch.stack([total, norm]).float().cpu().tolist()
                if not all(math.isfinite(x) for x in pair) or pair[1] <= 0:
                    raise FloatingPointError('Nonfinite or zero gradients in disposable probe')
                optimizer.step()
                report['disposable_updates'] += 1
                torch.cuda.synchronize()
                duration = time.perf_counter() - started
                losses_seen.append(pair[0])
                gradients.append(pair[1])
                if index >= perf['warmup_steps']:
                    seconds.append(duration)
                del images, image, features, target, predicted, loss, total, norm, values
            result.update(stage='passed', seconds=seconds,
                          images_per_second=profile['effective_batch'] * len(seconds) / sum(seconds),
                          peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
                          data_wait_seconds=waits, losses=losses_seen, gradients=gradients)
        except torch.cuda.OutOfMemoryError:
            result.update(stage='oom', peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30)
            # A recorded OOM excludes this profile; never silently shrink the production batch.
        finally:
            images = image = features = target = predicted = loss = total = norm = values = None
            del images, image, features, target, predicted, loss, total, norm, values
            optimizer.zero_grad(set_to_none=True)
            del iterator, loader
            gc.collect()
        report['rows'].append(result)
        atomic_json(report, report_path)
        if time.time() - report['started_at'] > perf.get('max_probe_seconds', 600):
            raise TimeoutError('Throughput benchmark budget reached; inspect, no automatic retry')
    aggregate = []
    for profile in perf['profiles']:
        trials = [r for r in report['rows'] if all(r[k] == v for k, v in profile.items())]
        if len(trials) != 2 or any(r['stage'] != 'passed' for r in trials):
            continue
        seconds = [v for r in trials for v in r['seconds']]
        aggregate.append(dict(profile, stage='passed',
                              images_per_second=profile['effective_batch'] * len(seconds) / sum(seconds),
                              peak_memory_gib=max(r['peak_memory_gib'] for r in trials)))
    report['paired_rounds'] = aggregate
    chosen = choose_profile(aggregate)
    reset()
    torch.cuda.empty_cache()
    # CPU checkpoint is authoritative; all disposable updates and their RNG are discarded.
    for key, value in raw_adapter.state_dict().items():
        if not torch.equal(value.detach().cpu(), state['adapter'][key]):
            raise AssertionError('Adapter restoration failed after throughput tuning')
    for module in (system.radio, system.encoder, system.decoder):
        if any(p.requires_grad or p.grad is not None for p in module.parameters()):
            raise AssertionError('Frozen backbone changed trainability')
    base_batch = config['train']['batch_size'] * config['train']['accumulation']
    selected = dict(authorized=True, autotune=False,
                    original_started_at=perf['original_started_at'],
                    **{k:chosen[k] for k in ('workers', 'effective_batch', 'compute_batch')},
                    anchor_step=state['step'], anchor_cursor=state['cursor'],
                    anchor_schedule_position=float(state['step']), base_effective_batch=base_batch,
                    schedule_ratio=chosen['effective_batch'] / base_batch,
                    selection='fastest finite end-to-end throughput; smaller batch within 3%',
                    lr_policy='same LR at transition; cosine advances by consumed images',
                    data_policy='same logical sampler and augmented images, grouped inside update',
                    numerical_equivalence='not bitwise; effective batch change changes optimization')
    report.update(stage='passed', finished_at=time.time(), selected=selected,
                  adapter_restored_exactly=True, optimizer_scheduler_rng_restored=True,
                  production_optimizer_updates=0)
    atomic_json(report, report_path)
    return selected
