"""Bounded actual same-resolution batch throughput and memory probe; no Adam updates."""
import argparse
import fcntl
import gc
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from radio_moge.checkpoint import atomic_json, restore_rng, rng_state
from radio_moge.corpus_admission import checked_path, file_ref
from radio_moge.corpus_model_probe import assert_frozen, module_digest
from radio_moge.data import TrainingImages
from radio_moge.losses import FeatureStatistics, weighted_loss
from radio_moge.prefetch import DeviceWindows, FloatLoader, PreparedWindows
from radio_moge.rgb_input_audit import ReferenceRecords, pixels_sha
from radio_moge.system import build_system
from radio_moge.upstream import extract_radio, radio_input


def probe(args):
    plan = json.loads(args.plan.read_text())
    root = args.plan.parent
    if (root / 'report.json').exists():
        raise FileExistsError('Probe already started; inspect its result')
    with Path(plan['training_lock']).open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.88)
        start = time.monotonic()
        report = dict(stage='loading', rows=[], production_optimizer_updates=0,
                      optimizer_steps_by_probe=0, started_at=time.time(), plan=file_ref(args.plan))
        atomic_json(report, root / 'report.json')
        if plan.get('reuse_64'):
            prior = json.loads(checked_path(plan['reuse_64']).read_text())
            prior_plan = json.loads(checked_path(prior['plan']).read_text())
            if prior_plan['checkpoint'] != plan['checkpoint'] or prior_plan['input_prefix'] != plan['input_prefix']:
                raise ValueError('Reused batch64 measurements have different inputs/weights')
            reused = [r for r in prior['rows'] if r['batch'] == 64 and r['stage'] == 'passed']
            if len(reused) != 3 or {tuple(r['bucket']) for r in reused} != {(480,640),(640,480),(400,640)}:
                raise ValueError('Incomplete batch64 measurements')
            report.update(rows=reused, reused_64=plan['reuse_64'])
        state = torch.load(checked_path(plan['checkpoint']), map_location='cpu',
                           mmap=True, weights_only=True)
        config = state['config']
        samples = json.loads(checked_path(plan['input_prefix']).read_text())
        if len(samples) < 640 or config['loss'] != dict(spatial=1., summary=1., cosine=1., summary_cosine=1.):
            raise ValueError('Expected saved actual RGB inputs and original feature objective')
        records = ReferenceRecords([dict(image=r['image'], sha256=r['sha256'],
            image_rotation_cw=r.get('image_rotation_cw', 0), dataset=r['dataset']) for r in samples[:640]])
        system, adapter = build_system(config, 'cuda')
        adapter.load_state_dict(state['adapter'])
        stats = FeatureStatistics(adapter.config['target_dim'], adapter.config['target_summary_dim']).cuda()
        stats.load_state_dict(state['statistics'])
        # Allocate the real inherited Adam moments, without taking an optimizer step.
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=config['train']['lr'],
                                      weight_decay=config['train']['weight_decay'])
        optimizer.load_state_dict(state['optimizer'])
        modules = dict(radio=system.radio, encoder=system.encoder, decoder=system.decoder,
                       adapter=adapter, statistics=stats)
        before = {k: module_digest(v) for k, v in modules.items()}
        original_rng = rng_state()
        initial_file_identity = (Path(plan['checkpoint']['path']).stat().st_size,
                                 Path(plan['checkpoint']['path']).stat().st_mtime_ns)
        report.update(stage='running', torch_version=torch.__version__,
                      gpu=torch.cuda.get_device_name(), source_step=state['step'],
                      allocated_optimizer_states=len(optimizer.state),
                      same_resolution_per_forward=True, gradient_accumulation=False,
                      memory_fraction_limit=.88, viewer_preserved=True)
        atomic_json(report, root / 'report.json')
        del state
        for batch in ((128,) if plan.get('reuse_64') else (64, 128)):
            batch_failed = False
            for height, width in config['data']['buckets']:
                if time.monotonic() - start > plan['max_seconds']:
                    raise TimeoutError('Bounded batch probe expired')
                row = dict(batch=batch, bucket=[height, width], stage='running')
                report['current'] = row
                atomic_json(report, root / 'report.json')
                windows = None
                image = features = target = prediction = loss = losses = raw = None
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                observed = []
                try:
                    keys = [(i, samples[i]['key'][1], height, width) for i in range(640)]
                    loader = DataLoader(TrainingImages(records, True, compact_transport=True),
                        batch_sampler=[keys[i:i+4] for i in range(0, 640, 4)], num_workers=4,
                        pin_memory=True, generator=torch.Generator().manual_seed(43))
                    def observe(tensors):
                        if len(tensors) != 1 or list(tensors[0].shape) != [batch, 3, height, width]:
                            raise ValueError('Configured batch was split into smaller forwards')
                        observed.append(dict(shape=list(tensors[0].shape), pixel_sha256=pixels_sha(tensors[0])))
                    cpu = PreparedWindows(FloatLoader(loader), batch//4, batch, 640//batch,
                        pin=True, measure=True, observer=observe)
                    windows = DeviceWindows(cpu, 'cuda', measure=True)
                    warm = 128//batch
                    times, waits, forwards, backwards, totals, norms = [], [], [], [], [], []
                    adapter.train()
                    for i in range(640//batch):
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.current_stream().synchronize()
                        began = time.perf_counter()
                        tensors = next(windows)
                        wait = time.perf_counter()-began
                        if len(tensors) != 1 or len(tensors[0]) != batch:
                            raise ValueError('Actual GPU forward batch differs')
                        image = tensors[0]
                        a, b, c = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                        a.record()
                        with torch.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
                            features, target = system.encode(image, adapter.indices)
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            prediction = adapter(features)
                        losses = stats.loss(prediction, target)
                        loss = weighted_loss(losses, config['loss'])
                        b.record()
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), config['train']['clip_grad'])
                        c.record()
                        if not torch.isfinite(loss) or not torch.isfinite(norm) or norm <= 0:
                            raise FloatingPointError('Nonfinite loss/gradient in true batch probe')
                        torch.cuda.current_stream().synchronize()
                        if i >= warm:
                            times.append(time.perf_counter()-began)
                            waits.append(wait)
                            forwards.append(a.elapsed_time(b)/1000)
                            backwards.append(b.elapsed_time(c)/1000)
                            totals.append(float(loss.detach()))
                            norms.append(float(norm))
                        # Do not retain the previous graph during the next forward.
                        del tensors
                        image = features = target = prediction = loss = losses = raw = None
                    row.update(stage='passed', training_images_per_second=512/sum(times),
                        measured_images=512, warmup_images=128, seconds_per_batch=times,
                        consumer_wait_seconds=waits, forward_seconds=forwards, backward_seconds=backwards,
                        losses=totals, gradient_norms=norms, actual_compute_batches=observed,
                        future_ready_before_consume=windows.ready_before_consume,
                        preparation_timings=windows.preparation_timings,
                        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                    atomic_json(dict(report, current=row), root / 'report.json')
                    windows.close()
                    windows = None
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    # Student inference excludes the MoGe teacher encoder and loss/backward.
                    compact = torch.stack([TrainingImages(records, True, compact_transport=True)[k]
                                           for k in keys[:batch]])
                    image = compact.to('cuda', dtype=torch.float32).div_(255)
                    adapter.eval()
                    infer = []
                    for i in range(4):
                        torch.cuda.synchronize()
                        began = time.perf_counter()
                        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                            features = extract_radio(system.radio, radio_input(image, system.radio, None), adapter.indices)
                            prediction = adapter(features)
                        with torch.no_grad():
                            raw = system.decoder(prediction, image.shape[-2:])
                        if not all(torch.isfinite(v).all() for v in raw.values() if torch.is_tensor(v)):
                            raise FloatingPointError('Nonfinite batched student inference')
                        torch.cuda.synchronize()
                        if i:
                            infer.append(time.perf_counter()-began)
                        features = prediction = raw = None
                    row['student_raw_inference'] = dict(batch=batch, seconds=infer,
                        images_per_second=batch*len(infer)/sum(infer),
                        scope='resident RGB through RADIO, adapter and raw MoGe decoder; excludes postprocess and I/O')
                    image = None
                except torch.cuda.OutOfMemoryError as e:
                    row.update(stage='oom', error=str(e).split('\n')[0],
                               peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                    batch_failed = True
                except RuntimeError as e:
                    if row['stage'] != 'passed' or '32-bit index math' not in str(e):
                        raise
                    row['student_raw_inference'] = dict(batch=batch, stage='unsupported',
                        error=str(e), training_forward_backward_passed=True,
                        action='Keep actual student depth inference at tested batch64; do not silently split training')
                finally:
                    image = features = target = prediction = loss = losses = raw = None
                    optimizer.zero_grad(set_to_none=True)
                    if windows is not None:
                        windows.close()
                    gc.collect()
                    torch.cuda.empty_cache()
                report['rows'].append(row)
                atomic_json(report, root / 'report.json')
                print(json.dumps(row), flush=True)
                if batch_failed:
                    break
        assert_frozen(system)
        after = {k: module_digest(v) for k, v in modules.items()}
        if before != after:
            raise ValueError('Probe modified the model')
        restore_rng(original_rng)
        if initial_file_identity != (Path(plan['checkpoint']['path']).stat().st_size,
                                     Path(plan['checkpoint']['path']).stat().st_mtime_ns):
            raise ValueError('Source checkpoint changed')
        passed = [n for n in (64, 128) if sum(r['batch']==n and r['stage']=='passed'
                  for r in report['rows']) == len(config['data']['buckets'])]
        report.update(stage='complete', selected_actual_batch=max(passed) if passed else None,
            model_states_unchanged=True, rng_restored=True, finished_at=time.time(), current=None,
            interpretation='Throughput and finite gradients only; no optimization or quality gain claim')
        atomic_json(report, root / 'report.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    probe(p.parse_args())
