"""Compare per-frame and cached PPD calibration on one contiguous video.

Four diffusion steps, weights and canonical RGB normalization stay fixed. The
camera is fixed to the first estimate in every controlled comparison. Offline
auditing may read tensors back; normal cached inference does not.
"""

import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import signal
import sys
import time

import cv2
import numpy as np
import torch

from benchmark_ppd_latency import gpu_status, processes, stats
from probe_ppd_feature_swap import digest, publish
from radio_moge.adapters import build_adapter
from radio_moge.decoder import MoGeDecoder, postprocess
from radio_moge.pointcloud import unproject
from radio_moge.ppd_fast import StaticPPDSampler, PPDGraph
from radio_moge.ppd_geometry import (CachedPPDGeometry, GeometryGraph,
                                     apply_log_depth_numpy, fit_log_depth)
from radio_moge.upstream import extract_radio, load_moge, load_radio, radio_input


MODES = {'cpu_every': 1, 'gpu_every': 1, 'gpu_10': 10, 'gpu_30': 30, 'gpu_first': 0}
SNAPSHOTS = [0, 29, 59, 89, 119, 149, 179, 209, 239, 269, 299, 359]


def expire(*_):
    raise TimeoutError('Cached calibration experiment exceeded its 20-minute limit')


def save_npz(path, **arrays):
    tmp = path.with_name(path.name + '.new')
    with tmp.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, path)


def discrepancy(candidate, reference, mask):
    ratio = candidate[mask] / reference[mask]
    rel = np.abs(ratio - 1)
    return {'abs_rel': float(rel.mean()), 'p95_pixel_rel': float(np.quantile(rel, .95)),
            'median_depth_ratio': float(np.median(ratio)), 'valid_pixels': int(mask.sum())}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('base', 'clouds', 'compiled', 'output', 'vendor', 'ppd-source', 'video'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGALRM, expire)
    signal.alarm(1200)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    base = json.loads((args.base / 'plan.json').read_text())
    for key in ('checkpoint', 'ppd_checkpoint'):
        assert digest(base[key]['path']) == base[key]['sha256']
    capture = cv2.VideoCapture(str(args.video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB))
    capture.release()
    assert len(frames) == 360 and fps == 30, (len(frames), fps)
    plan = {
        'base_plan_sha256': digest(args.base / 'plan.json'), 'script_sha256': digest(__file__),
        'geometry_source_sha256': digest(Path(__file__).resolve().parents[1] / 'src/radio_moge/ppd_geometry.py'),
        'video': {'path': str(args.video), 'sha256': digest(args.video), 'frames': len(frames), 'fps': fps},
        'shape_hw': [480, 640], 'batch': 1, 'ppd_steps': 4, 'optimizer_updates': 0,
        'precision': 'FP32 weights, BF16 autocast, FP32 decoder and geometry',
        'noise': 'One fixed seeded GPU noise map shared by all frames and all modes',
        'calibration': 'Official degree-one RANSAC objective against current adapter-MoGe depth; no GT',
        'camera': 'First-frame normalized K and pixel-center rays fixed for ALL modes; no world pose',
        'mask': 'Every-frame modes use current MoGe mask; cached modes use only current finite-depth validity, never a stale learned mask',
        'quality': 'All 360 consecutive frames; compare cached to fresh calibration of exactly the same PPD map on current reference-valid support, not ground-truth accuracy',
        'timing': 'CPU decoded uint8 RGB input; cpu_every returns compact CPU XYZ/RGB; GPU modes return organized GPU XYZ/depth/valid plus GPU RGB; disk, decode, rendering and browser delivery excluded',
        'timing_frames_per_round': 120, 'timing_rounds': 2, 'ordering': 'Forward then reversed mode order',
        'startup': 'Loading, first camera estimate and compilation excluded; frame-zero calibration included in each timed clip',
        'modes': MODES, 'snapshots': SNAPSHOTS, 'hardware_before': gpu_status(), 'processes_before': processes(),
    }
    publish(args.output / 'plan.json', plan)
    sys.path.insert(0, str(args.ppd_source))
    from ppd.models.dit import DiT
    from ppd.utils.align_depth_func import recover_metric_depth_ransac

    # Reuse saved predictions to verify the replacement arithmetic independently
    # from any change in image content or neural inference.
    cpu_parity = []
    for selected in base['selected']:
        index, row = selected['index'], selected['row']
        name = f"{index:02d}-{row['dataset']}-480x640"
        with np.load(args.base / (name + '.npz')) as z:
            ref = z['moge_adapter'].copy()
        with np.load(args.clouds / (name + '.npz')) as z:
            mask, camera = z['adapter_mask'].copy(), z['adapter_camera'].copy()
        with np.load(args.compiled / 'compiled-steps-4' / (name + '.npz')) as z:
            relative = z['raw'].copy()
        mask &= np.isfinite(ref) & (ref > 0)
        calibration = fit_log_depth(relative, ref, mask, base['seed'] + index)
        ours = apply_log_depth_numpy(relative, calibration)
        np.random.seed(base['seed'] + index)
        official = recover_metric_depth_ransac(relative, ref, mask)
        np.testing.assert_array_equal(ours, official)
        cpu_parity.append({'scene': name, 'official_cpu_exact': True})
    publish(args.output / 'cpu-parity.json', cpu_parity)

    checkpoint = torch.load(base['checkpoint']['path'], map_location='cpu', weights_only=True, mmap=True)
    models = dict(checkpoint['config']['models'], moge_source=str(args.vendor / 'MoGe'), radio_source=str(args.vendor / 'RADIO'))
    moge, _ = load_moge(models, 'cuda')
    decoder = MoGeDecoder(moge)
    del moge
    radio = load_radio(models, 'cuda')
    adapter = build_adapter(**checkpoint['adapter_config'])
    adapter.load_state_dict(checkpoint['adapter'], strict=True)
    adapter.cuda().eval().requires_grad_(False)
    del checkpoint
    state = torch.load(base['ppd_checkpoint']['path'], map_location='cpu', weights_only=True, mmap=True)
    with torch.device('meta'):
        dit = DiT()
    dit.load_state_dict({k[4:]: v for k, v in state.items()}, strict=True, assign=True)
    dit.cuda().eval().requires_grad_(False)
    del state
    gc.collect()
    torch.cuda.synchronize()
    print(json.dumps({'event': 'loaded', 'gpu': gpu_status()}), flush=True)

    def encode(rgb):
        image = (torch.from_numpy(rgb).float().permute(2, 0, 1) / 255)[None].cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            fitted = adapter(extract_radio(radio, radio_input(image, radio), adapter.indices))
        return image, fitted, fitted.spatial.flatten(2).transpose(1, 2)

    noise = torch.randn((1, 1, 480, 640), generator=torch.Generator().manual_seed(base['seed'])).cuda()
    image, fitted, semantics = encode(frames[0])
    fast = StaticPPDSampler(dit, steps=4)
    start = time.perf_counter()
    fast.velocity = torch.compile(fast.velocity, fullgraph=True, dynamic=False,
                                  options={'max_autotune': False, 'triton.cudagraphs': False})
    with torch.autocast('cuda', dtype=torch.bfloat16):
        fast(image, semantics, noise)
    torch.cuda.synchronize()
    ppd = PPDGraph(fast, image, semantics, noise)
    compilation_seconds = time.perf_counter() - start

    def calibrate(relative, fitted, image, index):
        prediction = postprocess(decoder(fitted, (480, 640)), image, models['moge_source'])
        raw = relative.float().cpu().numpy().copy()
        reference = prediction['depth'][0].cpu().numpy()
        mask = prediction['mask'][0].cpu().numpy().astype(bool)
        mask &= np.isfinite(reference) & (reference > 0)
        camera = prediction['intrinsics'][0].cpu().numpy().copy()
        calibration = fit_log_depth(raw, reference, mask, base['seed'] + index)
        return calibration, camera, mask, raw

    relative = ppd(image, semantics, noise)[0, 0]
    first, fixed_k, _, _ = calibrate(relative, fitted, image, 0)
    geometry = CachedPPDGeometry(480, 640, fixed_k, first)
    geometry_graph = GeometryGraph(geometry, relative)
    print(json.dumps({'event': 'graphs_ready', 'compile_seconds': compilation_seconds,
                      'fixed_camera': fixed_k.tolist(), 'first_calibration': asdict(first)}), flush=True)

    # Check graph parameter updates, current masks, default-mask reset, and XYZ.
    parity = []
    for selected in base['selected']:
        index, row = selected['index'], selected['row']
        name = f"{index:02d}-{row['dataset']}-480x640"
        with np.load(args.base / (name + '.npz')) as z:
            ref = z['moge_adapter'].copy()
        with np.load(args.clouds / (name + '.npz')) as z:
            mask = z['adapter_mask'].copy()
        with np.load(args.compiled / 'compiled-steps-4' / (name + '.npz')) as z:
            raw = z['raw'].copy()
        mask &= np.isfinite(ref) & (ref > 0)
        c = fit_log_depth(raw, ref, mask, base['seed'] + index)
        expected_depth = apply_log_depth_numpy(raw, c)
        geometry.update_calibration(c)
        d, p, m = geometry_graph(torch.from_numpy(raw).cuda(), torch.from_numpy(mask).cuda())
        d, p, m = d.cpu().numpy(), p.cpu().numpy(), m.cpu().numpy()
        expected_points = unproject(expected_depth, fixed_k)
        np.testing.assert_array_equal(m, mask)
        np.testing.assert_allclose(d, expected_depth, rtol=3e-5, atol=2e-5)
        np.testing.assert_allclose(p, expected_points, rtol=3e-5, atol=2e-5)
        parity.append({'scene': name, 'max_depth_abs': float(np.max(np.abs(d - expected_depth))),
                       'max_xyz_abs': float(np.max(np.abs(p - expected_points))),
                       'mask_exact': True})
    _, _, m = geometry_graph(torch.from_numpy(raw).cuda())
    assert m.all().item(), 'Default call reused a stale mask'
    raw_bad = torch.from_numpy(raw).cuda()
    raw_bad[1, 1] = torch.nan
    raw_bad[2, 2] = torch.inf
    _, _, m = geometry_graph(raw_bad)
    assert not m[1, 1].item() and not m[2, 2].item() and m.sum().item() == 480 * 640 - 2
    publish(args.output / 'gpu-parity.json', {'rows': parity, 'parameter_updates_observed': True,
            'default_mask_reset': True, 'nonfinite_current_pixels_excluded': True})

    # Offline audit is deliberately separate from the fast-path timing. It runs
    # the reference on EVERY frame solely to measure caching-induced drift.
    quality, cached = [], {}
    (args.output / 'snapshots').mkdir()
    for index, rgb in enumerate(frames):
        image, fitted, semantics = encode(rgb)
        relative = ppd(image, semantics, noise)[0, 0]
        fresh, camera, mask, raw = calibrate(relative, fitted, image, index)
        reference = apply_log_depth_numpy(raw, fresh)
        row = {'frame': index, 'seconds': index / fps, 'fresh_calibration': asdict(fresh),
               'recovered_camera': camera.tolist(), 'reference_valid_pixels': int(mask.sum()), 'modes': {}}
        maps = {'fresh': reference}
        for mode in ('gpu_10', 'gpu_30', 'gpu_first'):
            period = MODES[mode]
            update = index == 0 or (period and index % period == 0)
            jump = None
            if update:
                if mode in cached:
                    jump = discrepancy(apply_log_depth_numpy(raw, cached[mode]), reference, mask)
                cached[mode] = fresh
            candidate = apply_log_depth_numpy(raw, cached[mode])
            row['modes'][mode] = {**discrepancy(candidate, reference, mask), 'calibration_updated': bool(update),
                                  'same_frame_old_to_new_jump': jump}
            maps[mode] = candidate
        quality.append(row)
        if index in SNAPSHOTS:
            save_npz(args.output / 'snapshots' / f'frame-{index:03d}.npz', rgb=rgb, raw=raw,
                     intrinsics=fixed_k, current_reference_mask=mask, **maps)
        if (index + 1) % 30 == 0:
            print(json.dumps({'event': 'audit', 'frames': index + 1, 'current_errors': row['modes']}), flush=True)
    publish(args.output / 'quality-records.json', quality)

    def run_frame(rgb, index, mode, instrument=False):
        period = MODES[mode]
        keyframe = index == 0 or (period and index % period == 0)
        events = [torch.cuda.Event(enable_timing=True) for _ in range(5)] if instrument else []
        torch.cuda.synchronize()
        begin = time.perf_counter()
        if instrument:
            events[0].record()
        image = (torch.from_numpy(rgb).float().permute(2, 0, 1) / 255)[None].cuda()
        if instrument:
            events[1].record()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            fitted = adapter(extract_radio(radio, radio_input(image, radio), adapter.indices))
        if instrument:
            events[2].record()
        relative = ppd(image, fitted.spatial.flatten(2).transpose(1, 2), noise)[0, 0]
        if instrument:
            events[3].record()
        mask = None
        if keyframe:
            c, _, mask, raw = calibrate(relative, fitted, image, index)
            geometry.update_calibration(c)
        if mode == 'cpu_every':
            depth = apply_log_depth_numpy(raw, c)
            xyz = unproject(depth, fixed_k)[mask].astype(np.float32)
            colors = rgb[mask].copy()
        else:
            current_mask = torch.from_numpy(mask).cuda() if mode == 'gpu_every' else None
            depth, xyz, valid = geometry_graph(relative, current_mask)
            colors = image
        if instrument:
            events[4].record()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - begin) * 1000
        if mode != 'cpu_every':
            assert xyz.is_cuda and depth.is_cuda and valid.is_cuda and colors.is_cuda
        result = {'frame': index, 'mode': mode, 'keyframe': bool(keyframe), 'ms': elapsed}
        if instrument:
            result['stream_ms'] = {name: a.elapsed_time(b) for name, a, b in zip(
                ('input_prepare_and_h2d', 'radio_and_adapter', 'ppd', 'geometry'), events, events[1:])}
        return result

    for i in range(16):
        run_frame(frames[i], i, 'gpu_every')
    timing = {mode: [] for mode in MODES}
    for reverse in (False, True):
        for mode in (list(reversed(MODES)) if reverse else MODES):
            geometry.update_calibration(first)
            for index, rgb in enumerate(frames[:120]):
                timing[mode].append(run_frame(rgb, index, mode))
            print(json.dumps({'event': 'timing', 'mode': mode, 'round': int(reverse),
                              'ms': stats([x['ms'] for x in timing[mode]])}), flush=True)
    publish(args.output / 'timing-records.json', timing)
    # Instrument only a separate normal-frame sample, so the headline timings
    # are not inflated by extra CUDA event recording.
    geometry.update_calibration(first)
    stages = [run_frame(frames[i], i, 'gpu_first', instrument=True) for i in range(1, 33)]
    publish(args.output / 'stage-records.json', stages)
    relative = ppd(image, semantics, noise)[0, 0]
    geometry_times = []
    for _ in range(64):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        geometry_graph(relative)
        torch.cuda.synchronize()
        geometry_times.append((time.perf_counter() - begin) * 1000)

    timings = {}
    for mode, rows in timing.items():
        timings[mode] = {'all_ms': stats([r['ms'] for r in rows]),
                         'calibration_calls': sum(r['keyframe'] for r in rows)}
        for keyframe, name in ((True, 'keyframe_ms'), (False, 'ordinary_frame_ms')):
            values = [r['ms'] for r in rows if r['keyframe'] == keyframe]
            timings[mode][name] = stats(values) if values else None
    errors = {}
    for mode in ('gpu_10', 'gpu_30', 'gpu_first'):
        rows = [r['modes'][mode] for r in quality]
        errors[mode] = {'frame_abs_rel': stats([r['abs_rel'] for r in rows]),
                        'median_depth_ratio': stats([r['median_depth_ratio'] for r in rows]),
                        'frame_p95_pixel_rel': stats([r['p95_pixel_rel'] for r in rows]),
                        'worst_frame': int(np.argmax([r['abs_rel'] for r in rows])),
                        'calibration_calls': sum(r['calibration_updated'] for r in rows)}
    cameras = np.array([r['recovered_camera'] for r in quality])
    report = {'completed': True, 'plan_sha256': digest(args.output / 'plan.json'),
              'timings': timings, 'geometry_only_ms': stats(geometry_times),
              'ordinary_frame_stream_stage_ms': {key: stats([r['stream_ms'][key] for r in stages])
                                                for key in stages[0]['stream_ms']},
              'quality_vs_fresh_calibration_not_gt': errors, 'fixed_camera': fixed_k.tolist(),
              'fresh_fx_over_first_fx': stats(cameras[:, 0, 0] / fixed_k[0, 0]),
              'cpu_parity_exact_scenes': len(cpu_parity), 'gpu_parity': parity,
              'compilation_seconds': compilation_seconds,
              'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
              'gpu_after': gpu_status(), 'processes_after': processes(), 'optimizer_updates': 0}
    publish(args.output / 'report.json', report)
    signal.alarm(0)
    print(json.dumps({'event': 'complete', 'timings': timings, 'quality': errors}), flush=True)


if __name__ == '__main__':
    main()
