"""Bounded H200 benchmark: RADIO + adapter + MoGe + identical depth-edge masks.

GPU profiles return organized XYZ/RGB/mask or compact XYZ/RGB. CPU profiles
return compact XYZ/RGB. Official per-image camera recovery is retained in all
full-pipeline profiles. No PPD model or encoder is run.
"""

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import platform
import signal
import time

import numpy as np
import torch
import utils3d_moge

from benchmark_ppd_latency import gpu_status, processes, stats
from export_ppd_depth_edges import digest, publish
from radio_moge.adapters import build_adapter
from radio_moge.decoder import MoGeDecoder, postprocess
from radio_moge.depth_edges import depth_edge_keep_mask, depth_edge_keep_mask_torch
from radio_moge.pointcloud import unproject
from radio_moge.upstream import extract_radio, load_moge, load_radio, radio_input


MODES = ["gpu_original", "gpu_filtered", "gpu_compact_filtered",
         "cpu_original", "cpu_filtered", "cpu_official_filtered", "gpu_cpu_filtered"]
THRESHOLDS = [.01, .02, .04, .08]


def expire(*_):
    raise TimeoutError("MoGe filtering benchmark exceeded 10 minutes")


def official_keep(depth, valid):
    return valid & ~utils3d_moge.np.depth_map_edge(depth, rtol=.04)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["base", "clouds", "edges", "output", "vendor"]:
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--skip-filter-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGALRM, expire)
    signal.alarm(600)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    base = json.loads((args.base / "plan.json").read_text())
    edge_report = json.loads((args.edges / "report.json").read_text())
    edge_parity = json.loads((args.edges / "upstream-parity.json").read_text())
    assert edge_parity["passed"] and edge_parity["report_sha256"] == digest(args.edges / "report.json")
    assert digest(base["checkpoint"]["path"]) == base["checkpoint"]["sha256"]
    publish(args.output / "plan.json", {
        "resolution_hw": [480, 640], "batch": 1, "threshold": .04, "kernel_size": 3,
        "modes": args.modes, "rounds": 4, "frames_per_mode_per_round": 32,
        "profiles_per_mode": 128, "instrumented_gpu_filtered_frames": 64,
        "filter_only_samples_per_implementation": 0 if args.skip_filter_only else 256,
        "precision": "FP32 weights; BF16 autocast for RADIO and adapter; FP32 MoGe decoder, camera, filter and XYZ",
        "input": "CPU uint8 RGB already decoded and sized to 640x480; CPU normalization matches original predictions",
        "gpu_output": "Organized FP32 OpenGL XYZ + normalized RGB + validity mask on GPU; compact mode gathers valid XYZ/RGB",
        "cpu_output": "Compact FP32 OpenGL XYZ + uint8 RGB on CPU",
        "included": ["RGB normalization and H2D", "RADIO and adapter", "all original MoGe heads including normal",
                     "official per-image focal/shift recovery and projection", "current mask and coordinate convention",
                     "edge filter when selected", "CPU transfers and compaction only for CPU profiles"],
        "excluded": ["file/image decoding", "model loading", "warmup", "network transport", "PLY writing", "rendering"],
        "timing": "Per-request CUDA synchronization and wall clock; separate instrumented frames; reversed mode order on alternate rounds",
        "ppd_calls": 0, "moge_encoder_calls": 0, "optimizer_updates": 0,
        "compile": False, "cuda_graphs": False, "interframe_caching": False,
        "checkpoint": base["checkpoint"], "source_base_plan_sha256": digest(args.base / "plan.json"),
        "source_edge_report_sha256": digest(args.edges / "report.json"),
        "source_edge_parity_sha256": digest(args.edges / "upstream-parity.json"),
        "script_sha256": digest(__file__),
        "filter_source_sha256": digest(Path(__file__).resolve().parents[1] / "src/radio_moge/depth_edges.py"),
        "gpu_before": gpu_status(), "processes_before": processes(),
        "memory_events_before": Path('/sys/fs/cgroup/memory.events').read_text(),
    })

    # Verify the GPU implementation against ALL already verified full-size masks
    # for both branches/resolutions, not just the inference benchmark's 16 scenes.
    filter_parity = []
    for row in edge_report["results"]:
        name = row["id"]
        with np.load(args.base / (name + ".npz")) as z:
            moge = z["moge_adapter"].copy()
        with np.load(args.clouds / (name + ".npz")) as z:
            ppd, valid = z["ppd_adapter"].copy(), z["adapter_mask"].copy()
        masks_path = args.edges / "masks" / (name + ".npz")
        assert digest(masks_path) == row["masks_sha256"]
        valid_gpu = torch.from_numpy(valid).cuda()
        with np.load(masks_path) as masks:
            for method, depth in [("moge_adapter", moge), ("ppd_adapter", ppd)]:
                depth_gpu = torch.from_numpy(depth).cuda()
                for i, threshold in enumerate(THRESHOLDS):
                    actual = depth_edge_keep_mask_torch(depth_gpu, valid_gpu, threshold).cpu().numpy()
                    np.testing.assert_array_equal(actual, masks[f"{method}_keep_{i}"])
                    filter_parity.append({"scene": name, "method": method, "threshold": threshold,
                                          "different_pixels": 0})
    invalid = np.array([[np.nan, 1, 1], [1, np.inf, 2], [1, 2, 0]], np.float32)
    valid = np.isfinite(invalid) & (invalid > 0)
    for array in [invalid, np.ones((7, 9), np.float32), np.array([[1, 1, 1, 2, 2]] * 5, np.float32)]:
        valid = np.isfinite(array) & (array > 0)
        for threshold in THRESHOLDS:
            expected = depth_edge_keep_mask(array, valid, threshold)
            for batched in [False, True]:
                d, v = torch.from_numpy(array).cuda(), torch.from_numpy(valid).cuda()
                if batched:
                    d, v = d[None], v[None]
                got = depth_edge_keep_mask_torch(d, v, threshold).cpu().numpy().reshape(array.shape)
                np.testing.assert_array_equal(got, expected)
    publish(args.output / "filter-parity.json", {"passed": True, "comparisons": len(filter_parity),
        "different_pixels": 0, "batched_and_nonfinite_fixtures_passed": True, "rows": filter_parity})
    del depth_gpu, valid_gpu
    print(json.dumps({"event": "filter_parity", "comparisons": len(filter_parity), "different_pixels": 0}), flush=True)

    start = time.perf_counter()
    checkpoint = torch.load(base["checkpoint"]["path"], map_location="cpu", weights_only=True, mmap=True)
    assert checkpoint["step"] == 272646
    models = dict(checkpoint["config"]["models"], moge_source=str(args.vendor / "MoGe"), radio_source=str(args.vendor / "RADIO"))
    moge, _ = load_moge(models, "cuda")
    decoder = MoGeDecoder(moge)
    del moge
    radio = load_radio(models, "cuda")
    adapter = build_adapter(**checkpoint["adapter_config"])
    adapter.load_state_dict(checkpoint["adapter"], strict=True)
    adapter.cuda().eval().requires_grad_(False)
    del checkpoint
    gc.collect()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    hardware = {"torch": torch.__version__, "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(), "gpu": gpu_status(), "python": platform.python_version(),
                "cpu": next(x.split(':', 1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),
                "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                "cpu_quota": Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
                "load_seconds_excluded": load_seconds,
                "parameters": {name: sum(p.numel() for p in module.parameters()) for name, module in
                               [("radio", radio), ("adapter", adapter), ("moge_decoder", decoder)]}}
    publish(args.output / "loaded.json", hardware)
    print(json.dumps({"event": "loaded", "load_seconds": load_seconds, "gpu": hardware["gpu"]}), flush=True)
    flip_gpu = torch.tensor([1, -1, -1], dtype=torch.float32, device="cuda")
    flip_cpu = np.array([1, -1, -1], dtype=np.float32)
    scenes, parity = [], []

    def run(rgb, mode, instrument=False, return_prediction=False):
        names = ["h2d", "radio_grid_and_backbone", "adapter", "moge_decoder",
                 "camera_recovery_and_projection", "validity_and_rgb", "edge_filter", "output_xyz"]
        events = [torch.cuda.Event(enable_timing=True) for _ in range(len(names) + 1)] if instrument else []
        cursor = 0

        def mark():
            nonlocal cursor
            if instrument:
                events[cursor].record()
                cursor += 1

        torch.cuda.synchronize()
        begin = time.perf_counter()
        image_cpu = (torch.from_numpy(rgb).float().permute(2, 0, 1) / 255)[None]
        cpu_prepared = time.perf_counter()
        mark()
        image = image_cpu.cuda()
        mark()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            features = extract_radio(radio, radio_input(image, radio), adapter.indices)
            mark()
            fitted = adapter(features)
            mark()
        raw = decoder(fitted, (480, 640))
        mark()
        prediction = postprocess(raw, image, models['moge_source'])
        mark()
        depth, valid = prediction['depth'][0], prediction['mask'][0]
        valid = valid & torch.isfinite(depth) & (depth > 0)
        rgb_gpu = image[0].permute(1, 2, 0)
        mark()
        if mode.startswith('gpu'):
            keep = valid if mode == 'gpu_original' else depth_edge_keep_mask_torch(depth, valid)
            mark()
            if mode == 'gpu_cpu_filtered':
                xyz = prediction['points'][0][keep] * flip_gpu
                colors = (rgb_gpu[keep] * 255).round().to(torch.uint8)
                output = (xyz.cpu().numpy(), colors.cpu().numpy())
            elif mode == 'gpu_compact_filtered':
                output = (prediction['points'][0][keep] * flip_gpu, rgb_gpu[keep])
            else:
                output = (prediction['points'][0] * flip_gpu, rgb_gpu, keep)
        else:
            # The official MoGe postprocess already computed XYZ; reuse it
            # instead of doing a second CPU unprojection.
            depth_cpu = depth.cpu().numpy() if mode != 'cpu_original' else None
            valid_cpu = valid.cpu().numpy()
            xyz_cpu = prediction['points'][0].cpu().numpy()
            if mode == 'cpu_original':
                keep = valid_cpu
            elif mode == 'cpu_filtered':
                keep = depth_edge_keep_mask(depth_cpu, valid_cpu)
            else:
                keep = official_keep(depth_cpu, valid_cpu)
            mark()
            output = (xyz_cpu[keep] * flip_cpu, rgb[keep].copy())
        mark()
        torch.cuda.synchronize()
        end = time.perf_counter()
        row = {"mode": mode, "instrumented": instrument, "e2e_ms": (end - begin) * 1000,
               "cpu_normalization_ms": (cpu_prepared - begin) * 1000}
        if instrument:
            assert cursor == len(events)
            row['stream_stage_ms'] = {name: a.elapsed_time(b) for name, a, b in zip(names, events, events[1:])}
            row['stream_total_ms'] = events[0].elapsed_time(events[-1])
            row['outside_stream_ms'] = row['e2e_ms'] - row['stream_total_ms']
        if return_prediction:
            return output, row, prediction, keep
        return output, row

    torch.cuda.reset_peak_memory_stats()
    for selected in base['selected']:
        index, record = selected['index'], selected['row']
        name = f"{index:02d}-{record['dataset']}-480x640"
        with np.load(args.base / (name + '.npz')) as z:
            rgb, expected_depth = z['rgb'].copy(), z['moge_adapter'].copy()
        with np.load(args.clouds / (name + '.npz')) as z:
            expected_camera, expected_mask = z['adapter_camera'].copy(), z['adapter_mask'].copy()
        parity_mode = 'gpu_cpu_filtered' if 'gpu_cpu_filtered' in args.modes else 'gpu_filtered'
        output, timing, prediction, kept = run(rgb, parity_mode, return_prediction=True)
        depth = prediction['depth'][0].cpu().numpy()
        camera = prediction['intrinsics'][0].cpu().numpy()
        mask = prediction['mask'][0].cpu().numpy() & np.isfinite(depth) & (depth > 0)
        np.testing.assert_array_equal(depth, expected_depth)
        np.testing.assert_array_equal(camera, expected_camera)
        np.testing.assert_array_equal(mask, expected_mask)
        keep = kept.cpu().numpy()
        with np.load(args.edges / 'masks' / (name + '.npz')) as z:
            np.testing.assert_array_equal(keep, z['moge_adapter_keep_2'])
        expected_xyz = unproject(expected_depth, expected_camera) * flip_cpu
        xyz = output[0] if parity_mode == 'gpu_cpu_filtered' else output[0].cpu().numpy()[keep]
        np.testing.assert_allclose(xyz, expected_xyz[keep], rtol=2e-5, atol=2e-5)
        if parity_mode == 'gpu_cpu_filtered':
            np.testing.assert_array_equal(output[1], rgb[keep])
        parity.append({'scene': name, 'depth_camera_mask_exact': True, 'edge_mask_exact': True,
                       'parity_mode': parity_mode, 'cpu_colors_exact': parity_mode == 'gpu_cpu_filtered',
                       'maximum_kept_xyz_difference': float(np.max(np.abs(xyz - expected_xyz[keep]))),
                       'warmup_ms': timing['e2e_ms']})
        scenes.append({'name': name, 'rgb': rgb, 'depth': depth, 'valid': mask,
                       'depth_gpu': torch.from_numpy(depth.copy()).cuda(), 'valid_gpu': torch.from_numpy(mask.copy()).cuda()})
    publish(args.output / 'prediction-parity.json', {'passed': True, 'scenes': parity})
    for mode in args.modes:
        for scene in scenes[:4]:
            run(scene['rgb'], mode)
    print(json.dumps({'event': 'warmup_and_parity_complete', 'scenes': len(scenes)}), flush=True)

    rows = []
    log = (args.output / 'timings.jsonl').open('x')
    for round_index in range(4):
        modes = args.modes if round_index % 2 == 0 else list(reversed(args.modes))
        for mode in modes:
            for i in range(32):
                scene = scenes[i % len(scenes)]
                output, row = run(scene['rgb'], mode)
                row.update(scene=scene['name'], round=round_index)
                log.write(json.dumps(row) + '\n')
                rows.append(row)
                del output
            log.flush()
        print(json.dumps({'event': 'timing_round_complete', 'round': round_index + 1}), flush=True)
    profiles = []
    for i in range(64):
        scene = scenes[i % len(scenes)]
        output, row = run(scene['rgb'], 'gpu_filtered', instrument=True)
        row['scene'] = scene['name']
        profiles.append(row)
        log.write(json.dumps(row) + '\n')
        del output
    log.close()

    # Filter-only measurements have pre-resident input arrays/tensors and no
    # network or XYZ work. Each timing includes a completed mask result.
    filters = {'numpy_scipy': lambda s: depth_edge_keep_mask(s['depth'], s['valid']),
               'numpy_official': lambda s: official_keep(s['depth'], s['valid']),
               'gpu_torch': lambda s: depth_edge_keep_mask_torch(s['depth_gpu'], s['valid_gpu'])}
    if args.skip_filter_only:
        filters = {}
    for function in filters.values():
        for scene in scenes:
            function(scene)
    filter_rows = []
    for round_index in range(4):
        keys = list(filters) if round_index % 2 == 0 else list(reversed(filters))
        for key in keys:
            for i in range(64):
                scene = scenes[i % len(scenes)]
                torch.cuda.synchronize()
                begin = time.perf_counter()
                keep = filters[key](scene)
                if key == 'gpu_torch':
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - begin) * 1000
                filter_rows.append({'implementation': key, 'scene': scene['name'], 'round': round_index, 'ms': elapsed})
    publish(args.output / 'filter-timings.json', filter_rows)
    stage_values = defaultdict(list)
    for row in profiles:
        for key, value in row['stream_stage_ms'].items():
            stage_values[key].append(value)
        stage_values['outside_stream'].append(row['outside_stream_ms'])
    final = {
        'completed': True, 'plan_sha256': digest(args.output / 'plan.json'),
        'resolution_hw': [480, 640], 'batch': 1, 'threshold': .04,
        'e2e_ms': {mode: stats([r['e2e_ms'] for r in rows if r['mode'] == mode]) for mode in args.modes},
        'rounds_ms': {mode: [stats([r['e2e_ms'] for r in rows if r['mode'] == mode and r['round'] == i]) for i in range(4)] for mode in args.modes},
        'filter_only_ms': {key: stats([r['ms'] for r in filter_rows if r['implementation'] == key]) for key in filters},
        'gpu_filtered_stage_ms': {key: stats(values) for key, values in stage_values.items()},
        'gpu_filtered_instrumented_e2e_ms': stats([r['e2e_ms'] for r in profiles]),
        'filter_gpu_matches_saved_masks': True, 'predictions_match_saved_depth_camera_masks': True,
        'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved(),
        'gpu_after': gpu_status(), 'processes_after': processes(),
        'memory_events_after': Path('/sys/fs/cgroup/memory.events').read_text(),
        'ppd_calls': 0, 'moge_encoder_calls': 0, 'optimizer_updates': 0,
    }
    publish(args.output / 'report.json', final)
    print(json.dumps({'event': 'complete', 'e2e_ms': final['e2e_ms'], 'filter_only_ms': final['filter_only_ms']}), flush=True)
    signal.alarm(0)


if __name__ == '__main__':
    main()
