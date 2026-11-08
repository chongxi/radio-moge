"""Time the existing 640x480 RADIO -> fitted features -> PPD inference path.

Two outputs are measured: raw relative PPD depth, and the metric-calibrated depth
used by the point-cloud viewer. No teacher encoder is called and no weights change.
"""

import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from probe_ppd_feature_swap import digest, publish
from radio_moge.adapters import build_adapter
from radio_moge.decoder import MoGeDecoder, postprocess
from radio_moge.pointcloud import unproject
from radio_moge.upstream import extract_radio, load_moge, load_radio, radio_input


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {"n": len(a), "mean": float(a.mean()), "p50": float(np.median(a)),
            "p90": float(np.quantile(a, .90)), "p95": float(np.quantile(a, .95)),
            "min": float(a.min()), "max": float(a.max()), "std": float(a.std())}


def gpu_status():
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.used,memory.total,utilization.gpu,power.draw,clocks.sm",
        "--format=csv,noheader"], text=True).strip()


def processes():
    return subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader"], text=True).strip()


class Runner:
    def __init__(self, radio, adapter, decoder, dit, timesteps, sampler, models, align, seed):
        self.radio, self.adapter, self.decoder, self.dit = radio, adapter, decoder, dit
        self.timesteps, self.sampler, self.models, self.align, self.seed = timesteps, sampler, models, align, seed

    @torch.inference_mode()
    def run(self, scene, metric=False, instrument=False):
        names = ["h2d", "radio_resize", "radio", "adapter", "ppd_setup"]
        for i in range(len(self.timesteps)):
            names += [f"dit_step_{i + 1}", f"euler_step_{i + 1}"]
        names += ["ppd_output"]
        if metric:
            names += ["moge_decoder", "camera_recovery_mixed_cpu_gpu"]
        names += ["d2h"]
        events = [torch.cuda.Event(enable_timing=True) for _ in range(len(names) + 1)] if instrument else []
        cursor = 0

        def mark():
            nonlocal cursor
            if instrument:
                events[cursor].record()
                cursor += 1

        torch.cuda.synchronize()
        begin = time.perf_counter()
        # Identical to the main experiment: divide on CPU, then transfer.
        image_cpu = (torch.from_numpy(scene["rgb"]).float().permute(2, 0, 1) / 255)[None]
        prepared = time.perf_counter()
        mark()
        image = image_cpu.cuda()
        mark()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            resized = radio_input(image, self.radio)
            mark()
            features = extract_radio(self.radio, resized, self.adapter.indices)
            mark()
            fitted = self.adapter(features)
            mark()
            semantics = fitted.spatial.flatten(2).transpose(1, 2)
            noise_start = time.perf_counter()
            noise = torch.randn((1, 1, 480, 640), generator=torch.Generator().manual_seed(self.seed + scene["index"])).cuda()
            noise_ms = (time.perf_counter() - noise_start) * 1000
            latent, conditioning = noise.clone(), image - .5
            mark()
            for timestep in self.timesteps:
                velocity = self.dit(x=torch.cat((latent, conditioning), 1), semantics=semantics, timestep=timestep)
                mark()
                latent = self.sampler.step(pred=velocity, x_t=latent, t=timestep)
                mark()
            relative = latent + .5
            mark()
            if metric:
                raw = self.decoder(fitted, (480, 640))
                mark()
                prediction = postprocess(raw, image, self.models["moge_source"])
                mark()
            output = {"relative": relative[0, 0].float().cpu().numpy()}
            if metric:
                output.update(moge=prediction["depth"][0].cpu().numpy(),
                              camera=prediction["intrinsics"][0].cpu().numpy(),
                              mask=prediction["mask"][0].cpu().numpy().astype(bool))
                output["mask"] &= np.isfinite(output["moge"]) & (output["moge"] > 0)
            mark()
        torch.cuda.synchronize()
        network_done = time.perf_counter()
        align_ms = None
        if metric:
            start = time.perf_counter()
            np.random.seed(self.seed + scene["index"])
            output["metric"] = self.align(output["relative"], output["moge"], output["mask"])
            align_ms = (time.perf_counter() - start) * 1000
        end = time.perf_counter()
        row = {"scene": scene["name"], "index": scene["index"], "metric": metric,
               "instrumented": instrument, "e2e_ms": (end - begin) * 1000,
               "cpu_rgb_normalization_ms": (prepared - begin) * 1000,
               "network_to_cpu_ms": (network_done - begin) * 1000,
               "cpu_noise_creation_and_copy_host_ms_nonadditive": noise_ms}
        if instrument:
            assert cursor == len(events)
            row["stream_stage_ms"] = {name: a.elapsed_time(b) for name, a, b in zip(names, events, events[1:])}
            row["stream_total_ms"] = events[0].elapsed_time(events[-1])
            assert abs(sum(row["stream_stage_ms"].values()) - row["stream_total_ms"]) < .1
            row["outside_stream_ms"] = row["network_to_cpu_ms"] - row["stream_total_ms"]
        if metric:
            row["cpu_ransac_ms"] = align_ms
            start = time.perf_counter()
            points = unproject(output["metric"], output["camera"])[output["mask"]].astype(np.float32)
            points *= np.array([1, -1, -1], dtype=np.float32)
            rgb = scene["rgb"][output["mask"]].copy()
            row["optional_cpu_pointcloud_ms_excluded_from_depth_e2e"] = (time.perf_counter() - start) * 1000
            assert len(points) == len(rgb) and np.isfinite(points).all()
        return output, row


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "clouds", "output", "vendor", "ppd-source"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--per-block", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--deadline-seconds", type=int, default=600)
    args = parser.parse_args()
    assert args.per_block in (16, 32) and args.rounds == 2 and args.deadline_seconds <= 900
    args.output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.deadline_seconds
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    base = json.loads((args.base / "plan.json").read_text())
    assert base["sampling_steps"] == 4
    for key in ("checkpoint", "ppd_checkpoint"):
        assert digest(base[key]["path"]) == base[key]["sha256"]
    plan = {"script_sha256": digest(__file__), "base_plan_sha256": digest(args.base / "plan.json"),
            "cloud_report_sha256": digest(args.clouds / "report.json"), "resolution_hw": [480, 640],
            "batch_size": 1, "sampling_steps": 4, "precision": "FP32 weights; BF16 autocast; MoGe decoder FP32",
            "warmup": "16 original scenes; all three saved output types must match before timing",
            "per_block": args.per_block, "rounds": args.rounds, "scenes": 16,
            "profiles": ["relative_plain", "metric_plain", "relative_instrumented", "metric_instrumented"],
            "input": "CPU uint8 RGB at 640x480 already decoded and cropped; original CPU normalization",
            "output": "CPU float32 relative depth OR calibrated metric depth plus intrinsics/mask",
            "included": ["normalization", "host/device transfers", "RADIO", "adapter", "CPU noise creation",
                         "four PPD DiT calls and Euler updates", "MoGe decoder/camera/RANSAC for metric profile"],
            "excluded": ["model/checkpoint loading", "disk/image decode/crop/resize", "network transport", "PLY/image serialization", "viewer rendering"],
            "gpu_stage_timing": "CUDA events at consecutive boundaries, no added interstage synchronization; mixed camera stage includes CPU-induced stream idle time",
            "e2e_timing": "Synchronized wall clock; plain profile inserts no CUDA events; CPU input to CPU output",
            "compile": False, "tensorrt": False, "cuda_graphs": False, "optimizer_updates": 0,
            "teacher_encoder_forward_calls": 0, "radio_forward_calls_per_frame": 1,
            "shared_features": "Metric profile reuses the one adapter output for both PPD and MoGe heads",
            "device_before": gpu_status(), "processes_before": processes()}
    publish(args.output / "plan.json", plan)
    sys.path.insert(0, str(args.ppd_source))
    from ppd.models.dit import DiT
    from ppd.utils.diffusion.schedule import LinearSchedule
    from ppd.utils.diffusion.timesteps import Timesteps
    from ppd.utils.diffusion.sampler import EulerSampler
    from ppd.utils.align_depth_func import recover_metric_depth_ransac

    load_start = time.perf_counter()
    checkpoint = torch.load(base["checkpoint"]["path"], map_location="cpu", weights_only=True, mmap=True)
    assert checkpoint["step"] == 272646
    models = dict(checkpoint["config"]["models"], moge_source=str(args.vendor / "MoGe"), radio_source=str(args.vendor / "RADIO"))
    moge, _ = load_moge(models, "cuda")
    decoder = MoGeDecoder(moge)
    del moge  # The MoGe encoder is not part of either timed pipeline.
    radio = load_radio(models, "cuda")
    adapter = build_adapter(**checkpoint["adapter_config"])
    adapter.load_state_dict(checkpoint["adapter"], strict=True)
    adapter.cuda().eval().requires_grad_(False)
    del checkpoint
    state = torch.load(base["ppd_checkpoint"]["path"], map_location="cpu", weights_only=True, mmap=True)
    assert all(k.startswith("dit.") for k in state)
    with torch.device("meta"):
        dit = DiT()
    dit.load_state_dict({k[4:]: v for k, v in state.items()}, strict=True, assign=True)
    dit.cuda().eval().requires_grad_(False)
    del state
    gc.collect()
    torch.cuda.synchronize()
    load_ms = (time.perf_counter() - load_start) * 1000
    timesteps = Timesteps(T=1000, steps=4, device="cuda")
    sampler = EulerSampler(LinearSchedule(T=1000), timesteps, prediction_type="velocity")
    runner = Runner(radio, adapter, decoder, dit, timesteps, sampler, models, recover_metric_depth_ransac, base["seed"])
    hardware = {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "gpu": torch.cuda.get_device_name(), "python": platform.python_version(),
                "cpu": next(x.split(':', 1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),
                "cpu_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                "cpu_quota": Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
                "parameters": {k: sum(p.numel() for p in m.parameters()) for k, m in
                               (("radio", radio), ("adapter", adapter), ("dit", dit), ("moge_decoder", decoder))},
                "radio_min_resolution_step": radio.min_resolution_step, "radio_patch_size": radio.patch_size,
                "model_loading_ms_not_in_inference": load_ms}
    publish(args.output / "loaded.json", hardware)
    print(json.dumps({"event": "loaded", **hardware}), flush=True)
    scenes, parity = [], []
    torch.cuda.reset_peak_memory_stats()
    for selected in base["selected"]:
        index, row = selected["index"], selected["row"]
        name = f"{index:02d}-{row['dataset']}-480x640"
        with np.load(args.base / (name + ".npz")) as z:
            scene = {"index": index, "name": name, "rgb": z["rgb"].copy()}
            expected_relative, expected_moge = z["ppd_adapter"].copy(), z["moge_adapter"].copy()
        scenes.append(scene)
        output, timed = runner.run(scene, metric=True)
        with np.load(args.clouds / (name + ".npz")) as z:
            checks = {"relative": np.array_equal(output["relative"], expected_relative),
                      "moge": np.array_equal(output["moge"], expected_moge),
                      "metric": np.array_equal(output["metric"], z["ppd_adapter"]),
                      "camera": np.array_equal(output["camera"], z["adapter_camera"]),
                      "mask": np.array_equal(output["mask"], z["adapter_mask"])}
        if not all(checks.values()):
            raise AssertionError(f"Saved-output parity failed: {name}: {checks}")
        parity.append({"scene": name, "exact": checks, "warmup_e2e_ms": timed["e2e_ms"]})
        print(json.dumps({"event": "warmup", **parity[-1]}), flush=True)
        if time.monotonic() >= deadline:
            raise TimeoutError("Benchmark deadline during warmup")
    publish(args.output / "parity.json", {"all_exact": True, "cases": parity})
    records, telemetry = [], []
    orders = [["relative_plain", "metric_plain", "relative_instrumented", "metric_instrumented"],
              ["metric_instrumented", "relative_instrumented", "metric_plain", "relative_plain"]]
    with (args.output / "measurements.jsonl").open("x") as journal:
        for round_index, order in enumerate(orders):
            for profile in order:
                telemetry.append({"round": round_index, "profile": profile, "gpu": gpu_status(), "processes": processes()})
                block = []
                for iteration in range(args.per_block):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Benchmark deadline")
                    scene = scenes[(iteration + round_index * 7) % len(scenes)]
                    _, row = runner.run(scene, metric=profile.startswith("metric"), instrument=profile.endswith("instrumented"))
                    row.update(round=round_index, profile=profile, iteration=iteration)
                    records.append(row)
                    block.append(row["e2e_ms"])
                    journal.write(json.dumps(row) + "\n")
                    journal.flush()
                print(json.dumps({"event": "block", "round": round_index, "profile": profile, "e2e_ms": stats(block)}), flush=True)
    summary = {}
    for profile in orders[0]:
        rows = [r for r in records if r["profile"] == profile]
        result = {"e2e_ms": stats([r["e2e_ms"] for r in rows]),
                  "network_to_cpu_ms": stats([r["network_to_cpu_ms"] for r in rows]),
                  "per_round_e2e_ms": [stats([r["e2e_ms"] for r in rows if r["round"] == i]) for i in range(2)]}
        result["serial_fps_from_mean"] = 1000 / result["e2e_ms"]["mean"]
        if rows[0]["metric"]:
            result["cpu_ransac_ms"] = stats([r["cpu_ransac_ms"] for r in rows])
            key = "optional_cpu_pointcloud_ms_excluded_from_depth_e2e"
            result[key] = stats([r[key] for r in rows])
        if rows[0]["instrumented"]:
            result["stream_stage_ms"] = {key: stats([r["stream_stage_ms"][key] for r in rows]) for key in rows[0]["stream_stage_ms"]}
            result["outside_stream_ms"] = stats([r["outside_stream_ms"] for r in rows])
            result["stream_total_ms"] = stats([r["stream_total_ms"] for r in rows])
        summary[profile] = result
    result = {"completed": True, "plan_sha256": digest(args.output / "plan.json"), "hardware": hardware,
              "all_saved_outputs_bit_exact": True, "parity_cases": len(parity), "summary": summary,
              "records": len(records), "telemetry": telemetry,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
              "processes_after": processes(), "gpu_after": gpu_status(),
              "optimizer_updates": 0, "output_file_writes_inside_timed_region": False}
    publish(args.output / "report.json", result)
    print(json.dumps({"event": "complete", "report": str(args.output / "report.json"), "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
