"""Bounded H200 speed/quality comparison for fixed-grid PPD inference."""

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch

from benchmark_ppd_latency import gpu_status, processes, stats
from probe_ppd_feature_swap import digest, publish, sample, scalar_metrics
from radio_moge.adapters import build_adapter
from radio_moge.decoder import MoGeDecoder, postprocess
from radio_moge.ppd_fast import StaticPPDSampler, PPDGraph
from radio_moge.upstream import extract_radio, load_moge, load_radio, radio_input


def expire(*_):
    raise TimeoutError("PPD speedup experiment exceeded its fixed time budget")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "clouds", "output", "vendor", "ppd-source"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGALRM, expire)
    signal.alarm(900 if args.compile else 600)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    base = json.loads((args.base / "plan.json").read_text())
    for key in ("checkpoint", "ppd_checkpoint"):
        assert digest(base[key]["path"]) == base[key]["sha256"]
    publish(args.output / "plan.json", {
        "base_plan_sha256": digest(args.base / "plan.json"), "script_sha256": digest(__file__),
        "fast_source_sha256": digest(Path(__file__).resolve().parents[1] / "src/radio_moge/ppd_fast.py"),
        "compile_velocity": args.compile, "resolution_hw": [480, 640], "batch": 1,
        "dtype": "FP32 weights; BF16 autocast; MoGe decoder FP32", "steps_tested": [4, 2, 1],
        "scenes": 16, "quality_gt_scenes": 12, "optimizer_updates": 0,
        "ppd_measurements": "GPU-resident RGB/features/noise to GPU depth, synchronized wall time, 64 per mode",
        "end_to_end": "CPU uint8 RGB to CPU depth, real RADIO+adapter per request; 32 per mode and output type",
        "full_pipeline_warmup": "All 16 scenes through original four-step metric pipeline before end-to-end timing",
        "metric_branch": "Reuse adapter features for MoGe heads, own metric reference/camera/mask, then official CPU RANSAC",
        "quality": "Same 12 NYU validation images, GT-assisted relative alignment as original probe; also report model-calibrated metric errors",
        "hardware_before": gpu_status(), "processes_before": processes(),
    })
    sys.path.insert(0, str(args.ppd_source))
    from ppd.models.dit import DiT
    from ppd.utils.diffusion.timesteps import Timesteps
    from ppd.utils.diffusion.schedule import LinearSchedule
    from ppd.utils.diffusion.sampler import EulerSampler
    from ppd.utils.align_depth_func import recover_metric_depth_ransac

    checkpoint = torch.load(base["checkpoint"]["path"], map_location="cpu", weights_only=True, mmap=True)
    models = dict(checkpoint["config"]["models"], moge_source=str(args.vendor / "MoGe"), radio_source=str(args.vendor / "RADIO"))
    moge, _ = load_moge(models, "cuda")
    decoder = MoGeDecoder(moge)
    del moge
    radio = load_radio(models, "cuda")
    adapter = build_adapter(**checkpoint["adapter_config"])
    adapter.load_state_dict(checkpoint["adapter"], strict=True)
    adapter.cuda().eval().requires_grad_(False)
    del checkpoint
    state = torch.load(base["ppd_checkpoint"]["path"], map_location="cpu", weights_only=True, mmap=True)
    with torch.device("meta"):
        dit = DiT()
    dit.load_state_dict({k[4:]: v for k, v in state.items()}, strict=True, assign=True)
    dit.cuda().eval().requires_grad_(False)
    del state
    gc.collect()
    torch.cuda.synchronize()
    print(json.dumps({"event": "loaded", "gpu": gpu_status()}), flush=True)
    schedules = {n: Timesteps(T=1000, steps=n, device="cuda") for n in (4, 2, 1)}
    samplers = {n: EulerSampler(LinearSchedule(T=1000), t, prediction_type="velocity") for n, t in schedules.items()}

    def original(n, image, semantics, noise):
        return sample(dit, image, semantics, noise, schedules[n], samplers[n])

    scenes, references, quality = [], {n: [] for n in (4, 2, 1)}, {}
    for selected in base["selected"]:
        index, row = selected["index"], selected["row"]
        name = f"{index:02d}-{row['dataset']}-480x640"
        with np.load(args.base / (name + ".npz")) as z:
            cpu = {k: z[k].copy() for k in ("rgb", "target", "valid", "moge_adapter", "ppd_adapter")}
        image = (torch.from_numpy(cpu["rgb"]).float().permute(2, 0, 1) / 255)[None].cuda()
        noise = torch.randn((1, 1, 480, 640), generator=torch.Generator().manual_seed(base["seed"] + index)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = adapter(extract_radio(radio, radio_input(image, radio), adapter.indices))
            semantics = encoded.spatial.flatten(2).transpose(1, 2)
            for n in (4, 2, 1):
                raw = original(n, image, semantics, noise)[0, 0].float().cpu().numpy().copy()
                assert np.isfinite(raw).all()
                if n == 4:
                    assert np.array_equal(raw, cpu["ppd_adapter"]), name
                references[n].append(raw)
        with np.load(args.clouds / (name + ".npz")) as z:
            cpu["camera"], cpu["model_mask"] = z["adapter_camera"].copy(), z["adapter_mask"].copy()
        scenes.append({"index": index, "name": name, "image": image, "semantics": semantics,
                       "noise": noise, "cpu": cpu, "has_gt": row["dataset"] == "nyuv2"})
        print(json.dumps({"event": "prepared", "scene": name, "four_step_saved_exact": True}), flush=True)

    def score(label, predictions, save=False):
        rows = []
        if save:
            (args.output / label).mkdir()
        for scene, raw in zip(scenes, predictions):
            cpu, seed = scene["cpu"], base["seed"] + scene["index"]
            np.random.seed(seed)
            metric = recover_metric_depth_ransac(raw, cpu["moge_adapter"], cpu["model_mask"])
            result = {"name": scene["name"], "has_gt": scene["has_gt"],
                      "raw_four_step_rmse": float(np.sqrt(np.mean((raw - references[4][scene["index"]]) ** 2)))}
            if scene["has_gt"]:
                np.random.seed(seed)
                aligned = recover_metric_depth_ransac(raw, cpu["target"], cpu["valid"])
                result["relative_metrics"] = scalar_metrics(aligned, cpu["target"], cpu["valid"])
                result["model_calibrated_metrics"] = scalar_metrics(metric, cpu["target"], cpu["valid"])
            if save:
                p = args.output / label / (scene["name"] + ".npz")
                q = p.with_name(p.name + ".new")
                with q.open("xb") as f:
                    np.savez_compressed(f, raw=raw, metric=metric, intrinsics=cpu["camera"], mask=cpu["model_mask"])
                os.replace(q, p)
            rows.append(result)
        aggregate = {}
        for kind in ("relative_metrics", "model_calibrated_metrics"):
            aggregate[kind] = {k: float(np.mean([r[kind][k] for r in rows if r["has_gt"]]))
                               for k in ("abs_rel", "rmse", "delta1", "log1p_gradient_mae")}
        result = {"rows": rows, "aggregate": aggregate, "images_with_gt": 12}
        publish(args.output / (label + "-quality.json"), result)
        print(json.dumps({"event": "quality", "label": label, "aggregate": aggregate}), flush=True)
        return result

    for n in (4, 2, 1):
        quality[f"steps_{n}"] = score(f"steps-{n}", references[n], save=n != 4)

    # Preserve an untouched native model for interleaved reference timings.
    fast_dit = copy.deepcopy(dit)
    fast = {n: StaticPPDSampler(fast_dit, steps=n) for n in (4, 2, 1)}
    compilation = None
    if args.compile:
        from torch._inductor import config as inductor_config
        inductor_config.compile_threads = 2
        start = time.perf_counter()
        compiled_velocity = torch.compile(fast[4].velocity, fullgraph=True, dynamic=False,
                                          options={"max_autotune": False, "triton.cudagraphs": False})
        for module in fast.values():
            module.velocity = compiled_velocity
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fast[4](scenes[0]["image"], scenes[0]["semantics"], scenes[0]["noise"])
        torch.cuda.synchronize()
        compilation = {"seconds": time.perf_counter() - start, "backend": "inductor", "fullgraph": True,
                       "dynamic": False, "max_autotune": False, "compile_threads": 2}
        print(json.dumps({"event": "compiled", **compilation}), flush=True)
    variants = {"original4": lambda im, se, no: original(4, im, se, no), "cached4": fast[4]}
    parity = {}
    graphs = {}
    for n in (4, 2, 1):
        errors, predictions = [], []
        for scene, reference in zip(scenes, references[n]):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = fast[n](scene["image"], scene["semantics"], scene["noise"])[0, 0].float().cpu().numpy().copy()
            errors.append({"scene": scene["name"], "exact": bool(np.array_equal(p, reference)),
                           "max_abs": float(np.max(np.abs(p - reference))), "rmse": float(np.sqrt(np.mean((p - reference) ** 2)))})
            predictions.append(p)
        if not args.compile:
            assert all(x["exact"] for x in errors), errors
        else:
            quality[f"compiled_steps_{n}"] = score(f"compiled-steps-{n}", predictions, save=True)
        parity[f"cached{n}"] = errors
        graphs[n] = PPDGraph(fast[n], scenes[0]["image"], scenes[0]["semantics"], scenes[0]["noise"])
        errors = []
        for scene, expected in zip(scenes, predictions):
            p = graphs[n](scene["image"], scene["semantics"], scene["noise"])[0, 0].float().cpu().numpy().copy()
            errors.append({"scene": scene["name"], "exact": bool(np.array_equal(p, expected)),
                           "max_abs": float(np.max(np.abs(p - expected)))})
        assert all(x["exact"] for x in errors), errors
        parity[f"graph{n}"] = errors
        variants[f"graph{n}"] = graphs[n]
        print(json.dumps({"event": "graph_ready", "steps": n, "graph_matches_eager_exactly": True,
                          "matches_original_exactly": all(x["exact"] for x in parity[f"cached{n}"])}), flush=True)
    publish(args.output / "parity.json", parity)
    # Check the first graph again after all shared-model graphs have been created.
    p = graphs[4](scenes[3]["image"], scenes[3]["semantics"], scenes[3]["noise"])[0, 0].cpu().numpy()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = fast[4](scenes[3]["image"], scenes[3]["semantics"], scenes[3]["noise"])[0, 0].cpu().numpy()
    assert np.array_equal(p, expected)

    timings = {key: [] for key in variants}
    orders = [list(variants), list(reversed(variants))]
    for order in orders:
        for key in order:
            fn = variants[key]
            for i in range(32):
                scene = scenes[i % 16]
                torch.cuda.synchronize()
                begin = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = fn(scene["image"], scene["semantics"], scene["noise"])
                torch.cuda.synchronize()
                timings[key].append((time.perf_counter() - begin) * 1000)
            print(json.dumps({"event": "timing", "mode": key, "ppd_ms": stats(timings[key])}), flush=True)

    def whole(scene, fn, metric):
        torch.cuda.synchronize()
        start = time.perf_counter()
        image = (torch.from_numpy(scene["cpu"]["rgb"]).float().permute(2, 0, 1) / 255)[None].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fitted = adapter(extract_radio(radio, radio_input(image, radio), adapter.indices))
            semantics = fitted.spatial.flatten(2).transpose(1, 2)
            noise = torch.randn((1, 1, 480, 640), generator=torch.Generator().manual_seed(base["seed"] + scene["index"])).cuda()
            relative = fn(image, semantics, noise)
            if metric:
                prediction = postprocess(decoder(fitted, (480, 640)), image, models["moge_source"])
            relative = relative[0, 0].float().cpu().numpy()
            if metric:
                reference = prediction["depth"][0].cpu().numpy()
                mask = prediction["mask"][0].cpu().numpy().astype(bool)
                camera = prediction["intrinsics"][0].cpu().numpy()
                mask &= np.isfinite(reference) & (reference > 0)
        torch.cuda.synchronize()
        if metric:
            np.random.seed(base["seed"] + scene["index"])
            relative = recover_metric_depth_ransac(relative, reference, mask)
        elapsed = (time.perf_counter() - start) * 1000
        assert np.isfinite(relative).all()
        return elapsed

    # Model-only checks above do not exercise the MoGe camera solve. Warm that
    # complete path explicitly, so its first-use cost cannot bias the baseline.
    warmup_ms = [whole(scene, variants["original4"], True) for scene in scenes]
    publish(args.output / "full-pipeline-warmup.json", {"untimed": True, "scene_ms": warmup_ms})
    print(json.dumps({"event": "full_pipeline_warmed", "scenes": len(warmup_ms)}), flush=True)
    e2e = {key: {"relative": [], "metric": []} for key in ("original4", "graph4", "graph2", "graph1")}
    for reverse in (False, True):
        for key in (list(reversed(e2e)) if reverse else e2e):
            for metric in (False, True):
                values = e2e[key]["metric" if metric else "relative"]
                for scene in scenes:
                    values.append(whole(scene, variants[key], metric))
            print(json.dumps({"event": "e2e", "mode": key, "times_ms": {k: stats(v) for k, v in e2e[key].items()}}), flush=True)
    publish(args.output / "timing-records.json", {"ppd_ms": timings, "e2e_ms": e2e})
    summary = {"completed": True, "compile": compilation, "ppd_ms": {k: stats(v) for k, v in timings.items()},
               "e2e_ms": {key: {k: stats(v) for k, v in values.items()} for key, values in e2e.items()},
               "quality": {key: value["aggregate"] for key, value in quality.items()},
               "saved_four_step_reference_exact": True, "raw_parity": parity,
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
               "gpu_after": gpu_status(), "processes_after": processes(), "optimizer_updates": 0,
               "plan_sha256": digest(args.output / "plan.json")}
    publish(args.output / "report.json", summary)
    signal.alarm(0)
    print(json.dumps({"event": "complete", "ppd_ms": summary["ppd_ms"], "e2e_ms": summary["e2e_ms"]}), flush=True)


if __name__ == "__main__":
    main()
