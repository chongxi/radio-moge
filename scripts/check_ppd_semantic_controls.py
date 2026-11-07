"""Supplement a completed PPD probe with zero semantics and a second noise seed."""

import argparse
import gc
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

from probe_ppd_feature_swap import digest, publish, sample, scalar_metrics
from radio_moge.adapters import build_adapter
from radio_moge.upstream import extract_radio, load_moge, load_radio, radio_input


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ppd-source", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = json.loads((args.base / "plan.json").read_text())
    report = json.loads((args.base / "report.json").read_text())
    assert report["completed"] and all(p["exact"] for p in report["teacher_boundary_parity"])
    assert digest(args.base / "plan.json") == report["plan_sha256"]
    for key in ("checkpoint", "ppd_checkpoint"):
        assert digest(base[key]["path"]) == base[key]["sha256"]
    plan = {
        "base_plan_sha256": digest(args.base / "plan.json"),
        "base_report_sha256": digest(args.base / "report.json"),
        "script_sha256": digest(__file__), "resolution_hw": [768, 1024],
        "sampling_steps": base["sampling_steps"], "images": 12,
        "controls": ["Zero semantics with the original noise", "Native and fitted semantics with seed offset +100000"],
        "alignment_seed": "Same original per-image seed in every condition, independent of diffusion noise seed",
        "interpretation": "Zero conditioning is out-of-distribution; it measures reliance on semantic input, not performance of a retrained RGB-only model",
        "optimizer_steps": 0,
    }
    publish(args.output / "plan.json", plan)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(args.ppd_source))
    from ppd.models.dit import DiT
    from ppd.utils.diffusion.schedule import LinearSchedule
    from ppd.utils.diffusion.timesteps import Timesteps
    from ppd.utils.diffusion.sampler import EulerSampler
    from ppd.utils.align_depth_func import recover_metric_depth_ransac

    device = torch.device("cuda")
    checkpoint = torch.load(base["checkpoint"]["path"], map_location="cpu", weights_only=True, mmap=True)
    config = dict(checkpoint["config"]["models"], moge_source=str(args.vendor / "MoGe"), radio_source=str(args.vendor / "RADIO"))
    moge, _ = load_moge(config, device)
    encoder = moge.encoder
    del moge
    radio = load_radio(config, device)
    adapter = build_adapter(**checkpoint["adapter_config"])
    adapter.load_state_dict(checkpoint["adapter"], strict=True)
    adapter.to(device).eval().requires_grad_(False)
    del checkpoint
    state = torch.load(base["ppd_checkpoint"]["path"], map_location="cpu", weights_only=True, mmap=True)
    assert all(k.startswith("dit.") for k in state)
    with torch.device("meta"):
        dit = DiT()
    dit.load_state_dict({key[4:]: value for key, value in state.items()}, strict=True, assign=True)
    dit.to(device).eval().requires_grad_(False)
    del state
    gc.collect()
    timesteps = Timesteps(T=1000, steps=base["sampling_steps"], device=device)
    sampler = EulerSampler(LinearSchedule(T=1000), timesteps, prediction_type="velocity")

    results = []
    for index in range(12):
        name = f"{index:02d}-nyuv2-768x1024"
        with np.load(args.base / (name + ".npz")) as saved:
            image = torch.from_numpy(saved["rgb"].copy()).permute(2, 0, 1)[None].float().to(device) / 255
            target, mask = saved["target"].copy(), saved["valid"].copy()
        noise_seed = base["seed"] + index
        output = {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            spatial, _ = encoder(image, 48, 64, return_class_token=True)
            native = spatial.float().flatten(2).transpose(1, 2)
            fitted = adapter(extract_radio(radio, radio_input(image, radio), adapter.indices)).spatial
            fitted = fitted.flatten(2).transpose(1, 2)
            for key, semantics, seed in (
                ("ppd_zero", torch.zeros_like(native), noise_seed),
                ("ppd_native_seed2", native, noise_seed + 100000),
                ("ppd_adapter_seed2", fitted, noise_seed + 100000),
            ):
                noise = torch.randn((1, 1, 768, 1024), generator=torch.Generator().manual_seed(seed)).to(device)
                output[key] = sample(dit, image, semantics, noise, timesteps, sampler)[0, 0].float().cpu().numpy()
        aligned, metrics = {}, {}
        for key, value in output.items():
            assert np.isfinite(value).all()
            small = cv2.resize(value, (640, 480), interpolation=cv2.INTER_LINEAR)
            np.random.seed(noise_seed)
            aligned[key] = recover_metric_depth_ransac(small, target, mask)
            assert np.isfinite(aligned[key]).all()
            metrics[key] = scalar_metrics(aligned[key], target, mask)
        row = {"name": name, "index": index, "metrics": metrics}
        temporary = args.output / (name + ".npz.new")
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **output, **{"aligned_" + k: v for k, v in aligned.items()})
        os.replace(temporary, args.output / (name + ".npz"))
        publish(args.output / (name + ".json"), row)
        results.append(row)
        print(json.dumps(row), flush=True)
    aggregate = {method: {metric: float(np.mean([r["metrics"][method][metric] for r in results]))
                          for metric in ("abs_rel", "rmse", "delta1", "log1p_gradient_mae")}
                 for method in ("ppd_zero", "ppd_native_seed2", "ppd_adapter_seed2")}
    publish(args.output / "report.json", {"completed": True, "plan_sha256": digest(args.output / "plan.json"),
                                          "results": results, "aggregate": aggregate, "optimizer_steps": 0})
    print(json.dumps({"complete": True, "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
