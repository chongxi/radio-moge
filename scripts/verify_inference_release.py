"""Real-weight parity check for the installed inference wheel, run on a GPU.

No training, optimizer or reference files are modified. The output is a JSON
verification report, not a new accuracy benchmark.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from radio_moge.adapters import build_adapter
from radio_moge.checkpoint import atomic_json
from radio_moge.decoder import postprocess
from radio_moge.deployment import RadioMoGe
from radio_moge.upstream import extract_radio, radio_input


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("adapter", "checkpoint", "reference", "radio-source", "moge-source", "output"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = RadioMoGe.from_pretrained(args.adapter, device="cuda",
                                     radio_source=args.radio_source, moge_source=args.moge_source)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    original = build_adapter(**state["adapter_config"]).eval().to("cuda")
    original.load_state_dict(state["adapter"], strict=True)
    for key, tensor in model.adapter.state_dict().items():
        torch.testing.assert_close(tensor, original.state_dict()[key], rtol=0, atol=0)
    if model.metadata["training_step"] != state["step"]:
        raise AssertionError("training step mismatch")
    rows = []
    for path in sorted(Path(args.reference).glob("*.npz"))[:args.limit]:
        with np.load(path, allow_pickle=False) as saved:
            image = torch.from_numpy(saved["rgb"].copy()).permute(2, 0, 1)[None].float().cuda() / 255
        torch.cuda.synchronize()
        start = time.perf_counter()
        with model._autocast():
            features = extract_radio(model.radio, radio_input(image, model.radio), model.indices)
            expected = postprocess(model.decoder(original(features), image.shape[-2:]),
                                   image, model.moge_source)
        actual = model.infer_from_features(features, image)
        from_rgb = model.infer(image)
        torch.cuda.synchronize()
        row = dict(sample=path.name, hw=list(image.shape[-2:]),
                   check_seconds=round(time.perf_counter() - start, 3), outputs={})
        for key, value in expected.items():
            # BF16 inference is deterministic on this fixed CUDA device and shape.
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0, equal_nan=True)
            torch.testing.assert_close(from_rgb[key], actual[key], rtol=0, atol=0, equal_nan=True)
            error = (actual[key].float() - value.float()).abs()
            row["outputs"][key] = dict(shape=list(value.shape),
                                       max_abs_diff=float(torch.nan_to_num(error).max()))
        row["valid_percent"] = float(actual["mask"].float().mean()) * 100
        rows.append(row)
        print(json.dumps(row), flush=True)
    if not rows:
        raise RuntimeError("No reference RGBs found; refusing an empty verification")
    report = dict(training_step=model.metadata["training_step"],
                  adapter_sha256=sha256(args.adapter), adapter_bytes=Path(args.adapter).stat().st_size,
                  original_weights_exact=True, gpu=torch.cuda.get_device_name(),
                  torch_version=torch.__version__, samples=rows,
                  max_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
                  note="Packaging/inference parity only, not a new held-out accuracy evaluation")
    atomic_json(report, args.output)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
