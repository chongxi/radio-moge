"""Measure released RADIO activations on training images without changing weights."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from radio_moge.config import RADIO_COMMIT, RADIO_REVISION, load_config
from radio_moge.data import TrainingImages, read_manifest
from radio_moge.upstream import extract_radio, load_radio, radio_input


class ChannelStats:
    def __init__(self):
        self.means, self.squares, self.maxima, self.spatial_vars = [], [], [], []
        self.token_rms = []
        self.shapes = set()

    def add(self, value):
        value = value.detach().float().flatten(2)
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite RADIO activation")
        self.shapes.add(tuple(value.shape[1:]))
        self.means.append(value.mean(-1).cpu())
        self.squares.append(value.square().mean(-1).cpu())
        self.maxima.append(value.abs().amax(-1).cpu())
        self.spatial_vars.append(value.var(-1, correction=0).cpu())
        self.token_rms.append(value.square().mean(1).sqrt().cpu().flatten())

    def result(self):
        means, squares, maxima, spatial_vars = [
            torch.cat(x) for x in (self.means, self.squares, self.maxima, self.spatial_vars)
        ]
        energy = squares.mean(0)
        rms = energy.sqrt()
        mean = means.mean(0)
        order = energy.argsort(descending=True)
        median = rms.median()
        token_rms = torch.cat(self.token_rms)
        quantiles = [0.0, 0.5, 0.9, 0.99, 1.0]
        q = torch.tensor(quantiles)
        top_counts = Counter(squares.argmax(1).tolist())
        return {
            "num_images": len(means),
            "channel_token_shapes": sorted(self.shapes),
            "median_channel_rms": median.item(),
            "max_channel_rms_over_median": (rms.max() / median).item(),
            "top_energy_fraction": {
                str(k): (energy[order[:k]].sum() / energy.sum()).item()
                for k in (1, 2, 4, 8, 16, 32)
            },
            "channels_over_10x_median_rms": torch.where(rms > median * 10)[0].tolist(),
            "token_rms_quantiles": dict(zip(map(str, quantiles), token_rms.quantile(q).tolist())),
            "top_channels": [
                {
                    "channel": c,
                    "rms": rms[c].item(),
                    "mean": mean[c].item(),
                    "std": (energy[c] - mean[c].square()).clamp_min(0).sqrt().item(),
                    "within_image_spatial_std_rms": spatial_vars[:, c].mean().sqrt().item(),
                    "between_image_mean_std": means[:, c].std(correction=0).item(),
                    "max_abs": maxima[:, c].max().item(),
                    "energy_fraction": (energy[c] / energy.sum()).item(),
                    "top1_image_count": top_counts[c],
                    "image_rms_min_max": [
                        squares[:, c].sqrt().min().item(),
                        squares[:, c].sqrt().max().item(),
                    ],
                }
                for c in order[:16].tolist()
            ],
            "all_channel_rms": rms.tolist(),
            "all_channel_mean": mean.tolist(),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if min(args.count, args.batch_size) < 1:
        parser.error("count and batch size must be positive")
    torch.set_num_threads(4)
    config = load_config(args.config)
    records = read_manifest(config["data"]["train_manifest"])
    order = list(range(len(records)))
    random.Random(42).shuffle(order)
    selected, groups = [], set()
    for i in order:
        group = records[i].get("group", records[i]["image"])
        if group not in groups:
            selected.append(i)
            groups.add(group)
        if len(selected) == args.count:
            break
    dataset = TrainingImages(records, augment=False)
    started = time.monotonic()
    radio = load_radio(config["models"], "cuda")
    modules = {
        name: repr(getattr(radio, name, None))
        for name in ("feature_normalizer", "inter_feature_normalizer", "input_conditioner")
    }
    modules["backbone_norm"] = repr(getattr(radio.model, "norm", None))
    print(json.dumps({"normalizers": modules}), flush=True)
    indices = (7, 15, 23, 31)
    names = [f"block_{i + 1}" for i in indices] + ["final_spatial", "summary"]
    stats = {name: ChannelStats() for name in names}
    with torch.inference_mode():
        for start in range(0, len(selected), args.batch_size):
            image = torch.stack(
                [dataset[(i, 42, 480, 640)] for i in selected[start : start + args.batch_size]]
            ).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                features = extract_radio(radio, radio_input(image, radio), indices)
            values = list(features.intermediates) + [
                features.spatial,
                features.summary.unsqueeze(-1),
            ]
            for name, value in zip(names, values):
                stats[name].add(value)
            print(
                f"Measured {min(start + args.batch_size, len(selected))}/{len(selected)} training images",
                flush=True,
            )
    report = {
        "radio_source_commit": RADIO_COMMIT,
        "radio_weights_revision": RADIO_REVISION,
        "manifest": config["data"]["train_manifest"],
        "sample_indices": selected,
        "sample_groups": [records[i].get("group") for i in selected],
        "method": "Distinct training scenes, seed42; centered training crop; RGB [0,1]; 480x640; bf16 autocast; frozen unmodified RADIO",
        "num_images": len(selected),
        "manifest_sha256": hashlib.sha256(
            Path(config["data"]["train_manifest"]).read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "normalizers": modules,
        "elapsed_seconds": time.monotonic() - started,
        "layers": {name: value.result() for name, value in stats.items()},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name("." + output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)
    print(
        json.dumps(
            {
                name: {
                    k: v
                    for k, v in row.items()
                    if k not in ("all_channel_rms", "all_channel_mean", "top_channels")
                }
                | {"top_channels": row["top_channels"][:4]}
                for name, row in report["layers"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
