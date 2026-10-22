"""Read-only spatial/summary swaps at a completed architecture-screen rung.

Uses existing validation images only. Oracle inputs diagnose output sensitivity;
they are not deployable predictions or evidence that a training change will work.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from radio_moge.adapters import EncoderFeatures, GeometryAdapter
from radio_moge.checkpoint import atomic_json
from radio_moge.data import evaluation_image, read_manifest
from radio_moge.decoder import postprocess
from radio_moge.metrics import aggregate, depth_metrics
from radio_moge.system import build_system


def choose_records(records, per_source):
    """Hash-ordered round-robin scenes; retain all NYU sensor validation images."""
    if per_source == 0:
        return records
    groups = defaultdict(lambda: defaultdict(list))
    for row in records:
        groups[row["dataset"]][row["group"]].append(row)

    def key(value):
        return hashlib.sha256(str(value).encode()).hexdigest()

    chosen = set()
    for source, scenes in sorted(groups.items()):
        buckets = [
            sorted(scenes[g], key=lambda r: key(Path(r["image"]).name))
            for g in sorted(scenes, key=key)
        ]
        limit = sum(map(len, buckets)) if source == "nyuv2" else per_source
        count, index = 0, 0
        while count < limit and any(index < len(b) for b in buckets):
            for bucket in buckets:
                if index < len(bucket) and count < limit:
                    chosen.add(bucket[index]["image"])
                    count += 1
            index += 1
    return [r for r in records if r["image"] in chosen]


def summarize(rows, heads):
    result = {}
    for head in heads:
        result[head] = {}
        for variant in ("student", "teacher_summary", "teacher_spatial", "teacher"):
            sources = {}
            for source in sorted({r["dataset"] for r in rows}):
                selected = [r[head][variant] for r in rows if r["dataset"] == source]
                sources[source] = aggregate(selected)
            result[head][variant] = {
                "sources": sources,
                "macro_teacher_abs_rel": sum(s["teacher_depth/abs_rel"] for s in sources.values())
                / len(sources),
            }
    return result


@torch.no_grad()
def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    status_path = args.output.with_suffix(".status.json")
    atomic_json({"stage": "loading", "started": time.time()}, status_path)
    torch.set_num_threads(2)
    adapters, configs, checkpoints, reference = {}, {}, {}, {}
    system = None
    for head in args.heads:
        directory = args.root / head / "features"
        path = directory / "rungs" / str(args.step) / "latest.pt"
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        if state["step"] != args.step:
            raise ValueError(f"Wrong checkpoint step: {path}")
        configs[head] = state["config"]
        if system is None:
            system, adapter = build_system(configs[head], "cuda")
        else:
            adapter = GeometryAdapter(**state["adapter_config"]).cuda()
            for field in ("models", "data", "train", "loss", "evaluation"):
                if configs[head][field] != configs[args.heads[0]][field]:
                    raise ValueError(f"Unmatched {field}: {head}")
        adapter.load_state_dict(state["adapter"])
        adapters[head] = adapter.eval().requires_grad_(False)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        checkpoints[head] = {"path": str(path), "sha256": digest, "step": state["step"]}
        saved = directory / "train" / f"eval-{args.step:07d}" / "per_image.json"
        reference[head] = {r["image"]: r for r in json.loads(saved.read_text())}
        del state
    config = configs[args.heads[0]]
    if config["train"]["precision"] != "bf16":
        raise ValueError("This diagnostic is for the current BF16 screen")
    records = choose_records(read_manifest(config["data"]["val_manifest"]), args.per_source)
    rows, started, direct_checks = [], time.time(), 0
    for index, record in enumerate(records):
        image, gt = evaluation_image(record, config["evaluation"]["max_side"])
        image = image.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            radio, target = system.encode(
                image,
                adapters[args.heads[0]].indices,
                num_tokens=config["evaluation"]["num_tokens"],
            )
        target_raw = system.decoder(target, image.shape[-2:])
        teacher = postprocess(target_raw, image, config["models"].get("moge_source"))
        valid = teacher["mask"] & torch.isfinite(teacher["depth"]) & (teacher["depth"] > 0)
        row = {k: record[k] for k in ("image", "dataset", "group")}
        for head, adapter in adapters.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted = adapter(radio)
            raw = system.decoder(predicted, image.shape[-2:])
            # The only route from summary to this decoder's output is metric_scale.
            mixed_raw = dict(raw, metric_scale=target_raw["metric_scale"])
            reverse_raw = dict(target_raw, metric_scale=raw["metric_scale"])
            if index < 2:
                for encoded, expected in (
                    (EncoderFeatures(predicted.spatial, target.summary), mixed_raw),
                    (EncoderFeatures(target.spatial, predicted.summary), reverse_raw),
                ):
                    direct = system.decoder(encoded, image.shape[-2:])
                    for key in expected:
                        torch.testing.assert_close(direct[key], expected[key], rtol=0, atol=0)
                    direct_checks += 1
            variants = {
                "student": raw,
                "teacher_summary": mixed_raw,
                "teacher_spatial": reverse_raw,
                "teacher": target_raw,
            }
            row[head] = {}
            for variant, value in variants.items():
                output = (
                    teacher
                    if variant == "teacher"
                    else postprocess(value, image, config["models"].get("moge_source"))
                )
                scores = {
                    f"teacher_depth/{k}": v
                    for k, v in depth_metrics(output["depth"], teacher["depth"], valid).items()
                }
                scores["scale_log_error"] = float(
                    (value["metric_scale"].log() - target_raw["metric_scale"].log()).abs().mean()
                )
                if gt is not None:
                    depth_gt = gt.cuda().unsqueeze(0)
                    scores.update(
                        {
                            f"gt_student/{k}": v
                            for k, v in depth_metrics(
                                output["depth"], depth_gt, depth_gt > 0
                            ).items()
                        }
                    )
                if not scores or not all(math.isfinite(v) for v in scores.values()):
                    raise ValueError(f"Missing/nonfinite metrics: {record['image']}")
                row[head][variant] = scores
            original = reference[head][record["image"]]
            for key, value in row[head]["student"].items():
                if not math.isclose(value, original[key], rel_tol=1e-4, abs_tol=2e-6):
                    raise ValueError(f"Saved evaluation parity failed: {head} {key}")
        rows.append(row)
        if (index + 1) % 16 == 0:
            progress = {"stage": "evaluating", "images": index + 1, "total": len(records)}
            atomic_json(progress, status_path)
            print(json.dumps(progress), flush=True)
    report = {
        "started": started,
        "finished": time.time(),
        "checkpoints": checkpoints,
        "records_policy": "Hash-ordered round-robin scenes per source; all NYU val images",
        "per_source_limit": args.per_source,
        "images": len(rows),
        "direct_decoder_swap_parity_checks": direct_checks,
        "saved_evaluation_parity": True,
        "rows": rows,
        "summary": summarize(rows, args.heads),
        "limits": "Inference-only oracle swaps; no training or native-depth supervision. "
        "Conditional internal validation diagnostic, not a blind test. Component effects "
        "can compensate and are not additive. Does not prove a new summary architecture "
        "or scale loss can achieve the oracle improvement.",
    }
    atomic_json(report, args.output)
    atomic_json({"stage": "complete", "images": len(rows), "finished": time.time()}, status_path)
    print(json.dumps({"stage": "complete", "summary": report["summary"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--heads", nargs="+", choices=["fusion", "attention", "dpt"], default=["attention", "dpt"]
    )
    parser.add_argument("--step", type=int, default=3000)
    parser.add_argument("--per-source", type=int, default=32)
    args = parser.parse_args()
    if args.per_source < 0 or not args.heads or len(set(args.heads)) != len(args.heads):
        parser.error("Nonnegative source limit and unique heads required")
    if args.output.exists():
        parser.error("Existing diagnostic output must be preserved; choose a new output path")
    try:
        run(args)
    except Exception as error:
        atomic_json(
            {"stage": "failed", "error": str(error)}, args.output.with_suffix(".status.json")
        )
        raise
