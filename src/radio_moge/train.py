import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import signal
import time

from accelerate import Accelerator
from accelerate.utils import set_seed
import torch
from torch.utils.data import DataLoader

from .checkpoint import atomic_json, atomic_save, restore_rng, rng_state
from .config import load_config, RADIO_COMMIT, MOGE_COMMIT, RADIO_REVISION, MOGE_REVISION
from .data import DeterministicBatches, TrainingImages, check_split, manifest_digest, read_manifest
from .evaluate import evaluate, synchronize
from .losses import FeatureStatistics, geometry_losses, weighted_loss
from .system import build_system


def make_loader(records, config, start, count, calibration=False):
    data, train = config["data"], config["train"]
    sampler = DeterministicBatches(
        len(records), train["batch_size"], data["buckets"], train["seed"], start, count
    )
    return DataLoader(
        TrainingImages(records, data["augment"] and not calibration),
        batch_sampler=sampler,
        num_workers=data["workers"],
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(train["seed"]),
    )


def lr_factor(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1, max(0, (step - warmup) / max(1, total - warmup)))
    return 0.05 + 0.95 * (1 + math.cos(math.pi * progress)) / 2


def validate_config(config):
    train, data = config["train"], config["data"]
    if train["precision"] not in {"no", "bf16"}:
        raise ValueError("Supported precision modes: no, bf16")
    for name in ("steps", "batch_size", "accumulation", "log_every", "save_every", "eval_every"):
        if train[name] < 1:
            raise ValueError(f"{name} must be positive")
    if data["workers"] < 0 or data["workers"] > 4:
        raise ValueError("Use 0–4 data workers; respect the workspace's CPU/memory limits")
    if any(min(pair) < 16 or max(pair) > 2048 for pair in data["buckets"]):
        raise ValueError("Invalid training resolution bucket")
    if any(v < 0 for v in config["loss"].values()):
        raise ValueError("Loss weights must be nonnegative")


def train(config, resume=None, init=None, stop_after=None, system_adapter=None):
    """Single-device experiment runner. system_adapter allows small CPU integration tests."""
    validate_config(config)
    if resume and init:
        raise ValueError("Choose resume (same run) or init (new stage), not both")
    settings = config["train"]
    accelerator = Accelerator(
        cpu=settings.get("device") == "cpu", mixed_precision=settings["precision"]
    )
    if accelerator.num_processes != 1:
        raise ValueError("This initial trainer supports one process/GPU, matching the H200 plan")
    if system_adapter is None and accelerator.device.type != "cuda":
        raise RuntimeError(
            "Full-model training requires a CUDA GPU. Run it on the H200 machine; "
            "the small CPU test suite does not load pretrained backbones."
        )
    set_seed(settings["seed"])
    records = read_manifest(config["data"]["train_manifest"])
    validation = read_manifest(config["data"]["val_manifest"])
    check_split(records, validation)
    digests = {"train": manifest_digest(records), "val": manifest_digest(validation)}
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()) and not resume:
        raise FileExistsError(f"Run directory is not empty: {output}; use a new name or --resume")
    system, adapter = system_adapter or build_system(config, accelerator.device)
    raw_adapter = adapter
    statistics = FeatureStatistics(
        adapter.config["target_dim"], adapter.config["target_summary_dim"]
    )
    statistics.to(accelerator.device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, settings["steps"], settings["warmup"])
    )
    state = (
        torch.load(resume or init, map_location="cpu", weights_only=True)
        if resume or init
        else None
    )
    step, cursor, best = 0, 0, None
    if state:
        if state["adapter_config"] != raw_adapter.config:
            raise ValueError("Checkpoint adapter architecture differs from configuration")
        if (
            state["model_config"] != system.model_config
            or state["config"]["models"] != config["models"]
        ):
            raise ValueError("Checkpoint teacher/RADIO configuration differs")
        raw_adapter.load_state_dict(state["adapter"])
        statistics.load_state_dict(state["statistics"])
        if resume:
            previous = {k: v for k, v in state["config"].items() if k != "output_dir"}
            current = {k: v for k, v in config.items() if k != "output_dir"}
            if previous != current or state["data_digests"] != digests:
                raise ValueError(
                    "Resume needs identical config and manifests; use --init for a new stage"
                )
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            step, cursor, best = state["step"], state["cursor"], state["best"]
            restore_rng(state["rng"])
    adapter, optimizer = accelerator.prepare(adapter, optimizer)
    # Manual accumulation uses complete windows and a known data cursor. Accelerate's
    # accumulation factor stays at one; each microbatch loss is divided exactly once.
    amp = accelerator.autocast if settings["precision"] == "bf16" else nullcontext
    if state is None:

        def calibration_features():
            for image in make_loader(records, config, 0, settings["calibration_batches"], True):
                with amp(), torch.no_grad():
                    _, target = system.encode(image.to(accelerator.device), raw_adapter.indices)
                yield target

        statistics.fit(calibration_features())
    revisions = dict(
        radio_source=RADIO_COMMIT,
        moge_source=MOGE_COMMIT,
        radio_weights=RADIO_REVISION,
        moge_weights=MOGE_REVISION,
    )
    atomic_json(config, output / "config.json")
    atomic_json(
        {
            "revisions": revisions,
            "data_digests": digests,
            "adapter": raw_adapter.config,
            "torch": str(torch.__version__),
            "device": str(accelerator.device),
        },
        output / "provenance.json",
    )
    writer = None
    if settings.get("tensorboard", True):
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(output / "tensorboard", purge_step=step + 1 if resume else None)
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True
        print("Stop requested; finishing this optimizer step and saving a checkpoint.", flush=True)

    old_handlers = {
        sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }

    def save(filename):
        atomic_save(
            {
                "format_version": 1,
                "step": step,
                "cursor": cursor,
                "best": best,
                "adapter": accelerator.unwrap_model(adapter).state_dict(),
                "adapter_config": raw_adapter.config,
                "statistics": statistics.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": rng_state(),
                "config": config,
                "model_config": system.model_config,
                "data_digests": digests,
                "revisions": revisions,
            },
            output / filename,
        )

    total = min(settings["steps"], stop_after) if stop_after else settings["steps"]
    accumulation = settings["accumulation"]
    loader = make_loader(records, config, cursor, max(0, total - step) * accumulation)
    iterator = iter(loader)
    geometry_enabled = any(
        config["loss"].get(k, 0) for k in ("points", "mask", "scale", "gradient", "normal")
    )
    try:
        while step < total and not stop:
            adapter.train()
            optimizer.zero_grad(set_to_none=True)
            synchronize(accelerator.device)
            started = time.perf_counter()
            totals = {}
            for _ in range(accumulation):
                image = next(iterator).to(accelerator.device, non_blocking=True)
                with amp(), torch.no_grad():
                    features, target = system.encode(image, raw_adapter.indices)
                with amp():
                    prediction = adapter(features)
                losses = statistics.loss(prediction, target)
                if geometry_enabled:
                    with torch.no_grad():
                        target_raw = system.decoder(target, image.shape[-2:])
                    # Do not put this path in no_grad: decoder input gradients train the adapter.
                    predicted_raw = system.decoder(prediction, image.shape[-2:])
                    losses.update(geometry_losses(predicted_raw, target_raw))
                loss = weighted_loss(losses, config["loss"])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss before optimizer step {step + 1}")
                accelerator.backward(loss / accumulation)
                for key, value in {**losses, "total": loss}.items():
                    totals[key] = totals.get(key, 0.0) + value.detach().item() / accumulation
                cursor += 1
            norm = accelerator.clip_grad_norm_(adapter.parameters(), settings["clip_grad"])
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite adapter gradients; optimizer update cancelled")
            optimizer.step()
            scheduler.step()
            step += 1
            synchronize(accelerator.device)
            elapsed = time.perf_counter() - started
            row = {
                "step": step,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": elapsed,
                "images_per_second": settings["batch_size"] * accumulation / elapsed,
                "gradient_norm": norm.item(),
                **totals,
            }
            if accelerator.device.type == "cuda":
                row["peak_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
            if step % settings["log_every"] == 0 or step == 1:
                with (output / "train.jsonl").open("a") as file:
                    file.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(row), flush=True)
                if writer:
                    for key, value in row.items():
                        writer.add_scalar(f"train/{key}", value, step)
            if step % settings["save_every"] == 0:
                save("latest.pt")
            if step % settings["eval_every"] == 0 and not stop:
                metrics = evaluate(
                    system,
                    raw_adapter,
                    statistics,
                    validation,
                    config,
                    output / f"eval-{step:07d}",
                    config["evaluation"]["limit"],
                )
                if writer:
                    for key, value in metrics.items():
                        writer.add_scalar(f"eval/{key}", value, step)
                # Do not select a checkpoint whose focal recovery failed on some images.
                score = metrics.get("teacher_depth/abs_rel")
                if score is not None and metrics.get("postprocess_failed", 1) == 0:
                    if best is None or score < best:
                        best = score
                        save("best.pt")
        save("latest.pt")
        for module in (system.radio, system.encoder, system.decoder):
            if any(p.grad is not None or p.requires_grad for p in module.parameters()):
                raise AssertionError("Frozen models accumulated gradients or became trainable")
        return output / "latest.pt"
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        if writer:
            writer.close()
        accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="Train a RADIO → MoGe-2 adapter on one GPU")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument(
        "--init", help="Initialize adapter/statistics from stage 1; reset optimizer"
    )
    parser.add_argument(
        "--stop-after", type=int, help="Save and exit early without changing LR schedule"
    )
    args = parser.parse_args()
    print(train(load_config(args.config), args.resume, args.init, args.stop_after))


if __name__ == "__main__":
    main()
