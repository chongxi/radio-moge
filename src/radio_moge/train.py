import argparse
from copy import deepcopy
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
from .config import (
    load_config,
    RADIO_COMMIT,
    MOGE_COMMIT,
    RADIO_REVISION,
    MOGE_REPO,
    MOGE_REVISION,
)
from .data import (
    DeterministicBatches,
    MixtureBatches,
    TrainingImages,
    check_split,
    manifest_digest,
    read_manifest,
)
from .evaluate import evaluate, synchronize
from .decoder import geometry_points_with_grad, postprocess
from .losses import FeatureStatistics, geometry_losses, weighted_loss
from .planes import FLATNESS_MODES, PlaneConfig, detect_planes, plane_losses
from .system import build_system
from .tracking import ExperimentTracker, preserve_rng


def make_loader(records, config, start, count, calibration=False):
    data, train = config["data"], config["train"]
    if data.get("mixture_weights"):
        sampler = MixtureBatches(
            records,
            data["mixture_weights"],
            train["batch_size"],
            data["buckets"],
            train["seed"],
            start,
            count,
        )
    else:
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
    if train.get("lr_schedule", "cosine") not in {"cosine", "constant"}:
        raise ValueError("Unknown learning-rate schedule")
    if not math.isfinite(train["lr"]) or train["lr"] <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if train.get("update_norm_every", 0) < 0:
        raise ValueError("update_norm_every must be nonnegative")
    for name in ("steps", "batch_size", "accumulation", "log_every", "save_every", "eval_every"):
        if train[name] < 1:
            raise ValueError(f"{name} must be positive")
    if data["workers"] < 0 or data["workers"] > 4:
        raise ValueError("Use 0–4 data workers; respect the workspace's CPU/memory limits")
    if any(min(pair) < 16 or max(pair) > 2048 for pair in data["buckets"]):
        raise ValueError("Invalid training resolution bucket")
    if any(v < 0 for v in config["loss"].values()):
        raise ValueError("Loss weights must be nonnegative")
    if config.get('plane_flatness_mode', 'teacher_axis') not in FLATNESS_MODES:
        raise ValueError('Unknown plane flatness mode')
    if any(config["loss"].get(k, 0) for k in ("plane_flatness", "plane_normal", "plane_anchor")):
        PlaneConfig(**config.get("planes", {}))
        if config["loss"].get("plane_anchor", 0) <= 0:
            raise ValueError("Plane losses require a positive plane_anchor against collapse")
    if train.get("max_seconds") is not None and (
        not math.isfinite(train["max_seconds"]) or train["max_seconds"] <= 0
    ):
        raise ValueError("max_seconds must be finite and positive")


def continued_limits(settings, state, extension=None):
    """Explicitly extend a resumed run without resetting its LR schedule or state."""
    limits = dict(
        (state or {}).get("runtime_limits")
        or {"steps": settings["steps"], "max_seconds": settings.get("max_seconds")}
    )
    if extension is not None:
        if state is None:
            raise ValueError("Budget extension requires a resume checkpoint")
        seconds, steps = extension
        if not math.isfinite(seconds) or seconds <= 0 or steps < 1:
            raise ValueError("Extension seconds and update allowance must be positive")
        elapsed = state.get("wall_clock_elapsed_seconds", 0.0)
        limits = {
            "steps": state["step"] + steps,
            "max_seconds": elapsed + seconds,
            "extension": {
                "base_step": state["step"],
                "base_cursor": state["cursor"],
                "base_elapsed_seconds": elapsed,
                "additional_seconds": seconds,
                "additional_update_allowance": steps,
                "lr_policy": "Preserve original scheduler; retain its floor after cosine decay",
            },
        }
    return limits


def check_lr_fork(config, state, digests):
    """Only LR policy, a bounded new budget, tracking and diagnostics may differ."""
    previous, current = deepcopy(state["config"]), deepcopy(config)
    for candidate in (previous, current):
        for key in ("output_dir", "tracking"):
            candidate.pop(key, None)
        for key in ("lr", "lr_schedule", "steps", "max_seconds", "update_norm_every"):
            candidate["train"].pop(key, None)
    if previous != current or state["data_digests"] != digests:
        raise ValueError("LR fork requires identical model, data, loss and optimizer settings")
    if config["train"].get("lr_schedule") != "constant":
        raise ValueError("LR fork requires an explicit constant learning rate")
    if config["train"].get("max_seconds") is None or config["train"]["steps"] <= state["step"]:
        raise ValueError("LR fork requires a bounded positive new budget")


def train(
    config,
    resume=None,
    init=None,
    stop_after=None,
    system_adapter=None,
    extension=None,
    execution_revision=None,
    fork_from=None,
):
    """Single-device experiment runner. system_adapter allows small CPU integration tests."""
    invocation_started = time.monotonic()
    elapsed_before = 0.0
    validate_config(config)
    if sum(bool(x) for x in (resume, init, fork_from)) > 1:
        raise ValueError("Choose exactly one of resume, init or fork-from")
    if extension is not None and not resume:
        raise ValueError("Budget extension requires --resume")
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
    val_path = config["data"].get("val_manifest")
    validation = read_manifest(val_path) if val_path else []
    check_split(records, validation)
    probe_path = config['data'].get('train_probe_manifest')
    train_probe = read_manifest(probe_path) if probe_path else []
    if train_probe and not {r['image'] for r in train_probe} <= {r['image'] for r in records}:
        raise ValueError('Training diagnostics must be a subset of the training images')
    if train_probe and not validation:
        raise ValueError('Training diagnostics require a separate validation manifest')
    digests = {
        "train": manifest_digest(records),
        "val": manifest_digest(validation) if validation else None,
    }
    if train_probe:
        digests['train_probe'] = manifest_digest(train_probe)
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
        optimizer,
        lambda step: 1.0 if settings.get("lr_schedule") == "constant"
        else lr_factor(step, settings["steps"], settings["warmup"]),
    )
    state = (
        torch.load(resume or init or fork_from, map_location="cpu", weights_only=True)
        if resume or init or fork_from
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
        if fork_from:
            check_lr_fork(config, state, digests)
            optimizer.load_state_dict(state["optimizer"])
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] = settings["lr"]
            # Keep Adam moments and its update counter. Only the LR policy is reset.
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            step, cursor = state["step"], state["cursor"]
            restore_rng(state["rng"])
        elif resume:
            previous = {k: v for k, v in state["config"].items() if k != "output_dir"}
            current = {k: v for k, v in config.items() if k != "output_dir"}
            if previous != current or state["data_digests"] != digests:
                raise ValueError(
                    "Resume needs identical config and manifests; use --init for a new stage"
                )
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            step, cursor, best = state["step"], state["cursor"], state["best"]
            elapsed_before = state.get("wall_clock_elapsed_seconds", 0.0)
            restore_rng(state["rng"])
    limits = continued_limits(settings, state if resume else None, extension)
    fork_provenance = (state or {}).get("fork_provenance") if resume else None
    if fork_from:
        fork_provenance = {
            "checkpoint": str(Path(fork_from).resolve()),
            "step": step, "cursor": cursor,
            "parent_lr": state["optimizer"]["param_groups"][0]["lr"],
            "optimizer_lr": optimizer.param_groups[0]["lr"],
            "lr_schedule": "constant", "runtime_limits": limits,
            "restored": ["adapter", "statistics", "optimizer_moments", "rng", "cursor"],
            "reset": ["lr_schedule", "wall_clock_budget", "best_selection", "tracking_run"],
        }
        atomic_json(fork_provenance, output / "fork-state.json")
    if resume:
        atomic_json(
            {
                "checkpoint": str(Path(resume).resolve()),
                "step": step,
                "cursor": cursor,
                "best": best,
                "optimizer_lr": optimizer.param_groups[0]["lr"],
                "scheduler_last_epoch": scheduler.last_epoch,
                "elapsed_before_seconds": elapsed_before,
                "runtime_limits": limits,
                "config_and_manifest_match": True,
                "execution_revision": execution_revision,
                "restored": ["adapter", "statistics", "optimizer", "scheduler", "rng", "cursor"],
            },
            output / "resume-state.json",
        )
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
        moge_repo=config["models"].get("moge_repo", MOGE_REPO),
        moge_weights=config["models"].get("moge_revision", MOGE_REVISION),
    )
    atomic_json(config, output / "config.json")
    provenance = {
        "revisions": revisions,
        "data_digests": digests,
        "adapter": raw_adapter.config,
        "torch": str(torch.__version__),
        "device": str(accelerator.device),
        "train_images": len(records),
        "validation_images": len(validation),
        "code_revision": config.get("code_revision"),
        "execution_revision": execution_revision or (state or {}).get("execution_revision"),
        "runtime_limits": limits,
        "fork_provenance": fork_provenance,
    }
    atomic_json(provenance, output / "provenance.json")
    tracker = ExperimentTracker(config, output, resume=bool(resume), provenance=provenance)
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

    def elapsed_seconds():
        # Includes loading, validation and checkpoint I/O in each invocation.
        # Time between invocations does not consume the experiment budget.
        return elapsed_before + time.monotonic() - invocation_started

    def budget_exhausted():
        limit = limits.get("max_seconds")
        return limit is not None and elapsed_seconds() >= limit

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
                "wall_clock_elapsed_seconds": elapsed_seconds(),
                "runtime_limits": limits,
                "execution_revision": provenance["execution_revision"],
                "fork_provenance": fork_provenance,
            },
            output / filename,
        )

    total = min(limits["steps"], stop_after) if stop_after else limits["steps"]
    accumulation = settings["accumulation"]
    loader = make_loader(records, config, cursor, max(0, total - step) * accumulation)
    iterator = iter(loader)
    plane_enabled = any(config["loss"].get(k, 0)
                        for k in ("plane_flatness", "plane_normal", "plane_anchor"))
    plane_config = PlaneConfig(**config.get("planes", {})) if plane_enabled else None
    geometry_enabled = plane_enabled or any(
        config["loss"].get(k, 0) for k in ("points", "mask", "scale", "gradient", "normal")
    )
    success = False
    last_validation_step = None

    def validate():
        nonlocal best, last_validation_step
        destination = output / f"eval-{step:07d}"
        metrics = evaluate(
            system,
            raw_adapter,
            statistics,
            validation,
            config,
            destination,
            config["evaluation"]["limit"],
        )
        metrics["feature/total"] = sum(
            metrics.get(f"feature/{key}", 0) * config["loss"].get(key, 0)
            for key in ("spatial", "summary", "cosine")
        )
        with (output / "validation.jsonl").open("a") as file:
            file.write(json.dumps({"step": step, **metrics}, allow_nan=False) + "\n")
        print(json.dumps({"validation_step": step, **metrics}), flush=True)
        if writer:
            for key, value in metrics.items():
                writer.add_scalar(f"eval/{key}", value, step)
        tracker.log("validation", metrics, step, image_dir=destination)
        last_validation_step = step
        score = metrics.get(config["evaluation"].get("selection_metric", "teacher_depth/abs_rel"))
        if score is not None and metrics.get("postprocess_failed", 1) == 0:
            if best is None or score < best:
                best = score
                save("best.pt")
        if train_probe:
            # Diagnostics never affect best selection, gradients, or training randomness.
            destination = output / f'train-probe-{step:07d}'
            with preserve_rng():
                probe_metrics = evaluate(system, raw_adapter, statistics, train_probe, config,
                                         destination, None)
                probe_metrics['feature/total'] = sum(
                    probe_metrics.get(f'feature/{key}', 0) * config['loss'].get(key, 0)
                    for key in ('spatial', 'summary', 'cosine'))
                with (output / 'train-probe.jsonl').open('a') as file:
                    file.write(json.dumps({'step': step, **probe_metrics}, allow_nan=False) + '\n')
                tracker.log('train_probe', probe_metrics, step)
                print(json.dumps({'train_probe_step': step, **probe_metrics}), flush=True)

    try:
        if not resume:
            # Preserve the actual stage-start weights for decoded before/after comparisons.
            save("initial.pt")
        if validation and not resume and not fork_from and settings.get("eval_at_start", False):
            validate()
        while step < total and not stop and not budget_exhausted():
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
                    if plane_enabled:
                        # Teacher masks/planes are fixed pseudo-labels. Never filter on student fit.
                        target_geometry = postprocess(
                            target_raw, image, config["models"].get("moge_source"))
                        patches = [detect_planes(p.cpu().numpy(), m.cpu().numpy(), plane_config)
                                   for p, m in zip(target_geometry["points"],
                                                   target_geometry["mask"], strict=True)]
                        student_points = geometry_points_with_grad(
                            predicted_raw, config["models"].get("moge_source"))
                        losses.update(plane_losses(student_points, target_geometry["points"],
                                                   patches, config.get('plane_flatness_mode',
                                                                       'teacher_axis')))
                        losses["plane_regions"] = student_points.new_tensor(
                            sum(map(len, patches)) / len(patches))
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
            measure_every = settings.get("update_norm_every", 0)
            measure_update = measure_every and (step + 1) % measure_every == 0
            parameters = list(adapter.parameters()) if measure_update else []
            before = [p.detach().clone() for p in parameters]
            lr_used = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            step += 1
            synchronize(accelerator.device)
            elapsed = time.perf_counter() - started
            row = {
                "step": step,
                "lr": optimizer.param_groups[0]["lr"],
                "lr_used": lr_used,
                "seconds": elapsed,
                "images_per_second": settings["batch_size"] * accumulation / elapsed,
                "gradient_norm": norm.item(),
                **totals,
            }
            if measure_update:
                update_norm = torch.stack([
                    (p.detach() - old).float().square().sum()
                    for p, old in zip(parameters, before)
                ]).sum().sqrt()
                weight_norm = torch.stack([old.float().square().sum() for old in before]).sum().sqrt()
                row.update(update_norm=update_norm.item(),
                           relative_update_norm=(update_norm / weight_norm.clamp_min(1e-12)).item())
                del before, parameters
            if accelerator.device.type == "cuda":
                row["peak_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
            if step % settings["log_every"] == 0 or step == 1:
                with (output / "train.jsonl").open("a") as file:
                    file.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(row), flush=True)
                if writer:
                    for key, value in row.items():
                        writer.add_scalar(f"train/{key}", value, step)
                tracker.log("train", row, step)
            if step % settings["save_every"] == 0:
                save("latest.pt")
            if (
                validation
                and step % settings["eval_every"] == 0
                and not stop
                and not budget_exhausted()
            ):
                validate()
        save("latest.pt")
        if limits.get("max_seconds") is not None:
            reason = (
                "signal" if stop else "wall_clock_budget" if budget_exhausted() else "step_limit"
            )
            stopped_at = elapsed_seconds()
            atomic_json(
                dict(
                    reason=reason,
                    step=step,
                    elapsed_seconds=stopped_at,
                    max_seconds=limits["max_seconds"],
                    final_evaluation="pending",
                ),
                output / "stop-report.json",
            )
            # A deadline can fall between validation boundaries. Evaluate that
            # final checkpoint once; no additional optimizer steps are taken.
            # On resume, an existing completed evaluation must not be appended again.
            existing = output / "validation.jsonl"
            already_validated = last_validation_step == step or (
                existing.exists()
                and any(
                    json.loads(line)["step"] == step
                    for line in existing.read_text().splitlines()
                    if line.strip()
                )
            )
            if validation and not stop and not already_validated:
                validate()
            save("latest.pt")
            atomic_json(
                dict(
                    reason=reason,
                    step=step,
                    elapsed_seconds=elapsed_seconds(),
                    optimizer_stopped_elapsed_seconds=stopped_at,
                    max_seconds=limits["max_seconds"],
                    final_evaluation="skipped_on_signal" if stop else "complete",
                ),
                output / "stop-report.json",
            )
        for module in (system.radio, system.encoder, system.decoder):
            if any(p.grad is not None or p.requires_grad for p in module.parameters()):
                raise AssertionError("Frozen models accumulated gradients or became trainable")
        success = True
        return output / "latest.pt"
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        if writer:
            writer.close()
        tracker.finish(success=success)
        accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="Train a RADIO → MoGe-2 adapter on one GPU")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--fork-from", help="New constant-LR experiment preserving optimizer, RNG and data cursor")
    parser.add_argument(
        "--init", help="Initialize adapter/statistics from stage 1; reset optimizer"
    )
    parser.add_argument(
        "--stop-after", type=int, help="Save and exit early without changing LR schedule"
    )
    parser.add_argument(
        "--extend-seconds",
        type=float,
        help="Explicit additional live wall time from the resume checkpoint",
    )
    parser.add_argument(
        "--extend-updates",
        type=int,
        help="Explicit additional update ceiling; keeps original LR schedule",
    )
    parser.add_argument("--execution-revision", help="Revision of this trainer invocation")
    args = parser.parse_args()
    if (args.extend_seconds is None) != (args.extend_updates is None):
        parser.error("Both extension limits are required")
    extension = (
        (args.extend_seconds, args.extend_updates) if args.extend_seconds is not None else None
    )
    print(
        train(
            load_config(args.config),
            args.resume,
            args.init,
            args.stop_after,
            extension=extension,
            execution_revision=args.execution_revision,
            fork_from=args.fork_from,
        )
    )


if __name__ == "__main__":
    main()
