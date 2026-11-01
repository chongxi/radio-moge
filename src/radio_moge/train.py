import argparse
from copy import deepcopy
from contextlib import nullcontext
import json
import math
import os
import re
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
from .losses import FeatureStatistics, feature_loss_total, geometry_losses, weighted_loss
from .gt_geometry import GT_LOSS_KEYS, ground_truth_losses
from .gt_data import GroundTruthImages, geometry_digest, validate_gt_records
from .planes import FLATNESS_MODES, PlaneConfig, detect_planes, plane_losses
from .system import build_system
from .tracking import ExperimentTracker, preserve_rng
from .selection import validate_selections, selection_winners


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
    compact = data.get('compact_transport', False)
    dataset = (GroundTruthImages(records, data['augment'])
               if config.get('geometry_supervision') and not calibration else
               TrainingImages(records, data['augment'] and not calibration,
                              compact_transport=compact))
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=data["workers"],
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(train["seed"]),
    )
    if compact:
        from .prefetch import FloatLoader
        return FloatLoader(loader)
    return loader


def lr_factor(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1, max(0, (step - warmup) / max(1, total - warmup)))
    return 0.05 + 0.95 * (1 + math.cos(math.pi * progress)) / 2


def validate_config(config):
    validate_selections(config.get('evaluation', {}).get('checkpoint_metrics', {}))
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
    if any(not math.isfinite(v) or v < 0 for v in config["loss"].values()):
        raise ValueError("Loss weights must be finite and nonnegative")
    gt = config.get('geometry_supervision')
    if gt:
        if gt != {'target': 'ground_truth', 'camera': 'known_intrinsics'}:
            raise ValueError('Unsupported GT geometry supervision protocol')
        if data.get('compact_transport'):
            raise ValueError('GT pairs require the paired loader, not RGB-only compact transport')
        if config['loss'].get('gt_depth', 0) <= 0 or config['loss'].get('gt_points', 0) <= 0:
            raise ValueError('GT training requires positive metric gt_depth and gt_points anchors')
        if any(config['loss'].get(k, 0) for k in
               ('points', 'mask', 'scale', 'gradient', 'normal',
                'plane_flatness', 'plane_normal', 'plane_anchor')):
            raise ValueError('GT geometry cannot silently mix teacher pseudo-geometry losses')
    elif any(config['loss'].get(k, 0) for k in GT_LOSS_KEYS):
        raise ValueError('GT loss weights require explicit geometry_supervision')
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
        for key in ("lr", "lr_schedule", "steps", "max_seconds", "update_norm_every",
                    "save_every", "eval_every"):
            candidate["train"].pop(key, None)
        candidate.get('evaluation', {}).pop('checkpoint_metrics', None)
    if previous != current or state["data_digests"] != digests:
        raise ValueError("LR fork requires identical model, data, loss and optimizer settings")
    if config["train"].get("lr_schedule") != "constant":
        raise ValueError("LR fork requires an explicit constant learning rate")
    if config["train"].get("max_seconds") is None or config["train"]["steps"] <= state["step"]:
        raise ValueError("LR fork requires a bounded positive new budget")


def check_resolution_fork(config, state, digests):
    """Add training buckets and fixed validation views without silently changing the science."""
    previous = state['config']['data']['buckets']
    current = config['data']['buckets']
    old, new = {tuple(x) for x in previous}, {tuple(x) for x in current}
    if not old < new or len(new) != len(current):
        raise ValueError('Resolution fork must add unique buckets and preserve every old bucket')
    if any(len(x) != 2 or any(type(d) is not int or d < 16 or d > 2048 or d % 16
                             for d in x) for x in current):
        raise ValueError('Resolution fork requires supported patch-aligned buckets')
    variants = config['evaluation'].get('resolution_variants', {})
    if not variants or len(variants) > 3:
        raise ValueError('Resolution fork needs one to three explicit validation views')
    for name, view in variants.items():
        if (not re.fullmatch(r'[a-z0-9-]{1,32}', name) or set(view) != {'bucket', 'num_tokens'}
                or view['num_tokens'] is not None or tuple(view['bucket']) not in new - old):
            raise ValueError('Each new validation view must use a new native training bucket')
    if {tuple(v['bucket']) for v in variants.values()} != new - old:
        raise ValueError('Every added bucket needs a matching validation view')
    normalized = deepcopy(config)
    normalized['data']['buckets'] = previous
    normalized['evaluation'].pop('resolution_variants', None)
    check_lr_fork(normalized, state, digests)
    if config['train']['lr'] != state['config']['train']['lr']:
        raise ValueError('Resolution fork preserves the source learning rate')


def train(
    config,
    resume=None,
    init=None,
    stop_after=None,
    system_adapter=None,
    extension=None,
    execution_revision=None,
    fork_from=None,
    performance=None,
    budget_started_at=None,
    resolution_fork=False,
    data_fork=False,
    lr_adjustment=None,
):
    """Single-device experiment runner. system_adapter allows small CPU integration tests."""
    invocation_started = time.monotonic()
    elapsed_before = 0.0
    validate_config(config)
    if config.get('geometry_supervision') and performance:
        raise ValueError('GT paired batches do not support the RGB-only performance pipeline')
    if sum(bool(x) for x in (resume, init, fork_from)) > 1:
        raise ValueError("Choose exactly one of resume, init or fork-from")
    if extension is not None and not resume:
        raise ValueError("Budget extension requires --resume")
    if lr_adjustment is not None and (not resume or extension or performance):
        raise ValueError('LR adjustment requires exact resume with inherited performance and budget')
    if resolution_fork and (not fork_from or extension or init or resume):
        raise ValueError('Resolution fork requires a separate --fork-from experiment')
    if data_fork and (not fork_from or extension or init or resume or resolution_fork):
        raise ValueError('Data fork requires a separate --fork-from experiment')
    if budget_started_at is not None and (not fork_from or not math.isfinite(budget_started_at)
                                         or not 0 < budget_started_at <= time.time()):
        raise ValueError('An original wall-clock start requires a constant-LR fork and a past timestamp')
    settings = config["train"]
    if performance is not None:
        from .performance import validate_performance
        validate_performance(performance, config, constant_fork=resolution_fork or data_fork)
        if not (resolution_fork or data_fork) and (not resume or extension or fork_from or init):
            raise ValueError("Performance tuning requires an unextended exact-state resume")
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
    if config.get('geometry_supervision'):
        validate_gt_records(records)
    val_path = config["data"].get("val_manifest")
    validation = read_manifest(val_path) if val_path else []
    check_split(records, validation)
    dataset_validation = {}
    for name, view in config.get('evaluation', {}).get('dataset_variants', {}).items():
        if (not re.fullmatch(r'[a-z0-9-]{1,32}', name)
                or set(view) != {'manifest', 'num_tokens', 'max_side'}):
            raise ValueError('Invalid fixed dataset validation view')
        extra_records = read_manifest(view['manifest'])
        check_split(records, extra_records)
        check_split(validation, extra_records)
        dataset_validation[name] = extra_records
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
    if config.get('geometry_supervision'):
        digests['geometry'] = geometry_digest(records)
    if train_probe:
        digests['train_probe'] = manifest_digest(train_probe)
    for name, extra_records in dataset_validation.items():
        digests['dataset/' + name] = manifest_digest(extra_records)
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
            from .data_fork import check_data_fork
            check = check_data_fork if data_fork else check_resolution_fork if resolution_fork else check_lr_fork
            check(config, state, digests)
            if (resolution_fork or data_fork) and performance:
                parent_effective = (state.get('performance') or {}).get(
                    'effective_batch', state['config']['train']['batch_size'] *
                    state['config']['train']['accumulation'])
                if (performance['effective_batch'] != parent_effective
                        or performance.get('autotune') or performance.get('data_autotune')):
                    raise ValueError('Resolution fork preserves effective batch and forbids tuning')
            optimizer.load_state_dict(state["optimizer"])
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] = settings["lr"]
            # Keep Adam moments and its update counter. Only the LR policy is reset.
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            step, cursor = state["step"], state["cursor"]
            if data_fork:
                cursor = 0
                if performance and (performance['anchor_cursor'] != 0
                                    or performance['anchor_step'] != step):
                    raise ValueError('Data fork needs a fresh stream cursor and source step anchor')
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
    lr_adjustments = deepcopy((state or {}).get('lr_adjustments', [])) if resume else []
    if lr_adjustment is not None:
        from .lr_adjustment import apply_lr_adjustment
        lr_adjustments = apply_lr_adjustment(config, state, optimizer, scheduler, lr_adjustment)
    limits = continued_limits(settings, state if resume else None, extension)
    # A new objective initialized with --init does not inherit a parent's RGB-only pipeline.
    performance = performance or ((state or {}).get('performance') if not init else None)
    if config.get('geometry_supervision') and performance:
        raise ValueError('GT paired batches do not support the RGB-only performance pipeline')
    constant_fork = bool(fork_from or (resume and (state or {}).get('fork_provenance')))
    if performance is not None:
        from .performance import validate_performance
        validate_performance(performance, config, constant_fork=constant_fork)
    selections = config.get('evaluation', {}).get('checkpoint_metrics', {})
    best_metrics = deepcopy((state or {}).get('best_metrics', {})) if resume else {
        name: dict(value=spec['baseline'], step=step, metric=spec['metric'], mode=spec['mode'])
        for name, spec in selections.items() if 'baseline' in spec
    }
    if fork_from:
        primary = config['evaluation'].get('selection_metric', 'teacher_depth/abs_rel')
        for spec in selections.values():
            if spec['metric'] == primary and spec['mode'] == 'min' and 'baseline' in spec:
                best = spec['baseline']
    optimizer_seconds = (state or {}).get('optimizer_elapsed_seconds', 0.0) if resume else 0.0
    fork_provenance = (state or {}).get("fork_provenance") if resume else None
    initialization_provenance = ((state or {}).get('initialization_provenance')
                                 if resume else None)
    if init:
        initialization_provenance = {
            'checkpoint': str(Path(init).resolve()), 'source_step': state['step'],
            'restored': ['adapter', 'statistics'],
            'reset': ['optimizer', 'scheduler', 'rng', 'data_cursor', 'step', 'tracking_run'],
            'geometry_supervision': config.get('geometry_supervision'),
        }
        atomic_json(initialization_provenance, output / 'initialization-state.json')
    if fork_from:
        fork_provenance = {
            "checkpoint": str(Path(fork_from).resolve()),
            "step": step, "cursor": cursor,
            "parent_lr": state["optimizer"]["param_groups"][0]["lr"],
            "adam_steps": sorted({int(row['step'].item()) if torch.is_tensor(row['step'])
                                   else int(row['step'])
                                   for row in state['optimizer']['state'].values() if 'step' in row}),
            "optimizer_lr": optimizer.param_groups[0]["lr"],
            "lr_schedule": "constant", "runtime_limits": limits,
            "budget_started_at": budget_started_at,
            "restored": ["adapter", "statistics", "optimizer_moments", "rng", "cursor"],
            "reset": ["lr_schedule", "wall_clock_budget", "best_selection", "tracking_run"],
        }
        if resolution_fork:
            fork_provenance['resolution_change'] = dict(
                parent_buckets=state['config']['data']['buckets'], buckets=config['data']['buckets'],
                sampling='uniform bucket per logical microbatch; same source/scene sampler',
                augmentation='same seeded random aspect crop and horizontal flip',
                validation_views=config['evaluation']['resolution_variants'])
        if data_fork:
            fork_provenance['data_change'] = dict(
                parent_cursor=state['cursor'], new_cursor=0,
                parent_digests=state['data_digests'], digests=digests,
                parent_manifest=state['config']['data']['train_manifest'],
                manifest=config['data']['train_manifest'],
                sampling='permutation' if config['data'].get('coverage_epochs') else 'source/scene mixture')
            fork_provenance['restored'].remove('cursor')
            fork_provenance['reset'].append('data_cursor_for_new_manifest')
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
                "lr_adjustments": lr_adjustments,
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
    if performance is not None and performance.get('autotune'):
        from .performance import autotune
        performance = autotune(system, adapter, raw_adapter, statistics, optimizer, scheduler,
                               accelerator, amp, records, config, state, performance, output)
    if performance is not None and performance.get('data_autotune'):
        from .data_pipeline_probe import probe
        performance = probe(system, adapter, raw_adapter, statistics, optimizer, scheduler,
                            accelerator, amp, records, config, state, performance, output)
    if performance is not None:
        from .performance import schedule_position, transition_limits, interval_due
        if not constant_fork:
            limits = transition_limits(limits, settings, performance)
            scheduler.lr_lambdas = [lambda epoch: lr_factor(
                schedule_position(epoch, performance), settings['steps'], settings['warmup'])]
        atomic_json(performance, output / 'performance.json')
        state_file = output / ('fork-state.json' if fork_from else 'resume-state.json')
        resumed = json.loads(state_file.read_text())
        resumed.update(performance=performance, runtime_limits=limits,
                       original_absolute_deadline=performance['original_started_at'] + settings['max_seconds']
                       if settings.get('max_seconds') is not None and not constant_fork else None)
        atomic_json(resumed, state_file)
    revisions = dict(
        radio_source=RADIO_COMMIT,
        moge_source=MOGE_COMMIT,
        radio_weights=RADIO_REVISION,
        moge_repo=config["models"].get("moge_repo", MOGE_REPO),
        moge_weights=config["models"].get("moge_revision", MOGE_REVISION),
    )
    atomic_json(config, output / "config.json")
    if lr_adjustments:
        atomic_json(lr_adjustments, output / 'lr-adjustments.json')
    provenance = {
        "revisions": revisions,
        "data_digests": digests,
        "adapter": raw_adapter.config,
        "torch": str(torch.__version__),
        "device": str(accelerator.device),
        "train_images": len(records),
        "validation_images": len(validation),
        "dataset_validation_images": {k: len(v) for k, v in dataset_validation.items()},
        "code_revision": config.get("code_revision"),
        "execution_revision": execution_revision or (state or {}).get("execution_revision"),
        "runtime_limits": limits,
        "fork_provenance": fork_provenance,
        "initialization_provenance": initialization_provenance,
        "performance": performance,
        "lr_adjustments": lr_adjustments,
    }
    atomic_json(provenance, output / "provenance.json")
    tracker = ExperimentTracker(config, output, resume=bool(resume), provenance=provenance)
    if lr_adjustments:
        tracker.record_lr_adjustments(lr_adjustments)
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
        measured = elapsed_before + time.monotonic() - invocation_started
        if performance is not None and not constant_fork:
            measured = max(measured, time.time() - performance['original_started_at'])
        if fork_provenance and fork_provenance.get('budget_started_at') is not None:
            measured = max(measured, time.time() - fork_provenance['budget_started_at'])
        return measured

    def budget_exhausted():
        limit = limits.get("max_seconds")
        return limit is not None and elapsed_seconds() >= limit

    def save(filename):
        if coverage is not None:
            atomic_json(coverage.report(), output / 'coverage.json')
        atomic_save(
            {
                "format_version": 1,
                "step": step,
                "cursor": cursor,
                "best": best,
                "best_metrics": best_metrics,
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
                "optimizer_elapsed_seconds": optimizer_seconds,
                "runtime_limits": limits,
                "execution_revision": provenance["execution_revision"],
                "fork_provenance": fork_provenance,
                "initialization_provenance": initialization_provenance,
                "performance": performance,
                "lr_adjustments": lr_adjustments,
            },
            output / filename,
        )

    total = min(limits["steps"], stop_after) if stop_after else limits["steps"]
    accumulation = (performance['effective_batch'] // settings['batch_size']
                    if performance else settings['accumulation'])
    coverage = None
    if config['data'].get('coverage_epochs'):
        from .data_fork import CoverageTracker
        if not fork_provenance or 'data_change' not in fork_provenance:
            raise ValueError('A coverage epoch requires an explicit new data stream')
        expected = math.ceil(len(records) / (settings['batch_size'] * accumulation))
        if limits['steps'] - fork_provenance['step'] != expected:
            raise ValueError('Coverage budget must be exactly one full permutation plus final padding')
        coverage = CoverageTracker(len(records), settings['batch_size'], settings['seed'], cursor)
    loader_config = deepcopy(config)
    if performance:
        loader_config['data']['workers'] = performance['workers']
        loader_config['data']['compact_transport'] = performance.get('compact_transport', False)
    loader = make_loader(records, loader_config, cursor, max(0, total - step) * accumulation)
    prepared_windows = None
    gpu_prefetch = bool(performance and performance.get('gpu_prefetch', False))
    if performance and performance.get('data_pipeline') == 'prepared':
        from .prefetch import DeviceWindows, PreparedWindows
        if gpu_prefetch and accelerator.device.type != 'cuda':
            raise ValueError('GPU prefetch requires CUDA')
        prepared_windows = PreparedWindows(loader, accumulation, performance['compute_batch'],
                                           max(0, total - step), pin=accelerator.device.type == 'cuda')
        if gpu_prefetch:
            prepared_windows = DeviceWindows(prepared_windows, accelerator.device)
        iterator = None
    else:
        iterator = iter(loader)

    def synchronize_update():
        if gpu_prefetch:
            # Future H2D work on the copy stream need not finish with this update.
            torch.cuda.current_stream(accelerator.device).synchronize()
        else:
            synchronize(accelerator.device)

    plane_enabled = any(config["loss"].get(k, 0)
                        for k in ("plane_flatness", "plane_normal", "plane_anchor"))
    plane_config = PlaneConfig(**config.get("planes", {})) if plane_enabled else None
    geometry_enabled = plane_enabled or any(
        config["loss"].get(k, 0) for k in ("points", "mask", "scale", "gradient", "normal")
    )
    success = False
    last_validation_step = None

    def link_selection(source, name):
        temporary = output / (name + '.incoming')
        os.link(output / source, temporary)
        temporary.replace(output / name)

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
        metrics["feature/total"] = feature_loss_total(metrics, config['loss'])
        for name, view in config['evaluation'].get('resolution_variants', {}).items():
            variant = deepcopy(config)
            variant['evaluation'].update(view)
            with preserve_rng():
                extra = evaluate(system, raw_adapter, statistics, validation, variant,
                                 destination / name, config['evaluation']['limit'])
            extra['feature/total'] = feature_loss_total(extra, config['loss'])
            metrics.update({f'resolution/{name}/{key}': value for key, value in extra.items()})
        for name, extra_records in dataset_validation.items():
            variant = deepcopy(config)
            view = config['evaluation']['dataset_variants'][name]
            variant['evaluation'].update({k: v for k, v in view.items() if k != 'manifest'})
            with preserve_rng():
                extra = evaluate(system, raw_adapter, statistics, extra_records, variant,
                                 destination / ('dataset-' + name), None)
            extra['feature/total'] = feature_loss_total(extra, config['loss'])
            metrics.update({f'dataset/{name}/{key}': value for key, value in extra.items()})
        if performance:
            metrics.update(schedule_position=schedule_position(step, performance),
                           logical_data_cursor=cursor, effective_batch=performance['effective_batch'])
        if settings.get('record_cosine_speed', False):
            metrics.update(optimizer_seconds=optimizer_seconds, elapsed_seconds=elapsed_seconds())
        with (output / "validation.jsonl").open("a") as file:
            file.write(json.dumps({"step": step, **metrics}, allow_nan=False) + "\n")
        print(json.dumps({"validation_step": step, **metrics}), flush=True)
        if writer:
            for key, value in metrics.items():
                writer.add_scalar(f"eval/{key}", value, step)
        tracker.log("validation", metrics, step, image_dir=destination)
        last_validation_step = step
        score = metrics.get(config["evaluation"].get("selection_metric", "teacher_depth/abs_rel"))
        winners = selection_winners(selections, best_metrics, metrics, step)
        primary_improved = False
        if score is not None and metrics.get("postprocess_failed", 1) == 0:
            if best is None or score < best:
                best = score
                primary_improved = True
        if winners or primary_improved:
            # Write one complete checkpoint, even if several metrics improve together.
            save('selection.pt')
            for name in winners:
                link_selection('selection.pt', f'best-{name}.pt')
            if primary_improved:
                link_selection('selection.pt', 'best.pt')
            (output / 'selection.pt').unlink()
            atomic_json(best_metrics, output / 'checkpoint-selection.json')
        if train_probe:
            # Diagnostics never affect best selection, gradients, or training randomness.
            destination = output / f'train-probe-{step:07d}'
            with preserve_rng():
                probe_metrics = evaluate(system, raw_adapter, statistics, train_probe, config,
                                         destination, None)
                probe_metrics['feature/total'] = feature_loss_total(probe_metrics, config['loss'])
                if settings.get('record_cosine_speed', False):
                    probe_metrics.update(optimizer_seconds=optimizer_seconds,
                                         elapsed_seconds=elapsed_seconds())
                with (output / 'train-probe.jsonl').open('a') as file:
                    file.write(json.dumps({'step': step, **probe_metrics}, allow_nan=False) + '\n')
                tracker.log('train_probe', probe_metrics, step)
                print(json.dumps({'train_probe_step': step, **probe_metrics}), flush=True)

    try:
        if not resume:
            # Preserve the actual stage-start weights for decoded before/after comparisons.
            save("initial.pt")
            for name in best_metrics:
                link_selection('initial.pt', f'best-{name}.pt')
            if best is not None:
                link_selection('initial.pt', 'best.pt')
            if best_metrics:
                atomic_json(best_metrics, output / 'checkpoint-selection.json')
        if validation and not resume and (not fork_from or resolution_fork or data_fork) and settings.get("eval_at_start", False):
            validate()
        while step < total and not stop and not budget_exhausted():
            adapter.train()
            optimizer.zero_grad(set_to_none=True)
            synchronize_update()
            started = time.perf_counter()
            totals = {}
            if prepared_windows is not None:
                batches = next(prepared_windows)
            elif performance:
                from .performance import grouped_window
                batches = grouped_window(iterator, accumulation, performance['compute_batch'])
            else:
                batches = (next(iterator) for _ in range(accumulation))
            for cpu_batch in batches:
                cpu_image = cpu_batch['image'] if isinstance(cpu_batch, dict) else cpu_batch
                weight = len(cpu_image) / (settings['batch_size'] * accumulation)
                image = cpu_image.to(accelerator.device, non_blocking=True)
                with amp(), torch.no_grad():
                    features, target = system.encode(image, raw_adapter.indices)
                with amp():
                    prediction = adapter(features)
                losses = statistics.loss(prediction, target)
                if config.get('geometry_supervision'):
                    gt_target = {k: v.to(accelerator.device, non_blocking=True)
                                 for k, v in cpu_batch.items() if k != 'image'}
                    predicted_raw = system.decoder(prediction, image.shape[-2:])
                    losses.update(ground_truth_losses(predicted_raw, gt_target))
                    losses['feature_weighted'] = sum(losses[k] * config['loss'].get(k, 0)
                                                     for k in ('spatial', 'summary', 'cosine',
                                                               'summary_cosine'))
                    losses['geometry_weighted'] = sum(losses[k] * config['loss'].get(k, 0)
                                                      for k in GT_LOSS_KEYS)
                elif geometry_enabled:
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
                if not performance and not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss before optimizer step {step + 1}")
                accelerator.backward(loss * weight)
                for key, value in {**losses, "total": loss}.items():
                    detached = value.detach() if performance else value.detach().item()
                    totals[key] = totals.get(key, 0.0) + detached * weight
                cursor += len(cpu_image) // settings['batch_size']
            if performance:
                keys = list(totals)
                values = torch.stack([totals[k] for k in keys]).float().cpu().tolist()
                totals = dict(zip(keys, values, strict=True))
                if not all(math.isfinite(v) for v in values):
                    raise FloatingPointError("Nonfinite loss; optimizer update cancelled")
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
            if coverage is not None:
                coverage.advance(cursor)
            synchronize_update()
            elapsed = time.perf_counter() - started
            optimizer_seconds += elapsed
            row = {
                "step": step,
                "lr": optimizer.param_groups[0]["lr"],
                "lr_used": lr_used,
                "seconds": elapsed,
                "images_per_second": settings["batch_size"] * accumulation / elapsed,
                "gradient_norm": norm.item(),
                **totals,
            }
            if performance:
                row.update(effective_batch=performance['effective_batch'],
                           compute_batch_limit=performance['compute_batch'],
                           logical_data_cursor=cursor,
                           schedule_position=schedule_position(step, performance))
            if settings.get('record_cosine_speed', False):
                row.update(optimizer_seconds=optimizer_seconds, elapsed_seconds=elapsed_seconds(),
                           spatial_cosine=1 - totals['cosine'],
                           summary_cosine_similarity=1 - totals['summary_cosine'])
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
            cadence_step = step - fork_provenance['step'] if constant_fork else step
            save_due = (interval_due(step, settings['save_every'], performance) if performance and not constant_fork
                        else cadence_step % settings['save_every'] == 0)
            eval_due = (interval_due(step, settings['eval_every'], performance) if performance and not constant_fork
                        else cadence_step % settings['eval_every'] == 0)
            if save_due:
                save("latest.pt")
            if (
                validation
                and eval_due
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
        if prepared_windows is not None:
            prepared_windows.close()
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
    parser.add_argument('--lr-adjustment', help='Reviewed constant-LR change on exact-state resume')
    parser.add_argument("--fork-from", help="New constant-LR experiment preserving optimizer, RNG and data cursor")
    parser.add_argument('--resolution-fork', action='store_true',
                        help='Explicit bucket additions with old/new validation; preserve Adam and LR')
    parser.add_argument('--data-fork', action='store_true',
                        help='Explicit dataset change preserving Adam; reset cursor and evaluate baseline')
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
    parser.add_argument("--performance-config", help="Reviewed runtime tuning and bounded benchmark")
    parser.add_argument("--execution-revision", help="Revision of this trainer invocation")
    parser.add_argument('--budget-started-at', type=float,
                        help='Original stage start; includes initialization failures and downtime')
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
            performance=json.loads(Path(args.performance_config).read_text())
            if args.performance_config else None,
            budget_started_at=args.budget_started_at,
            resolution_fork=args.resolution_fork,
            data_fork=args.data_fork,
            lr_adjustment=json.loads(Path(args.lr_adjustment).read_text())
            if args.lr_adjustment else None,
        )
    )


if __name__ == "__main__":
    main()
