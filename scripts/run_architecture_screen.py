"""Serial scratch screening on a common frozen feature cache; no teacher during updates."""

import argparse
from collections import defaultdict
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from radio_moge.adapters import EncoderFeatures, RadioFeatures
from radio_moge.architecture_screen import conditional_predict, flow_training, make_candidate
from radio_moge.capacity_screen import initialization_seed
from radio_moge.checkpoint import atomic_json, atomic_save, rng_state
from radio_moge.losses import FeatureStatistics, weighted_loss
from radio_moge.tracking import ExperimentTracker, preserve_rng


def load_cache(root, partition):
    index = json.loads((root / "cache-index.json").read_text())
    assert index["stage"] == "passed"
    pieces = defaultdict(list)
    rows = [json.loads(x) for x in (root / f"{partition}.jsonl").read_text().splitlines()]
    for shard in index["shards"]:
        if shard["partition"] != partition:
            continue
        path = Path(shard["path"])
        # One read-back checksum when loading for the screen, not every model.
        with path.open("rb") as file:
            assert hashlib.file_digest(file, "sha256").hexdigest() == shard["sha256"]
        tensors = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        for name, value in tensors.items():
            assert torch.isfinite(value).all()
            pieces[name].append(value.cuda())
    result = {name: torch.cat(values) for name, values in pieces.items()}
    assert len(result["summary"]) == len(rows)
    return result, rows


def batch(cache, indices):
    data = {k: v[indices] for k, v in cache.items()}
    return (
        RadioFeatures(data["spatial"], data["summary"], tuple(data[f"tap{i}"] for i in range(4))),
        EncoderFeatures(data["target_spatial"], data["target_summary"]),
    )


def fit_statistics(cache):
    result = FeatureStatistics(
        cache["target_spatial"].shape[1], cache["target_summary"].shape[1]
    ).cuda()
    # Avoid a full float64 copy of the cache.
    result.fit(
        EncoderFeatures(cache["target_spatial"][i : i + 4], cache["target_summary"][i : i + 4])
        for i in range(0, len(cache["summary"]), 4)
    )
    return result


def deterministic_noise(indices, spatial_shape, summary_dim, seed):
    spatial, summary = [], []
    for index in indices:
        generator = torch.Generator(device="cuda").manual_seed(seed + int(index) * 104729)
        spatial.append(torch.randn(spatial_shape, generator=generator, device="cuda"))
        summary.append(torch.randn(summary_dim, generator=generator, device="cuda"))
    return EncoderFeatures(torch.stack(spatial), torch.stack(summary))


def metrics_per_image(prediction, target, stats):
    result = {}
    for name, scale in [
        ("spatial", stats.spatial_scale[None, :, None, None]),
        ("summary", stats.summary_scale[None]),
    ]:
        p, t = getattr(prediction, name).float(), getattr(target, name).float()
        cosine = F.cosine_similarity(p, t, dim=1).reshape(len(p), -1).mean(-1)
        ratio = p.norm(dim=1).clamp_min(1e-8) / t.norm(dim=1).clamp_min(1e-8)
        result[name + "/cosine"] = cosine.tolist()
        result[name + "/norm_ratio"] = ratio.reshape(len(p), -1).mean(-1).tolist()
        result[name + "/abs_log_norm_ratio"] = (
            ratio.log().abs().reshape(len(p), -1).mean(-1).tolist()
        )
        result[name + "/normalized_mse"] = (
            ((p - t) / scale).square().reshape(len(p), -1).mean(-1).tolist()
        )
    return result


def aggregate(rows):
    keys = [k for k, v in rows[0].items() if type(v) in (int, float) and k != "cache_index"]
    result = {key: sum(r[key] for r in rows) / len(rows) for key in keys}
    sources = sorted({r["dataset"] for r in rows})
    for source in sources:
        group = [r for r in rows if r["dataset"] == source]
        result.update(
            {f"dataset/{source}/{key}": sum(r[key] for r in group) / len(group) for key in keys}
        )
    result.update(
        {
            f"macro/{key}": sum(result[f"dataset/{s}/{key}"] for s in sources) / len(sources)
            for key in keys
        }
    )
    result["images"] = len(rows)
    return result


@torch.no_grad()
def evaluate(model, spec, cache, records, indices, stats, output, step, nfe, clocks, seed):
    with preserve_rng():
        model.eval()
        rows = []
        forward_seconds = 0.0
        for start in range(0, len(indices), 16):
            selected = indices[start : start + 16]
            features, target = batch(cache, selected)
            noise = (
                deterministic_noise(
                    selected, target.spatial.shape[1:], target.summary.shape[1], seed
                )
                if spec["family"].startswith("conditional_")
                else None
            )
            torch.cuda.synchronize()
            tick = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = (
                    conditional_predict(model, features, stats, noise, nfe)
                    if noise is not None
                    else model(features)
                )
            torch.cuda.synchronize()
            forward_seconds += time.perf_counter() - tick
            values = metrics_per_image(prediction, target, stats)
            for j, index in enumerate(selected):
                row = records[index]
                rows.append(
                    dict(
                        image=row["image"],
                        group=row["group"],
                        dataset=row["dataset"],
                        screen_id=row["screen_id"],
                        cache_index=index,
                        **{k: v[j] for k, v in values.items()},
                    )
                )
        assert all(math.isfinite(v) for row in rows for v in row.values() if isinstance(v, float))
        result = dict(
            step=step,
            nfe=nfe,
            **clocks,
            **aggregate(rows),
            inference_ms_per_image=forward_seconds * 1000 / len(rows),
        )
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(rows, output / "per_image.json")
        atomic_json(result, output / "metrics.json")
        digest = hashlib.sha256((output / "per_image.json").read_bytes()).hexdigest()
        atomic_json(
            dict(stage="written", per_image_sha256=digest, image_count=len(rows)),
            output / "integrity.json",
        )
        return result


def learning_rate(step, plan):
    return plan["lr"] * min(1.0, (step + 1) / plan["warmup_updates"])


def preflight(root, plan, dimensions, cache, stats):
    proofs = {}
    features, target = batch(cache, [0, 256, 512, 768])
    for spec in plan["candidates"]:
        torch.manual_seed(initialization_seed(spec, plan))
        model = make_candidate(spec, dimensions).cuda()
        torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if spec["family"].startswith("conditional_"):
                noise = deterministic_noise(
                    [0, 256, 512, 768], target.spatial.shape[1:], target.summary.shape[1], 903
                )
                times = torch.tensor([0.0, 0.2, 0.5, 0.9], device="cuda")
                losses = flow_training(model, features, target, stats, noise, times)
            else:
                losses = stats.loss(model(features), target)
            loss = weighted_loss(losses, plan["loss_weights"])
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.numel()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert any(torch.count_nonzero(g) > 0 for g in gradients)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        torch.cuda.synchronize()
        proofs[spec["name"]] = dict(
            parameters=sum(p.numel() for p in model.parameters()),
            finite_loss=float(loss.detach()),
            gradient_norm=float(norm),
            finite_nonzero_gradients=True,
            optimizer_updates=0,
            seconds=time.perf_counter() - tick,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        )
        atomic_json(dict(stage="running", candidates=proofs), root / "preflight-verification.json")
        del model, gradients, losses, loss
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(
        dict(stage="passed", candidates=proofs, optimizer_updates=0),
        root / "preflight-verification.json",
    )


def run(root, revision):
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("Real screening is GPU only")
    lock = (root / ".training.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "status.json").exists():
        raise FileExistsError("Inspect prior screen; never blindly restart")
    plan = json.loads((root / "plan.json").read_text())
    cache_info = json.loads((root / "cache-index.json").read_text())
    started = time.time()
    status = dict(
        stage="loading_cache",
        pid=os.getpid(),
        started_at=started,
        revision=revision,
        completed=[],
        current=None,
        step=0,
    )
    atomic_json(status, root / "status.json")
    train_cache, train_rows = load_cache(root, "train")
    val_cache, val_rows = load_cache(root, "validation")
    stats = fit_statistics(train_cache)
    atomic_save(stats.state_dict(), root / "statistics.pt")
    dimensions = dict(
        spatial_dim=train_cache["spatial"].shape[1],
        summary_dim=train_cache["summary"].shape[1],
        target_dim=train_cache["target_spatial"].shape[1],
        target_summary_dim=train_cache["target_summary"].shape[1],
    )
    generator = torch.Generator().manual_seed(plan["data_seed"])
    order = torch.cat(
        [
            torch.randperm(len(train_rows), generator=generator)
            for _ in range(math.ceil(plan["updates"] * plan["effective_batch"] / len(train_rows)))
        ]
    )
    atomic_json(
        dict(
            stage="passed",
            dimensions=dimensions,
            training_images=len(train_rows),
            validation_images=len(val_rows),
            calibration_images=len(train_rows),
            cache_index_sha256=hashlib.sha256((root / "cache-index.json").read_bytes()).hexdigest(),
            sample_order_sha256=hashlib.sha256(order.numpy().tobytes()).hexdigest(),
            data_seed=plan["data_seed"],
            calibration_uses_validation=False,
            cache_gpu_bytes=sum(
                x.numel() * x.element_size() for c in (train_cache, val_cache) for x in c.values()
            ),
        ),
        root / "cache-load-verification.json",
    )
    status.update(stage="preflight", updated_at=time.time())
    atomic_json(status, root / "status.json")
    preflight(root, plan, dimensions, train_cache, stats)
    results = {}
    for spec in plan["candidates"]:
        if time.time() - started + cache_info["elapsed_seconds"] > plan["total_screen_seconds"]:
            status.update(stage="budget_exhausted", updated_at=time.time(), current=None)
            atomic_json(status, root / "status.json")
            return
        torch.manual_seed(initialization_seed(spec, plan))
        torch.cuda.manual_seed_all(initialization_seed(spec, plan))
        output = root / spec["name"]
        output.mkdir(exist_ok=False)
        arm_started = time.perf_counter()
        model = make_candidate(spec, dimensions).cuda()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=plan["lr"], weight_decay=plan["weight_decay"]
        )
        params = sum(p.numel() for p in model.parameters())
        config = dict(
            architecture=spec,
            dimensions=dimensions,
            fresh_initialization=True,
            seed=initialization_seed(spec, plan),
            train={
                k: plan[k]
                for k in ["effective_batch", "microbatch", "lr", "warmup_updates", "updates"]
            },
            loss=plan["loss_weights"],
            tracking=dict(
                enabled=True,
                project="radio-moge",
                entity="brainsim",
                mode="online",
                name=root.name + "-" + spec["name"],
                group=root.name,
                tags=["scratch", "architecture-screen", "seven-sources", spec["family"]],
                images=0,
            ),
        )
        atomic_json(config, output / "config.json")
        tracker = ExperimentTracker(
            config,
            output,
            provenance=dict(
                execution_revision=revision,
                parameters=params,
                cache_sha256=hashlib.sha256((root / "cache-index.json").read_bytes()).hexdigest(),
            ),
        )
        status.update(
            stage="training",
            current=spec["name"],
            step=0,
            parameters=params,
            updated_at=time.time(),
        )
        atomic_json(status, root / "status.json")
        optimizer_seconds, step = 0.0, 0
        nfe_list = [1, 2, 4] if spec["family"] == "conditional_flow" else [1]
        torch.cuda.reset_peak_memory_stats()
        success = False
        try:

            def assess(current_model=model):
                nonlocal results
                for nfe in nfe_list:
                    clocks = dict(
                        optimizer_seconds=optimizer_seconds,
                        elapsed_seconds=time.perf_counter() - arm_started,
                    )
                    for partition, cache, rows, ids in [
                        ("validation", val_cache, val_rows, list(range(len(val_rows)))),
                        ("train_probe", train_cache, train_rows, plan["train_probe_indices"]),
                    ]:
                        metric = evaluate(
                            current_model,
                            spec,
                            cache,
                            rows,
                            ids,
                            stats,
                            output / f"{partition}-{step:07d}-nfe{nfe}",
                            step,
                            nfe,
                            clocks,
                            seed=903,
                        )
                        with (output / f"{partition}-nfe{nfe}.jsonl").open("a") as file:
                            file.write(json.dumps(metric, allow_nan=False) + "\n")
                        tracker.log(
                            partition if nfe == 1 else partition + f"_nfe{nfe}", metric, step
                        )
                        if partition == "validation":
                            results[spec["name"] + f"/nfe{nfe}"] = dict(parameters=params, **metric)
                            atomic_json(results, root / "latest-results.json")
                current_model.train()

            assess()
            with (output / "train.jsonl").open("a", buffering=1) as log:
                while step < plan["updates"]:
                    if (
                        time.perf_counter() - arm_started >= plan["max_seconds_per_candidate"]
                        or time.time() - started + cache_info["elapsed_seconds"]
                        >= plan["total_screen_seconds"]
                    ):
                        break
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    lr = learning_rate(step, plan)
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.zero_grad(set_to_none=True)
                    values = defaultdict(float)
                    ids = order[
                        step * plan["effective_batch"] : (step + 1) * plan["effective_batch"]
                    ].tolist()
                    for start in range(0, len(ids), plan["microbatch"]):
                        selected = ids[start : start + plan["microbatch"]]
                        features, target = batch(train_cache, selected)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            if spec["family"].startswith("conditional_"):
                                noise = deterministic_noise(
                                    selected,
                                    target.spatial.shape[1:],
                                    target.summary.shape[1],
                                    201 + step * 17,
                                )
                                t = (
                                    torch.rand(len(selected), device="cuda")
                                    if spec["family"] == "conditional_flow"
                                    else torch.zeros(len(selected), device="cuda")
                                )
                                losses = flow_training(model, features, target, stats, noise, t)
                            else:
                                losses = stats.loss(model(features), target)
                            loss = weighted_loss(losses, plan["loss_weights"])
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite objective before update")
                        (loss * len(selected) / len(ids)).backward()
                        for k, v in {**losses, "total": loss}.items():
                            values[k] += float(v.detach()) * len(selected) / len(ids)
                    grad = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), plan["clip_grad"], error_if_nonfinite=True
                    )
                    optimizer.step()
                    torch.cuda.synchronize()
                    duration = time.perf_counter() - tick
                    optimizer_seconds += duration
                    step += 1
                    if step == 1 or step % 10 == 0:
                        row = dict(
                            step=step,
                            lr_used=lr,
                            gradient_norm=float(grad),
                            seconds=duration,
                            optimizer_seconds=optimizer_seconds,
                            elapsed_seconds=time.perf_counter() - arm_started,
                            **values,
                        )
                        assert all(math.isfinite(v) for v in row.values())
                        log.write(json.dumps(row) + "\n")
                        tracker.log("train", row, step)
                        status.update(step=step, updated_at=time.time())
                        atomic_json(status, root / "status.json")
                    if step in plan["evaluation_steps"]:
                        assess()
            if step not in plan["evaluation_steps"]:
                assess()
            torch.cuda.synchronize()
            state = dict(
                adapter={k: v.detach().cpu() for k, v in model.state_dict().items()},
                statistics={k: v.cpu() for k, v in stats.state_dict().items()},
                optimizer=optimizer.state_dict(),
                rng=rng_state(),
                spec=spec,
                dimensions=dimensions,
                step=step,
                cursor=step * plan["effective_batch"],
                config=config,
                execution_revision=revision,
                initialization="scratch; no parent checkpoint",
                optimizer_seconds=optimizer_seconds,
            )
            atomic_save(state, output / "latest.pt")
            del state
            stop = dict(
                reason="step_limit" if step == plan["updates"] else "time_limit",
                step=step,
                cursor=step * plan["effective_batch"],
                optimizer_seconds=optimizer_seconds,
                elapsed_seconds=time.perf_counter() - arm_started,
                parameters=params,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            )
            atomic_json(stop, output / "stop-report.json")
            success = True
        finally:
            tracker.finish(success)
        status["completed"].append(dict(name=spec["name"], **stop))
        atomic_json(status, root / "status.json")
        del assess, model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    status.update(
        stage="complete",
        current=None,
        finished_at=time.time(),
        updated_at=time.time(),
        confirmation_evaluated=False,
        automatic_extension=False,
    )
    atomic_json(status, root / "status.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--revision", required=True)
    args = p.parse_args()
    try:
        run(args.root, args.revision)
    except Exception as error:
        atomic_json(
            dict(
                stage="failed",
                type=type(error).__name__,
                message=str(error),
                checked_at=time.time(),
            ),
            args.root / "training-error.json",
        )
        raise
