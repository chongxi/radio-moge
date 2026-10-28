"""Read-only reserved-feature and decoded-depth confirmation of a frozen shortlist."""

import argparse
from collections import Counter
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from radio_moge.architecture_screen import conditional_predict, make_candidate
from radio_moge.checkpoint import atomic_json
from radio_moge.data import TrainingImages, evaluation_image
from radio_moge.decoder import MoGeDecoder, postprocess
from radio_moge.losses import FeatureStatistics
from radio_moge.metrics import aggregate, dataset_metrics, depth_metrics, fov_error
from radio_moge.planes import PlaneConfig, detect_planes, planarity_metrics, preservation_metrics
from radio_moge.upstream import load_moge
from run_architecture_screen import batch, deterministic_noise, evaluate, load_cache


def identity(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def center_box(width, height, output_height=480, output_width=640):
    cw = min(width, max(1, round(height * output_width / output_height)))
    ch = min(height, max(1, round(width * output_height / output_width)))
    left, top = (width - cw) // 2, (height - ch) // 2
    return left, top, left + cw, top + ch


def geometry_view(record):
    # RGB is exactly the cache's transform. GT receives the same oriented crop,
    # with nearest-neighbour resizing to preserve invalid depth pixels.
    image = TrainingImages([record], augment=False)[(0, 0, 480, 640)].unsqueeze(0)
    gt = None
    if "depth" in record:
        full, depth = evaluation_image(record, max_side=1_000_000)
        height, width = full.shape[-2:]
        gt = torch.from_numpy(
            np.array(
                Image.fromarray(depth.numpy())
                .crop(center_box(width, height))
                .resize((640, 480), Image.Resampling.NEAREST),
                copy=True,
            )
        )
    return image, gt


def geometry_indices(records):
    counts, result = Counter(), []
    for index, row in enumerate(records):
        if counts[row["dataset"]] < 4 or row["dataset"] == "nyuv2":
            result.append(index)
        counts[row["dataset"]] += 1
    return result


def finite_rows(rows):
    for row in rows:
        for key, value in list(row.items()):
            if isinstance(value, float) and not np.isfinite(value):
                row[key] = None
    return rows


@torch.no_grad()
def confirm(root, revision):
    torch.set_num_threads(2)
    lock = (root.parent / "attention-24h/.training.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "confirmation-status.json").exists():
        raise FileExistsError("Inspect existing confirmation; do not duplicate")
    plan = json.loads((root / "plan.json").read_text())
    shortlist_path = root / "shortlist.json"
    shortlist = json.loads(shortlist_path.read_text())
    assert (
        shortlist["stage"] == "frozen_before_confirmation"
        and not shortlist["confirmation_consulted"]
    )
    queue = json.loads((root / "queue-status.json").read_text())
    assert queue["child_pid"] is None and queue["child_exit"] == 0
    assert json.loads((root / "status.json").read_text())["stage"] == "complete"
    for relative, digest in shortlist["screen_metric_sha256"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest
    assert (
        hashlib.sha256((root / "confirmation.jsonl").read_bytes()).hexdigest()
        == plan["manifest_sha256"]["confirmation"]
    )
    started = time.time()
    status = dict(
        stage="loading",
        pid=os.getpid(),
        started_at=started,
        revision=revision,
        shortlist_sha256=hashlib.sha256(shortlist_path.read_bytes()).hexdigest(),
        optimizer_updates=0,
        completed=[],
    )
    atomic_json(status, root / "confirmation-status.json")

    def progress(stage, **kwargs):
        if time.time() - started > 1800:
            raise TimeoutError("Read-only confirmation budget exhausted; no automatic retry")
        status.update(stage=stage, updated_at=time.time(), **kwargs)
        atomic_json(status, root / "confirmation-status.json")

    cache, records = load_cache(root, "confirmation")
    cache_index = json.loads((root / "cache-index.json").read_text())
    image_identities = {
        image["image"]: image
        for shard in cache_index["shards"]
        if shard["partition"] == "confirmation"
        for image in shard["images"]
    }
    dims = json.loads((root / "cache-load-verification.json").read_text())["dimensions"]
    stats = FeatureStatistics(dims["target_dim"], dims["target_summary_dim"]).cuda()
    stats.load_state_dict(torch.load(root / "statistics.pt", map_location="cpu", weights_only=True))
    model, _ = load_moge(plan["models"], "cuda")
    decoder = MoGeDecoder(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    ids = geometry_indices(records)
    assert len(ids) == 56
    export_ids = [
        next(i for i in ids if records[i]["dataset"] == source)
        for source in ("nyuv2", "arkitscenes", "structured3d", "coco2017")
    ]
    plane_config = PlaneConfig()
    teachers, teacher_rows = {}, []
    for index in ids:
        progress("teacher_geometry", geometry_index=index)
        source = Path(records[index]["image"])
        source_stat = source.stat()
        recorded = image_identities[str(source)]
        assert (source_stat.st_size, source_stat.st_mtime_ns) == (
            recorded["bytes"],
            recorded["mtime_ns"],
        )
        image, gt = geometry_view(records[index])
        assert (source.stat().st_size, source.stat().st_mtime_ns) == (
            recorded["bytes"],
            recorded["mtime_ns"],
        )
        image = image.cuda()
        _, target = batch(cache, [index])
        row = dict(
            image=records[index]["image"],
            dataset=records[index]["dataset"],
            group=records[index]["group"],
            cache_index=index,
            input_image_sha256=recorded["sha256"],
        )
        if "depth" in records[index]:
            depth_path = Path(records[index]["depth"])
            with depth_path.open("rb") as handle:
                row["gt_source_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
        try:
            teacher = postprocess(
                decoder(target, (480, 640)), image, plan["models"].get("moge_source")
            )
            valid = teacher["mask"] & torch.isfinite(teacher["depth"]) & (teacher["depth"] > 0)
            assert valid.any(), "No valid teacher pixels"
            points = teacher["points"][0].cpu().numpy()
            patches = detect_planes(points, valid[0].cpu().numpy(), plane_config)
            labels = np.zeros((480, 640), dtype=np.uint8)
            for number, patch in enumerate(patches, 1):
                labels[patch.mask] = number
            row.update(
                {f"planes/teacher/{k}": v for k, v in planarity_metrics(points, patches).items()}
            )
            row.update(
                {
                    "planes/regions": len(patches),
                    "planes/mask_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
                }
            )
            if gt is not None:
                row.update(
                    {
                        f"gt_teacher/{k}": v
                        for k, v in depth_metrics(
                            teacher["depth"], gt.cuda().unsqueeze(0), gt.cuda().unsqueeze(0) > 0
                        ).items()
                    }
                )
            row["teacher_failed"] = 0
            teachers[index] = dict(
                image=image.cpu(),
                gt=gt,
                prediction={k: v.cpu() for k, v in teacher.items() if torch.is_tensor(v)},
                valid=valid.cpu(),
                patches=patches,
                labels=labels,
            )
        except (ValueError, AssertionError, np.linalg.LinAlgError) as error:
            row.update(teacher_failed=1, teacher_error=str(error))
        teacher_rows.append(row)
    destination = root / "confirmation"
    destination.mkdir(exist_ok=True)
    atomic_json(finite_rows(teacher_rows), destination / "teacher-per-image.json")
    sources = {}
    for item in shortlist["selected"]:
        name, nfe = item["name"], item["nfe"]
        progress("candidate", current=name, nfe=nfe)
        path = root / name / "latest.pt"
        before = identity(path)
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        assert state["step"] == 1000 and state["cursor"] == 16000
        assert state["execution_revision"] == plan["execution_revision"]
        assert all(
            torch.equal(v.cpu(), state["statistics"][k]) for k, v in stats.state_dict().items()
        )
        adam_steps = sorted({int(s["step"]) for s in state["optimizer"]["state"].values()})
        assert adam_steps == [1000]
        candidate = make_candidate(state["spec"], state["dimensions"]).cuda().eval()
        candidate.load_state_dict(state["adapter"])
        assert identity(path) == before
        sources[name] = dict(
            source=str(path),
            bytes=before[2],
            sha256=digest,
            identity=before,
            step=state["step"],
            cursor=state["cursor"],
            adam_steps=adam_steps,
        )
        atomic_json(sources, destination / "checkpoint-sources.json")
        spec = state["spec"]
        del state
        arm = destination / name
        metrics = evaluate(
            candidate,
            spec,
            cache,
            records,
            list(range(len(records))),
            stats,
            arm / f"features-nfe{nfe}",
            1000,
            nfe,
            dict(optimizer_seconds=0.0, elapsed_seconds=time.time() - started),
            seed=903,
        )
        rows = []
        for teacher_row in teacher_rows:
            index = teacher_row["cache_index"]
            progress("candidate_geometry", current=name, geometry_index=index)
            row = dict(teacher_row)
            if index not in teachers:
                row["student_not_evaluated"] = 1
                rows.append(row)
                continue
            info = teachers[index]
            features, target = batch(cache, [index])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if spec["family"].startswith("conditional_"):
                    noise = deterministic_noise(
                        [index], target.spatial.shape[1:], target.summary.shape[1], 903
                    )
                    predicted_features = conditional_predict(candidate, features, stats, noise, nfe)
                else:
                    predicted_features = candidate(features)
            image = info["image"].cuda()
            try:
                student = postprocess(
                    decoder(predicted_features, (480, 640)),
                    image,
                    plan["models"].get("moge_source"),
                )
                teacher = info["prediction"]
                row.update(
                    {
                        f"teacher_depth/{k}": v
                        for k, v in depth_metrics(
                            student["depth"], teacher["depth"].cuda(), info["valid"].cuda()
                        ).items()
                    }
                )
                row["fov_error_deg"] = fov_error(
                    student["intrinsics"], teacher["intrinsics"].cuda()
                )
                if info["gt"] is not None:
                    gt = info["gt"].cuda().unsqueeze(0)
                    row.update(
                        {
                            f"gt_student/{k}": v
                            for k, v in depth_metrics(student["depth"], gt, gt > 0).items()
                        }
                    )
                row["postprocess_failed"] = 0
                # Preserve depth accuracy rows when a plane or edge metric is
                # undefined, rather than silently excluding difficult images.
                try:
                    row.update(
                        {
                            f"planes/student/{k}": v
                            for k, v in planarity_metrics(
                                student["points"][0].cpu().numpy(), info["patches"]
                            ).items()
                        }
                    )
                    row["planes/failed"] = 0
                except (ValueError, np.linalg.LinAlgError) as error:
                    row.update({"planes/failed": 1, "planes/error": str(error)})
                try:
                    row.update(
                        preservation_metrics(
                            student["depth"][0].cpu().numpy(),
                            teacher["depth"][0].numpy(),
                            info["valid"][0].numpy(),
                            info["patches"],
                        )
                    )
                    row["preservation/failed"] = 0
                except ValueError as error:
                    row.update({"preservation/failed": 1, "preservation/error": str(error)})
                if index in export_ids:
                    p = arm / f"geometry-{index:03d}.npz"
                    temporary = p.with_suffix(".incoming")
                    with temporary.open("wb") as file:
                        np.savez_compressed(
                            file,
                            rgb=(info["image"][0].permute(1, 2, 0).numpy() * 255)
                            .clip(0, 255)
                            .astype(np.uint8),
                            labels=info["labels"],
                            teacher_points=teacher["points"][0].numpy(),
                            student_points=student["points"][0].cpu().numpy(),
                            teacher_depth=teacher["depth"][0].numpy(),
                            student_depth=student["depth"][0].cpu().numpy(),
                            teacher_intrinsics=teacher["intrinsics"][0].numpy(),
                            student_intrinsics=student["intrinsics"][0].cpu().numpy(),
                        )
                    temporary.replace(p)
            except (ValueError, np.linalg.LinAlgError) as error:
                row.update(postprocess_failed=1, postprocess_error=str(error))
            rows.append(row)
        rows = finite_rows(rows)
        atomic_json(rows, arm / "geometry-per-image.json")
        atomic_json({**aggregate(rows), **dataset_metrics(rows)}, arm / "geometry-metrics.json")
        assert identity(path) == before
        status["completed"].append(
            dict(
                name=name,
                nfe=nfe,
                spatial=metrics["macro/spatial/cosine"],
                summary=metrics["macro/summary/cosine"],
            )
        )
        del candidate
        gc.collect()
        torch.cuda.empty_cache()
    progress("complete", finished_at=time.time(), current=None, optimizer_updates=0)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--revision", required=True)
    a = p.parse_args()
    try:
        confirm(a.root, a.revision)
    except Exception as error:
        atomic_json(
            dict(
                stage="failed",
                type=type(error).__name__,
                message=str(error),
                checked_at=time.time(),
            ),
            a.root / "confirmation-error.json",
        )
        raise
