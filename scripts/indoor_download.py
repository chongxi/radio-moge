"""Resumable, scene-balanced RGB download from the official indoor dataset archives.

Uses HTTP Range to read selected ZIP members, with ZIP CRC and image decode checks.
Only downloads data: this script never invokes a model, training, or GPU lifecycle API.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import time
import zipfile

import numpy as np
from PIL import Image
import requests


BASE = "https://docs-assets.developer.apple.com/ml-research/datasets"
METADATA_URLS = {
    "arkitscenes-raw-metadata.csv": f"{BASE}/arkitscenes/v1/raw/metadata.csv",
    "hypersim-scene-splits.csv": (
        "https://raw.githubusercontent.com/apple-aiml-research/ml-hypersim/main/"
        "evermotion_dataset/analysis/metadata_images_split_scene_v1.csv"
    ),
    "ARKitScenes-LICENSE.txt": (
        "https://raw.githubusercontent.com/apple-aiml-research/ARKitScenes/main/LICENSE"
    ),
    "Hypersim-LICENSE.txt": (
        "https://raw.githubusercontent.com/apple-aiml-research/ml-hypersim/main/LICENSE.txt"
    ),
}


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_bytes(data if isinstance(data, bytes) else data.encode())
    os.replace(temporary, path)


def write_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


class RangeFile(io.RawIOBase):
    """Bounded seekable HTTP file; refuse servers that ignore or mis-cache ranges."""

    def __init__(self, url, session=None):
        self.url = url
        self.session = session or requests.Session()
        self.position = 0
        self.cache_start, self.cache = 0, b""
        self.bytes_received = 0
        response = self.session.head(url, timeout=(15, 60), allow_redirects=True)
        response.raise_for_status()
        self.size = int(response.headers["Content-Length"])
        response.close()

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = {0: 0, 1: self.position, 2: self.size}[whence] + offset
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def fetch(self, start, length):
        end = min(self.size, start + length) - 1
        if end < start:
            return b""
        expected = f"bytes {start}-{end}/{self.size}"
        for attempt in range(6):
            try:
                with self.session.get(
                    self.url,
                    headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                    timeout=(15, 90),
                    stream=True,
                ) as response:
                    response.raise_for_status()
                    if (
                        response.status_code != 206
                        or response.headers.get("Content-Range") != expected
                    ):
                        raise ValueError(f"Server did not honor range {expected}")
                    data = response.raw.read(end - start + 2)
                    if len(data) != end - start + 1:
                        raise OSError("Incomplete HTTP range")
                    self.bytes_received += len(data)
                    return data
            except (requests.RequestException, OSError, ValueError):
                if attempt == 5:
                    raise
                time.sleep(min(2**attempt, 20))

    def prefetch(self, start, length):
        self.cache_start, self.cache = start, self.fetch(start, length)

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        if size > 64 * 1024**2:
            raise ValueError("Unexpected read larger than 64 MiB")
        if not (
            self.cache_start <= self.position
            and self.position + size <= self.cache_start + len(self.cache)
        ):
            self.prefetch(self.position, max(size, min(65536, self.size - self.position)))
        offset = self.position - self.cache_start
        data = self.cache[offset : offset + size]
        self.position += len(data)
        return data


class RemoteZip:
    def __init__(self, url, session=None):
        self.file = RangeFile(url, session)
        self.zip = zipfile.ZipFile(self.file)

    def read(self, member):
        info = self.zip.getinfo(member)
        if info.file_size > 64 * 1024**2:
            raise ValueError("Unexpectedly large ZIP member")
        # Usually a single request includes header, filename, extra fields and data.
        self.file.prefetch(
            info.header_offset, info.compress_size + len(info.filename.encode()) + 4096
        )
        return self.zip.read(info)  # zipfile verifies decompression and CRC32.

    def names(self):
        return self.zip.namelist()


def fetch_small(session, url, limit=16 * 1024**2):
    for attempt in range(6):
        try:
            with session.get(url, timeout=(15, 60), stream=True) as response:
                response.raise_for_status()
                data = response.raw.read(limit + 1)
                if len(data) > limit:
                    raise ValueError("Unexpectedly large metadata file")
                return data
        except (requests.RequestException, OSError):
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 20))


def image_info(data):
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        width, height = image.size
        gray = np.asarray(image.convert("L").resize((160, 120)), dtype=np.float32)
        small = np.asarray(image.convert("L").resize((9, 8)))
    lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1]
    bits = small[:, 1:] > small[:, :-1]
    return {
        "width": width,
        "height": height,
        "sharpness": float(lap.var()),
        "dhash": f"{int.from_bytes(np.packbits(bits).tobytes(), 'big'):016x}",
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def verify_existing(root, path):
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    for row in result["records"]:
        data = (root / row["image"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"Existing file hash mismatch: {row['image']}")
    return result


def allocate(total, count):
    return [total // count + int(i < total % count) for i in range(count)]


def make_tasks(metadata, target, seed, validation_per_visit):
    ark = list(csv.DictReader((metadata / "arkitscenes-raw-metadata.csv").open()))

    def known(row):
        return row["visit_id"] not in {"", "NA", "nan"}

    validation_visits = {r["visit_id"] for r in ark if r["fold"] == "Validation" and known(r)}
    groups = defaultdict(list)
    for row in ark:
        if not known(row) or (row["fold"] == "Training" and row["visit_id"] in validation_visits):
            continue
        groups[(row["fold"], row["visit_id"])].append(row)
    tasks = []
    training = sorted(key for key in groups if key[0] == "Training")
    quotas = dict(zip(training, allocate(target, len(training))))
    for (fold, visit), rows in sorted(groups.items()):
        split = "train" if fold == "Training" else "val"
        quota = quotas[(fold, visit)] if split == "train" else validation_per_visit
        random.Random(f"{seed}:{visit}").shuffle(rows)
        tasks.append(
            {
                "dataset": "arkitscenes",
                "group": f"arkitscenes:visit:{visit}",
                "split": split,
                "id": visit,
                "captures": rows,
                "target": quota,
            }
        )
    hypo = defaultdict(list)
    for row in csv.DictReader((metadata / "hypersim-scene-splits.csv").open()):
        if row["included_in_public_release"] == "True" and row["split_partition_name"] in {
            "train",
            "val",
        }:
            hypo[(row["split_partition_name"], row["scene_name"])].append(row)
    for (split, scene), rows in sorted(hypo.items()):
        tasks.append(
            {
                "dataset": "hypersim",
                "group": "hypersim:" + scene,
                "split": split,
                "id": scene,
                "frames": rows,
                "target": len(rows),
            }
        )
    # Interleave datasets, so progress and quality checks cover both early.
    random.Random(seed).shuffle(tasks)
    return tasks


def time_of(name):
    return float(Path(name).stem.split("_")[-1])


def farthest_order(length):
    """Deterministic time-spread ordering, allowing quality rejection without clustering."""
    queue = [(0, length)]
    while queue:
        next_queue = []
        for low, high in queue:
            if low >= high:
                continue
            middle = (low + high) // 2
            yield middle
            next_queue.extend([(low, middle), (middle + 1, high)])
        queue = next_queue


def read_trajectory(data):
    rows = []
    for line in data.decode().splitlines():
        values = [float(v) for v in line.split()]
        if len(values) == 7 and all(math.isfinite(v) for v in values):
            rows.append(values)
    return np.asarray(sorted(rows), dtype=np.float64).reshape(-1, 7)


def closest_pose(trajectory, timestamp):
    if not len(trajectory):
        return None
    index = int(np.argmin(abs(trajectory[:, 0] - timestamp)))
    return trajectory[index].tolist() if abs(trajectory[index, 0] - timestamp) <= 0.15 else None


def download_arkit(task, root, session, sharpness_min):
    records, rejected = [], defaultdict(int)
    seen_hashes = set()
    # Two captures per scene first; additional captures fill quality-filter shortfalls.
    initial_count = min(2, len(task["captures"]))
    first_quotas = allocate(task["target"], initial_count)
    for capture_index, capture in enumerate(task["captures"]):
        remaining = task["target"] - len(records)
        if remaining <= 0:
            break
        quota = min(
            remaining, first_quotas[capture_index] if capture_index < initial_count else remaining
        )
        video, fold = capture["video_id"], capture["fold"]
        prefix = f"{BASE}/arkitscenes/v1/raw/{fold}/{video}"
        directory = root / "arkitscenes" / task["split"] / task["id"] / video
        directory.mkdir(parents=True, exist_ok=True)
        try:
            trajectory_data = fetch_small(session, prefix + "/lowres_wide.traj")
            atomic_write(directory / "lowres_wide.traj", trajectory_data)
            trajectory = read_trajectory(trajectory_data)
            rgb = RemoteZip(prefix + "/vga_wide.zip", session)
            intrinsics = RemoteZip(prefix + "/vga_wide_intrinsics.zip", session)
            names = sorted(
                (n for n in rgb.names() if n.endswith(".png") and not n.startswith("__MACOSX")),
                key=time_of,
            )
            intrinsic_names = {Path(n).stem: n for n in intrinsics.names() if n.endswith(".pincam")}
            times, hashes, selected = [], [], 0
            for index in farthest_order(len(names)):
                if selected >= quota:
                    break
                name = names[index]
                timestamp = time_of(name)
                if any(abs(timestamp - t) < 0.5 for t in times):
                    rejected["temporal_neighbor"] += 1
                    continue
                pose = closest_pose(trajectory, timestamp)
                if pose is None:
                    rejected["missing_pose"] += 1
                    continue
                intrinsic_name = intrinsic_names.get(Path(name).stem)
                if not intrinsic_name:
                    rejected["missing_intrinsics"] += 1
                    continue
                destination = directory / Path(name).name
                data = destination.read_bytes() if destination.exists() else rgb.read(name)
                info = image_info(data)
                if info["width"] != 640 or info["height"] != 480:
                    raise ValueError(f"Unexpected VGA dimensions: {info}")
                if info["sharpness"] < sharpness_min:
                    rejected["blur_or_low_detail"] += 1
                    continue
                if info["sha256"] in seen_hashes:
                    rejected["exact_duplicate"] += 1
                    continue
                perceptual = int(info["dhash"], 16)
                if any((perceptual ^ old).bit_count() <= 2 for old in hashes):
                    rejected["near_duplicate_in_capture"] += 1
                    continue
                camera_data = intrinsics.read(intrinsic_name)
                camera = [float(v) for v in camera_data.decode().split()]
                if len(camera) != 6 or not all(math.isfinite(v) for v in camera):
                    raise ValueError("Invalid camera intrinsics")
                atomic_write(destination, data)
                atomic_write(directory / Path(intrinsic_name).name, camera_data)
                records.append(
                    dict(
                        image=str(destination.relative_to(root)),
                        dataset="arkitscenes",
                        group=task["group"],
                        split=task["split"],
                        video_id=video,
                        timestamp=timestamp,
                        pose_axis_angle_translation=pose,
                        camera_width_height_fx_fy_cx_cy=camera,
                        sky_direction=capture["sky_direction"],
                        source_url=prefix + "/vga_wide.zip",
                        source_member=name,
                        **info,
                    )
                )
                times.append(timestamp)
                hashes.append(perceptual)
                seen_hashes.add(info["sha256"])
                selected += 1
        except requests.HTTPError as error:
            if error.response.status_code == 404:
                rejected["unavailable_capture"] += 1
                continue
            raise
    if not records:
        raise ValueError("No valid RGB frames in scene")
    return {
        "records": records,
        "rejections": dict(rejected),
        "target": task["target"],
        "shortfall": max(0, task["target"] - len(records)),
    }


def download_hypersim(task, root, session):
    scene = task["id"]
    url = f"{BASE}/hypersim/v1/scenes/{scene}.zip"
    archive = RemoteZip(url, session)
    names = set(archive.names())
    records = []
    for frame in task["frames"]:
        member = (
            f"{scene}/images/scene_{frame['camera_name']}_final_preview/"
            f"frame.{int(frame['frame_id']):04d}.tonemap.jpg"
        )
        if member not in names:
            raise ValueError(f"Released RGB frame missing: {member}")
        destination = root / "hypersim" / task["split"] / member
        data = destination.read_bytes() if destination.exists() else archive.read(member)
        info = image_info(data)
        atomic_write(destination, data)
        records.append(
            dict(
                image=str(destination.relative_to(root)),
                dataset="hypersim",
                group=task["group"],
                split=task["split"],
                camera=frame["camera_name"],
                frame_id=int(frame["frame_id"]),
                source_url=url,
                source_member=member,
                **info,
            )
        )
    # Small pose/scale metadata, useful for later geometry evaluation.
    for member in sorted(names):
        if "/_detail/" in member and (
            member.endswith("metadata_scene.csv")
            or member.endswith("metadata_cameras.csv")
            or "camera_keyframe_" in member
        ):
            if not member.endswith("/"):
                atomic_write(root / "hypersim" / task["split"] / member, archive.read(member))
    return {"records": records, "target": task["target"], "shortfall": 0, "rejections": {}}


def task_key(task):
    return f"{task['dataset']}-{task['split']}-{task['id']}"


def run_task(task, root, sharpness_min):
    result_path = root / "receipts" / (task_key(task) + ".json")
    previous = verify_existing(root, result_path)
    if previous is not None:
        return previous
    started = time.time()
    with requests.Session() as session:
        result = (
            download_arkit(task, root, session, sharpness_min)
            if task["dataset"] == "arkitscenes"
            else download_hypersim(task, root, session)
        )
    result.update(task=task_key(task), elapsed_seconds=time.time() - started)
    write_json(result_path, result)
    return result


def finalize(root, tasks):
    splits, hashes, groups = defaultdict(list), {}, defaultdict(set)
    dedup = defaultdict(int)
    for task in tasks:
        receipt = json.loads((root / "receipts" / (task_key(task) + ".json")).read_text())
        for row in receipt["records"]:
            split = row["split"]
            if row["sha256"] in hashes:
                if hashes[row["sha256"]] != split:
                    raise ValueError("Cross-split identical image detected")
                dedup[split] += 1
                continue
            hashes[row["sha256"]] = split
            groups[split].add(row["group"])
            splits[split].append(row)
    if groups["train"] & groups["val"]:
        raise ValueError("Cross-split scene leakage")
    counts = {}
    for split, rows in splits.items():
        rows.sort(key=lambda r: r["image"])
        atomic_write(root / f"{split}.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
        counts[split] = {
            "images": len(rows),
            "groups": len(groups[split]),
            "deduplicated": dedup[split],
        }
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arkit-target", type=int, default=200000)
    parser.add_argument("--validation-per-visit", type=int, default=24)
    parser.add_argument("--workers", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sharpness-min", type=float, default=4.0)
    parser.add_argument("--pilot", action="store_true", help="One scene from each dataset only")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / ".download.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with requests.Session() as session:
        for name, url in METADATA_URLS.items():
            source = args.metadata / name
            if not source.exists():
                atomic_write(source, fetch_small(session, url))
            atomic_write(args.output / "metadata" / name, source.read_bytes())
    tasks = make_tasks(args.metadata, args.arkit_target, args.seed, args.validation_per_visit)
    if args.pilot:
        tasks = [
            next(t for t in tasks if t["dataset"] == ds and t["split"] == "train")
            for ds in ["arkitscenes", "hypersim"]
        ]
    plan = {
        "schema": 1,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "tasks": tasks,
        "training_authorized": False,
        "metadata_urls": METADATA_URLS,
        "preprocessing_notes": [
            "RGB files preserve native source orientation and encoding.",
            "ARKit sky_direction and native intrinsics/poses are recorded; audit and apply upright orientation before training.",
            "Hypersim uses official tonemapped preview RGB, not HDR color arrays.",
            "No ground-truth depth is downloaded in this RGB distillation preparation.",
        ],
        "metadata_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.metadata.glob("*.csv")
        },
    }
    plan_path = args.output / "plan.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        if old != plan:
            raise ValueError("Existing plan differs; use a new output directory")
    else:
        write_json(plan_path, plan)
    status = {
        "stage": "downloading",
        "pid": os.getpid(),
        "started": time.time(),
        "tasks_total": len(tasks),
        "tasks_done": 0,
        "images_verified": {},
        "failures": [],
        "target_images": {},
        "training_started": False,
    }
    for task in tasks:
        key = task["dataset"] + "/" + task["split"]
        status["target_images"][key] = status["target_images"].get(key, 0) + task["target"]
    write_json(args.output / "status.json", status)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_task, task, args.output, args.sharpness_min): task for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                key = task["dataset"] + "/" + task["split"]
                status["images_verified"][key] = status["images_verified"].get(key, 0) + len(
                    result["records"]
                )
                status["tasks_done"] += 1
                print(
                    json.dumps(
                        {
                            "task": task_key(task),
                            "images": len(result["records"]),
                            "shortfall": result["shortfall"],
                            "done": status["tasks_done"],
                        }
                    ),
                    flush=True,
                )
            except Exception as error:
                failure = {"task": task_key(task), "error": f"{type(error).__name__}: {error}"}
                status["failures"].append(failure)
                print(json.dumps(failure), flush=True)
            status["updated"] = time.time()
            write_json(args.output / "status.json", status)
    if status["failures"]:
        status["stage"] = "incomplete_retryable"
    else:
        status["manifests"] = finalize(args.output, tasks)
        status["stage"] = "complete"
    status["finished"] = time.time()
    write_json(args.output / "status.json", status)
    if status["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
