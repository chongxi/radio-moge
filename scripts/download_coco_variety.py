"""Resume a fixed COCO RGB subset, with ZIP CRC, decode and exact-hash checks.

The S3 path addresses the official images.cocodataset.org bucket with valid TLS.
Only RGB is used for teacher distillation. COCO has image splits, not room IDs.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import random
import threading
import time
import zlib

from PIL import Image
import requests

from indoor_download import RemoteZip, atomic_write, write_json

BASE = "https://s3.amazonaws.com/images.cocodataset.org"
LOCAL = threading.local()


def make_plan(seed, train_images, val_images, test_images):
    result = []
    for official, counts in [
        ("train2017", [("train", train_images)]),
        ("val2017", [("val", val_images), ("test", test_images)]),
    ]:
        archive = RemoteZip(f"{BASE}/zips/{official}.zip")
        members = sorted(
            [i for i in archive.zip.infolist() if i.filename.endswith(".jpg")],
            key=lambda i: i.filename,
        )
        random.Random(f"{seed}:{official}").shuffle(members)
        cursor = 0
        for split, count in counts:
            if cursor + count > len(members):
                raise ValueError("Requested subset exceeds official split")
            for member in members[cursor : cursor + count]:
                result.append(
                    dict(
                        split=split,
                        member=member.filename,
                        crc32=member.CRC,
                        bytes=member.file_size,
                        official_split=official,
                    )
                )
            cursor += count
    return result


def fetch(task, output):
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    path = output / "rgb" / task["member"]
    url = f"{BASE}/{task['member']}"
    for attempt in range(5):
        try:
            if path.exists() and attempt == 0:
                data = path.read_bytes()
            else:
                response = LOCAL.session.get(url, timeout=(15, 60))
                response.raise_for_status()
                data = response.content
            if len(data) != task["bytes"] or zlib.crc32(data) != task["crc32"]:
                raise ValueError("Image does not match official ZIP CRC/size")
            with Image.open(io.BytesIO(data)) as picture:
                width, height = picture.size
                picture.verify()
            if min(width, height) < 16:
                raise ValueError("Image too small")
            if not path.exists() or attempt > 0:
                atomic_write(path, data)
            return dict(
                image=str(path.relative_to(output)),
                dataset="coco2017",
                group="coco2017:image:" + path.stem,
                split=task["split"],
                sha256=hashlib.sha256(data).hexdigest(),
                sha256_kind="encoded_file_bytes",
                width=width,
                height=height,
                source_url=url,
                source_member=task["member"],
                official_split=task["official_split"],
                crc32=task["crc32"],
            )
        except (requests.RequestException, OSError, ValueError):
            if attempt == 4:
                raise
            time.sleep(min(2**attempt, 8))


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".download.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    settings = dict(
        seed=args.seed,
        train_images=args.train_images,
        val_images=args.val_images,
        test_images=args.test_images,
    )
    plan_path = output / "plan.json"
    write_json(
        output / "status.json",
        dict(stage="planning", pid=os.getpid(), started=time.time(), settings=settings),
    )
    try:
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            if plan["settings"] != settings:
                raise ValueError("Resume requires identical subset settings")
        else:
            plan = dict(settings=settings, tasks=make_plan(**settings))
            write_json(plan_path, plan)
    except BaseException as error:
        write_json(
            output / "status.json",
            dict(stage="failed", pid=os.getpid(), error=f"{type(error).__name__}: {error}"),
        )
        raise
    state = dict(
        stage="downloading",
        pid=os.getpid(),
        started=time.time(),
        total=len(plan["tasks"]),
        done=0,
        workers=args.workers,
        settings=settings,
    )
    write_json(output / "status.json", state)
    records = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            # executor.map preserves the preselected order; no completion-order sampling.
            for row in pool.map(lambda task: fetch(task, output), plan["tasks"]):
                records.append(row)
                if len(records) % 100 == 0:
                    state.update(done=len(records), updated=time.time())
                    write_json(output / "status.json", state)
        # Held-out records win over train duplicates. Cross-held-out duplicates go to test.
        seen, kept, excluded = set(), [], 0
        for split in ["test", "val", "train"]:
            for row in records:
                if row["split"] != split:
                    continue
                if row["sha256"] in seen:
                    excluded += 1
                    continue
                seen.add(row["sha256"])
                kept.append(row)
        counts = {}
        for split in ["train", "val", "test"]:
            rows = [r for r in kept if r["split"] == split]
            counts[split] = len(rows)
            atomic_write(output / f"{split}.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
        info = dict(
            stage="ready",
            counts=counts,
            settings=settings,
            exact_duplicates=excluded,
            source="COCO 2017 official images.cocodataset.org S3 bucket",
            verification="Every selected RGB checked against official ZIP CRC and decoded; "
            "SHA256 recorded and exact duplicates removed.",
            split_policy="Official train2017 for train; disjoint fixed image subsets of "
            "val2017 for validation and test; no scene identity guarantee.",
            supervision="RGB teacher distillation only; no ground-truth depth.",
        )
        write_json(output / "dataset_info.json", info)
        state.update(stage="complete", done=len(records), finished=time.time(), counts=counts)
    except BaseException as error:
        state.update(stage="failed", done=len(records), error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "status.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-images", type=int, default=40000)
    parser.add_argument("--val-images", type=int, default=256)
    parser.add_argument("--test-images", type=int, default=256)
    parser.add_argument("--workers", type=int, choices=[1, 2, 3, 4], default=2)
    run(parser.parse_args())
