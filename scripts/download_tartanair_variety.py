"""Download a reproducible subset or all TartanAir left RGB from official ZIPs.

Whole environment families are held out. Only left RGB is fetched, using HTTP
ranges, with ZIP CRC, complete image decoding, and encoded/decoded SHA256 checks.
No training or GPU lifecycle operations are performed here.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import random
import threading
import time
import zlib

from PIL import Image
import requests

from indoor_download import RemoteZip, atomic_write, write_json

BASE = (
    "https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/"
    "AUTH_ac8533a83cff4d48bc8c608ad222d330/tartanair"
)
CATALOG = (
    "https://raw.githubusercontent.com/castacks/tartanair_tools/"
    "master/download_training_zipfiles.txt"
)
FAMILIES = {
    "abandonedfactory_night": "abandonedfactory",
    "seasonsforest_winter": "seasonsforest",
    # Conservatively keep both office worlds in the same split.
    "office2": "office",
}
VALIDATION = {"office", "japanesealley"}
TEST = {"seasonsforest", "amusement"}
LOCAL = threading.local()


def fetch_catalog(url=CATALOG):
    # GitHub can gzip metadata. Decode transport compression before caching UTF-8.
    with requests.Session() as session:
        with session.get(url, timeout=(15, 60), stream=True) as response:
            response.raise_for_status()
            data = response.raw.read(16 * 1024**2 + 1, decode_content=True)
            if len(data) > 16 * 1024**2:
                raise ValueError("Archive catalog exceeds metadata size bound")
            data.decode("utf-8")
            return data


def rgb_archives(catalog):
    # The official list contains '<relative ZIP path> <size in GB>' per line.
    paths = [line.split()[0] for line in catalog.splitlines() if line.strip()]
    return sorted({path for path in paths if path.endswith("/image_left.zip")})


def family(environment):
    return FAMILIES.get(environment, environment)


def split_for(environment):
    name = family(environment)
    return "val" if name in VALIDATION else "test" if name in TEST else "train"


def quotas(capacities, total):
    """Water-fill integer budgets without exceeding a small group's capacity."""
    result = dict.fromkeys(sorted(capacities), 0)
    remaining = min(total, sum(capacities.values()))
    while remaining:
        for key in result:
            if result[key] < capacities[key]:
                result[key] += 1
                remaining -= 1
                if not remaining:
                    break
    return result


def select(index, targets, seed, stride):
    if stride < 1 or any(n < 1 for n in targets.values()):
        raise ValueError("Positive targets and frame stride required")
    pools = defaultdict(lambda: defaultdict(list))
    for row in index:
        pools[row["family"]][row["trajectory"]].append(row)
    eligible = {}
    for fam, trajectories in sorted(pools.items()):
        eligible[fam] = {}
        for trajectory, rows in sorted(trajectories.items()):
            rng = random.Random(f"{seed}:{trajectory}")
            # All chosen frames are separated by at least 'stride' frames.
            rows.sort(key=lambda r: r["member"])
            eligible[fam][trajectory] = rows[rng.randrange(min(stride, len(rows))) :: stride]
    selected = []
    for split, target in targets.items():
        families = {f: t for f, t in eligible.items() if split_for(f) == split}
        counts = quotas({f: sum(map(len, t.values())) for f, t in families.items()}, target)
        for fam, count in counts.items():
            trajectories = families[fam]
            for trajectory, n in quotas(
                {t: len(v) for t, v in trajectories.items()}, count
            ).items():
                rows = trajectories[trajectory]
                rng = random.Random(f"{seed}:bins:{trajectory}")
                # One random frame per time bin; never take just a trajectory prefix.
                for i in range(n):
                    selected.append(
                        rows[rng.randrange(i * len(rows) // n, (i + 1) * len(rows) // n)]
                    )
    return sorted(selected, key=lambda r: (r["archive"], r["member"]))


def select_all(index):
    """Keep every indexed frame and preserve the established family holdouts."""
    identities = {(r["archive"], r["member"]) for r in index}
    if len(identities) != len(index):
        raise ValueError("Repeated archive/member in full RGB inventory")
    for row in index:
        if row["family"] != family(row["environment"]) or row["split"] != split_for(
            row["environment"]
        ):
            raise ValueError("Full inventory violates environment-family split policy")
    return sorted(index, key=lambda r: (r["archive"], r["member"]))


def archive_index(archive, output):
    parts = PurePosixPath(archive).parts
    if len(parts) != 3 or parts[1] not in {"Easy", "Hard"} or parts[2] != "image_left.zip":
        raise ValueError(f"Unexpected archive: {archive}")
    environment, difficulty, _ = parts
    cache = output / "indices" / f"{environment}-{difficulty}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    remote = RemoteZip(f"{BASE}/{archive}")
    rows = []
    for member in remote.zip.infolist():
        p = PurePosixPath(member.filename)
        if p.suffix != ".png":
            continue
        if (
            len(p.parts) != 5
            or p.parts[:2] != (environment, difficulty)
            or p.parts[3] != "image_left"
            or ".." in p.parts
            or p.is_absolute()
        ):
            raise ValueError(f"Unexpected RGB member: {member.filename}")
        rows.append(
            dict(
                archive=archive,
                member=member.filename,
                environment=environment,
                family=family(environment),
                trajectory="/".join(p.parts[:3]),
                split=split_for(environment),
                crc32=member.CRC,
                bytes=member.file_size,
            )
        )
    remote.zip.close()
    remote.file.session.close()
    if not rows:
        raise ValueError(f"Empty RGB archive: {archive}")
    write_json(cache, rows)
    return rows


def validate_rgb(data, task):
    if len(data) != task["bytes"] or zlib.crc32(data) != task["crc32"]:
        raise ValueError("RGB size/CRC mismatch against ZIP directory")
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        rgb = image.convert("RGB")
        width, height = rgb.size
        if min(width, height) < 16:
            raise ValueError("RGB image too small")
        decoded = hashlib.sha256(f"{width}x{height}:RGB:".encode() + rgb.tobytes()).hexdigest()
    return dict(
        width=width,
        height=height,
        decoded_rgb_sha256=decoded,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def fetch(task, output, reuse_root=None):
    path = output / "rgb" / task["member"]
    if path.exists():
        data = path.read_bytes()
        info = validate_rgb(data, task)  # A corrupted resume fails closed.
    elif reuse_root is not None and (reuse_root / "rgb" / task["member"]).is_file():
        source = reuse_root / "rgb" / task["member"]
        data = source.read_bytes()
        info = validate_rgb(data, task)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, path)
        except OSError as error:
            import errno

            if error.errno != errno.EXDEV:
                raise
            atomic_write(path, data)
    else:
        if getattr(LOCAL, "archive", None) != task["archive"]:
            if hasattr(LOCAL, "remote"):
                LOCAL.remote.zip.close()
                LOCAL.remote.file.session.close()
            LOCAL.remote = RemoteZip(f"{BASE}/{task['archive']}")
            LOCAL.archive = task["archive"]
        data = LOCAL.remote.read(task["member"])
        info = validate_rgb(data, task)
        atomic_write(path, data)
    return dict(
        image=str(path.relative_to(output)),
        dataset="tartanair",
        split=task["split"],
        group="tartanair:environment:" + task["family"],
        environment=task["environment"],
        trajectory=task["trajectory"],
        source_url=f"{BASE}/{task['archive']}",
        source_member=task["member"],
        sha256_kind="encoded_file_bytes",
        crc32=task["crc32"],
        **info,
    )


def deduplicate(records):
    # Held-out wins. Decoded hashes also catch differently encoded exact RGBs.
    encoded, decoded, kept, rejected = set(), set(), [], []
    for split in ("test", "val", "train"):
        for row in records:
            if row["split"] != split:
                continue
            if row["sha256"] in encoded or row["decoded_rgb_sha256"] in decoded:
                rejected.append(dict(image=row["image"], split=split, reason="exact_duplicate"))
                continue
            encoded.add(row["sha256"])
            decoded.add(row["decoded_rgb_sha256"])
            kept.append(row)
    return kept, rejected


def wait_for_reuse_source(source, output, state, wait):
    """Serialize the full download after the existing subset, without signaling it."""
    while True:
        previous = json.loads((source / "status.json").read_text())
        if previous["stage"] == "failed":
            raise ValueError("Reuse source download failed; inspect it before continuing")
        if previous["stage"] == "complete":
            with (source / ".download.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    if json.loads((source / "status.json").read_text())["stage"] == "complete":
                        return
        if not wait:
            raise ValueError("Reuse source must finish downloading first")
        state.update(
            stage="waiting_for_reuse_source",
            reuse_source=str(source),
            source_done=previous.get("done"),
            source_total=previous.get("total"),
            updated=time.time(),
        )
        write_json(output / "status.json", state)
        time.sleep(30)


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".download.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    all_frames = getattr(args, "all_frames", False)
    reuse_root = getattr(args, "reuse_root", None)
    if reuse_root is not None:
        reuse_root = reuse_root.resolve()
        if reuse_root == output:
            raise ValueError("Reuse source must differ from destination")
    settings = (
        dict(selection="all_left_rgb_frames", seed=args.seed)
        if all_frames
        else dict(
            seed=args.seed,
            stride=args.stride,
            targets=dict(train=args.train_images, val=args.val_images, test=args.test_images),
        )
    )
    plan_path = output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text())["settings"] != settings:
        raise ValueError("Resume requires identical selection settings; use a new directory")
    state = dict(
        stage="planning",
        pid=os.getpid(),
        started=time.time(),
        done=0,
        settings=settings,
        workers=args.workers,
    )
    write_json(output / "status.json", state)
    try:
        if reuse_root is not None:
            wait_for_reuse_source(reuse_root, output, state, getattr(args, "wait_for_reuse", False))
            state.update(stage="planning", updated=time.time())
            write_json(output / "status.json", state)
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
        else:
            catalog_path = output / "official-archive-list.txt"
            if not catalog_path.exists():
                reused_catalog = reuse_root / catalog_path.name if reuse_root else None
                atomic_write(
                    catalog_path,
                    reused_catalog.read_bytes()
                    if reused_catalog is not None and reused_catalog.exists()
                    else fetch_catalog(),
                )
            archives = rgb_archives(catalog_path.read_text())
            if not archives:
                raise ValueError("No official RGB archive entries")
            index = []
            for i, archive in enumerate(archives):
                if reuse_root:
                    environment, difficulty, _ = PurePosixPath(archive).parts
                    cache_name = f"{environment}-{difficulty}.json"
                    source_cache = reuse_root / "indices" / cache_name
                    target_cache = output / "indices" / cache_name
                    if source_cache.exists() and not target_cache.exists():
                        atomic_write(target_cache, source_cache.read_bytes())
                index += archive_index(archive, output)
                state.update(
                    indexed_archives=i + 1, archives_total=len(archives), updated=time.time()
                )
                write_json(output / "status.json", state)
            tasks = (
                select_all(index)
                if all_frames
                else select(index, settings["targets"], args.seed, args.stride)
            )
            plan = dict(
                settings=settings,
                tasks=tasks,
                archive_catalog_sha256=hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
                selected_counts=dict(Counter(t["split"] for t in tasks)),
                environment_splits={t["environment"]: t["split"] for t in tasks},
            )
            write_json(plan_path, plan)
        state.update(stage="downloading", total=len(plan["tasks"]))
        write_json(output / "status.json", state)
        records = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for start in range(0, len(plan["tasks"]), 128):
                for row in pool.map(
                    lambda task: fetch(task, output, reuse_root), plan["tasks"][start : start + 128]
                ):
                    records.append(row)
                state.update(
                    done=len(records), updated=time.time(), last_image=records[-1]["image"]
                )
                write_json(output / "status.json", state)
        kept, rejected = deduplicate(records)
        counts = {}
        for split in ("train", "val", "test"):
            rows = [r for r in kept if r["split"] == split]
            if not rows:
                raise ValueError(f"No surviving images in {split}")
            counts[split] = len(rows)
            atomic_write(output / f"{split}.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
        write_json(output / "excluded.json", rejected)
        write_json(
            output / "dataset_info.json",
            dict(
                stage="ready",
                counts=counts,
                settings=settings,
                selection_mode="all_left_rgb_frames" if all_frames else "bounded_subset",
                reuse_source=str(reuse_root) if reuse_root is not None else None,
                selected_counts=plan["selected_counts"],
                exact_duplicates=len(rejected),
                environment_splits=plan["environment_splits"],
                supervision="RGB only; frozen MoGe-v2 pseudo targets, no native depth loaded.",
                split_policy="Whole environment families; office/office2, day/night and season variants grouped. "
                "Internal split, not official test; shared synthetic assets may remain across worlds.",
                verification="All selected files: ZIP CRC, full RGB decode, encoded and decoded SHA256. "
                "Held-out wins exact duplicates. Cross-source audit required before mixing.",
            ),
        )
        state.update(stage="complete", counts=counts, finished=time.time())
    except BaseException as error:
        state.update(stage="failed", error=f"{type(error).__name__}: {error}", updated=time.time())
        raise
    finally:
        write_json(output / "status.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--all-frames", action="store_true", help="All left RGB; no quotas or temporal stride"
    )
    parser.add_argument(
        "--reuse-root", type=Path, help="Completed previous dataset; verified files are reused"
    )
    parser.add_argument(
        "--wait-for-reuse", action="store_true", help="Wait for reuse source to finish; no overlap"
    )
    parser.add_argument("--train-images", type=int, default=50000)
    parser.add_argument("--val-images", type=int, default=512)
    parser.add_argument("--test-images", type=int, default=512)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    run(parser.parse_args())
