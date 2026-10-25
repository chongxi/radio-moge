"""Read back staged RGB against pinned manifests; never copy or publish data."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_json(path, value):
    temporary = path.with_name(path.name + f".new-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def relative_path(value):
    path = PurePosixPath(value)
    require(
        not path.is_absolute()
        and value == path.as_posix()
        and bool(path.parts)
        and all(part not in (".", "..") for part in path.parts),
        f"Invalid relative path: {value}",
    )
    return path


class Reader:
    def __init__(self, mib_per_second):
        self.rate = mib_per_second * 1024**2
        self.bytes = 0
        self.started = time.monotonic()

    def digest(self, path):
        sha = hashlib.sha256()
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            require(stat.S_ISREG(before.st_mode), f"Not a regular file: {path}")
            while block := handle.read(1024**2):
                sha.update(block)
                self.bytes += len(block)
                if self.rate:
                    delay = self.bytes / self.rate - (time.monotonic() - self.started)
                    if delay > 0:
                        time.sleep(delay)
            require(identity(before) == identity(os.fstat(handle.fileno())), f"Changed: {path}")
        require(identity(before) == identity(path.lstat()), f"Replaced: {path}")
        return sha.hexdigest(), before


def rgb_inventory(root):
    require(root.is_dir() and not root.is_symlink(), "RGB root missing or symlinked")
    paths = set()
    for folder, directories, files in os.walk(root, followlinks=False):
        for name in directories:
            require(not (Path(folder) / name).is_symlink(), "Symlink directory in RGB tree")
        for name in files:
            path = Path(folder) / name
            require(stat.S_ISREG(path.lstat().st_mode), "Nonregular file in RGB tree")
            paths.add(path.relative_to(root.parent).as_posix())
    return paths


def declared_exclusions(root, metadata):
    """Preparation preserves duplicate RGB files outside the retained manifests."""
    excluded = set()
    if "excluded.json" not in metadata:
        return excluded
    for row in json.loads((root / "excluded.json").read_text()):
        path = relative_path(row["image"])
        require(path.parts[0] == "rgb", "Excluded image is outside the RGB tree")
        require(row.get("reason") == "exact_duplicate", "Unknown exclusion reason")
        require(row.get("split") in {"train", "val", "test"}, "Unknown exclusion split")
        require(row["image"] not in excluded, "Duplicate exclusion path")
        excluded.add(row["image"])
    if "dataset_info.json" in metadata:
        info = json.loads((root / "dataset_info.json").read_text())
        require(
            len(excluded) == info.get("excluded_exact_duplicates", 0),
            "Excluded image count mismatch",
        )
    return excluded


def verify(root, pins, report, mib_per_second=20):
    require(root.is_dir() and not root.is_symlink(), "Dataset root missing or symlinked")
    reader = Reader(mib_per_second)
    metadata = {}
    identities = {}
    report(stage="checking_metadata")
    for name, expected in pins["metadata_files"].items():
        relative_path(name)
        require(Path(name).name == name, "Metadata must be at dataset root")
        sha, info = reader.digest(root / name)
        require(
            sha == expected["sha256"] and info.st_size == expected["bytes"],
            f"Metadata SHA256/size mismatch: {name}",
        )
        metadata[name] = {"sha256": sha, "bytes": info.st_size}
        identities[name] = identity(info)
    excluded = declared_exclusions(root, metadata)
    report(stage="checking_rgb_inventory", metadata=metadata)
    remaining = rgb_inventory(root / "rgb")
    initial_inventory = remaining.copy()
    require(excluded <= remaining, "Declared excluded RGB files are missing")
    inventory_count = len(remaining)
    counts = {}
    verified = rgb_bytes = 0
    last_report = time.monotonic()
    report(stage="checking_rgb_hashes", inventory_count=inventory_count, verified_images=0)
    for name in sorted(n for n in metadata if n.endswith(".jsonl")):
        split = name.removesuffix(".jsonl")
        counts[split] = 0
        with (root / name).open() as manifest:
            for line in manifest:
                row = json.loads(line)
                relative = relative_path(row["image"])
                require(relative.parts[0] == "rgb", "Image is outside the dataset RGB tree")
                require(row["image"] not in excluded, "Retained image is also excluded")
                require(row.get("sha256_kind") == "encoded_file_bytes", "Missing encoded SHA256")
                require(row["image"] in remaining, f"Missing or duplicate image: {row['image']}")
                sha, info = reader.digest(root / relative)
                require(sha == row["sha256"], f"RGB SHA256 mismatch: {row['image']}")
                remaining.remove(row["image"])
                verified += 1
                rgb_bytes += info.st_size
                counts[split] += 1
                if time.monotonic() - last_report >= 30:
                    report(
                        stage="checking_rgb_hashes",
                        verified_images=verified,
                        verified_rgb_bytes=rgb_bytes,
                        counts=counts,
                        current_manifest=name,
                        last_image=row["image"],
                    )
                    last_report = time.monotonic()
        require(identity((root / name).lstat()) == identities[name], f"Manifest changed: {name}")
    require(remaining == excluded, f"Unreferenced RGB files: {len(remaining - excluded)}")
    for split, count in pins["counts"].items():
        require(counts.get(split) == count, f"Split count mismatch: {split}")
    for name, expected in identities.items():
        require(identity((root / name).lstat()) == expected, f"Metadata changed: {name}")
    report(
        stage="checking_final_inventory",
        verified_images=verified,
        verified_rgb_bytes=rgb_bytes,
        counts=counts,
    )
    require(rgb_inventory(root / "rgb") == initial_inventory, "RGB inventory changed")
    return {
        "stage": "passed",
        "published": False,
        "metadata": metadata,
        "counts": counts,
        "verified_images": verified,
        "verified_rgb_bytes": rgb_bytes,
        "inventory_count": inventory_count,
        "excluded_images": len(excluded),
        "excluded_bytes_verified": False,
        "verification_scope": "Retained RGB bytes; pinned exclusions account for other RGB paths",
        "failures": 0,
        "checks": [
            "pinned metadata SHA256 and size",
            "every retained encoded RGB SHA256",
            "relative paths within RGB tree",
            "no symlinks or special files",
            "unique retained paths and exact retained-plus-excluded RGB inventory",
            "pinned split counts",
            "stable file identity while reading",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--mib-per-second", type=float, default=20)
    args = parser.parse_args()
    require(args.mib_per_second > 0, "Read rate must be positive")
    plan_bytes = args.plan.read_bytes()
    require(hashlib.sha256(plan_bytes).hexdigest() == args.plan_sha256, "Plan SHA256 mismatch")
    plan = json.loads(plan_bytes)
    require(args.dataset in plan["datasets"], "Dataset not in backup plan")
    relative_path(args.dataset)
    require(Path(args.dataset).name == args.dataset, "Invalid dataset name")
    lock = args.report.with_suffix(".lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not args.report.exists(), "Verification report already exists; inspect it first")
    root = Path(plan["staging"]) / args.dataset
    state = {
        "pid": os.getpid(),
        "started_at": time.time(),
        "dataset": args.dataset,
        "root": str(root),
        "plan_sha256": args.plan_sha256,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mib_per_second": args.mib_per_second,
        "published": False,
    }

    def report(**changes):
        state.update(changes, updated_at=time.time())
        atomic_json(args.status, state)

    def copied():
        controller = json.loads(
            (args.plan.parent / "prepared-dataset-backup-status.json").read_text()
        )
        require(controller["plan_sha256"] == args.plan_sha256, "Controller plan mismatch")
        require(args.dataset in controller["copied_pending_verification"], "Copy is incomplete")
        require(controller.get("dataset") != args.dataset, "Dataset is still being copied")

    try:
        copied()
        result = verify(
            root, plan["pinned_source_inventory"][args.dataset], report, args.mib_per_second
        )
        copied()
        report(**result)
        atomic_json(args.report, state)
    except Exception as error:
        report(stage="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
