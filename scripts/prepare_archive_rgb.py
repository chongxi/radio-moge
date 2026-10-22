"""Extract/audit the downloaded Structured3D and IRS RGB without native-depth losses.

One process, one archive at a time; no network or GPU operations. Resume uses
per-archive receipts, never a partly written final manifest. Inputs remain intact.
"""

import argparse
from collections import Counter, defaultdict
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import tempfile
import time
import zipfile

from PIL import Image


MAX_RGB_BYTES = 64 * 1024**2
IRS_ARCHIVES = (
    "Store.tar.gz",
    "Home-1.tar.gz",
    "Home-2.tar.gz",
    "Office-1.tar.gz",
    "Office-2.tar.gz",
    "Restaurant.tar.gz",
)
STRUCTURED_ARCHIVES = tuple(
    f"Structured3D_perspective_{kind}_{i:02d}.zip" for kind in ("full", "empty") for i in range(18)
) + ("Structured3D_annotation_3d.zip", "Structured3D_bbox.zip")


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def safe_member(name):
    value = PurePosixPath(name)
    if value.is_absolute() or ".." in value.parts or "\\" in name or "\x00" in name:
        raise ValueError("Unsafe archive member path")
    return value


def structured_row(name, archive_name):
    parts = safe_member(name).parts
    if not parts or not re.fullmatch(r"rgb.*\.(png|jpg|jpeg)", parts[-1], re.I):
        return None
    match = re.search(
        r"(?:^|/)scene_(\d{5})/2D_rendering/([^/]+)/perspective/(full|empty)/([^/]+)/"
        r"(rgb(?:_rawlight|_coldlight|_warmlight)?\.(?:png|jpg|jpeg))$",
        str(PurePosixPath(*parts)),
    )
    if not match:
        raise ValueError(f"Unexpected RGB member layout: {name}")
    scene, room, configuration, view, rgb_name = match.groups()
    if f"perspective_{configuration}_" not in archive_name or int(scene) >= 3500:
        raise ValueError("Unexpected Structured3D configuration or scene range")
    split = "train" if int(scene) < 3000 else "val" if int(scene) < 3250 else "test"
    return dict(
        dataset="structured3d",
        split=split,
        group=f"structured3d:scene:{scene}",
        scene=scene,
        room=room,
        configuration=configuration,
        view=view,
        image=f"rgb/scene_{scene}/2D_rendering/{room}/perspective/{configuration}/{view}/{rgb_name}",
    )


def irs_member(name):
    parts = safe_member(name).parts
    if not parts or not re.fullmatch(r"[lr]_\d+\.png", parts[-1]):
        return None
    matches = [
        i
        for i, p in enumerate(parts)
        if p in {"Home", "Office", "Restaurant", "Store"} and len(parts) - i == 3
    ]
    if len(matches) != 1:
        raise ValueError(f"Unexpected IRS RGB member layout: {name}")
    return str(PurePosixPath(*parts[matches[0] :]))


def irs_inventory(metadata, policy_path):
    metadata, policy_path = Path(metadata), Path(policy_path)
    policy = json.loads(policy_path.read_text())
    groups, splits = policy["groups"], policy["family_splits"]
    if set(groups.values()) != set(splits) or set(splits.values()) != {"train", "val", "test"}:
        raise ValueError("Invalid IRS family split policy")
    rows = {}
    observed_groups = set()
    inputs = {"family_policy": digest(policy_path)}
    for original in ("train", "test"):
        path = metadata / f"irs_{original}.list"
        inputs[path.name] = digest(path)
        for line in path.read_text().splitlines():
            fields = line.split()
            if len(fields) != 4:
                raise ValueError("IRS official rows must contain RGB pair/disparity/normal paths")
            left, right, disparity, normal = fields
            canonical = [irs_member(x) for x in (left, right)]
            if canonical != [left, right] or left.replace("/l_", "/r_") != right:
                raise ValueError("IRS official stereo pair mismatch")
            group = str(PurePosixPath(left).parent)
            if group not in groups:
                raise ValueError(f"Unreviewed IRS scene group: {group}")
            if any(str(safe_member(x).parent) != group for x in (disparity, normal)):
                raise ValueError("IRS label/RGB scene mismatch")
            observed_groups.add(group)
            family = groups[group]
            for eye, member in zip(("left", "right"), canonical):
                if member in rows:
                    raise ValueError("Duplicate/overlapping official IRS RGB reference")
                rows[member] = dict(
                    dataset="irs",
                    split=splits[family] if original == "train" else "official-test",
                    group="irs:family:" + family,
                    scene_directory=group,
                    family=family,
                    eye=eye,
                    frame=PurePosixPath(member).stem[2:],
                    official_split=original,
                    image="rgb/" + member,
                )
    if observed_groups != set(groups):
        raise ValueError("IRS scene policy and official directory inventory differ")
    return rows, dict(
        inputs=inputs,
        directory_groups=len(groups),
        families=len(splits),
        family_splits=splits,
        counts=dict(Counter(r["split"] for r in rows.values())),
        policy_note=policy.get("note"),
    )


def write_rgb(data, row, output, archive, member):
    if not 0 < len(data) <= MAX_RGB_BYTES:
        raise ValueError("Invalid RGB byte count")
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        native_mode = image.mode
        rgb = image.convert("RGB")
        width, height = rgb.size
        if min(width, height) < 16:
            raise ValueError("RGB image too small")
        pixel_sha = hashlib.sha256(f"{width}x{height}:RGB:".encode() + rgb.tobytes()).hexdigest()
    result = dict(
        row,
        sha256=hashlib.sha256(data).hexdigest(),
        sha256_kind="encoded_file_bytes",
        decoded_rgb_sha256=pixel_sha,
        width=width,
        height=height,
        native_mode=native_mode,
        bytes=len(data),
        source_archive=archive,
        source_member=member,
    )
    target = output / row["image"]
    if target.exists():
        if target.is_symlink() or digest(target) != result["sha256"]:
            raise ValueError(f"Existing extracted RGB changed: {row['image']}")
    else:
        atomic_bytes(target, data)
    return result


def extract_zip(path, receipt, output, progress=lambda **kw: None):
    if path.stat().st_size != receipt["bytes"] or digest(path) != receipt["sha256"]:
        raise ValueError("ZIP archive differs from verified download receipt")
    rows = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            safe_member(member.filename)
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Archive symbolic link is not permitted")
            row = structured_row(member.filename, path.name) if not member.is_dir() else None
            if row is None:
                continue
            if member.file_size > MAX_RGB_BYTES:
                raise ValueError("Oversized RGB member")
            # ZipFile.read checks the selected member's CRC at EOF.
            data = archive.read(member)
            rows.append(write_rgb(data, row, output, path.name, member.filename))
            if len(rows) % 256 == 0:
                progress(archive=path.name, rgb_done=len(rows))
    return rows


class HashReader:
    def __init__(self, file):
        self.file, self.sha = file, hashlib.sha256()

    def read(self, size=-1):
        data = self.file.read(size)
        self.sha.update(data)
        return data


def extract_tar(path, receipt, output, official, progress=lambda **kw: None):
    if path.stat().st_size != receipt["bytes"]:
        raise ValueError("IRS archive size differs from verified receipt")
    rows, unexpected = [], []
    with path.open("rb") as raw:
        reader = HashReader(raw)
        with gzip.GzipFile(fileobj=reader, mode="rb") as gz:
            with tarfile.open(fileobj=gz, mode="r|", ignore_zeros=True) as archive:
                while (member := archive.next()) is not None:
                    safe_member(member.name)
                    if not (member.isfile() or member.isdir()):
                        raise ValueError("Archive links/special files are not permitted")
                    canonical = irs_member(member.name) if member.isfile() else None
                    if canonical is not None:
                        if canonical not in official:
                            unexpected.append(canonical)
                        else:
                            if member.size > MAX_RGB_BYTES:
                                raise ValueError("Oversized RGB member")
                            with archive.extractfile(member) as file:
                                data = file.read(MAX_RGB_BYTES + 1)
                            if len(data) != member.size:
                                raise ValueError("Truncated IRS member")
                            rows.append(
                                write_rgb(data, official[canonical], output, path.name, member.name)
                            )
                            if len(rows) % 256 == 0:
                                progress(archive=path.name, rgb_done=len(rows))
                    # We only use next()/extractfile(TarInfo), never random lookup.
                    # Avoid caching hundreds of thousands of depth-label headers.
                    archive.members.clear()
            while gz.read(1024 * 1024):
                pass  # Verify the complete gzip trailer/CRC, beyond the tar EOF.
        while reader.read(1024 * 1024):
            pass
        if reader.sha.hexdigest() != receipt["sha256"]:
            raise ValueError("IRS compressed SHA256 differs from download receipt")
    if unexpected:
        atomic_json(output / (path.name + "-unlisted-rgb.json"), unexpected)
        raise ValueError(
            "IRS archive contains unlisted RGB; inspect inventory before declaring full scope"
        )
    return rows


def finalize(dataset, rows, output, provenance):
    # Multiple release archives may carry the same member. Keep one only if both
    # encoded/pixel hashes agree; preserve the duplicate inventory for auditing.
    unique, repeated_paths = {}, []
    for row in rows:
        if row["dataset"] != dataset or row["split"] not in {
            "train",
            "val",
            "test",
            "official-test",
        }:
            raise ValueError("Unexpected source/split during finalization")
        if row["image"] in unique:
            previous = unique[row["image"]]
            if any(
                row[k] != previous[k] for k in ("sha256", "decoded_rgb_sha256", "split", "group")
            ):
                raise ValueError("Conflicting RGB member across source archives")
            repeated_paths.append(dict(image=row["image"], archive=row["source_archive"]))
        else:
            unique[row["image"]] = row
    pools = defaultdict(list)
    for row in unique.values():
        pools[row["split"]].append(row)
    official = sorted(pools.pop("official-test", []), key=lambda r: r["image"])
    # Original official test images remain completely outside optimizer/selection,
    # including byte/pixel-identical copies with different filenames.
    encoded, decoded = (
        {r["sha256"] for r in official},
        {r["decoded_rgb_sha256"] for r in official},
    )
    excluded, kept = [], {}
    for split in ("test", "val", "train"):
        kept[split] = []
        for row in sorted(pools[split], key=lambda r: r["image"]):
            if row["sha256"] in encoded or row["decoded_rgb_sha256"] in decoded:
                excluded.append(dict(image=row["image"], split=split, reason="exact_duplicate"))
                continue
            encoded.add(row["sha256"])
            decoded.add(row["decoded_rgb_sha256"])
            kept[split].append(row)
        if not kept[split]:
            raise ValueError(f"No eligible {dataset}/{split} RGB")
    # Stereo views and illumination/material variants share one conservative group.
    owners = {}
    for split, values in kept.items():
        for row in values:
            if owners.setdefault(row["group"], split) != split:
                raise ValueError("Scene family leaked across primary partitions")
    manifests = dict(kept)
    if dataset == "irs":
        if not official:
            raise ValueError("IRS official test audit is missing")
        manifests["test-official"] = official
    for split, values in manifests.items():
        atomic_bytes(
            output / f"{split}.jsonl", ("".join(json.dumps(r) + "\n" for r in values)).encode()
        )
    atomic_json(output / "excluded.json", excluded)
    atomic_json(output / "repeated-archive-members.json", repeated_paths)
    info = dict(
        stage="ready",
        dataset=dataset,
        scope="all_eligible_rgb",
        pilot=False,
        counts={s: len(values) for s, values in kept.items()},
        raw_extracted_unique_rgb=len(unique),
        official_test_rgb=len(official),
        primary_group_counts={s: len({r["group"] for r in values}) for s, values in kept.items()},
        excluded_exact_duplicates=len(excluded),
        repeated_archive_paths=len(repeated_paths),
        provenance=provenance,
        verification=dict(
            rgb_decode_all=True, encoded_and_decoded_sha256=True, primary_scene_splits_disjoint=True
        ),
        manifest_sha256={s + ".jsonl": digest(output / f"{s}.jsonl") for s in manifests},
        extra_holdout_manifests=["test-official.jsonl"] if official else [],
        supervision="Frozen MoGe-v2 RGB feature targets; native labels remain in archives, unused by loss",
        limits="Family separation does not prove disjoint synthetic assets. IRS official frame-level "
        "test is audit-only and shares scene families; it is not primary validation/test.",
        prepared_at=time.time(),
    )
    atomic_json(output / "dataset_info.json", info)
    return info


def run(args):
    source, output = args.source.resolve(), args.output.resolve()
    if source == output:
        raise ValueError("Use a separate prepared RGB directory")
    state = json.loads((source / "status.json").read_text())
    if state["stage"] not in {"archives_complete", "archives_complete_with_missing_assets"}:
        raise ValueError("All source archives must finish downloading and verifying first")
    # Observe that the downloader finished and released its writer lock. Do not
    # hold it while reading: the next queued dataset uses that same release as
    # its start condition. Per-archive hashes/stats detect input changes below.
    with (source / ".download.lock").open("a") as source_lock:
        fcntl.flock(source_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    missing = state.get("missing", [])
    if args.dataset == "structured3d":
        if missing not in ([], ["Structured3D_perspective_full_09.zip"]):
            raise ValueError("Unexpected missing Structured3D release assets")
        names = [n for n in STRUCTURED_ARCHIVES if n not in missing]
        completed = [r["name"] for r in state["completed"]]
        official, split_plan = None, dict(policy="scene<3000 train;3000-3249 val;3250-3499 test")
    else:
        names, completed = list(IRS_ARCHIVES), state["completed"]
        official, split_plan = irs_inventory(args.metadata, args.family_policy)
    if len(completed) != len(names) or set(completed) != set(names):
        raise ValueError("Source completion does not cover the approved archive inventory")
    receipts = {n: json.loads((source / "receipts" / (n + ".json")).read_text()) for n in names}
    provenance = dict(
        source=str(source),
        dataset=args.dataset,
        archive_sha256={n: r["sha256"] for n, r in receipts.items()},
        missing_archives=missing,
        split_plan=split_plan,
    )
    output.mkdir(parents=True, exist_ok=True)
    output_lock = (output / ".prepare.lock").open("a")
    fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "preparation-plan.json").exists():
        if json.loads((output / "preparation-plan.json").read_text()) != provenance:
            raise ValueError("Source archives or split policy changed; use a new output")
    else:
        if any(p.name != ".prepare.lock" for p in output.iterdir()):
            raise ValueError("Unrecognized output contents; inspect before reuse")
        atomic_json(output / "preparation-plan.json", provenance)
    status = dict(stage="extracting", pid=os.getpid(), started_at=time.time(), completed=[])

    def record(**changes):
        status.update(changes, updated_at=time.time())
        atomic_json(output / "status.json", status)

    record()
    atomic_json(output / "dataset_info.json", dict(stage="preparing", dataset=args.dataset))
    rows = []
    try:
        for name in names:
            path, receipt = source / "archives" / name, receipts[name]
            before = (path.stat().st_size, path.stat().st_mtime_ns)
            if before[0] != receipt["bytes"]:
                raise ValueError("Archive size differs from source receipt")
            done = output / "receipts" / (name + ".json")
            record(archive=name, rgb_done=0)
            if done.exists():
                saved = json.loads(done.read_text())
                if saved["source_sha256"] != receipt["sha256"] or saved["source_stat"] != list(
                    before
                ):
                    raise ValueError("Previously prepared archive changed")
                extracted = saved["records"]
                for row in extracted:
                    if digest(output / row["image"]) != row["sha256"]:
                        raise ValueError("Previously verified RGB changed")
            else:
                extracted = (
                    extract_zip(path, receipt, output, record)
                    if args.dataset == "structured3d"
                    else extract_tar(path, receipt, output, official, record)
                )
                if (path.stat().st_size, path.stat().st_mtime_ns) != before:
                    raise ValueError("Archive changed during preparation")
                atomic_json(
                    done,
                    dict(source_sha256=receipt["sha256"], source_stat=before, records=extracted),
                )
            rows.extend(extracted)
            status["completed"].append(name)
            record(total_extracted_rgb=len(rows))
        if official is not None:
            found = {r["image"][len("rgb/") :] for r in rows}
            if found != set(official):
                atomic_json(output / "missing-official-rgb.json", sorted(set(official) - found))
                raise ValueError("IRS archives do not cover every official RGB reference")
        result = finalize(args.dataset, rows, output, provenance)
        record(stage="ready", counts=result["counts"], finished_at=time.time())
        return result
    except BaseException as error:
        atomic_json(output / "dataset_info.json", dict(stage="failed", dataset=args.dataset))
        record(stage="failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["structured3d", "irs"], required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--family-policy", type=Path)
    args = parser.parse_args()
    if args.dataset == "irs" and not (args.metadata and args.family_policy):
        parser.error("IRS preparation needs --metadata and --family-policy")
    print(json.dumps(run(args)))
