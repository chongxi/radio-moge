"""Inventory all eligible ARKitScenes Training captures without frame quotas.

This is an offline capture inventory. It does not transfer images or claim that
every RGB archive is available. The next stage must index ZIP members and verify
RGB, poses, and intrinsics; native Validation and existing held-out groups stay
excluded from training.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


BASE = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def manifest_arkit(path, heldout=False):
    counts, groups = Counter(), {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("dataset") != "arkitscenes":
                continue
            if (heldout and row["split"] not in {"val", "test"}) or (
                not heldout and row["split"] != "train"
            ):
                raise ValueError("Manifest split does not match its intended role")
            video, group = str(row["video_id"]), row["group"]
            if video in groups and groups[video] != group:
                raise ValueError("A capture appears in multiple scene groups")
            groups[video] = group
            counts[video] += 1
    return counts, groups


def build_inventory(metadata, existing_train, heldout_manifests):
    with metadata.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(r["fold"] not in {"Training", "Validation"} for r in rows):
        raise ValueError("Unexpected metadata fold")
    if len({r["video_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate capture IDs in metadata")
    def known(visit):
        return visit.lower() not in {"", "na", "nan"}
    reserved = {
        "arkitscenes:visit:" + r["visit_id"]
        for r in rows
        if r["fold"] == "Validation" and known(r["visit_id"])
    }
    heldout_videos = set()
    for path in heldout_manifests:
        _, groups = manifest_arkit(path, heldout=True)
        reserved.update(groups.values())
        heldout_videos.update(groups)
    existing_counts, existing_groups = manifest_arkit(existing_train)
    captures, excluded = [], []
    for row in sorted(rows, key=lambda r: r["video_id"]):
        if row["fold"] != "Training":
            continue
        video, visit = row["video_id"], row["visit_id"]
        group = "arkitscenes:visit:" + visit
        reason = (
            "unknown_visit"
            if not known(visit)
            else "heldout_visit"
            if group in reserved
            else "heldout_capture"
            if video in heldout_videos
            else None
        )
        if reason:
            excluded.append(dict(video_id=video, visit_id=visit, reason=reason))
            continue
        if not video.isdecimal() or not visit.isdecimal():
            raise ValueError("Unexpected capture/visit identifier")
        if video in existing_groups and existing_groups[video] != group:
            raise ValueError("Existing training group disagrees with official metadata")
        prefix = f"{BASE}/Training/{video}"
        captures.append(
            dict(
                video_id=video,
                visit_id=visit,
                group=group,
                split="train",
                sky_direction=row.get("sky_direction"),
                is_in_upsampling=row.get("is_in_upsampling") == "True",
                is_in_threedod=row.get("is_in_threedod") == "True",
                rgb_url=prefix + "/vga_wide.zip",
                intrinsics_url=prefix + "/vga_wide_intrinsics.zip",
                trajectory_url=prefix + "/lowres_wide.traj",
                existing_training_images=existing_counts[video],
            )
        )
    eligible_ids = {r["video_id"] for r in captures}
    if set(existing_counts) - eligible_ids:
        raise ValueError("Existing training contains a capture excluded by the full-data split")
    return dict(
        generated_at=time.time(),
        stage="capture_inventory_ready",
        training_ready=False,
        metadata_sha256=digest(metadata),
        existing_train_manifest_sha256=digest(existing_train),
        heldout_manifest_sha256={str(p): digest(p) for p in heldout_manifests},
        official_captures_by_fold=dict(Counter(r["fold"] for r in rows)),
        eligible_training_captures=len(captures),
        eligible_training_visits=len({r["visit_id"] for r in captures}),
        eligible_upsampling_captures=sum(r["is_in_upsampling"] for r in captures),
        existing_training_captures=len(existing_counts),
        existing_training_images=sum(existing_counts.values()),
        unseen_training_captures=len(eligible_ids - set(existing_counts)),
        exclusion_counts=dict(Counter(r["reason"] for r in excluded)),
        excluded=excluded,
        captures=captures,
        policy={
            "global_frame_cap": None,
            "per_visit_frame_cap": None,
            "frame_stride": None,
            "native_validation_preserved": True,
            "existing_heldout_bytes_unchanged": True,
            "next_stage": "Index every eligible VGA RGB ZIP; count frames and bytes before bulk transfer. Reuse existing RGB only after CRC/hash validation. No request or member-count verification performed by this offline inventory.",
            "pose_policy": "Retain raw RGB inventory, explicitly record missing trajectories/intrinsics. Orientation-dependent training eligibility is determined during preparation.",
            "count_caveat": "449K is not used as a quota or asserted raw frame count.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--existing-train", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_inventory(args.metadata, args.existing_train, args.heldout)
    atomic_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"captures", "excluded"}}))


if __name__ == "__main__":
    main()
