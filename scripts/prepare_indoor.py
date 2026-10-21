"""Audit downloaded indoor RGB and publish upright, scene-disjoint training manifests."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from scipy.spatial.transform import Rotation

from indoor_download import atomic_write, write_json


def quarter_turn(rotation_cw):
    turns = rotation_cw // 90
    base = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return np.linalg.matrix_power(base, turns)


def pose_orientation(row):
    # ARKitScenes' released trajectory is world-to-camera; its scene frame is Z-up.
    # Source: official threedod/benchmark_scripts/utils/tenFpsDataLoader.py.
    vector = np.asarray(row["pose_axis_angle_translation"][1:4], dtype=float)
    up = Rotation.from_rotvec(vector).as_matrix()[:, 2]
    projection = float(np.linalg.norm(up[:2]))
    dominance = abs(abs(up[0]) - abs(up[1])) / max(projection, 1e-8)
    rotation = max((0, 90, 180, 270), key=lambda q: -(quarter_turn(q) @ up)[1])
    return rotation, projection, float(dominance)


def orient_record(row, rotation, source):
    row = dict(row)
    row.update(image_rotation_cw=rotation, orientation_source=source)
    width, height, fx, fy, cx, cy = row["camera_width_height_fx_fy_cx_cy"]
    camera = [width, height, fx, fy, cx, cy]
    for _ in range(rotation // 90):
        width, height, fx, fy, cx, cy = height, width, fy, fx, height - 1 - cy, cx
    row["native_camera_width_height_fx_fy_cx_cy"] = camera
    row["camera_width_height_fx_fy_cx_cy"] = [width, height, fx, fy, cx, cy]
    row["native_width"], row["native_height"] = row["width"], row["height"]
    row["width"], row["height"] = int(width), int(height)
    pose = row["pose_axis_angle_translation"]
    row["native_pose_axis_angle_translation"] = pose
    matrix = quarter_turn(rotation)
    orientation = matrix @ Rotation.from_rotvec(pose[1:4]).as_matrix()
    row["pose_axis_angle_translation"] = [
        pose[0],
        *Rotation.from_matrix(orientation).as_rotvec().tolist(),
        *(matrix @ pose[4:7]).tolist(),
    ]
    return row


def balanced_subset(records, per_dataset, seed):
    by_dataset = defaultdict(lambda: defaultdict(list))
    for row in records:
        by_dataset[row["dataset"]][row["group"]].append(row)
    selected = []
    for dataset, groups in sorted(by_dataset.items()):
        rng = random.Random(f"{seed}:{dataset}")
        keys = sorted(groups)
        rng.shuffle(keys)
        for key in keys:
            rng.shuffle(groups[key])
        count = 0
        while count < per_dataset:
            advanced = False
            for key in keys:
                if groups[key] and count < per_dataset:
                    selected.append(groups[key].pop())
                    count += 1
                    advanced = True
            if not advanced:
                break
    random.Random(seed).shuffle(selected)
    return selected


def prepare(source, output, validation_per_dataset=256, seed=42, pilot=False):
    status = json.loads((source / "status.json").read_text())
    if status["stage"] != "complete" or status.get("failures"):
        raise ValueError("Download must finish without unresolved failures before preparation")
    if (output / "dataset_info.json").exists():
        raise FileExistsError("Prepared dataset already exists; inspect it before reusing")
    raw = {}
    for split in ("train", "val"):
        path = source / f"{split}.jsonl"
        raw[split] = (
            [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        )
    grouped = defaultdict(list)
    for rows in raw.values():
        for row in rows:
            if row["dataset"] == "arkitscenes":
                grouped[row["video_id"]].append(row)
    consensus = {}
    for video, rows in grouped.items():
        votes = Counter()
        for row in rows:
            rotation, projection, dominance = pose_orientation(row)
            if projection >= 0.35 and dominance >= 0.15:
                votes[rotation] += 1
        if votes:
            best, count = votes.most_common(1)[0]
            if count / sum(votes.values()) >= 0.7:
                consensus[video] = best
    output.mkdir(parents=True, exist_ok=True)
    prepared, seen, groups = defaultdict(list), {}, defaultdict(set)
    counts, excluded, rotations, disagreements = Counter(), Counter(), Counter(), Counter()
    total_bytes = 0
    labels = {"Up": 0, "Left": 90, "Down": 180, "Right": 270}
    source_digests = {}
    for split, rows in raw.items():
        manifest = source / f"{split}.jsonl"
        if manifest.exists():
            source_digests[split] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        for index, row in enumerate(rows):
            image = source / row["image"]
            data = image.read_bytes()
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError(f"Image content changed: {image}")
            total_bytes += len(data)
            if row["sha256"] in seen:
                if seen[row["sha256"]] != split:
                    raise ValueError("Cross-split duplicate image")
                excluded["duplicate"] += 1
                continue
            seen[row["sha256"]] = split
            if row["dataset"] == "arkitscenes":
                rotation, projection, dominance = pose_orientation(row)
                method = "world_Z_up_pose"
                if projection < 0.35 or dominance < 0.15:
                    if row["video_id"] not in consensus:
                        excluded["ambiguous_orientation"] += 1
                        continue
                    rotation, method = consensus[row["video_id"]], "capture_pose_consensus"
                if rotation != labels.get(row["sky_direction"]):
                    disagreements[split] += 1
                row = orient_record(row, rotation, method)
                rotations[str(rotation)] += 1
            else:
                row = dict(row, image_rotation_cw=0, orientation_source="official_render")
            row["image"] = os.path.relpath(image, output)
            prepared[split].append(row)
            groups[split].add(row["group"])
            counts[f"{row['dataset']}/{split}"] += 1
            if index % 10000 == 0:
                write_json(
                    output / "progress.json",
                    dict(
                        stage="verifying",
                        split=split,
                        row=index,
                        counts=dict(counts),
                        bytes_verified=total_bytes,
                    ),
                )
    if groups["train"] & groups["val"]:
        raise ValueError("Cross-split physical scene overlap")
    if not pilot:
        if counts["arkitscenes/train"] < 180000 or counts["hypersim/train"] < 55000:
            raise ValueError(
                f"Unexpected training shortfall requires investigation: {dict(counts)}"
            )
        if any(counts[f"{ds}/val"] < validation_per_dataset for ds in ("arkitscenes", "hypersim")):
            raise ValueError("Insufficient independent validation images")
    selected = balanced_subset(prepared["val"], validation_per_dataset, seed)
    for name, rows in [
        ("train", prepared["train"]),
        ("val-full", prepared["val"]),
        ("val", selected),
    ]:
        atomic_write(output / f"{name}.jsonl", "".join(json.dumps(row) + "\n" for row in rows))
    report = dict(
        stage="ready",
        generated_at=time.time(),
        source=str(source),
        counts=dict(counts),
        groups={k: len(v) for k, v in groups.items()},
        rotation_counts=dict(rotations),
        metadata_orientation_disagreements=dict(disagreements),
        excluded=dict(excluded),
        validation_selected=len(selected),
        source_manifest_sha256=source_digests,
        image_bytes_verified=total_bytes,
        pilot=pilot,
        rgb_policy="Lossless quarter-turn at load time; original RGB is preserved.",
        validation_policy="Balanced datasets, round-robin scenes; full validation remains separate.",
        orientation_policy="World-Z-up pose; low projection uses >=70% capture consensus; otherwise exclude.",
    )
    write_json(output / "dataset_info.json", report)
    write_json(output / "progress.json", dict(stage="complete", counts=dict(counts)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-per-dataset", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.source.resolve(),
                args.output.resolve(),
                args.validation_per_dataset,
                args.seed,
                args.pilot,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
