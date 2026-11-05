"""Verify closed custom structural scenes without relabelling their projection."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from render_procthor_empty_pilot import BUILD, publish, sha
from render_structural_room_batch import (
    FRAME_POLICY,
    POLICY,
    box_mesh,
    camera_slots,
    canonical_sha,
    mesh_for,
    scene_spec,
    validate_rgb,
    vec,
)


def bound(ref, parent=None):
    path = Path(ref["path"])
    if parent is not None and not path.resolve().is_relative_to(parent.resolve()):
        raise ValueError("Evidence path escapes its source batch")
    if sha(path) != ref["sha256"]:
        raise ValueError("Evidence hash differs: " + str(path))
    return path


def read_rgb(path):
    with Image.open(path) as image:
        if image.mode != "RGB" or image.size != (640, 480) or image.getexif().get(274, 1) != 1:
            raise ValueError("Unexpected RGB format or orientation")
        return np.array(image)


def check_camera(actual, expected):
    if actual.get("thirdPartyCameraId", 0) != 0:
        raise ValueError("Unexpected camera identity")
    for key in ("position", "rotation"):
        for axis in ("x", "y", "z"):
            delta = actual[key][axis] - expected[key][axis]
            if key == "rotation":
                delta = (delta + 180) % 360 - 180
            if not np.isfinite(delta) or abs(delta) >= 1e-4:
                raise ValueError("Actual camera differs from planned pose")
    if abs(actual.get("fieldOfView", float("nan")) - expected["fieldOfView"]) >= 1e-4:
        raise ValueError("Actual camera FOV differs")
    if not np.isfinite(actual.get("fieldOfView", float("nan"))):
        raise ValueError("Missing actual camera FOV")


def check_assets(ref, batch, spec):
    path = bound(ref, batch / "assets")
    record = json.loads(path.read_text())
    if record["scene_identity"] != spec["identity"] or len(record["objects"]) != len(
        spec["structures"]
    ):
        raise ValueError("Asset scene or object count differs")
    stem = f"{spec['split']}-{spec['scene_ordinal']:06d}"
    for item, saved in zip(spec["structures"], record["objects"]):
        mesh = mesh_for(item)
        if canonical_sha(saved["specification"]) != canonical_sha(item) or saved[
            "mesh_sha256"
        ] != canonical_sha(mesh):
            raise ValueError("Structural asset changed")
        texture = bound(saved["texture"], batch / "assets")
        expected_color = np.broadcast_to(
            np.rint(np.array(item["color"]) * 255).astype(np.uint8), (8, 8, 3)
        )
        with Image.open(texture) as image:
            if image.mode != "RGB" or not np.array_equal(np.array(image), expected_color):
                raise ValueError("Structural material differs")
        points = np.array([list(v.values()) for v in mesh["vertices"]])
        collider = box_mesh(points.max(axis=0) - points.min(axis=0))
        request = dict(
            action="CreateObjectPrefab",
            name=f"{stem}-{item['name']}",
            **mesh,
            texturePath=str(texture),
            visibilityPoints=[vec((0, 0, 0))],
            colliders=[{k: collider[k] for k in ("vertices", "triangles")}],
            physicalProperties=dict(useGravity=False, isKinematic=True),
            annotations=dict(
                objectType="Undefined", primaryProperty="Undefined", secondaryProperties=[]
            ),
        )
        if saved["created_prefab_request_sha256"] != canonical_sha(request):
            raise ValueError("Prefab request differs")


def freeze(selection_path, selection_sha, output):
    if sha(selection_path) != selection_sha:
        raise ValueError("Selection hash differs")
    selection = json.loads(selection_path.read_text())
    if selection["split"] not in ("train", "val") or selection["policy"] != POLICY:
        raise ValueError("Explicit custom-source policy and split required")
    split = selection["split"]
    plan = json.loads(bound(selection["generation_plan"]).read_text())
    if plan["policy"] != POLICY or plan["frame_policy"] != FRAME_POLICY:
        raise ValueError("Generation plan differs")
    planned = {(r["split"], r["ordinal"]): r for r in plan["scenes"]}
    legacy = plan["preserved_validation"]
    seen_scenes, pixels, paths, layouts = set(), set(), set(), set()
    output_rows, scene_proofs, rejection_proofs = [], [], []
    projection_counts = dict(perspective=0, orthographic=0)
    for source in selection["sources"]:
        batch = Path(source["batch"])
        source_manifest = bound(source["manifest"], batch)
        bound(source["termination"], batch)
        old = source["frame_policy"] is None
        if old:
            if (
                split != "val"
                or batch.name != legacy["batch"]
                or source["scene_ordinals"] != legacy["scene_ordinals"]
            ):
                raise ValueError("Legacy scope differs from preserved closed validation prefix")
        elif source["frame_policy"] != FRAME_POLICY:
            raise ValueError("Unknown frame-filter policy")
        # Metadata outside the selected closed range is not promoted or decoded.
        rows_by_scene = {i: {} for i in source["scene_ordinals"]}
        for line in source_manifest.read_text().splitlines(keepends=True):
            row = json.loads(line)
            if row["scene_ordinal"] not in rows_by_scene:
                continue
            target = rows_by_scene[row["scene_ordinal"]]
            if row["slot"] in target:
                raise ValueError("Duplicate scene slot")
            target[row["slot"]] = (row, hashlib.sha256(line.encode()).hexdigest())
        for ordinal, rows in rows_by_scene.items():
            spec = scene_spec(split, ordinal)
            identity = spec["identity"]
            if identity in seen_scenes or spec["layout_sha256"] in layouts:
                raise ValueError("Scene/layout duplicated")
            seen_scenes.add(identity)
            layouts.add(spec["layout_sha256"])
            if not old:
                expected = planned.get((split, ordinal))
                if (
                    expected is None
                    or expected["scene_sha256"] != canonical_sha(spec)
                    or expected["camera_plan_sha256"] != canonical_sha(camera_slots(spec, 20))
                ):
                    raise ValueError("Scene/camera identity absent from frozen generation plan")
            stem = f"{split}-{ordinal:06d}"
            spec_path = batch / "scenes" / (stem + ".json")
            spec_ref = dict(path=str(spec_path), sha256=sha(spec_path))
            if canonical_sha(json.loads(spec_path.read_text())) != canonical_sha(spec):
                raise ValueError("Scene differs from deterministic generator")
            receipt_path = batch / "scene-reports" / (stem + ".json")
            receipt = json.loads(receipt_path.read_text())
            if (
                receipt.get("closed") is not True
                or receipt["scene"] != identity
                or receipt["scene_file"] != spec_ref
                or receipt["layout_sha256"] != spec["layout_sha256"]
                or receipt["windows"] != 0
                or receipt["doors"] != 0
            ):
                raise ValueError("Scene is not a closed bound zero-window room")
            if receipt["camera_policy"] != "18_perspective_2_orthographic_per_20":
                raise ValueError("Scene camera policy changed")
            check_assets(receipt["assets"], batch, spec)
            accepted = {r["slot"]: r for r in receipt["images"]}
            if len(accepted) != len(receipt["images"]) or set(accepted) != set(rows):
                raise ValueError("Receipt and manifest slots differ")
            rejected = set()
            for ref in receipt.get("rejected", []):
                r = json.loads(bound(ref, batch / "rejected").read_text())
                if (
                    r["scene"] != identity
                    or r["slot"] in rejected
                    or r["training_admission"] is not False
                    or r["frame_policy"] != FRAME_POLICY
                ):
                    raise ValueError("Rejected slot identity differs")
                slot = camera_slots(spec, 20)[r["slot"]]
                if canonical_sha(r["requested_camera"]) != canonical_sha(slot["request"]):
                    raise ValueError("Rejected camera replaced")
                image = bound(r["image"], batch / "rejected")
                arr = read_rgb(image)
                if hashlib.sha256(arr.tobytes()).hexdigest() != r["decoded_rgb_sha256"]:
                    raise ValueError("Rejected RGB changed")
                try:
                    validate_rgb(arr)
                except ValueError as error:
                    if str(error) != r["reason"]:
                        raise ValueError("Rejection reason differs") from error
                else:
                    raise ValueError("Recorded rejection is not reproduced")
                rejected.add(r["slot"])
                rejection_proofs.append(ref)
            if set(rows) & rejected or set(rows) | rejected != set(range(20)) or len(rejected) > 2:
                raise ValueError("Slots missing, replaced or excessive rejections")
            if old and rejected:
                raise ValueError("Legacy closed scenes cannot have rejected slots")
            if not old and (
                receipt["frame_policy"] != FRAME_POLICY
                or receipt["attempted_slots"] != list(range(20))
            ):
                raise ValueError("Incomplete attempted camera plan")
            for slot_index in sorted(rows):
                row, line_sha = rows[slot_index]
                slot = camera_slots(spec, 20)[slot_index]
                request = slot["request"]
                projection = "orthographic" if request["orthographic"] else "perspective"
                if (
                    row["dataset"] != "structural_rooms"
                    or row["policy"] != POLICY
                    or row["split"] != split
                    or row["group"] != identity
                    or row["scene"] != spec_ref
                    or row["scene_layout_sha256"] != spec["layout_sha256"]
                    or row["assets"] != receipt["assets"]
                    or row.get("frame_policy") != source["frame_policy"]
                ):
                    raise ValueError("RGB source identity differs")
                if (
                    row["projection"] != projection
                    or row["camera_model"] != ("parallel" if request["orthographic"] else "pinhole")
                    or row["camera_pattern_index"] != slot["camera_pattern_index"]
                    or row["camera_kind"] != slot["kind"]
                    or canonical_sha(row["requested_camera"]) != canonical_sha(request)
                ):
                    raise ValueError("Projection or planned camera changed")
                if (
                    (row["width"], row["height"]) != (640, 480)
                    or row["vertical_fov_degrees"]
                    != (None if request["orthographic"] else request["fieldOfView"])
                    or row["orthographic_half_height_m"] != request.get("orthographicSize")
                ):
                    raise ValueError("Projection parameters differ")
                if (
                    row["wall_distance_m"] != spec["depth"] - request["position"]["z"]
                    or row["intrinsics"] is not None
                ):
                    raise ValueError("Unexpected geometry metadata")
                if (
                    any(
                        row[k] != 0
                        for k in ("window_count", "door_count", "exterior_background_pixels")
                    )
                    or row["agent_visible"] is not False
                    or row["geometry_label"] is not None
                    or row["label_role"] != "feature_only_rgb"
                    or row["training_admission"] is not False
                    or row["renderer_build"] != BUILD
                ):
                    raise ValueError("Feature-only closed-room contract differs")
                check_camera(row["actual_camera"], request)
                image = bound(dict(path=row["image"], sha256=row["sha256"]), batch / "rgb")
                if accepted[slot_index] != dict(
                    image=str(image), sha256=row["sha256"], slot=slot_index
                ):
                    raise ValueError("Receipt RGB differs")
                arr = read_rgb(image)
                std = validate_rgb(arr)
                pixel_sha = hashlib.sha256(arr.tobytes()).hexdigest()
                if (
                    pixel_sha != row["decoded_rgb_sha256"]
                    or pixel_sha in pixels
                    or str(image) in paths
                    or abs(std - row["spatial_rgb_std"]) > 1e-7
                ):
                    raise ValueError("RGB bytes/statistics changed or duplicated")
                pixels.add(pixel_sha)
                paths.add(str(image))
                projection_counts[projection] += 1
                output_rows.append(
                    dict(
                        row,
                        candidate_record_sha256=line_sha,
                        training_admission=split == "train",
                        geometry_admission=False,
                    )
                )
            scene_proofs.append(
                dict(
                    group=identity,
                    layout_sha256=spec["layout_sha256"],
                    receipt=dict(path=str(receipt_path), sha256=sha(receipt_path)),
                    accepted=len(rows),
                    rejected=len(rejected),
                )
            )
    isolation = []
    for other in selection.get("isolate_from", []):
        p = bound(other)
        other_pixels, other_groups, other_layouts = set(), set(), set()
        for line in p.read_text().splitlines():
            row = json.loads(line)
            other_pixels.add(row["decoded_rgb_sha256"])
            other_groups.add(row["group"])
            if row.get("scene_layout_sha256"):
                other_layouts.add(row["scene_layout_sha256"])
        overlaps = dict(
            pixels=len(pixels & other_pixels),
            groups=len(seen_scenes & other_groups),
            layouts=len(layouts & other_layouts),
        )
        if any(overlaps.values()):
            raise ValueError("Training/validation exact overlap")
        isolation.append(dict(manifest=other, overlaps=overlaps, images=len(other_pixels)))
    if not output_rows or len(seen_scenes) != selection["expected_scenes"]:
        raise ValueError("Incomplete selected scene count")
    output.mkdir(exist_ok=False)
    manifest = output / "rgb.jsonl"
    incoming = manifest.with_suffix(".jsonl.incoming")
    incoming.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in output_rows))
    incoming.replace(manifest)
    report = dict(
        stage="frozen_verified_structural_feature_rgb",
        policy=POLICY,
        split=split,
        selection=dict(path=str(selection_path), sha256=selection_sha),
        images=len(output_rows),
        scenes=len(seen_scenes),
        projections=projection_counts,
        rejected_frames=len(rejection_proofs),
        scene_receipts=scene_proofs,
        rejection_proofs=rejection_proofs,
        exact_pixel_duplicates=0,
        isolation=isolation,
        manifest=dict(path=str(manifest), sha256=sha(manifest)),
        every_selected_rgb_decoded_and_hashed=True,
        source_role="custom_parameterized_scene_not_official_procthor_house",
        geometry_labels=False,
        feature_training_eligible=split == "train",
        geometry_admission=False,
        visual_review_scope="Separate sampled review; no all-image certification",
        optimizer_started=False,
        code={
            p.name: sha(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("render_structural_room_batch.py"),
                Path(__file__).with_name("render_structural_room_samples.py"),
                Path(__file__).with_name("render_procthor_empty_pilot.py"),
            )
        },
    )
    publish(output / "report.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("selection", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--selection-sha256", required=True)
    args = p.parse_args()
    r = freeze(args.selection, args.selection_sha256, args.output)
    print(
        json.dumps(
            {
                k: r[k]
                for k in ("stage", "images", "scenes", "projections", "rejected_frames", "manifest")
            }
        )
    )
