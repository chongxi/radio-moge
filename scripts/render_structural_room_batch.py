"""Bounded RGB-only structural-room generation with user-approved perspective and orthographic cameras."""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from render_procthor_empty_pilot import BUILD, create_controller, publish, sha
from render_structural_room_samples import box_mesh, material, mesh_for, room, save_image, vec


POLICY = "structural-room-rgb-multiview-v1"
FRAME_POLICY = "strict-rgb-frame-filter-v2"
PALETTES = (
    (0.84, 0.82, 0.76),
    (0.76, 0.80, 0.83),
    (0.67, 0.68, 0.68),
    (0.83, 0.78, 0.69),
    (0.76, 0.79, 0.73),
    (0.9, 0.89, 0.86),
)


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def scene_spec(split, ordinal):
    if split not in ("train", "val") or not 0 <= ordinal < 100000:
        raise ValueError("Explicit scene split and bounded nonnegative ordinal required")
    identity = f"{POLICY}/{split}/{ordinal:06d}"
    seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little")
    rng = np.random.Generator(np.random.PCG64(seed))
    width, depth, height = (
        float(rng.uniform(8, 12)),
        float(rng.uniform(13, 20)),
        float(rng.uniform(3.8, 5.4)),
    )
    color = np.clip(
        np.array(PALETTES[ordinal % len(PALETTES)]) + rng.uniform(-0.025, 0.025, 3), 0.3, 0.95
    ).tolist()
    object_color = np.clip(np.array(color) * rng.uniform(0.72, 0.96), 0.25, 0.9).tolist()
    wall_material = ("PureWhite", "WallDrywallWhite3")[ordinal % 2]
    house = room(0)
    house["id"] = identity
    floor = [vec(p) for p in ((0, 0, 0), (0, 0, depth), (width, 0, depth), (width, 0, 0))]
    house["rooms"][0]["floorPolygon"] = floor
    house["rooms"][0]["floorMaterial"] = material((0.62, 0.61, 0.59), "LightWoodCounters2")
    walls = []
    for i, (a, b) in enumerate(zip(floor, floor[1:] + floor[:1])):
        mat = material(color, wall_material)
        mat.update(
            tilingDivisorX=float(rng.uniform(0.7, 2.0)), tilingDivisorY=float(rng.uniform(0.7, 2.0))
        )
        walls.append(
            dict(
                id=f"wall|{split}-{ordinal:06d}|{i}",
                roomId="room|0",
                polygon=[copy.deepcopy(a), copy.deepcopy(b), dict(a, y=height), dict(b, y=height)],
                thickness=0.15,
                material=mat,
            )
        )
    house["walls"] = walls
    house["proceduralParameters"]["ceilingMaterial"] = material(color, "WallDrywallWhite3")
    lights = []
    for i in range(2 + ordinal % 2):
        lights.append(
            dict(
                id=f"light_{i}",
                type="point",
                position=vec(
                    (rng.uniform(0.2, 0.8) * width, height - 0.25, rng.uniform(0.4, 0.85) * depth)
                ),
                intensity=float(rng.uniform(0.35, 0.85)),
                range=24.0,
                rgb=dict(
                    zip(
                        ("r", "g", "b"),
                        [1.0, float(rng.uniform(0.89, 1.0)), float(rng.uniform(0.8, 1.0))],
                    )
                ),
                shadow=dict(
                    type="Soft",
                    strength=float(rng.uniform(0.45, 0.8)),
                    normalBias=0.0,
                    bias=0.03,
                    nearPlane=0.2,
                    resolution="FromQualitySettings",
                ),
            )
        )
    house["proceduralParameters"]["lights"] = lights
    pose = dict(
        position=vec((width / 2, 0.95, 1.5)), rotation=vec((0, 0, 0)), horizon=0, standing=True
    )
    house["metadata"].update(agent=pose, agentPoses=dict(default=pose))
    span = float(rng.uniform(0.72, 0.87) * height)
    left, right = width / 2 - span / 2, width / 2 + span / 2
    z = depth - float(rng.uniform(2.5, 4.0))
    square_width = float(rng.uniform(0.45, 0.75))
    radius = float(rng.uniform(0.25, 0.39))
    beam_thickness = float(rng.uniform(0.28, 0.55))
    # Every scene contains both kinds of grounded pillars and connected spanning members.
    pieces = [
        dict(
            name="square_column",
            kind="box",
            size=(square_width, height, square_width),
            center=(left, height / 2, z),
        ),
        dict(
            name="round_column",
            kind="cylinder_y",
            radius=radius,
            length=height,
            center=(right, height / 2, z),
        ),
        dict(
            name="rectangular_beam",
            kind="box",
            size=(span + 0.65, beam_thickness, 0.55),
            center=(width / 2, height * float(rng.uniform(0.77, 0.84)), z),
        ),
        dict(
            name="horizontal_round_member",
            kind="cylinder_x",
            radius=float(rng.uniform(0.11, 0.23)),
            length=span,
            center=(width / 2, height * float(rng.uniform(0.28, 0.53)), z),
        ),
    ]
    layout = ordinal % 4
    if layout:
        rear_x = width / 2 + float(rng.uniform(-0.7, 0.7))
        rear_z = depth - 0.7
        if layout == 1:
            pieces.append(
                dict(
                    name="rear_square_column",
                    kind="box",
                    size=(0.5, height, 0.5),
                    center=(rear_x, height / 2, rear_z),
                )
            )
        else:
            pieces.append(
                dict(
                    name="rear_round_column",
                    kind="cylinder_y",
                    radius=0.3,
                    length=height,
                    center=(rear_x, height / 2, rear_z),
                )
            )
            pieces.append(
                dict(
                    name="depth_beam",
                    kind="box",
                    size=(0.4, 0.35, depth - z - 0.1),
                    center=(right, height * 0.82, (depth + z - 0.1) / 2),
                )
            )
    if layout == 3:
        pieces.append(
            dict(
                name="back_horizontal_beam",
                kind="box",
                size=(span + 0.7, 0.32, 0.4),
                center=(width / 2, height * 0.65, depth - 0.65),
            )
        )
    pieces = [dict(item, color=object_color) for item in pieces]
    result = dict(
        policy=POLICY,
        identity=identity,
        split=split,
        scene_ordinal=ordinal,
        seed=seed,
        source_role="custom_parameterized_scene_not_official_procthor_house",
        width=width,
        depth=depth,
        height=height,
        layout=layout,
        house=house,
        structures=pieces,
        camera_policy="18_perspective_2_orthographic_per_20",
        requested_windows=0,
        requested_doors=0,
        user_approved_projection=True,
        feature_only=True,
        geometry_label=None,
    )
    result["layout_sha256"] = canonical_sha(dict(house=house, structures=pieces))
    return result


def camera_slots(spec, count):
    if count not in (4, 20):
        raise ValueError("Use four preflight views or twenty batch views")
    width, height, depth = spec["width"], spec["height"], spec["depth"]
    offset = (spec["seed"] % 1009) / 1009
    # Front, left/right oblique, high/low and farther/closer cameras. All are
    # ordinary pinhole views except the final two explicitly marked comparisons.
    positions = [
        (0.50, 1.5 / depth, 0.50),
        (0.16, 0.12, 0.50),
        (0.84, 0.12, 0.50),
        (0.14, 0.28, 0.50),
        (0.86, 0.28, 0.50),
        (0.50, 0.33, 0.50),
        (0.22, 0.22, 0.68),
        (0.78, 0.22, 0.30),
        (0.50, 0.12, 0.32),
        (0.50, 0.12, 0.68),
        (0.12, 0.40, 0.50),
        (0.88, 0.40, 0.50),
        (0.25, 0.15, 0.42),
        (0.75, 0.15, 0.58),
        (0.30, 0.30, 0.35),
        (0.70, 0.30, 0.65),
        (0.40, 0.20, 0.60),
        (0.60, 0.20, 0.40),
    ]
    selected = range(20) if count == 20 else (0, 10, 11, 18)
    result = []
    front_z = spec["structures"][0]["center"][2]
    for slot, i in enumerate(selected):
        base = dict(nearClippingPlane=0.05, farClippingPlane=50.0, skyboxColor="#ff00ff")
        if i < 18:
            px, pz, py = positions[i]
            position = np.array((width * px, height * py, depth * pz))
            target = np.array((width / 2, height * 0.5, front_z))
            difference = target - position
            yaw = math.degrees(math.atan2(difference[0], difference[2])) % 360
            horizontal = math.hypot(difference[0], difference[2])
            pitch = -math.degrees(math.atan2(difference[1], horizontal)) % 360
            fov = float(
                np.clip(math.degrees(2 * math.atan(height * 0.60 / horizontal)), 32.0, 58.0)
            )
            request = dict(
                base,
                position=vec(position),
                rotation=vec((pitch, yaw, 0)),
                fieldOfView=fov,
                orthographic=False,
            )
            kind = "perspective_multiview"
        else:
            u = (offset + i * 0.6180339887498949) % 1
            v = (offset + i * 0.4142135623730951) % 1
            x = width / 2 + (u - 0.5) * 0.65
            y = height / 2 + (v - 0.5) * height * 0.06
            maximum = min(y, height - y) - 0.12
            scale = maximum * (0.9 + 0.085 * ((i * 0.7320508075688772 + offset) % 1))
            request = dict(
                base,
                position=vec((x, y, 1.5)),
                rotation=vec((0, 0, 0)),
                fieldOfView=32.0,
                orthographic=True,
                orthographicSize=scale,
            )
            half_width = scale * 4 / 3
            if not (
                0 < x - half_width < x + half_width < width and 0 < y - scale < y + scale < height
            ):
                raise ValueError("Orthographic ray origins extend outside the closed room")
            kind = "orthographic_supplement"
        result.append(dict(index=slot, camera_pattern_index=i, kind=kind, request=request))
    return result


class RejectedRGB(ValueError):
    """A retained diagnostic, never a training RGB or an implicit replacement."""


def validate_rgb(rgb):
    if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
        raise ValueError("Unexpected RGB framebuffer")
    if ((rgb[..., 0] > 240) & (rgb[..., 1] < 15) & (rgb[..., 2] > 240)).any():
        raise RejectedRGB("Exterior/background is visible")
    std = float(rgb.astype(np.float32).std(axis=(0, 1)).mean())
    if std < 0.3:
        raise RejectedRGB("Near-uniform RGB")
    return std


def spawn_structures(controller, spec, output, stem):
    items = []
    for item in spec["structures"]:
        mesh = mesh_for(item)
        name = f"{stem}-{item['name']}"
        texture = output / "assets" / (name + "-material.png")
        array = np.broadcast_to(
            np.rint(np.asarray(item["color"]) * 255).astype(np.uint8), (8, 8, 3)
        ).copy()
        save_image(texture, array)
        points = np.array([list(v.values()) for v in mesh["vertices"]])
        collider = box_mesh(points.max(axis=0) - points.min(axis=0))
        request = dict(
            action="CreateObjectPrefab",
            name=name,
            **mesh,
            texturePath=str(texture),
            visibilityPoints=[vec((0, 0, 0))],
            colliders=[{k: collider[k] for k in ("vertices", "triangles")}],
            physicalProperties=dict(useGravity=False, isKinematic=True),
            annotations=dict(
                objectType="Undefined", primaryProperty="Undefined", secondaryProperties=[]
            ),
        )
        controller.step(**request, raise_for_failure=True)
        event = controller.step(
            action="SpawnAsset",
            assetId=name,
            generatedId=name,
            position=vec(item["center"]),
            rotation=vec((0, 0, 0)),
            raise_for_failure=True,
        )
        items.append(
            dict(
                specification=item,
                mesh_sha256=canonical_sha(mesh),
                texture=dict(path=str(texture), sha256=sha(texture)),
                created_prefab_request_sha256=canonical_sha(request),
                spawn_result=event.metadata.get("actionReturn"),
            )
        )
    path = output / "assets" / (stem + ".json")
    publish(path, dict(objects=items, scene_identity=spec["identity"]))
    return dict(path=str(path), sha256=sha(path))


def run(root, output, split, start, count, views, maximum_seconds):
    if (
        split not in ("train", "val")
        or start < 0
        or not 1 <= count <= 500
        or start + count > 100000
    ):
        raise ValueError("Explicit bounded scene range and split required")
    if views not in (4, 20) or not 60 <= maximum_seconds <= 7200:
        raise ValueError("Use four/twenty views, at most two hours")
    output.mkdir(exist_ok=False)
    for subdir in ("rgb", "scenes", "assets", "scene-reports", "rejected"):
        (output / subdir).mkdir()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
    started, controller = time.monotonic(), None
    totals = dict(
        scenes_closed=0,
        completed=0,
        duplicate_frames=0,
        perspective_frames=0,
        orthographic_frames=0,
        rejected_frames=0,
    )
    seen, summaries, next_scene = set(), [], start
    manifest = output / "rgb-candidates.jsonl"
    publish(output / "status.json", dict(stage="starting_cpu_renderer", **totals))
    try:
        with manifest.open("x") as records:
            for ordinal in range(start, start + count):
                if (
                    output / "STOP_AFTER_SCENE"
                ).exists() or time.monotonic() - started >= maximum_seconds:
                    break
                spec = scene_spec(split, ordinal)
                stem = f"{split}-{ordinal:06d}"
                spec_path = output / "scenes" / (stem + ".json")
                publish(spec_path, spec)
                event = None
                if controller is None:
                    controller = create_controller(root, output, spec["house"], field_of_view=32)
                    event = controller.last_event
                else:
                    event = controller.reset(scene=spec["house"], fieldOfView=32)
                if not event.metadata["lastActionSuccess"]:
                    raise RuntimeError(event.metadata.get("errorMessage", "CreateHouse failed"))
                if controller.initialization_parameters.get("makeAgentsVisible") is not False:
                    raise ValueError("Simulator agent visibility changed")
                assets = spawn_structures(controller, spec, output, stem)
                scene_rows, rejected_rows = [], []
                for slot in camera_slots(spec, views):
                    request = slot["request"]
                    event = controller.step(
                        action="AddThirdPartyCamera"
                        if slot["index"] == 0
                        else "UpdateThirdPartyCamera",
                        **({} if slot["index"] == 0 else dict(thirdPartyCameraId=0)),
                        **request,
                        raise_for_failure=True,
                    )
                    rgb = event.third_party_camera_frames[0]
                    try:
                        std = validate_rgb(rgb)
                    except RejectedRGB as error:
                        # Preserve the entire rejected image. Never repair pixels,
                        # relax the zero-background rule or select another camera.
                        image = output / "rejected" / f"{stem}-slot-{slot['index']:02d}.png"
                        save_image(image, rgb)
                        rejected = dict(
                            scene=spec["identity"], slot=slot["index"],
                            camera_pattern_index=slot["camera_pattern_index"],
                            requested_camera=request,
                            actual_camera=event.metadata.get("thirdPartyCameras"),
                            image=dict(path=str(image), sha256=sha(image)),
                            decoded_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                            reason=str(error), retained_for_diagnosis_only=True,
                            training_admission=False, frame_policy=FRAME_POLICY,
                        )
                        proof = image.with_suffix(".json")
                        publish(proof, rejected)
                        rejected_rows.append(dict(path=str(proof), sha256=sha(proof)))
                        totals["rejected_frames"] += 1
                        if len(rejected_rows) > (2 if views == 20 else 0):
                            raise RuntimeError("Excess rejected views; inspect this scene before continuing")
                        continue
                    pixel_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
                    if pixel_sha in seen:
                        totals["duplicate_frames"] += 1
                        raise ValueError("Duplicate RGB pixels; do not silently replace the slot")
                    seen.add(pixel_sha)
                    cameras = event.metadata.get("thirdPartyCameras")
                    if not cameras or len(cameras) != 1:
                        raise ValueError("Unexpected third-party camera state")
                    actual = cameras[0]
                    for k in ("position", "rotation"):
                        differences = [
                            actual[k][axis] - request[k][axis] for axis in ("x", "y", "z")
                        ]
                        if k == "rotation":
                            differences = [(v + 180) % 360 - 180 for v in differences]
                        if not all(abs(v) < 1e-4 for v in differences):
                            raise ValueError("Camera pose changed")
                    image = output / "rgb" / f"{stem}-slot-{slot['index']:02d}.png"
                    save_image(image, rgb)
                    row = dict(
                        image=str(image),
                        sha256=sha(image),
                        decoded_rgb_sha256=pixel_sha,
                        dataset="structural_rooms",
                        policy=POLICY,
                        frame_policy=FRAME_POLICY,
                        split=split,
                        group=spec["identity"],
                        scene_ordinal=ordinal,
                        slot=slot["index"],
                        width=640,
                        height=480,
                        projection="orthographic" if request["orthographic"] else "perspective",
                        camera_model="parallel" if request["orthographic"] else "pinhole",
                        intrinsics=None,
                        camera_kind=slot["kind"],
                        camera_pattern_index=slot["camera_pattern_index"],
                        requested_camera=request,
                        actual_camera=actual,
                        orthographic_half_height_m=request.get("orthographicSize"),
                        vertical_fov_degrees=None
                        if request["orthographic"]
                        else request["fieldOfView"],
                        wall_distance_m=spec["depth"] - request["position"]["z"],
                        scene=dict(path=str(spec_path), sha256=sha(spec_path)),
                        scene_layout_sha256=spec["layout_sha256"],
                        assets=assets,
                        window_count=0,
                        door_count=0,
                        exterior_background_pixels=0,
                        agent_visible=False,
                        spatial_rgb_std=std,
                        geometry_label=None,
                        label_role="feature_only_rgb",
                        training_admission=False,
                        user_approved_projection=True,
                        renderer_build=BUILD,
                    )
                    records.write(json.dumps(row, allow_nan=False) + "\n")
                    scene_rows.append(
                        dict(image=str(image), sha256=row["sha256"], slot=slot["index"])
                    )
                    totals["completed"] += 1
                    totals[
                        "orthographic_frames" if request["orthographic"] else "perspective_frames"
                    ] += 1
                records.flush()
                os.fsync(records.fileno())
                receipt = output / "scene-reports" / (stem + ".json")
                publish(
                    receipt,
                    dict(
                        scene=spec["identity"],
                        scene_file=dict(path=str(spec_path), sha256=sha(spec_path)),
                        layout_sha256=spec["layout_sha256"],
                        assets=assets,
                        images=scene_rows,
                        rejected=rejected_rows,
                        attempted_slots=list(range(views)),
                        frame_policy=FRAME_POLICY,
                        camera_policy="18_perspective_2_orthographic_per_20",
                        windows=0,
                        doors=0,
                        closed=True,
                    ),
                )
                summaries.append(
                    dict(
                        ordinal=ordinal,
                        scene=spec["identity"],
                        images=len(scene_rows),
                        rejected=len(rejected_rows),
                        receipt=dict(path=str(receipt), sha256=sha(receipt)),
                    )
                )
                totals["scenes_closed"] += 1
                next_scene = ordinal + 1
                status = dict(
                    stage="rendering_structural_multiview_rgb",
                    next_scene=next_scene,
                    elapsed_seconds=time.monotonic() - started,
                    **totals,
                )
                publish(output / "status.json", status)
                print(json.dumps(status), flush=True)
        report = dict(
            stage="bounded_rgb_batch_rendered_review_pending",
            **totals,
            policy=POLICY,
            frame_policy=FRAME_POLICY,
            split=split,
            start_scene=start,
            requested_scenes=count,
            requested_views=views,
            next_scene=next_scene,
            all_requested_scenes_closed=next_scene == start + count,
            maximum_seconds=maximum_seconds,
            elapsed_seconds=time.monotonic() - started,
            manifest=dict(path=str(manifest), sha256=sha(manifest)),
            scenes=summaries,
            camera_policy="18_perspective_2_orthographic_per_20",
            user_approved_projection=True,
            cpu_only=True,
            training_admission=False,
            geometry_labels=False,
            renderer_build=BUILD,
            source_role="custom_parameterized_scene_not_official_procthor_house",
            code={
                p.name: sha(p)
                for p in (
                    Path(__file__),
                    Path(__file__).with_name("render_structural_room_samples.py"),
                    Path(__file__).with_name("render_procthor_empty_pilot.py"),
                )
            },
        )
        publish(output / "report.json", report)
        publish(
            output / "status.json", dict(stage=report["stage"], **totals, next_scene=next_scene)
        )
        return report
    except Exception as error:
        publish(
            output / "status.json",
            dict(
                stage="failed_inspect_partial",
                next_scene=next_scene,
                **totals,
                error=type(error).__name__,
                message=str(error),
            ),
        )
        raise
    finally:
        if controller is not None:
            controller.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--split", choices=("train", "val"), required=True)
    p.add_argument("--start-scene", type=int, required=True)
    p.add_argument("--scenes", type=int, required=True)
    p.add_argument("--views", type=int, choices=(4, 20), required=True)
    p.add_argument("--maximum-seconds", type=int, default=7200)
    args = p.parse_args()
    run(
        args.root,
        args.output,
        args.split,
        args.start_scene,
        args.scenes,
        args.views,
        args.maximum_seconds,
    )
