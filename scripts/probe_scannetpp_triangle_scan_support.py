"""Check reviewed source triangles against retained scanner range samples.

Use actual stored azimuth/polar angles, a bounded coarse-to-fine angular search,
and the exact candidate triangle. No whole-mesh rerender, camera fitting, label
repair, model inference or panorama training. Same acquisition is not an
independent physical calibration standard.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from probe_scannetpp_native_near_surfaces import analytic_hits


def directions(azimuth, polar):
    azimuth, polar = (
        np.asarray(azimuth, dtype=np.float64) / 1000,
        np.asarray(polar, dtype=np.float64) / 1000,
    )
    return np.stack(
        (np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)), -1
    )


def nearest_angle_pixel(azimuth, polar, target, stride=32):
    """Search stored angles, including disconnected/seam-adjacent neighborhoods.

    This bounded search is not asserted to be a global nearest-neighbor proof.
    The final exact ray/triangle hit, angle and sampling neighborhood are saved.
    """
    target = np.asarray(target, dtype=np.float64)
    if (
        azimuth.shape != polar.shape
        or azimuth.ndim != 2
        or target.shape != (3,)
        or not np.isfinite(target).all()
        or np.linalg.norm(target) == 0
        or stride < 1
    ):
        raise ValueError("Matching angle grids and nonzero finite target required")
    target = target / np.linalg.norm(target)
    sampled = directions(azimuth[::stride, ::stride], polar[::stride, ::stride])
    error = np.linalg.norm(sampled - target, axis=-1)
    count = min(8, error.size)
    selected = np.argsort(error.ravel(), kind="stable")[:count]
    best = None
    for i in selected:
        cy, cx = np.unravel_index(i, error.shape)
        y0, y1 = max(0, cy * stride - stride), min(azimuth.shape[0], cy * stride + stride + 1)
        x0, x1 = max(0, cx * stride - stride), min(azimuth.shape[1], cx * stride + stride + 1)
        local = directions(azimuth[y0:y1, x0:x1], polar[y0:y1, x0:x1])
        distance = np.linalg.norm(local - target, axis=-1)
        j = int(np.argmin(distance))
        y, x = np.unravel_index(j, distance.shape)
        candidate = (float(distance[y, x]), y + y0, x + x0)
        if best is None or candidate < best:
            best = candidate
    chord, y, x = best
    ray = directions(azimuth[y, x], polar[y, x])
    return int(y), int(x), ray, float(2 * np.arcsin(min(1, chord / 2)))


def range_neighborhood(depth, mask, y, x):
    if depth.shape != mask.shape or depth.ndim != 2:
        raise ValueError("Range and anonymity grids must match")
    if y < 1 or x < 1 or y + 1 >= depth.shape[0] or x + 1 >= depth.shape[1]:
        return dict(stable_supported=False, reason="edge_pixel")
    patch = depth[y - 1 : y + 2, x - 1 : x + 2].astype(np.float64) / 1000
    allowed = mask[y - 1 : y + 2, x - 1 : x + 2] == 255
    center = float(patch[1, 1])
    stable = bool(
        np.all(allowed & (patch > 0.05) & (patch < 65.535))
        and np.all(abs(patch - center) <= 0.03 * center)
    )
    return dict(
        stable_supported=stable,
        range_m=center,
        ranges_m=patch.tolist(),
        anonymity_allowed=allowed.tolist(),
        relative_stability_tolerance=0.03,
    )


def compare_triangle(vertices, scanner_pose, direction_local, measurement):
    pose = np.asarray(scanner_pose, dtype=np.float64)
    if (
        pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6, rtol=0)
    ):
        raise ValueError("Finite rigid scanner pose required")
    direction = np.asarray(direction_local, dtype=np.float64) @ pose[:3, :3].T
    direction = direction / np.linalg.norm(direction)
    hits = analytic_hits(vertices, np.array([[0, 1, 2]]), pose[:3, 3], direction)["first_hits"]
    result = dict(
        intersects_exact_triangle=bool(hits),
        direction_world=direction.tolist(),
        scanner_origin=pose[:3, 3].tolist(),
        measurement=measurement,
    )
    if not hits:
        result["diagnostic"] = "stored_ray_misses_exact_triangle"
        return result
    hit = hits[0]
    result.update(
        triangle_range_m=hit["t"],
        barycentrics=hit["barycentrics"],
        boundary_hit=hit["boundary_hit"],
        intersection_residual_m=hit["ray_triangle_residual_m"],
    )
    if not measurement["stable_supported"] or hit["boundary_hit"]:
        result["diagnostic"] = "unsupported_or_boundary_measurement"
        return result
    residual = measurement["range_m"] - hit["t"]
    result["measured_minus_triangle_range_m"] = residual
    result["diagnostic"] = (
        "measurement_beyond_triangle"
        if residual > 0.05
        else "measurement_before_triangle"
        if residual < -0.05
        else "measurement_near_triangle"
    )
    return result


def run(probe_path, root, output, ordinals):
    from radio_moge.corpus_admission import checked_path, checked_json, file_ref
    from radio_moge.geometry_cache import publish_json

    probe_ref = file_ref(probe_path)
    probe = checked_json(probe_ref)
    if (
        probe["stage"] != "bounded_raw_mesh_ray_diagnosis_complete"
        or probe["training_admission"] is not False
        or not 1 <= len(set(ordinals)) == len(ordinals) <= 3
    ):
        raise ValueError("Completed raw-ray diagnosis and bounded explicit scope required")
    cases = []
    for ref in probe["cases"]:
        if int(Path(ref["path"]).stem) in ordinals:
            case = checked_json(ref)
            cases.append((ref, case))
    if len(cases) != len(ordinals):
        raise ValueError("Requested source cases missing")
    output.mkdir(exist_ok=False)
    results = []
    groups = sorted({case["group"] for _, case in cases})
    for group in groups:
        scene = group.split(":", 1)[1]
        subset = [(ref, case) for ref, case in cases if case["group"] == group]
        audit_ref = file_ref(root / "scannetpp-source-audit-01" / f"{scene}.json")
        audit = checked_json(audit_ref)
        pose_ref = file_ref(root / "scanner-poses/data" / scene / "scans/scanner_poses.json")
        poses = np.asarray(checked_json(pose_ref), dtype=np.float64)
        targets = []
        for ref, case in subset:
            source_refs = case["source_refs"]
            for source in source_refs:
                checked_path(source)
            for ray in case["rays"]:
                if not ray["cached_valid"] or not any(
                    "supported_depth_quantile" in s for s in ray["selection"]
                ):
                    continue
                hit = ray["analytical"]["first_hits"][0]
                targets.append(
                    dict(
                        case=ref,
                        corpus_ordinal=case["corpus_ordinal"],
                        pixel_yx=ray["pixel_yx"],
                        cached_z_m=ray["cached_z_m"],
                        face_id=hit["face_id"],
                        point_world=hit["point_world"],
                        vertices_world=hit["vertices_world"],
                        source_refs=source_refs,
                    )
                )
        for scan in audit["scanner_vs_mesh"]:
            if pose_ref["sha256"] != scan["scanner_poses_sha256"]:
                raise ValueError("Scanner pose version differs from retained audit")
            arrays = {
                k: np.asarray(Image.open(checked_path(ref))).copy()
                for k, ref in scan["inputs"].items()
            }
            if len({a.shape for a in arrays.values()}) != 1:
                raise ValueError("Scanner arrays differ in shape")
            pose = poses[int(scan["scan"])]
            measurements = []
            for target in targets:
                point = np.asarray(target["point_world"])
                local = (point - pose[:3, 3]) @ pose[:3, :3]
                y, x, direction, angle = nearest_angle_pixel(arrays["azim"], arrays["elev"], local)
                measure = range_neighborhood(arrays["range"], arrays["mask"], y, x)
                comparison = compare_triangle(target["vertices_world"], pose, direction, measure)
                measurements.append(
                    dict(
                        target=target,
                        pixel_yx=[y, x],
                        angular_error_rad=angle,
                        target_radial_distance_m=float(np.linalg.norm(local)),
                        stored_azimuth_millirad=int(arrays["azim"][y, x]),
                        stored_polar_millirad=int(arrays["elev"][y, x]),
                        comparison=comparison,
                    )
                )
            path = output / f"{scene}-scan{scan['scan']}.json"
            publish_json(
                path,
                dict(
                    scene=scene,
                    scan=scan["scan"],
                    audit=audit_ref,
                    scanner_poses=pose_ref,
                    inputs=scan["inputs"],
                    angular_search_stride=32,
                    coarse_candidates_refined=8,
                    full_grid_nearest_neighbor_claimed=False,
                    measurements=measurements,
                ),
            )
            results.append(file_ref(path))
            del arrays
            print(
                json.dumps(
                    dict(
                        completed_scans=len(results),
                        scene=scene,
                        scan=scan["scan"],
                        targets=len(targets),
                    )
                ),
                flush=True,
            )
    checked_path(probe_ref)
    report = dict(
        stage="bounded_triangle_scanner_support_complete",
        probe=probe_ref,
        ordinals=ordinals,
        scans=results,
        same_laser_acquisition=True,
        independent_physical_calibration=False,
        source_quality_accepted=False,
        training_admission=False,
        optimizer_updates=0,
        geometry_or_camera_edited=False,
        panoramas_added_to_training=False,
        gpu_used=False,
        code=file_ref(Path(__file__)),
    )
    publish_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ordinals", required=True, type=int, nargs="+")
    args = parser.parse_args()
    run(args.probe.resolve(), args.root.resolve(), args.output.resolve(), args.ordinals)
