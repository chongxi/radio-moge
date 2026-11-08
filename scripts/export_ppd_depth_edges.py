"""Apply the MoGe CLI edge filter to saved paired clouds; no model inference."""

import argparse
import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from radio_moge.depth_edges import relative_depth_edge_score


THRESHOLDS = [0.01, 0.02, 0.04, 0.08]
METHODS = ["moge_adapter", "ppd_adapter"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publish(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".new")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def save_ply(path, depth, camera, keep, rgb, base_mask, source_sha256):
    h, w = depth.shape
    u, v = np.meshgrid((np.arange(w) + .5) / w, (np.arange(h) + .5) / h)
    rays = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(camera).T
    points = (rays * depth[..., None])[keep].astype("<f4")
    points *= np.array([1, -1, -1], dtype=np.float32)
    vertices = np.empty(len(points), dtype=[("xyz", "<f4", (3,)), ("rgb", "u1", (3,))])
    vertices["xyz"], vertices["rgb"] = points, rgb[keep]
    header = ("ply\nformat binary_little_endian 1.0\n"
              "comment MoGe relative depth edge filter rtol=0.04 kernel=3; unchanged surviving points\n"
              "comment predicted meters; x right y up z backward\n"
              f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    temporary = path.with_name(path.name + ".new")
    with temporary.open("xb") as stream:
        stream.write(header.encode())
        stream.write(vertices.tobytes())
    os.replace(temporary, path)
    # Reconstruct the prior PLY and bind it to its already published SHA. Only
    # the compact depth/camera archives need to be mirrored to this machine.
    original = np.empty(int(base_mask.sum()), dtype=vertices.dtype)
    original["xyz"] = (rays * depth[..., None])[base_mask].astype("<f4")
    original["xyz"] *= np.array([1, -1, -1], dtype=np.float32)
    original["rgb"] = rgb[base_mask]
    original_header = ("ply\nformat binary_little_endian 1.0\n"
                       "comment predicted metric scale; x right y up z backward\n"
                       f"element vertex {len(original)}\nproperty float x\nproperty float y\nproperty float z\n"
                       "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    sha = hashlib.sha256(original_header.encode())
    sha.update(original.tobytes())
    assert sha.hexdigest() == source_sha256, "Reconstructed source PLY differs from published artifact"
    return vertices, original, {"file": str(path.name), "vertices": len(points), "sha256": digest(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ["masks", "ply", "previews"]:
        (args.output / name).mkdir()
    clouds, depths_root = args.source / "pointclouds-02", args.source / "results-01"
    old_report = json.loads((clouds / "report.json").read_text())
    old_data = json.loads((clouds / "data.json").read_text())
    assert old_report["completed"] and old_report["data_sha256"] == digest(clouds / "data.json")
    source_ply_shas = {p["file"]: p["sha256"] for p in old_report["ply_files"]}
    publish(args.output / "plan.json", {
        "source": str(args.source), "source_report_sha256": digest(clouds / "report.json"),
        "source_data_sha256": digest(clouds / "data.json"), "cases": 32,
        "filter": "utils3d_moge.np.depth_map_edge(depth, rtol=t), kernel_size=3; mask applied afterwards",
        "thresholds": THRESHOLDS, "default_threshold": .04,
        "moge_revision": "74fbce054ebed49800de42d0ad0e83495065719a",
        "utils3d_revision": "62f09d58509485564e24d5d9f6aac9ee9ebc0c37",
        "filter_source_sha256": digest(Path(__file__).resolve().parents[1] / "src/radio_moge/depth_edges.py"),
        "script_sha256": digest(__file__), "model_inference": False, "ground_truth_used": False,
        "depths_changed": False, "camera_changed": False, "calibration_repeated": False,
        "methods": METHODS, "camera_and_base_mask": "saved adapter MoGe for both methods",
        "ppd_sampling_steps": 4, "default_ppd_filtered": False,
        "preview_stride": "2 for 640x480; 4 for 1024x768", "filter_resolution": "full original resolution",
        "ply_resolution": "full original resolution, MoGe filtered at rtol=0.04",
        "meaning": "Removed fractions are filter selections, not verified erroneous-point fractions",
    })
    samples, rows, ply_files = [], [], []
    for old in old_data["samples"]:
        name = old["id"]
        with np.load(depths_root / (name + ".npz")) as raw:
            rgb, moge = raw["rgb"].copy(), raw["moge_adapter"].copy()
        with np.load(clouds / (name + ".npz")) as saved:
            ppd = saved["ppd_adapter"].copy()
            camera, base_mask = saved["adapter_camera"].copy(), saved["adapter_mask"].copy()
        h, w = moge.shape
        assert rgb.shape == (h, w, 3) and ppd.shape == (h, w)
        assert np.isfinite(camera).all() and camera[0, 0] > 0 and camera[1, 1] > 0
        assert camera[0, 1] == camera[1, 0] == 0
        assert base_mask.sum() == old["full_counts"]["moge_adapter"]
        assert all(np.isfinite(d[base_mask]).all() and (d[base_mask] > 0).all() for d in (moge, ppd))
        removal_bits, mask_arrays, counts = [], {}, {}
        for key, depth in zip(METHODS, [moge, ppd]):
            score = relative_depth_edge_score(depth)
            bits = np.zeros((h, w), dtype="u1")
            counts[key] = []
            previous = None
            for index, threshold in enumerate(THRESHOLDS):
                keep = base_mask & ~(score > threshold)
                if previous is not None:
                    assert not (previous & ~keep).any(), "Stronger threshold must remove a superset"
                previous = keep
                mask_arrays[f"{key}_keep_{index}"] = keep
                bits |= ((base_mask & ~keep).astype("u1") << index)
                removed = int((base_mask & ~keep).sum())
                counts[key].append({"threshold": threshold, "kept": int(keep.sum()),
                                    "removed": removed, "removed_fraction": removed / int(base_mask.sum())})
            removal_bits.append(bits)
        mask_path = args.output / "masks" / (name + ".npz")
        with mask_path.with_suffix(".npz.new").open("xb") as stream:
            np.savez_compressed(stream, base_mask=base_mask, **mask_arrays)
        os.replace(mask_path.with_suffix(".npz.new"), mask_path)
        keep = mask_arrays["moge_adapter_keep_2"]
        ply_path = args.output / "ply" / (name + "-moge-edge-004.ply")
        vertices, original, ply = save_ply(ply_path, moge, camera, keep, rgb, base_mask,
                                         source_ply_shas[name + "-moge_adapter.ply"])
        assert len(original) == base_mask.sum()
        assert vertices.tobytes() == original[keep[base_mask]].tobytes(), "Filtering changed retained vertices"
        ply_files.append(ply)
        stride = 2 if h == 480 else 4
        packed = b"".join(d[::stride, ::stride].astype("<f4").tobytes() for d in (moge, ppd))
        packed += base_mask[::stride, ::stride].astype("u1").tobytes()
        packed += b"".join(bits[::stride, ::stride].tobytes() for bits in removal_bits)
        packed += rgb[::stride, ::stride].tobytes()
        n = ((h + stride - 1) // stride) * ((w + stride - 1) // stride)
        assert len(packed) == n * 14
        preview = io.BytesIO()
        Image.fromarray(rgb).resize((320, 240)).save(preview, format="JPEG", quality=90)
        samples.append({
            "id": name, "index": old["index"], "scene": old["scene"], "height": h, "width": w,
            "stride": stride, "camera": camera.tolist(), "base_count": int(base_mask.sum()),
            "range": np.quantile(moge[base_mask], [.02, .98]).tolist(), "counts": counts,
            "rgb": "data:image/jpeg;base64," + base64.b64encode(preview.getvalue()).decode(),
            "payload": base64.b64encode(gzip.compress(packed, compresslevel=6, mtime=0)).decode(),
        })
        row = {"id": name, "index": old["index"], "scene": old["scene"], "height": h, "width": w,
               "base_count": int(base_mask.sum()), "counts": counts, "kept_vertices_byte_exact": True,
               "source_depth_sha256": digest(depths_root / (name + ".npz")),
               "source_geometry_sha256": digest(clouds / (name + ".npz")),
               "masks_sha256": digest(mask_path)}
        rows.append(row)
        print(json.dumps({"id": name, "moge_removed_percent": counts[METHODS[0]][2]["removed_fraction"] * 100,
                          "ppd_removed_percent": counts[METHODS[1]][2]["removed_fraction"] * 100}), flush=True)
    bundles = []
    for h, w in [(480, 640), (768, 1024)]:
        archive = args.output / f"moge-filtered-{w}x{h}.zip"
        temporary = archive.with_suffix(".zip.new")
        selected = [p for p in ply_files if f"-{h}x{w}-" in p["file"]]
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as stream:
            for ply in selected:
                stream.write(args.output / "ply" / ply["file"], ply["file"])
        os.replace(temporary, archive)
        bundles.append({"file": archive.name, "files": len(selected), "bytes": archive.stat().st_size,
                        "sha256": digest(archive)})
    aggregates = {}
    for h in [480, 768]:
        for group in ["nyuv2", "structured3d", "all"]:
            selected = [r for r in rows if r["height"] == h and (group == "all" or r["scene"].startswith(group))]
            aggregates[f"{h}/{group}"] = {}
            for method in METHODS:
                values = [r["counts"][method][2]["removed_fraction"] for r in selected]
                aggregates[f"{h}/{group}"][method] = {
                    "cases": len(values), "mean_removed_fraction": float(np.mean(values)),
                    "min_removed_fraction": min(values), "max_removed_fraction": max(values)}
    data_path = args.output / "data.json"
    publish(data_path, {"completed": True, "thresholds": THRESHOLDS, "samples": samples})
    publish(args.output / "report.json", {
        "completed": True, "cases": len(rows), "results": rows, "aggregates_at_004": aggregates,
        "data_sha256": digest(data_path), "plan_sha256": digest(args.output / "plan.json"),
        "ply_files": ply_files, "bundles": bundles, "source_depth_and_camera_preserved": True,
        "all_retained_ply_vertices_byte_exact": True, "model_inference": False, "ground_truth_used": False,
        "upstream_mask_parity": "Pending independent comparison against pinned utils3d_moge",
    })


if __name__ == "__main__":
    main()
