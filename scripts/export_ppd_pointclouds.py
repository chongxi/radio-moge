"""Recover cameras for saved PPD outputs and export RGB point clouds without GT.

PPD-native uses native MoGe metric depth/camera; PPD-adapter uses the adapter's
MoGe decoder for both. A separately labeled common-reference diagnostic is saved.
"""

import argparse
import base64
import gc
import gzip
import io
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image
import torch

from probe_ppd_feature_swap import digest, publish
from radio_moge.adapters import EncoderFeatures, build_adapter
from radio_moge.decoder import MoGeDecoder, postprocess
from radio_moge.pointcloud import unproject, verify_geometry
from radio_moge.upstream import extract_radio, load_moge, load_radio, radio_input


NAMES = ["moge_native", "moge_adapter", "ppd_native", "ppd_adapter", "ppd_adapter_shared"]
LABELS = ["原生 MoGe", "我们的特征 → MoGe", "原生特征 → PPD", "我们的特征 → PPD", "我们的特征 → PPD（共用原生校准）"]


def save_ply(path, depth, camera, mask, rgb):
    points = unproject(depth, camera)[mask].astype("<f4")
    points *= np.array([1, -1, -1], dtype=np.float32)
    assert np.isfinite(points).all() and len(points) > 1000
    vertices = np.empty(len(points), dtype=[("xyz", "<f4", (3,)), ("rgb", "u1", (3,))])
    vertices["xyz"], vertices["rgb"] = points, rgb[mask]
    header = ("ply\nformat binary_little_endian 1.0\n"
              "comment predicted metric scale; x right y up z backward\n"
              f"element vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    temporary = path.with_name(path.name + ".new")
    with temporary.open("xb") as stream:
        stream.write(header.encode())
        stream.write(vertices.tobytes())
    os.replace(temporary, path)
    assert path.stat().st_size == len(header.encode()) + 15 * len(points)
    return {"file": path.name, "vertices": len(points), "sha256": digest(path)}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "output", "vendor", "ppd-source"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    ply_root = args.output / "ply"
    ply_root.mkdir()
    plan = json.loads((args.base / "plan.json").read_text())
    report = json.loads((args.base / "report.json").read_text())
    assert report["completed"] and digest(args.base / "plan.json") == report["plan_sha256"]
    assert all(x["exact"] for x in report["teacher_boundary_parity"])
    assert digest(plan["checkpoint"]["path"]) == plan["checkpoint"]["sha256"]
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(args.ppd_source))
    from ppd.utils.align_depth_func import recover_metric_depth_ransac

    publish(args.output / "plan.json", {
        "source_report_sha256": digest(args.base / "report.json"),
        "source_plan_sha256": digest(args.base / "plan.json"),
        "script_sha256": digest(__file__), "checkpoint": plan["checkpoint"],
        "names": NAMES, "cases": len(report["results"]),
        "purpose": "3D reconstruction of the same saved PPD samples; no new diffusion sampling",
        "calibration": {
            "ppd_native": "Official PPD log1p RANSAC to native MoGe depth, native mask and camera",
            "ppd_adapter": "Official PPD log1p RANSAC to adapter-MoGe depth, adapter mask and camera",
            "ppd_adapter_shared": "Diagnostic only: adapter PPD aligned to native MoGe, same mask and camera as native PPD",
        },
        "camera_metadata": "Repeat frozen MoGe decoder inference; require saved metric depths to match before using cameras/masks",
        "projection": "Normalized pixel centers (u+0.5)/W,(v+0.5)/H; OpenGL coordinates x right,y up,z backward",
        "ground_truth_used_for_pointclouds": False,
        "outlier_filter": False, "smoothing": False, "pointcloud_registration": False,
        "viewer_stride": 4, "ply_stride": 1, "optimizer_updates": 0,
    })
    checkpoint = torch.load(plan["checkpoint"]["path"], map_location="cpu", weights_only=True, mmap=True)
    config = dict(checkpoint["config"]["models"], moge_source=str(args.vendor / "MoGe"), radio_source=str(args.vendor / "RADIO"))
    moge, _ = load_moge(config, "cuda")
    decoder = MoGeDecoder(moge)
    radio = load_radio(config, "cuda")
    adapter = build_adapter(**checkpoint["adapter_config"])
    adapter.load_state_dict(checkpoint["adapter"], strict=True)
    adapter.cuda().eval().requires_grad_(False)
    del checkpoint
    gc.collect()

    samples, ply_files, geometry = [], [], []
    for result in report["results"]:
        name = result["name"]
        with np.load(args.base / (name + ".npz")) as source:
            # Deliberately never read the previous GT-aligned display maps or GT.
            rgb = source["rgb"].copy()
            depths = {key: source[key].copy() for key in NAMES[:4]}
        height, width = rgb.shape[:2]
        image = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().cuda() / 255
        with torch.autocast("cuda", dtype=torch.bfloat16):
            spatial, summary = moge.encoder(image, height // 16, width // 16, return_class_token=True)
            native = EncoderFeatures(spatial.float(), summary.float())
            fitted = adapter(extract_radio(radio, radio_input(image, radio), adapter.indices))
            cameras, masks, errors, exacts = {}, {}, {}, {}
            for key, encoded in (("native", native), ("adapter", fitted)):
                prediction = postprocess(decoder(encoded, (height, width)), image, config["moge_source"])
                camera, mask, error = verify_geometry(depths["moge_" + key], prediction)
                cameras[key], masks[key], errors[key] = camera, mask, error
                current_depth = prediction["depth"][0].cpu().numpy()
                exacts[key] = bool(np.array_equal(current_depth, depths["moge_" + key]))
        raw_adapter = depths["ppd_adapter"].copy()
        for output, raw, reference in (
            ("ppd_native", depths["ppd_native"], "native"),
            ("ppd_adapter", raw_adapter, "adapter"),
            ("ppd_adapter_shared", raw_adapter, "native"),
        ):
            np.random.seed(plan["seed"] + result["index"])
            depths[output] = recover_metric_depth_ransac(raw, depths["moge_" + reference], masks[reference])
            if not np.isfinite(depths[output]).all() or np.any(depths[output] <= 0):
                raise ValueError(f"Invalid reconstructed depths: {name}/{output}")
        route_reference = {"moge_native": "native", "moge_adapter": "adapter", "ppd_native": "native",
                           "ppd_adapter": "adapter", "ppd_adapter_shared": "native"}
        for route in NAMES:
            reference = route_reference[route]
            entry = save_ply(ply_root / (name + "-" + route + ".ply"), depths[route], cameras[reference], masks[reference], rgb)
            ply_files.append(entry)
        temporary = args.output / (name + ".npz.new")
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, native_camera=np.asarray(cameras["native"]), adapter_camera=np.asarray(cameras["adapter"]),
                                native_mask=masks["native"], adapter_mask=masks["adapter"],
                                ppd_native=depths["ppd_native"], ppd_adapter=depths["ppd_adapter"],
                                ppd_adapter_shared=depths["ppd_adapter_shared"])
        os.replace(temporary, args.output / (name + ".npz"))
        stride = 4
        packed = b"".join(np.asarray(depths[key][::stride, ::stride], dtype="<f4").tobytes() for key in NAMES)
        packed += b"".join(masks[key][::stride, ::stride].astype("u1").tobytes() for key in ("native", "adapter"))
        packed += rgb[::stride, ::stride].tobytes()
        preview = io.BytesIO()
        Image.fromarray(rgb).resize((320, 240)).save(preview, format="JPEG", quality=90)
        sample = {
            "id": name, "index": result["index"], "scene": result["group"], "width": width, "height": height,
            "stride": stride, "cameras": cameras,
            "range": np.quantile(depths["moge_native"][masks["native"]], [.02, .98]).tolist(),
            "rgb": "data:image/jpeg;base64," + base64.b64encode(preview.getvalue()).decode(),
            "payload": base64.b64encode(gzip.compress(packed, compresslevel=6, mtime=0)).decode(),
            "full_counts": {key: int(masks[route_reference[key]].sum()) for key in NAMES},
            "features_cos": result["feature_metrics"]["mean_cosine"],
        }
        samples.append(sample)
        geometry.append({"id": name, "depths_exact": exacts, "unprojection_max_error_m": errors})
        print(json.dumps({"id": name, "depths_exact": exacts, "points": sample["full_counts"]}), flush=True)
        del prediction, native, fitted, image
    bundles = []
    for height, width in plan["resolutions_hw"]:
        archive = args.output / f"pointclouds-{width}x{height}.zip"
        temporary = archive.with_name(archive.name + ".new")
        selected = [p for p in ply_files if f"-{height}x{width}-" in p["file"]]
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as stream:
            for p in selected:
                stream.write(ply_root / p["file"], p["file"])
        os.replace(temporary, archive)
        bundles.append({"file": archive.name, "ply_files": len(selected), "bytes": archive.stat().st_size, "sha256": digest(archive)})
    publish(args.output / "data.json", {"completed": True, "names": NAMES, "labels": LABELS, "samples": samples})
    publish(args.output / "report.json", {
        "completed": True, "cases": len(samples), "ply_files": ply_files, "bundles": bundles,
        "plan_sha256": digest(args.output / "plan.json"), "data_sha256": digest(args.output / "data.json"),
        "geometry_checks": geometry, "ground_truth_used": False, "diffusion_inference_repeated": False,
        "optimizer_updates": 0, "full_resolution_ply": True, "viewer_stride": 4,
    })
    print(json.dumps({"complete": True, "cases": len(samples), "ply_files": len(ply_files), "bundles": bundles}), flush=True)


if __name__ == "__main__":
    main()
