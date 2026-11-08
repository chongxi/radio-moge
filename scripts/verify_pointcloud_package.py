"""Check the installed offline runtime against saved, accepted point-cloud inputs."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import socket
import sys
import time

import numpy as np
from PIL import Image
import torch

import radio_moge
from radio_moge import PointCloudPipeline
from radio_moge.checkpoint import atomic_json
from radio_moge.upstream import extract_radio, radio_input


def no_network(*args, **kwargs):
    raise AssertionError("Offline runtime attempted a network connection")


def stats(values):
    return dict(count=len(values), mean_ms=float(np.mean(values)),
                median_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)),
                max_ms=float(np.max(values)))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["weights", "checkpoint", "base", "clouds", "edges", "output"]:
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert "site-packages" in radio_moge.__file__, radio_moge.__file__
    assert radio_moge.__version__ == "0.2.0"
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    socket.socket.connect = no_network
    socket.create_connection = no_network
    start = time.perf_counter()
    pipeline = PointCloudPipeline.from_pretrained(args.weights, device="cuda")
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    assert pipeline.metadata["training_step"] == 272646
    assert not any("encoder" in k for k in pipeline.model.decoder.state_dict())
    original = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    for key, value in pipeline.model.adapter.state_dict().items():
        torch.testing.assert_close(value.cpu(), original["adapter"][key], rtol=0, atol=0)
    del original
    rows, timing_inputs = [], []
    for path in sorted(args.base.glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            rgb, depth = z["rgb"].copy(), z["moge_adapter"].copy()
        with np.load(args.clouds / path.name, allow_pickle=False) as z:
            expected_k, expected_base = z["adapter_camera"].copy(), z["adapter_mask"].copy()
        with np.load(args.edges / "masks" / path.name, allow_pickle=False) as z:
            expected_mask = z["moge_adapter_keep_2"].copy()
        result = pipeline(rgb)
        output = result.numpy()
        np.testing.assert_array_equal(output["depth"], depth)
        np.testing.assert_array_equal(output["base_mask"], expected_base)
        np.testing.assert_array_equal(output["mask"], expected_mask)
        np.testing.assert_array_equal(output["intrinsics_normalized"], expected_k)
        np.testing.assert_array_equal(output["rgb"], rgb)
        xyz, colors = result.pointcloud(numpy=True)
        np.testing.assert_array_equal(colors, rgb[expected_mask])
        h, w = depth.shape
        u, v = np.meshgrid((np.arange(w) + .5) / w, (np.arange(h) + .5) / h)
        rays = np.stack((u, v, np.ones_like(u)), -1) @ np.linalg.inv(expected_k).T
        expected_xyz = (rays * depth[..., None])[expected_mask]
        error = float(np.max(np.abs(xyz - expected_xyz)))
        np.testing.assert_allclose(xyz, expected_xyz, atol=5e-6, rtol=1e-6)
        # Inverse projection verifies the public pixel K (including the half-pixel convention).
        pixel_u = xyz[:, 0] / xyz[:, 2] * output["intrinsics"][0, 0] + output["intrinsics"][0, 2]
        pixel_v = xyz[:, 1] / xyz[:, 2] * output["intrinsics"][1, 1] + output["intrinsics"][1, 2]
        expected_v, expected_u = np.nonzero(expected_mask)
        np.testing.assert_allclose(pixel_u, expected_u, atol=3e-4, rtol=0)
        np.testing.assert_allclose(pixel_v, expected_v, atol=3e-4, rtol=0)
        rows.append(dict(scene=path.stem, height=h, width=w, depth_mask_intrinsics_rgb_exact=True,
                         retained_points=len(xyz), max_xyz_error=error))
        if (h, w) == (480, 640):
            timing_inputs.append(rgb)
        print(json.dumps(rows[-1]), flush=True)
    assert len(rows) == 32 and len(timing_inputs) == 16
    # PIL and tensor entry points, shared-feature bypass, disabled filter and export.
    rgb = timing_inputs[0]
    Image.fromarray(rgb).save(args.output / "example.png")
    reference = pipeline(rgb)
    image = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
    for variant in (Image.fromarray(rgb), image, image.cuda()):
        actual = pipeline(variant)
        torch.testing.assert_close(actual.depth, reference.depth, rtol=0, atol=0)
        torch.testing.assert_close(actual.mask, reference.mask, rtol=0, atol=0)
    with pipeline.model._autocast():
        features = extract_radio(pipeline.model.radio,
                                 radio_input(image[None].cuda(), pipeline.model.radio),
                                 pipeline.model.indices)
    backbone = pipeline.model.radio
    pipeline.model.radio = None
    actual = pipeline(image.cuda(), features=features, validate=False)
    pipeline.model.radio = backbone
    torch.testing.assert_close(actual.points, reference.points, rtol=0, atol=0)
    pipeline.edge_threshold = None
    unfiltered = pipeline(rgb)
    torch.testing.assert_close(unfiltered.mask, reference.base_mask, rtol=0, atol=0)
    torch.testing.assert_close(unfiltered.depth, reference.depth, rtol=0, atol=0)
    pipeline.edge_threshold = .04
    ply = reference.save_ply(args.output / "example.ply")
    header, data = ply.read_bytes().split(b"end_header\n", 1)
    vertices = np.frombuffer(data, dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
    xyz, colors = reference.pointcloud(numpy=True)
    assert f"element vertex {len(xyz)}".encode() in header
    np.testing.assert_array_equal(vertices["xyz"], xyz)
    np.testing.assert_array_equal(vertices["rgb"], colors)
    del reference, actual, unfiltered, features, result, output
    # Small installed-package regression check, alternating profiles across all scenes.
    timings = {"organized_gpu": [], "compact_cpu": []}
    for rgb in timing_inputs:
        pipeline(rgb)
    for round_ in range(4):
        modes = list(timings) if round_ % 2 == 0 else list(reversed(timings))
        for rgb in timing_inputs:
            for mode in modes:
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = pipeline(rgb)
                if mode == "compact_cpu":
                    result.pointcloud(numpy=True)
                torch.cuda.synchronize()
                timings[mode].append((time.perf_counter() - start) * 1000)
    report = dict(passed=True, version=radio_moge.__version__, training_step=272646,
                  installed_module=radio_moge.__file__, python=sys.executable,
                  network_connections_permitted=False, adapter_weights_bit_exact=True,
                  no_teacher_encoder=True, samples=rows, pil_and_tensor_parity=True,
                  shared_features_skip_radio=True, ply_roundtrip_exact=True,
                  load_including_hash_verification_seconds=load_seconds,
                  timing={k: stats(v) for k, v in timings.items()},
                  timing_scope="CPU decoded uint8 RGB to GPU organized or CPU compact XYZ/RGB; "
                               "640x480, one frame, per-frame camera recovery, no loading/IO/render",
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  max_gpu_allocated=torch.cuda.max_memory_allocated(),
                  installed_dependencies={name: importlib.metadata.version(name) for name in
                    ["radio-moge-inference", "torch", "torchvision", "timm", "utils3d-moge"]})
    atomic_json(timings, args.output / "timings.json")
    atomic_json(report, args.output / "report.json")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}), flush=True)


if __name__ == "__main__":
    main()
