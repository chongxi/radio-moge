"""Add verified camera intrinsics and validity masks to an existing depth export.

No training or metric selection is performed. The existing depth planes and test
scores remain untouched; a repeat forward pass recovers omitted camera metadata.
"""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch

from .checkpoint import atomic_json
from .compare import autocast, fingerprint
from .data import check_split, evaluation_image, read_manifest
from .decoder import postprocess
from .system import build_system


def unproject(depth, intrinsics):
    """Normalized, pixel-center intrinsics -> OpenCV camera coordinates, in meters."""
    k = np.asarray(intrinsics, dtype=np.float64)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0
            or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1])):
        raise ValueError("Invalid normalized camera intrinsics")
    h, w = depth.shape
    u, v = np.meshgrid((np.arange(w) + 0.5) / w, (np.arange(h) + 0.5) / h)
    rays = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(k).T
    return rays * depth[..., None]


def verify_geometry(saved_depth, prediction):
    depth = prediction["depth"][0].cpu().numpy()
    np.testing.assert_allclose(depth, saved_depth, rtol=1e-5, atol=1e-5, equal_nan=True,
                               err_msg="Depth differs from the existing evaluation export")
    k = prediction["intrinsics"][0].cpu().numpy()
    points = prediction["points"][0].cpu().numpy()
    projected = unproject(saved_depth, k)
    valid = np.isfinite(points).all(-1) & np.isfinite(projected).all(-1)
    if not valid.any():
        raise ValueError("No finite 3D points")
    np.testing.assert_allclose(projected[valid], points[valid], rtol=2e-5, atol=2e-5,
                               err_msg="Unprojection differs from official MoGe point map")
    mask = prediction.get("mask")
    mask = mask[0].cpu().numpy().astype(bool) if mask is not None else np.ones_like(depth, bool)
    mask &= np.isfinite(depth) & (depth > 0)
    return k.tolist(), mask, float(np.max(np.abs(projected[valid] - points[valid])))


@torch.no_grad()
def export_geometry(checkpoint, manifest, viewer):
    viewer = Path(viewer).resolve()
    destination = viewer / "geometry"
    if destination.exists():
        raise FileExistsError(f"Geometry already exists: {destination}")
    source = viewer / "manifest.json"
    exported = json.loads(source.read_text())
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (fingerprint(state["adapter"]) != exported["run"]["checkpoint_sha256"]
            or state["revisions"] != exported["run"]["revisions"]):
        raise ValueError("Checkpoint does not match the displayed evaluation")
    config = state["config"]
    records = read_manifest(manifest)
    if [r["id"] for r in records] != [s["id"] for s in exported["samples"]]:
        raise ValueError("Test image order differs from the existing export")
    for key in ("train_manifest", "val_manifest"):
        if config["data"].get(key):
            check_split(read_manifest(config["data"][key]), records)
    system, adapter = build_system(config, "cuda")
    adapter.load_state_dict(state["adapter"])
    adapter.eval()
    staging = Path(tempfile.mkdtemp(prefix=".geometry-", dir=viewer))
    samples = []
    for index, (record, sample) in enumerate(zip(records, exported["samples"], strict=True)):
        h, w = sample["height"], sample["width"]
        planes = np.frombuffer(gzip.decompress((viewer / sample["depth"]).read_bytes()),
                               dtype="<f4").reshape(len(exported["channels"]), h, w)
        image, _ = evaluation_image(record, config["evaluation"]["max_side"])
        image = image.cuda()
        with autocast(config, "cuda"):
            features, target = system.encode(image, adapter.indices,
                                              config["evaluation"]["num_tokens"])
            encodings = {"after": adapter(features), "teacher": target}
        geometry = {}
        for name, encoded in encodings.items():
            pred = postprocess(system.decoder(encoded, (h, w)), image,
                               config["models"].get("moge_source"))
            k, mask, error = verify_geometry(planes[exported["channels"].index(name)], pred)
            filename = f"{index:04d}-{name}-mask.png"
            Image.fromarray(mask.astype("uint8") * 255).save(staging / filename)
            geometry[name] = dict(intrinsics=k, mask=f"geometry/{filename}",
                                  valid_points=int(mask.sum()), projection_max_error_m=error)
        samples.append(dict(id=sample["id"], **geometry))
        if index % 20 == 0 or index + 1 == len(records):
            print(f"Verified 3D geometry {index + 1}/{len(records)}", flush=True)
    result = dict(schema=1, samples=samples,
                  depth_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  checkpoint_sha256=exported["run"]["checkpoint_sha256"],
                  coordinates="OpenCV: x right, y down, z forward; meters",
                  projection="Normalized intrinsics, pixel centers (x+0.5)/width, (y+0.5)/height",
                  policy="Each model's own recovered intrinsics and validity mask. No alignment.")
    atomic_json(result, staging / "manifest.json")
    os.rename(staging, destination)
    print(json.dumps({"output": str(destination), "images": len(samples),
                      "max_projection_error_m": max(s[b]["projection_max_error_m"]
                                                    for s in samples for b in encodings)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--viewer", required=True)
    args = parser.parse_args()
    export_geometry(args.checkpoint, args.manifest, args.viewer)


if __name__ == "__main__":
    main()
