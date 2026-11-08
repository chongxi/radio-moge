"""RGB -> RADIO-v4-H -> adapter -> MoGe-v2 decoder -> depth edges -> colored XYZ."""

from dataclasses import dataclass, fields
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn

from .depth_edges import depth_edge_keep_mask_torch
from .deployment import RadioMoGe


def _image_tensor(image, validate):
    """Keep the original view/resolution; normalize uint8 on its current device."""
    if isinstance(image, Image.Image):
        image = np.array(ImageOps.exif_transpose(image).convert("RGB"))
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("NumPy input must be uint8 RGB HxWx3 (convert OpenCV BGR first)")
        # Copy also handles read-only arrays and negative strides without mutating the caller.
        image = torch.from_numpy(image.copy()).permute(2, 0, 1)
    if not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Tensor input must be one RGB image, 3xHxW; NumPy uses HxWx3")
    if min(image.shape[-2:]) < 1:
        raise ValueError("Image dimensions must be nonempty")
    if image.dtype == torch.uint8:
        # For CPU inputs this matches the evaluated CPU division before H2D exactly.
        return image.float() / 255
    if not image.is_floating_point():
        raise ValueError("Tensor input must be uint8 [0,255] or floating point [0,1]")
    if validate and (not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1):
        raise ValueError("Floating RGB input must be finite and in [0,1]")
    return image.float()


@dataclass(frozen=True)
class PointCloudResult:
    """Organized tensors on the model device; invalid samples are excluded by mask.

    points/normal use OpenCV camera axes: x right, y down, z forward. Depth is
    optical-axis Z with MoGe's predicted metric scale, not a Euclidean range.
    intrinsics uses integer pixel centers (u,v); intrinsics_normalized uses
    ((u+.5)/W, (v+.5)/H). rgb is uint8 HxWx3. Arrays retain the input view.
    """

    depth: torch.Tensor
    points: torch.Tensor
    mask: torch.Tensor
    base_mask: torch.Tensor
    rgb: torch.Tensor
    intrinsics: torch.Tensor
    intrinsics_normalized: torch.Tensor
    normal: torch.Tensor | None = None

    def numpy(self):
        """Explicitly copy ALL organized output planes to CPU NumPy arrays."""
        return {f.name: None if (v := getattr(self, f.name)) is None else v.detach().cpu().numpy()
                for f in fields(self)}

    def pointcloud(self, *, numpy=False, coordinate_system="opencv"):
        """Return (XYZ float32 Nx3, RGB uint8 Nx3), compacted on the model device.

        numpy=True transfers only retained XYZ/RGB to CPU. OpenGL flips Y and Z;
        it does not estimate a world pose. The organized result is never mutated.
        """
        if coordinate_system not in ("opencv", "opengl"):
            raise ValueError("coordinate_system must be 'opencv' or 'opengl'")
        xyz, colors = self.points[self.mask], self.rgb[self.mask]
        if coordinate_system == "opengl":
            xyz = xyz * xyz.new_tensor([1, -1, -1])
        if numpy:
            return xyz.detach().cpu().numpy(), colors.detach().cpu().numpy()
        return xyz, colors

    def save_ply(self, path, *, coordinate_system="opencv"):
        """Atomically write the filtered colored cloud as binary little-endian PLY."""
        xyz, rgb = self.pointcloud(numpy=True, coordinate_system=coordinate_system)
        vertices = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                             ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        for i, name in enumerate(("x", "y", "z")):
            vertices[name] = xyz[:, i]
        for i, name in enumerate(("red", "green", "blue")):
            vertices[name] = rgb[:, i]
        header = ("ply\nformat binary_little_endian 1.0\n"
                  f"comment camera coordinates {coordinate_system}\nelement vertex {len(xyz)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(header.encode("ascii"))
                f.write(vertices.tobytes())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return path


class PointCloudPipeline(nn.Module):
    """Load once, call once per RGB frame. No resize/crop of the output view.

    Inputs: PIL image, NumPy uint8 RGB HWC, or torch RGB CHW (uint8/float [0,1]).
    A CUDA tensor stays on CUDA. validate=False skips tensor value checks for
    trusted streams, avoiding their host synchronization. No camera is cached
    across frames; the official MoGe recovery is retained on each invocation.
    """

    def __init__(self, model: RadioMoGe, *, edge_threshold=.04):
        super().__init__()
        if edge_threshold is not None and (not math.isfinite(edge_threshold) or edge_threshold <= 0):
            raise ValueError("edge_threshold must be positive and finite, or None to disable")
        self.model = model
        self.edge_threshold = edge_threshold
        self.eval().requires_grad_(False)

    @property
    def device(self):
        return self.model.device

    @property
    def metadata(self):
        return self.model.metadata

    @classmethod
    def from_pretrained(cls, directory, *, device=None, edge_threshold=.04,
                        verify=True, load_backbone=True, radio=None):
        """Load a complete local bundle; never fetch models or source from a network.

        verify=True checks bundled files' SHA256 once at load time. For a shared
        RADIO backbone use radio=existing_model or load_backbone=False, then pass
        features=RadioFeatures to each call, computed from exactly the same view.
        """
        from .bundle import read_bundle

        root, manifest = read_bundle(directory, verify=verify)
        model = RadioMoGe.from_pretrained(
            root / "adapter.pt", device=device, local_files_only=True,
            moge_source=root / "sources/MoGe", radio_source=root / "sources/RADIO",
            decoder_checkpoint=root / "decoder.pt", radio_checkpoint=str(root / "radio.pth.tar"),
            load_backbone=load_backbone, radio=radio,
        )
        if model.metadata["training_step"] != manifest["training_step"]:
            raise ValueError("Bundle manifest and adapter training step differ")
        return cls(model, edge_threshold=edge_threshold)

    @torch.inference_mode()
    def forward(self, image, *, features=None, validate=True):
        image = _image_tensor(image, validate).to(self.device)
        output = self.model.infer(image, features=features, validate=False)
        depth = output["depth"]
        base_mask = output["mask"] & torch.isfinite(depth) & (depth > 0)
        mask = (base_mask if self.edge_threshold is None else
                depth_edge_keep_mask_torch(depth, base_mask, self.edge_threshold))
        height, width = depth.shape
        normalized = output["intrinsics"]
        pixels = normalized * normalized.new_tensor([width, height, 1])[:, None]
        # Upstream UVs are pixel-center normalized; OpenCV integer centers start at 0.
        pixels = pixels.clone()
        pixels[0, 2] -= .5
        pixels[1, 2] -= .5
        rgb = (image.permute(1, 2, 0) * 255).round().to(torch.uint8)
        return PointCloudResult(depth, output["points"], mask, base_mask, rgb,
                                pixels, normalized, output.get("normal"))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RGB image to filtered colored PLY")
    parser.add_argument("--weights", required=True, help="complete local bundle directory")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True, help="output .ply")
    parser.add_argument("--device", default=None)
    parser.add_argument("--edge-threshold", type=float, default=.04)
    parser.add_argument("--coordinates", choices=["opencv", "opengl"], default="opencv")
    args = parser.parse_args()
    pipeline = PointCloudPipeline.from_pretrained(
        args.weights, device=args.device, edge_threshold=args.edge_threshold)
    with Image.open(args.image) as image:
        result = pipeline(image)
    print(result.save_ply(args.output, coordinate_system=args.coordinates))


if __name__ == "__main__":
    main()
