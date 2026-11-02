"""Adapter-only distribution and perception-style inference.

Importing this module does not download weights, fetch code, or allocate a GPU.
``from_pretrained`` downloads only pinned upstream releases and source revisions.
The head-only path never constructs a RADIO backbone or a MoGe encoder.
"""

from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

from filelock import FileLock
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn

from .adapters import RadioFeatures, build_adapter
from .checkpoint import atomic_json, atomic_save
from .config import MOGE_COMMIT, MOGE_REPO, MOGE_REVISION, RADIO_COMMIT, RADIO_REVISION
from .decoder import MoGeDecoder, postprocess
from .upstream import extract_radio, load_radio, radio_input, source_path


DEFAULT_REPO = "chongxi/radio-moge-adapter-step232288"
BUCKETS = ((480, 640), (640, 480), (400, 640))
PINS = dict(radio_source=RADIO_COMMIT, moge_source=MOGE_COMMIT,
            radio_weights=RADIO_REVISION, moge_repo=MOGE_REPO, moge_weights=MOGE_REVISION)
SOURCES = {"MoGe": ("https://github.com/microsoft/MoGe.git", MOGE_COMMIT),
           "RADIO": ("https://github.com/NVlabs/RADIO.git", RADIO_COMMIT)}


def ensure_source(name, override=None, cache_dir=None, local_files_only=False):
    """Cache an immutable, allow-listed source checkout; never update user code."""
    url, revision = SOURCES[name]
    if override is not None:
        return source_path(name, revision, override)
    root = Path(cache_dir or os.environ.get(
        "RADIO_MOGE_CACHE", Path.home() / ".cache" / "radio-moge")) / "sources"
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{name}-{revision}"
    with FileLock(str(target) + ".lock"):
        if target.exists():
            return source_path(name, revision, target)
        if local_files_only:
            raise FileNotFoundError(f"Pinned {name} source is not cached at {target}")
        # A failed fetch remains isolated in its temporary directory. No existing
        # source is reset or replaced, and other processes only see complete trees.
        with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=root) as temporary:
            commands = [
                ["git", "init", "-q", temporary],
                ["git", "-C", temporary, "fetch", "--quiet", "--depth=1", url, revision],
                ["git", "-C", temporary, "checkout", "--quiet", "--detach", "FETCH_HEAD"],
            ]
            for command in commands:
                subprocess.run(command, check=True, timeout=300)
            source_path(name, revision, temporary)
            os.rename(temporary, target)
    return target


def export_adapter(checkpoint, output_directory):
    """Export FP32 adapter tensors, without decoder/backbones/optimizer/dataset paths."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if state["revisions"] != PINS:
        raise ValueError("Checkpoint revisions differ from this inference release")
    if state["adapter_config"].get("kind") != "joint_attention":
        raise ValueError("This release format currently supports the direct joint adapter")
    metadata = dict(
        format="radio-moge-adapter", format_version=1,
        adapter_config=state["adapter_config"], model_config=state["model_config"],
        revisions=PINS, training_step=int(state["step"]),
        buckets=state["config"]["data"].get("buckets", BUCKETS),
        precision=state["config"]["train"].get("precision", "bf16"),
        window_size=state["config"]["models"].get("window_size"),
    )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    # Do not silently replace a release. Use a new staging directory for a new one.
    if (output / "adapter.pt").exists():
        raise FileExistsError(output / "adapter.pt")
    atomic_save(dict(metadata, adapter=state["adapter"]), output / "adapter.pt")
    atomic_json(metadata, output / "config.json")
    return metadata


def prepare_image(image, bucket=None, buckets=BUCKETS):
    """EXIF transpose, center crop, then resize to a training bucket (H,W).

    Returns RGB float32 CHW in [0,1]. Geometry is expressed in this *processed*
    view, not the uncropped original photograph. Use the identical tensor for
    a teacher comparison and for every shared-backbone consumer.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("prepare_image expects a PIL image")
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if bucket is None:
        bucket = min(buckets, key=lambda hw: abs(math.log(width / height) -
                                               math.log(hw[1] / hw[0])))
    out_h, out_w = map(int, bucket)
    if min(out_h, out_w) < 1 or out_h % 16 or out_w % 16:
        raise ValueError("Bucket dimensions must be positive multiples of 16")
    crop_w = min(width, max(1, round(height * out_w / out_h)))
    crop_h = min(height, max(1, round(width * out_h / out_w)))
    left, top = (width - crop_w) // 2, (height - crop_h) // 2
    image = image.crop((left, top, left + crop_w, top + crop_h))
    image = image.resize((out_w, out_h), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255


def _resolve_artifact(path_or_repo, revision, cache_dir, local_files_only):
    path = Path(path_or_repo)
    if path.is_dir():
        path = path / "adapter.pt"
    if path.is_file():
        return path
    if path.suffix == ".pt" or str(path_or_repo).startswith(("/", "./", "../")):
        raise FileNotFoundError(path)
    return Path(hf_hub_download(str(path_or_repo), "adapter.pt", revision=revision,
                               cache_dir=cache_dir, local_files_only=local_files_only))


class RadioMoGe(nn.Module):
    """Metric geometry from RGB or features already computed by perception.

    ``infer(image)``: image is RGB float CHW or BCHW in [0,1].
    ``infer(image, features=...)``: reuse normalized RADIO maps from the SAME
    view and pass, without calling a backbone. Features must include all four
    requested taps. No input tensor is modified.

    Outputs: depth (meters), points (camera XYZ, meters), intrinsics (normalized
    3x3), mask (bool), and normal (unit XYZ). A batched input keeps its batch
    axis. Invalid geometry is not overwritten; always honor the mask.
    """

    def __init__(self, adapter, decoder, metadata, moge_source, radio=None):
        super().__init__()
        self.adapter = adapter
        self.decoder = decoder
        self.radio = radio
        self.metadata = metadata
        self.moge_source = moge_source
        self.indices = tuple(adapter.indices)
        self.buckets = tuple(tuple(x) for x in metadata.get("buckets", BUCKETS))
        self.precision = metadata.get("precision", "bf16")
        self.eval().requires_grad_(False)

    @property
    def device(self):
        return next(self.adapter.parameters()).device

    @classmethod
    def from_pretrained(cls, path_or_repo=DEFAULT_REPO, *, device=None, revision=None,
                        load_backbone=True, radio=None, radio_source=None,
                        moge_source=None, radio_checkpoint=None, moge_checkpoint=None,
                        cache_dir=None, local_files_only=False):
        """Load a local adapter.pt or HF model id and pinned upstream dependencies.

        Set load_backbone=False for perception integration. A provided ``radio``
        is reused, never copied or reloaded. CPU is supported for debugging;
        CUDA with BF16 is the evaluated deployment path.
        """
        path = _resolve_artifact(path_or_repo, revision, cache_dir, local_files_only)
        artifact = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if artifact.get("format") != "radio-moge-adapter" or artifact.get("format_version") != 1:
            raise ValueError("Expected adapter-only release, not a training/geometry-head checkpoint")
        if artifact.get("revisions") != PINS:
            raise ValueError("Upstream revisions do not match the installed inference code")
        target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        moge_source = ensure_source("MoGe", moge_source, cache_dir, local_files_only)
        weight_path = moge_checkpoint or hf_hub_download(
            MOGE_REPO, "model.pt", revision=MOGE_REVISION,
            cache_dir=cache_dir, local_files_only=local_files_only)
        teacher = torch.load(weight_path, map_location="cpu", weights_only=True, mmap=True)
        if teacher["model_config"] != artifact["model_config"]:
            raise ValueError("MoGe decoder configuration differs from the trained teacher")
        decoder = MoGeDecoder.from_config(artifact["model_config"], moge_source)
        # Select only decoder weights. No DINOv2 encoder is ever instantiated here.
        keys = set(decoder.state_dict())
        decoder.load_state_dict({k: v for k, v in teacher["model"].items() if k in keys},
                                strict=True)
        adapter = build_adapter(**artifact["adapter_config"])
        adapter.load_state_dict(artifact["adapter"], strict=True)
        metadata = {k: v for k, v in artifact.items() if k != "adapter"}
        model = cls(adapter, decoder, metadata, moge_source)
        model.to(target)
        del teacher, artifact
        if radio is not None:
            if next(radio.parameters()).device != target:
                raise ValueError("Shared RADIO must already be on the requested device")
            if radio.training:
                raise ValueError("Put the shared RADIO in eval() mode before attaching it")
            model.radio = radio
        elif load_backbone:
            source = ensure_source("RADIO", radio_source, cache_dir, local_files_only)
            weight = radio_checkpoint or hf_hub_download(
                "nvidia/C-RADIOv4-H", "c-radio_v4-h_half.pth.tar", revision=RADIO_REVISION,
                cache_dir=cache_dir, local_files_only=local_files_only)
            model.radio = load_radio(dict(radio_source=str(source), radio_checkpoint=str(weight),
                                         window_size=metadata.get("window_size")), target)
        return model

    def preprocess(self, image, bucket=None):
        return prepare_image(image, bucket, self.buckets)

    def _autocast(self):
        if self.device.type != "cuda" or self.precision not in ("bf16", "fp16"):
            return nullcontext()
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        return torch.autocast("cuda", dtype=dtype)

    def _validate_features(self, features, batch):
        if not isinstance(features, RadioFeatures):
            raise TypeError("features must be radio_moge.adapters.RadioFeatures")
        cfg = self.adapter.config
        spatial, summary = features.spatial, features.summary
        if spatial.ndim != 4 or spatial.shape[:2] != (batch, cfg["spatial_dim"]):
            raise ValueError("RADIO spatial must be B,C,H,W with the trained channel count")
        if summary.shape != (batch, cfg["summary_dim"]):
            raise ValueError("RADIO summary must contain both original backbone summary tokens")
        if len(features.intermediates) != len(self.indices):
            raise ValueError(f"Expected RADIO intermediate layers {self.indices} from one pass")
        if any(x.shape != spatial.shape for x in features.intermediates):
            raise ValueError("All RADIO taps must have the same B,C,H,W shape as spatial")
        tensors = (spatial, summary, *features.intermediates)
        if any(not x.is_floating_point() for x in tensors):
            raise ValueError("RADIO features must be floating point")
        converted = [x.to(device=self.device, dtype=torch.float32) for x in tensors]
        return RadioFeatures(converted[0], converted[1], tuple(converted[2:]))

    @torch.inference_mode()
    def infer(self, image, *, features=None):
        if not isinstance(image, torch.Tensor) or image.ndim not in (3, 4):
            raise ValueError("image must be RGB CHW or BCHW; use preprocess() for PIL images")
        single = image.ndim == 3
        image = image[None] if single else image
        if image.shape[1] != 3 or min(image.shape[-2:]) < 1 or image.shape[0] < 1:
            raise ValueError("image must have three RGB channels and nonempty dimensions")
        if not image.is_floating_point() or not torch.isfinite(image).all() or \
                image.min() < 0 or image.max() > 1:
            raise ValueError("image must be finite floating-point RGB in [0,1]")
        image = image.to(device=self.device, dtype=torch.float32)
        with self._autocast():
            if features is None:
                if self.radio is None:
                    raise ValueError("Head-only model: pass features= from your perception backbone")
                features = extract_radio(self.radio, radio_input(image, self.radio), self.indices)
            features = self._validate_features(features, image.shape[0])
            raw = self.decoder(self.adapter(features), image.shape[-2:])
        result = postprocess(raw, image, self.moge_source)
        return {k: v[0] for k, v in result.items()} if single else result

    def infer_from_features(self, features, image):
        """Same argument order as GeometryHead.infer(features, image)."""
        return self.infer(image, features=features)

    def forward(self, image, *, features=None):
        return self.infer(image, features=features)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Export adapter-only weights for inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    info = export_adapter(args.checkpoint, args.output)
    print(json.dumps(dict(output=args.output, step=info["training_step"])))


if __name__ == "__main__":
    main()
