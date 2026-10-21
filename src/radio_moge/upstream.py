"""Pinned upstream loading. No checkpoints download at import time."""

import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from .adapters import RadioFeatures
from .config import ROOT, RADIO_COMMIT, MOGE_COMMIT, RADIO_REVISION, MOGE_REPO, MOGE_REVISION


def source_path(name, commit, override=None):
    path = Path(override or ROOT / "vendor" / name).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run python scripts/setup_upstream.py first.")
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != commit:
        raise ValueError(f"{name} source is {actual}; expected pinned revision {commit}")
    return path


def import_moge(path=None):
    directory = source_path("MoGe", MOGE_COMMIT, path)
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    from moge.model.v2 import MoGeModel

    return MoGeModel


def load_moge(config, device):
    cls = import_moge(config.get("moge_source"))
    checkpoint_path = config.get("moge_checkpoint") or hf_hub_download(
        config.get("moge_repo", MOGE_REPO),
        "model.pt",
        revision=config.get("moge_revision", MOGE_REVISION),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    model = cls(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval().requires_grad_(False).to(device), checkpoint["model_config"]


def load_radio(config, device):
    directory = source_path("RADIO", RADIO_COMMIT, config.get("radio_source"))
    checkpoint = config.get("radio_checkpoint") or hf_hub_download(
        "nvidia/C-RADIOv4-H", "c-radio_v4-h_half.pth.tar", revision=RADIO_REVISION
    )
    # The official RADIO loader uses pickle for its trusted release checkpoint.
    model = torch.hub.load(
        str(directory),
        "radio_model",
        source="local",
        version=checkpoint,
        adaptor_names=[],
        vitdet_window_size=config.get("window_size"),
    )
    # .float() does not change the conditioner's plain Python dtype attribute.
    # Keep parameter/input dtypes consistent when AMP is disabled.
    model.input_conditioner.dtype = torch.float32
    return model.eval().requires_grad_(False).float().to(device)


def radio_input(image, radio, num_tokens=None):
    """Resize the same view to a supported RADIO grid; never call the backbone twice."""
    h, w = image.shape[-2:]
    if num_tokens is not None:
        ratio = w / h
        h = max(1, round((num_tokens / ratio) ** 0.5)) * radio.patch_size
        w = max(1, round((num_tokens * ratio) ** 0.5)) * radio.patch_size
    step = radio.min_resolution_step
    h, w = max(step, round(h / step) * step), max(step, round(w / step) * step)
    if max(h, w) > 2048:
        raise ValueError("RADIO input exceeds the released model's 2048-pixel maximum")
    return F.interpolate(image, (h, w), mode="bilinear", align_corners=False, antialias=True)


def extract_radio(radio, image, indices=()):
    """Also works with a RADIO instance whose existing teacher adapters are enabled."""
    step = radio.min_resolution_step
    if any(n % step for n in image.shape[-2:]):
        raise ValueError(f"RADIO input dimensions must be divisible by {step}")
    if indices:
        final, intermediates = radio.forward_intermediates(
            image,
            indices=list(indices),
            output_fmt="NCHW",
            norm=True,
            aggregation="sparse",
            return_prefix_tokens=False,
            intermediates_only=False,
        )
    else:
        final, intermediates = radio(image, feature_fmt="NCHW"), ()
    if isinstance(final, dict):
        final = final["backbone"]
    return RadioFeatures(
        final.features.float(), final.summary.float(), tuple(x.float() for x in intermediates)
    )
