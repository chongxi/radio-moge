"""Portable, versioned offline inference bundles; export never changes source weights."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import tempfile

import torch

from .checkpoint import atomic_json, atomic_save
from .decoder import MoGeDecoder
from .deployment import PINS, SOURCES, export_adapter
from .upstream import source_path


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_bundle(directory, *, verify=True):
    root = Path(directory).resolve()
    manifest = json.loads((root / "bundle.json").read_text())
    if manifest.get("format") != "radio-moge-pointcloud" or manifest.get("format_version") != 1:
        raise ValueError("Unsupported point-cloud bundle format")
    if manifest.get("revisions") != PINS:
        raise ValueError("Bundle upstream revisions differ from the installed runtime")
    required = {"adapter.pt", "decoder.pt", "radio.pth.tar", "config.json"}
    if not required <= manifest["files"].keys():
        raise ValueError("Incomplete bundle manifest")
    for name, info in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Bundle path escapes its directory: {name}")
        if not path.is_file() or path.stat().st_size != info["bytes"]:
            raise ValueError(f"Missing or truncated bundle file: {name}")
        if verify and sha256(path) != info["sha256"]:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    return root, manifest


def export_bundle(checkpoint, output, *, radio_checkpoint, moge_checkpoint,
                  radio_source, moge_source, notices=None):
    """Export adapter + frozen decoder + original RADIO release and standalone sources.

    The parent training checkpoint, optimizer, teacher encoder and dataset paths
    are not part of the runtime bundle. Sources include independent .git metadata
    so the upstream pin checks work after moving the directory to another machine.
    """
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_paths = {"RADIO": radio_source, "MoGe": moge_source}
    for name, path in source_paths.items():
        source_path(name, SOURCES[name][1], path)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as tmp:
        staging = Path(tmp) / "bundle"
        metadata = export_adapter(checkpoint, staging)
        teacher = torch.load(moge_checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if teacher["model_config"] != metadata["model_config"]:
            raise ValueError("Teacher decoder configuration differs from adapter metadata")
        decoder = MoGeDecoder.from_config(metadata["model_config"], moge_source)
        keys = set(decoder.state_dict())
        selected = {k: v for k, v in teacher["model"].items() if k in keys}
        decoder.load_state_dict(selected, strict=True)
        atomic_save(dict(format="radio-moge-decoder", format_version=1, revisions=PINS,
                         model_config=metadata["model_config"], model=selected),
                    staging / "decoder.pt")
        del teacher, decoder, selected
        shutil.copyfile(radio_checkpoint, staging / "radio.pth.tar")
        (staging / "sources").mkdir()
        for name, path in source_paths.items():
            destination = staging / "sources" / name
            revision = SOURCES[name][1]
            # The source HEAD was checked above. file:// permits a shallow local clone
            # without alternates or hardlinks back to an external checkout.
            # file:// clone starts upload-pack in a subprocess; scope the shared-
            # volume ownership exception to that process and this one source.
            upload_pack = shlex.join(["git", "-c",
                                     f"safe.directory={Path(path).resolve() / '.git'}",
                                     "upload-pack"])
            subprocess.run(["git", "-c", "advice.detachedHead=false", "clone", "--quiet",
                            "--depth=1", "--no-hardlinks", "--upload-pack", upload_pack,
                            Path(path).resolve().as_uri(), str(destination)], check=True)
            source_path(name, revision, destination)
            subprocess.run(["git", "-c", f"safe.directory={destination}",
                            "-C", str(destination), "remote", "set-url", "origin",
                            SOURCES[name][0]], check=True)
        if notices:
            shutil.copyfile(notices, staging / "THIRD_PARTY.md")
        files = {str(p.relative_to(staging)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                 for p in sorted(staging.rglob("*")) if p.is_file()}
        manifest = dict(
            format="radio-moge-pointcloud", format_version=1, runtime_version="0.2.0",
            training_step=metadata["training_step"], revisions=PINS,
            checkpoint_sha256=sha256(checkpoint),
            original_radio_sha256=sha256(radio_checkpoint),
            original_moge_sha256=sha256(moge_checkpoint),
            precision=metadata["precision"], edge_threshold=.04, edge_kernel_size=3,
            coordinates="OpenCV camera: x right, y down, z forward",
            metric_scale="MoGe predicted metric scale; depth is optical-axis Z",
            files=files,
        )
        atomic_json(manifest, staging / "bundle.json")
        os.rename(staging, output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Export complete offline point-cloud weights")
    for name in ("checkpoint", "output", "radio-checkpoint", "moge-checkpoint",
                 "radio-source", "moge-source"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--notices")
    args = parser.parse_args()
    info = export_bundle(**vars(args))
    print(json.dumps({"output": args.output, "training_step": info["training_step"],
                      "files": len(info["files"]),
                      "bytes": sum(x["bytes"] for x in info["files"].values())}))


if __name__ == "__main__":
    main()
