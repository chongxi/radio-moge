"""CPU-only audit of a completed head and fixed-image viewer; never runs inference."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from radio_moge.compare import fingerprint
from radio_moge.export import GeometryHead


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(prediction, target):
    valid = np.isfinite(target) & (target > 0)
    assert valid.any()
    p, t = prediction[valid].astype("float64"), target[valid].astype("float64")
    bad = ~np.isfinite(p) | (p <= 0)
    p[bad] = 0
    ratio = np.maximum(np.maximum(p, 1e-8) / t, t / np.maximum(p, 1e-8))
    return dict(abs_rel=float(np.mean(np.abs(p - t) / t)),
                rmse=float(np.sqrt(np.mean((p - t) ** 2))),
                delta1=float(np.mean(ratio < 1.25)), invalid_fraction=float(bad.mean()))


def planes(root, manifest, sample):
    return np.frombuffer(gzip.decompress((root / sample["depth"]).read_bytes()),
                         dtype="<f4").reshape(len(manifest["channels"]),
                                              sample["height"], sample["width"])


def audit(args):
    torch.set_num_threads(1)
    root = args.root.resolve()
    viewer = root / "selected/heldout-20"
    assert (root / "export-retry-exit-code.txt").read_text().strip() == "0"
    assert sha(args.manifest) == args.manifest_sha256
    records = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    assert len(records) == 20
    manifest = json.loads((viewer / "manifest.json").read_text())
    old = json.loads((args.previous / "manifest.json").read_text())
    geometry = json.loads((viewer / "geometry/manifest.json").read_text())
    ids = [r["id"] for r in records]
    assert ids == [s["id"] for s in manifest["samples"]]
    assert ids == [s["id"] for s in old["samples"]]
    assert ids == [s["id"] for s in geometry["samples"]]
    assert manifest["channels"] == old["channels"] == [
        "before", "after", "teacher", "ground_truth"]
    assert manifest["dtype"] == "float32-le"
    assert manifest["run"]["checkpoint_role"] == "selected"
    assert manifest["run"]["steps"] == 83494
    assert geometry["depth_manifest_sha256"] == sha(viewer / "manifest.json")
    state = torch.load(root / "selected.pt", map_location="cpu", weights_only=True, mmap=True)
    head_path = root / "selected/head.pt"
    head = torch.load(head_path, map_location="cpu", weights_only=True, mmap=True)
    assert state["step"] == head["training_step"] == 83494
    assert head["format_version"] == 1
    for key in ["adapter_config", "model_config", "revisions"]:
        assert head[key] == state[key]
    assert head["models"] == state["config"]["models"]
    assert all(k.startswith(("adapter.", "decoder.")) for k in head["state_dict"])
    assert not any("encoder" in k or "backbone" in k for k in head["state_dict"])
    assert any(k.startswith("decoder.") for k in head["state_dict"])
    for key, value in state["adapter"].items():
        assert torch.equal(value, head["state_dict"]["adapter." + key]), key
    assert all(torch.isfinite(v).all() for v in head["state_dict"].values())
    # Strict construction/load checks every decoder and adapter key/shape on CPU.
    loaded = GeometryHead.load(head_path, device="cpu", moge_source=head["models"].get("moge_source"))
    assert all(p.device.type == "cpu" for p in loaded.parameters())
    del loaded
    adapter_sha = fingerprint(state["adapter"])
    assert manifest["run"]["checkpoint_sha256"] == adapter_sha
    assert geometry["checkpoint_sha256"] == adapter_sha
    assert manifest["run"]["initial_adapter_sha256"] == old["run"]["checkpoint_sha256"]
    assert manifest["run"]["revisions"] == state["revisions"]
    max_metric_error = 0.0
    max_roundtrip_error = 0.0
    rows = []
    for record, sample, previous, geom in zip(records, manifest["samples"], old["samples"],
                                               geometry["samples"], strict=True):
        assert not sample["failures"]
        assert sample["scene"] == previous["scene"] == record["scene"]
        assert sha(viewer / sample["rgb"]) == sha(args.previous / previous["rgb"])
        current = planes(viewer, manifest, sample)
        before = planes(args.previous, old, previous)
        for new_index, old_index in [(0, 1), (2, 2), (3, 3)]:
            assert np.array_equal(current[new_index], before[old_index], equal_nan=True)
        sensor = np.load((args.manifest.parent / record["depth"]).resolve())
        sensor = sensor * record.get("depth_scale", 1.0)
        assert np.array_equal(current[3], sensor, equal_nan=True)
        values = {}
        for index, channel in enumerate(manifest["channels"][:3]):
            values[channel] = metrics(current[index], current[3])
            for key, value in values[channel].items():
                error = abs(value - sample["metrics"][channel][key])
                max_metric_error = max(max_metric_error, error)
                assert error < 1e-5, (sample["id"], channel, key, error)
        for channel in ["after", "teacher"]:
            item = geom[channel]
            depth = current[manifest["channels"].index(channel)]
            k = np.asarray(item["intrinsics"], dtype="float64")
            assert k.shape == (3, 3) and np.isfinite(k).all()
            assert k[0, 0] > 0 and k[1, 1] > 0
            assert np.array_equal(k[2], [0, 0, 1])
            mask = np.asarray(Image.open(viewer / item["mask"])) > 0
            assert mask.shape == depth.shape and int(mask.sum()) == item["valid_points"]
            assert mask.any() and np.all(np.isfinite(depth[mask]) & (depth[mask] > 0))
            y, x = np.nonzero(mask)
            uv = np.stack([(x + .5) / depth.shape[1], (y + .5) / depth.shape[0],
                           np.ones_like(x)], axis=-1)
            points = (uv @ np.linalg.inv(k).T) * depth[mask, None]
            assert np.isfinite(points).all()
            projected = points @ k.T
            projected /= projected[:, 2, None]
            error = float(np.max(np.abs(projected - uv)))
            max_roundtrip_error = max(max_roundtrip_error, error)
            assert error < 1e-12
            assert 0 <= item["projection_max_error_m"] < 1e-3
        rows.append(dict(id=sample["id"], sensor_metrics=values))
    summary = {}
    for channel in ["before", "after", "teacher"]:
        summary[channel] = {k: float(np.mean([r["sensor_metrics"][channel][k] for r in rows]))
                            for k in ["abs_rel", "rmse", "delta1", "invalid_fraction"]}
        assert manifest["summary"][channel]["failures"] == 0
        for key, value in summary[channel].items():
            assert manifest["summary"][channel][key + "_count"] == 20
            assert abs(value - manifest["summary"][channel][key]) < 1e-5
    inventory = []
    for path in sorted((root / "selected").rglob("*")):
        if path.is_file():
            assert not path.is_symlink()
            inventory.append(dict(path=str(path.relative_to(root)), bytes=path.stat().st_size,
                                  sha256=sha(path)))
    report = dict(passed=True, checked_at=time.time(), stage="export_verified",
                  checkpoint_role="selected", training_step=83494, count=20,
                  fixed_manifest_sha256=args.manifest_sha256, head_sha256=sha(head_path),
                  adapter_sha256=adapter_sha, head_tensor_count=len(head["state_dict"]),
                  adapter_tensors_exact=len(state["adapter"]), head_cpu_strict_load=True,
                  head_has_encoder_or_backbone=False, exact_rgb_teacher_sensor_parity=True,
                  exact_before_equals_previous_attention6000=True,
                  sensor_depth_matches_source_npy=True, summary=summary,
                  max_metric_recomputation_error=max_metric_error,
                  max_projection_roundtrip_error=max_roundtrip_error,
                  export_recorded_max_official_point_error_m=max(
                      s[c]["projection_max_error_m"] for s in geometry["samples"]
                      for c in ["after", "teacher"]),
                  geometry_policy=geometry["policy"], rows=rows, inventory=inventory,
                  limits="No inference repeated. Projection round trips are independently checked; "
                  "official-point parity was checked by the exporter. Fixed20 is a reused qualitative "
                  "set, not a blind test. Teacher agreement and sensor accuracy are distinct.")
    assert not args.report.exists(), "Preserve existing verification report"
    temporary = args.report.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, args.report)
    print(json.dumps({k: v for k, v in report.items() if k not in ["rows", "inventory"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    audit(parser.parse_args())
