# Point-cloud API 0.2.0 release verification

Release artifact directory:
`/shared/radio-moge/chongxi/pointcloud-api-step272646-20261108-01`.
The matching GPU copy is under `/home/buildhub/shared/radio-moge/chongxi/`.
See [the API guide](pointcloud-api.md) for installation, inputs, coordinates,
outputs, GPU integration and CLI usage.

The complete **RADIO-v4-H → adapter → MoGe-v2 decoder → depth-edge filtering →
3D point cloud** implementation is in `radio_moge.pipeline.PointCloudPipeline`.
The default edge threshold is 0.04; every frame retains the original camera
recovery, metric scaling, force-projection and normal head. The API returns
organized GPU tensors, with explicit compact GPU/CPU cloud and binary PLY exports.

## Current weights

The bundle uses final step **272646**, whose training checkpoint SHA256 is
`943534362e85d5336cde00d6277cb550c2bbc0b180b1253054f1609de50639c8`.

| File | Bytes | SHA256 |
|---|---:|---|
| `adapter.pt` | 886,997,495 | `825906324a423df201ad2a58d549ad7d3344a76bc5fdc6e9d7f9849ed2bbc1f2` |
| `decoder.pt` | 89,392,239 | `e497940f59e945083b80044e717ed4acfcff5cc6cc861b50b940f563a4b1c7ae` |
| `radio.pth.tar` | 1,681,320,675 | `bace44df72e750bc8555ea6979cc19d1a87e12ade89582edfe090513d5d6aab9` |

The manifest binds 270 files totaling 2,733,949,456 bytes, excluding the manifest
itself. It includes independent pinned RADIO/MoGe source checkouts and notices.
No optimizer, training dataset paths, PPD model, or MoGe encoder weights are
needed by the runtime. Manifest SHA256:
`4d82fb16f8aba9f2a68b187923caf9eee07d0e411daef8a90f039cd7d10ca906`.

## Installed-package checks

The wheel was installed with its dependencies into a **new Python 3.11 virtual
environment**, with no system site packages or editable training installation.
Validation ran with `python -I -B`, from `/tmp`, with an empty model cache and
network connections explicitly rejected. The loaded module came from the new
environment's `site-packages`; all source and weights came from the bundle.

`scripts/verify_pointcloud_package.py` passed all of these checks:

- Every exported adapter tensor exactly matches the original checkpoint.
- All **32 saved cases**, comprising 16 scenes at 640×480 and 1024×768, have
  exactly matching depth, normalized intrinsics, base masks, filtered masks and
  RGB colors against the accepted saved predictions/filter artifacts.
- Retained XYZ matches the reference float64 camera unprojection within
  **1.768016e-6** predicted metric units; the difference is FP32 arithmetic.
- Pixel intrinsics reproject the cloud to the original integer pixel centers.
- PIL, NumPy, CPU tensor and pre-normalized CUDA tensor entry points agree.
- Passing shared RADIO features works with no backbone attached; disabling the
  filter restores the base mask without changing depth.
- PLY readback reproduces the selected XYZ/RGB exactly. The installed CLI also
  produced a byte-identical PLY (4,434,694 bytes,
  SHA256 `76fa56d84ff101d066193d33eb853e347408d58845d5d2141cd57d7d41e71ea1`).
- `uv pip check` accepts all 47 installed distributions.

Local CPU tests of the API contracts, projection/axis conventions, edge-mask
selection, empty clouds, RGB formats, bundle checksum failures, decoder-only
loading and the existing upstream/deployment paths also passed: **32 tests**.
Ruff and `git diff --check` passed. The existing viewer remained running; no
training, deployment replacement, or additional optimizer step was performed.

The GPU verification report is `validation-01/report.json`, SHA256
`117e11f7c080ce62beb6ee889f64903eca67028239e7f96ff52c991a88700b5e`.

## Installed API latency

NVIDIA H200, PyTorch 2.8.0+cu128, 640×480, batch one, BF16 RADIO/adapter and
FP32 decoder/camera/filter/XYZ. Each profile contains 64 synchronized wall-clock
measurements with two PyTorch CPU threads. Profiles cover all 16 scenes in four
rounds with alternating profile order,
after warmup. Both start from decoded CPU uint8 RGB, include normalization,
upload, all model stages, per-frame camera recovery and filtering, and exclude
file decoding, model loading, rendering and PLY writing.

| Output | Mean | Median | P95 |
|---|---:|---:|---:|
| Organized GPU depth/XYZ/RGB/masks | 24.59 ms | 24.31 ms | 26.64 ms |
| Compact CPU XYZ/RGB | 25.20 ms | 25.02 ms | 28.13 ms |

Loading plus default hash verification took 17.99 seconds in this new
environment; it happens once. Peak allocated GPU memory across **both tested
resolutions** was 5,127,683,584 bytes. Timings are a packaging regression check,
not a guarantee for another GPU or application. These checks preserve the
accepted output; they are not a new held-out depth-accuracy evaluation.
