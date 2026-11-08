# RADIO → MoGe → filtered point cloud: Python API

`radio-moge-inference` **0.2.0** packages the complete current chain:
**RADIO-v4-H → fitted feature adapter → frozen MoGe-v2 decoder → official-style
relative depth-edge mask → colored 3D points**. The supplied weights are the
Muon continuation's final **step 272646**, not the older step 232288 HF release.

## Install and call

Extract the release directory, then install its two wheels in your application's
Python environment (Python 3.11+ and `git` must be available):

```bash
python -m pip install /path/to/release/wheels/*.whl
```

This installs the runtime and the included `utils3d-moge` wheel built from commit
`62f09d58509485564e24d5d9f6aac9ee9ebc0c37`. Standard dependencies such as PyTorch
are downloaded by pip if absent. The model bundle itself is offline: no Hugging
Face login, source checkout, training repository, or weight download is needed
when loading it. An offline *Python dependency installation* additionally needs
your own package mirror/wheelhouse; CUDA/PyTorch are not packed into the weights.

```python
from PIL import Image
from radio_moge import PointCloudPipeline

# Load once at service/application startup. All weights are bundled here.
pipeline = PointCloudPipeline.from_pretrained("/path/to/release/weights", device="cuda")

result = pipeline(Image.open("frame.png"))
xyz, rgb = result.pointcloud(numpy=True)  # N×3 float32 XYZ, N×3 uint8 RGB
result.save_ply("frame.ply")             # filtered, colored, binary PLY
```

`pipeline(rgb_uint8)` also accepts an RGB NumPy array of shape **H×W×3**. For
OpenCV use `pipeline(bgr[..., ::-1])`. A torch input must be **3×H×W**, either
uint8 `[0,255]` or floating `[0,1]`. One call processes one frame; no batch axis.
PIL EXIF orientation is applied. Output retains the resulting input resolution
and field of view: **no automatic crop or resize to a training bucket**. RADIO
internally resizes to its supported patch grid, as in the verified pipeline.
Inputs exceeding RADIO's supported 2048-pixel grid are rejected.

## GPU integration and outputs

```python
result = pipeline(rgb_chw_cuda, validate=False)  # trusted float RGB in [0,1]
depth_gpu = result.depth
mask_gpu = result.mask                         # already excludes depth edges
xyz_gpu, rgb_gpu = result.pointcloud()          # compacts on the GPU
K_pixels = result.intrinsics

# Only if a consumer needs every plane on the CPU:
arrays = result.numpy()
```

By default `validate=True` rejects nonfinite/out-of-range floating inputs.
`validate=False` skips these value checks, avoiding their GPU-to-host
synchronization; dtype and shape checks remain. CPU uint8 normalization matches
the saved experiments exactly. Normalizing uint8 on CUDA can differ by a float
rounding bit, so bit-exact reproduction uses CPU uint8 or the same pre-normalized
float tensor. Neither input RGB nor shared features are modified.

| Result field | Shape / type | Meaning |
|---|---|---|
| `depth` | H×W float32 | Optical-axis Z, with MoGe's predicted metric scale |
| `points` | H×W×3 float32 | Organized camera XYZ; use `mask` to select valid points |
| `mask` | H×W bool | MoGe validity, finite positive depth, and depth-edge filtering |
| `base_mask` | H×W bool | Validity before edge filtering |
| `rgb` | H×W×3 uint8 | RGB colors aligned with the point map |
| `intrinsics` | 3×3 float32 | Pixel K for integer pixel centers `(u,v)` starting at 0 |
| `intrinsics_normalized` | 3×3 float32 | Upstream K for `((u+.5)/W, (v+.5)/H)` |
| `normal` | H×W×3 float32 | MoGe unit normals in camera coordinates |

All fields stay on the model device until an explicit CPU conversion. The
organized depth/XYZ planes keep their original values at excluded pixels;
filtering only changes the mask. `pointcloud()` and `save_ply()` return/write
only surviving points. GPU boolean compaction has a synchronization cost for
its data-dependent length; use organized tensors plus mask when your consumer
accepts them. There is no point-cloud CPU round trip in the default forward.

Coordinates default to **OpenCV camera axes: X right, Y down, Z forward**.
For OpenGL use `result.pointcloud(coordinate_system="opengl", numpy=True)` or
`result.save_ply("frame.ply", coordinate_system="opengl")`; this flips Y and Z.
Neither form estimates a world/camera pose. Pixel K applies to OpenCV axes:
`X=(u-cx)*Z/fx`, `Y=(v-cy)*Z/fy`. The principal point conversion is
`cx_pixel=cx_normalized*W-0.5` (similarly Y), consistent with the point map.
Scale is the monocular model's metric estimate, not sensor-measured distance.

Default edges use a **3×3 neighborhood and relative threshold 0.04**, matching
the accepted point-cloud comparison and the pinned MoGe utility. To adjust:

```python
pipeline = PointCloudPipeline.from_pretrained("weights", edge_threshold=0.02)
unfiltered = PointCloudPipeline.from_pretrained("weights", edge_threshold=None)
```

Prefer selecting a threshold at startup; do not repeatedly load the model.
The filter removes depth-discontinuity samples and does not alter or smooth
retained depth values. Camera focal/shift recovery and metric projection still
run **per frame**. The upstream recovery currently uses a small CPU solve;
this is not an entirely CUDA-only camera solver.

## Reuse an existing RADIO backbone

```python
from radio_moge.upstream import extract_radio, radio_input

# Attach an existing frozen, eval-mode RADIO-v4-H on the same CUDA device.
pipeline = PointCloudPipeline.from_pretrained("weights", device="cuda", radio=radio)

# Or supply the SAME frame's final map, both summary tokens, and four taps:
head = PointCloudPipeline.from_pretrained("weights", device="cuda", load_backbone=False)
with head.model._autocast():
    features = extract_radio(radio, radio_input(image_chw[None], radio), head.model.indices)
result = head(image_chw, features=features)
```

The trained intermediate layers are `[7,15,23,31]`; spatial and summary widths
are 1280 and 2560. Features must come from the same normalized RGB view, RADIO
checkpoint, preprocessing, and forward pass. `features=` bypasses RADIO entirely.
Normal callers should use `pipeline(rgb)` so these details stay internal.

## CLI and other libraries

```bash
radio-moge-pointcloud --weights /path/to/release/weights \
  --image frame.png --output frame.ply --device cuda
```

For Open3D (optional, not a runtime dependency):

```python
import open3d as o3d
xyz, rgb = result.pointcloud(numpy=True)
cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(xyz)
cloud.colors = o3d.utility.Vector3dVector(rgb.astype("float64") / 255)
```

## Weights, reproducibility, and performance

`weights/` contains `adapter.pt` (FP32 inference tensors, no optimizer),
`decoder.pt` (only MoGe decoder/heads), the original `radio.pth.tar`, two
standalone pinned upstream source trees, and `bundle.json` with file hashes.
No MoGe teacher encoder is instantiated. PPD is not used. `verify=True`
checks every manifest file once at load; `verify=False` skips hashing for a
previously checked, unchanged bundle. Model loading and verification are not
part of frame latency. Loading never resets or updates shared upstream sources.

Original training checkpoint SHA256:
`943534362e85d5336cde00d6277cb550c2bbc0b180b1253054f1609de50639c8`.
Weights use FP32 storage, BF16 autocast for RADIO/adapter, and FP32 MoGe decoding,
camera recovery, depth-edge filtering, and XYZ. H200 CUDA is the evaluated
deployment target; CPU is available for debugging, with different speed/numerics.

The underlying pipeline's prior H200 640×480 batch-1 measurements were **23.9 ms**
for organized GPU output and **25.6 ms** for compact CPU XYZ/RGB, including RGB
normalization/upload and per-frame camera recovery, excluding file decoding,
model loading, rendering and PLY writing. These are not latency promises for
other devices; release validation records installed-package measurements separately.

The release keeps the accepted visualization behavior. Its final primary feature
cosines improved slightly, while independent LiDAR AbsRel was 1.19% worse than
the immediate training parent. Packaging and parity checks are not a new depth
accuracy evaluation. Upstream sources/weights retain their respective licenses;
see `THIRD_PARTY.md` and the included source notices.

## Build from this repository

```bash
python scripts/build_pointcloud_package.py --output /new/path/package-build
python -m radio_moge.bundle --checkpoint checkpoint-step272646.pt \
  --output /new/path/release/weights \
  --radio-checkpoint c-radio_v4-h_half.pth.tar --moge-checkpoint model.pt \
  --radio-source vendor/RADIO --moge-source vendor/MoGe --notices THIRD_PARTY.md
```

Build the pinned `utils3d-moge` commit as a wheel alongside the runtime wheel.
The exporter refuses to replace an existing bundle, preserves the source
checkpoint, and publishes the finished directory atomically. Old
`radio_moge.deployment.RadioMoGe` calls remain available with their original
unfiltered tensor/dictionary API.
