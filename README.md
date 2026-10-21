# RADIO → MoGe-2 distillation

Train a geometry adapter on **frozen C-RADIOv4-H** and reuse MoGe-2's frozen
decoder. The MoGe encoder is a teacher during training only. Deployment accepts
precomputed RADIO features and does not own or run either backbone.

```text
Training:   RGB ── frozen RADIO ── adapter ── frozen MoGe decoder ── geometry
             └── frozen MoGe encoder ─────── feature/geometry targets

Deployment: existing RADIO pass ─┬─ existing perception adapters
                                └─ our adapter + MoGe decoder
```

This is an experimental codebase with an initial 400-step NYUv2 sanity run.
The default teacher is
[`Ruicheng/moge-2-vitl-normal`](https://huggingface.co/Ruicheng/moge-2-vitl-normal),
the largest released MoGe-2 checkpoint (**331M parameters**). It uses the largest
MoGe-2 backbone, DINOv2 ViT-L/14, and includes the normal-output decoder head.
The non-normal ViT-L model is 326M; the normal ViT-B and ViT-S models are 104M
and 35M respectively ([official model table](https://github.com/microsoft/MoGe#-pretrained-models)).
The encoder interface remains a 1024-channel spatial map and 1024-dimensional
class token. The frozen decoder/export includes the normal head; the initial
NYUv2 pilot still trains on encoder features. Its pinned teacher revision is
`cb0e8bbd6b1e243589717c78e750b1ba4c093acf`, recorded in each run configuration.
The initial run reached depth AbsRel 0.122 versus the teacher's 0.095 on the custom
290-image held-out split below. Training took 3m30s on one H200 with 4.07 GiB peak
allocated memory. These are a short feature-distillation pilot's results, not
teacher parity or an official NYUv2 benchmark score.
The current runner deliberately supports **one process / one GPU**.

## Setup

On the GPU machine, in this repository (Python 3.11 recommended):

```bash
uv sync --locked --extra dev --extra upstream
.venv/bin/python scripts/setup_upstream.py
```

The lock installs PyTorch 2.8 with its CUDA dependencies on Linux. The H200 needs
a compatible driver. Public model weights download on the first real model load;
an HF account is not required for these public releases. Source and model revisions
are pinned in `src/radio_moge/config.py` and recorded in every run. Upstream source
lives in ignored `vendor/`; checkpoint downloads use the standard HF cache.

For lightweight **CPU development/tests** instead of CUDA dependencies:

```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e '.[dev,upstream,data]'
.venv/bin/python scripts/setup_upstream.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/pytest -q
.venv/bin/ruff check src scripts tests
```

The CPU tests use small random models, including the actual upstream encoder,
decoder, RADIO intermediate extraction and MoGe postprocessing. They do not
download pretrained checkpoints or establish learned geometry quality.

The GitHub Actions configuration is provided as `ci/github-actions-tests.yml`.
BuildHub's repository credential cannot create workflow files, so this template
is inactive. To enable CI, use your own GitHub account to place it at
`.github/workflows/tests.yml`.

## Data

### First indoor sanity experiment: NYU Depth V2

The [original labeled release](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)
contains 1,449 RGB/depth pairs from 464 indoor scenes. Prepare it with:

```bash
uv pip install --python .venv/bin/python -e '.[data]'
.venv/bin/python scripts/prepare_nyuv2.py
```

This downloads the original ~2.8 GiB MAT file and extracts one frame at a time.
Allow approximately 6 GiB of disk space including the source. Data, manifests,
source checksum, scene counts and a training-only preview go under
`/shared/radio-moge/datasets/nyuv2-labeled-s42/`. The completed directory is
published atomically, and existing datasets are never overwritten. Use
`--source`, `--output`, `--seed`, or `--test-fraction` to prepare a different copy.

The **80/20 split groups whole scenes** (seed 42), removes identical RGB frames,
and conservatively merges scene suffixes. It targets the image ratio while
keeping each scene entirely in one split. Actual counts are in
`dataset_info.json`; `overfit.jsonl` contains 32 training images only.

The prepared seed-42 split has **1,159 training images / 375 scenes** and
**290 test images / 89 scenes** (79.99% / 20.01%). No duplicate RGB frames were
found, and paths, scene groups and decoded RGB hashes are disjoint across splits.

Training uses RGB and MoGe teacher targets. The extracted depth is the aligned,
measured **`rawDepths` in meters**, with missing/nonpositive/nonfinite values
masked out. Final evaluation uses all valid pixels at full-frame resolution,
without scale alignment or the standard NYUv2 benchmark crop. This is a custom
sanity split, **not** a comparable official NYUv2 benchmark result. Teacher
pretraining may include these scenes; the split tests the new adapter's transfer.
Use `--depth-key depths` and a new output directory if you explicitly want
inpainted depth targets instead.

On the GPU, first check learning on the 32-image training subset, then start a
fresh fixed-budget run on all 80% (do not initialize it from the overfit run):

```bash
.venv/bin/python -m radio_moge.train --config configs/nyuv2-overfit.yaml
.venv/bin/python -m radio_moge.train --config configs/nyuv2.yaml
.venv/bin/python -m radio_moge.evaluate \
  --checkpoint /shared/radio-moge/chongxi/nyuv2-mlp-s42/latest.pt \
  --manifest /shared/radio-moge/datasets/nyuv2-labeled-s42/test.jsonl \
  --output /shared/radio-moge/chongxi/nyuv2-mlp-s42/test-final
```

The main pilot uses 400 optimizer steps, effective batch 32, a 480×640 RADIO
view (1,200 tokens), and feature distillation only. **`val_manifest: null`
disables checkpoint selection and all periodic evaluation.** Only the final
checkpoint is evaluated on the full 20%; no `--limit` means all test images.
Report teacher–student agreement plus **both models' errors against measured
depth**, failures, scale, FOV and boundary examples. If this result motivates
tuning, create an internal validation split inside the 80% before comparing
architectures or selecting checkpoints; do not repeatedly tune on the 20% test.

### Other image collections

Use RGB images, with separate scene/sequence groups for training and validation.
No ground-truth depth is needed for distillation. With one folder per scene:

```bash
.venv/bin/python scripts/make_manifests.py /shared/my-images --output data
```

This writes `train.jsonl`, `val.jsonl`, and a 32-image `overfit.jsonl` subset.
It removes byte-identical duplicates and splits whole parent-directory groups.
Choose grouping manually if one scene spans multiple folders; near-duplicates
still require dataset curation. Existing manifests are never intentionally overwritten.

You can also write manifests directly. Paths resolve relative to the manifest:

```json
{"image": "/shared/images/scene01/001.jpg", "group": "scene01"}
```

For independent metric-depth evaluation, add an aligned depth PNG or `.npy` file:

```json
{"image": "rgb/001.jpg", "depth": "depth/001.png", "depth_scale": 0.001, "group": "scene01"}
```

Depth values are converted to meters using `depth_scale`; zero/negative/nonfinite
values are invalid. Depth must match the EXIF-oriented RGB image. The evaluator
resizes depth with nearest-neighbor sampling. There is no ground-truth camera
calibration schema in this first version; FOV is compared with the MoGe teacher.

## Train

Edit manifests and `output_dir` in the configs first. Run paths are interpreted
relative to the repository working directory. Use a **new output directory for
each experiment**. Defaults put artifacts under `/shared/radio-moge/chongxi/`.

1. Wiring/overfit check on 32 images:

   ```bash
   .venv/bin/python -m radio_moge.train --config configs/overfit.yaml
   ```

2. Profile the first 200 optimizer steps of the MLP pilot, then resume its exact schedule:

   ```bash
   .venv/bin/python -m radio_moge.train --config configs/mlp.yaml --stop-after 200
   .venv/bin/python -m radio_moge.train --config configs/mlp.yaml \
     --resume /shared/radio-moge/chongxi/mlp-pilot/latest.pt
   ```

3. Compare intermediate fusion with the same training data and feature objectives:

   ```bash
   .venv/bin/python -m radio_moge.train --config configs/fusion.yaml
   ```

4. Refine the chosen MLP through the frozen geometry decoder:

   ```bash
   .venv/bin/python -m radio_moge.train --config configs/refine.yaml \
     --init /shared/radio-moge/chongxi/mlp-pilot/best.pt
   ```

`best.pt` exists only after a validation with successful focal recovery on all
sampled images. Use `latest.pt` for inspection if no best checkpoint exists.
To refine fusion, create a config extending `fusion.yaml` with the refinement
learning rate/losses and a new output directory; `--init` requires the same adapter.

For a persistent run use `tmux`, with logs saved outside the run directory:

```bash
tmux new -s radio-moge
.venv/bin/python -m radio_moge.train --config configs/mlp.yaml 2>&1 | tee mlp-console.log
# Detach with Ctrl-b then d; tmux attach -t radio-moge reconnects.
```

The scripts do not start, stop or delete BuildHub machines. Save the final
artifacts before stopping the machine. A SIGINT/SIGTERM finishes the current
optimizer step and saves `latest.pt`; forced termination falls back to the most
recent periodic checkpoint.

### What is trained

| Component | Weights updated | Autograd |
|---|---|---|
| RADIO | No | Disabled |
| MoGe teacher encoder | No | Disabled |
| New adapter | Yes | Enabled |
| MoGe decoder | No | Enabled for student geometry refinement |

- `linear`: one spatial projection and one summary projection; set `adapter.kind: linear`.
- `mlp`: spatial `1280 → 2048 → 1024`, separate summary MLP.
- `fusion`: project and average RADIO blocks 8/16/24/32, then spatial MLP;
  summary branch still uses the final RADIO summary.
- Input/output channel dimensions are read from the loaded models, not guessed.
- Teacher targets are the **fused projected spatial map and final CLS token from
  the trained MoGe checkpoint**, not raw DINOv2 features.
- Teacher spatial/class losses use fixed per-channel RMS statistics fitted to
  training images. A small cosine term supplements absolute feature matching.
- Geometry refinement adds robust point-map, mask, log-scale and boundary-gradient
  losses. Normal loss is supported if a compatible normal teacher is configured.
- Encoders/adapters use BF16 autocast; decoder and losses use FP32. Frozen model
  parameters are never passed to the optimizer.

Both branches see the same crop and horizontal flip. RADIO runs at the configured
bucket resolution; MoGe resizes that view to `14h × 14w` for RADIO's `h × w` token
grid. Validation preserves the image aspect ratio and uses a supported RADIO grid
near the requested token count. MoGe's decoder receives the output image's aspect
ratio. For deployment, set buckets and `models.window_size` to match your existing
RADIO pass; change these together for ablations. No second RADIO pass is required.

### Outputs and resume

Runs contain `config.json`, `provenance.json`, `train.jsonl`, TensorBoard events,
`initial.pt`, `latest.pt`, optional `best.pt`, and validation directories with per-image metrics
and PNG comparisons. PNG columns are RGB, teacher depth, student depth, relative
depth error. Teacher/student panels share a depth range; magenta marks invalid pixels.

Checkpoints contain adapter weights, fixed feature statistics, optimizer, LR
scheduler, Python/NumPy/Torch RNG, optimizer step and consumed microbatch cursor.
The sampler and crop seeds are independent of DataLoader prefetch. Resume checks
configuration and manifest/file metadata; keep image contents immutable.
`--resume` continues the same run; `--init` starts a new stage with a fresh
optimizer and schedule. Checkpoint writes use a temporary file and atomic rename.

```bash
.venv/bin/tensorboard --logdir /shared/radio-moge/chongxi
```

## Evaluate

```bash
.venv/bin/python -m radio_moge.evaluate \
  --checkpoint /shared/radio-moge/chongxi/mlp-pilot/best.pt \
  --manifest data/val.jsonl \
  --output /shared/radio-moge/chongxi/mlp-eval-1200 \
  --num-tokens 1200
```

Repeat with larger token counts and the fusion checkpoint for resolution/design
ablations. `--limit N` provides a small diagnostic subset; default standalone
evaluation uses all manifest images. `--device cpu` is available for small tests,
but the full backbones should run on the GPU machine.

Metrics include feature errors, raw point/boundary errors, mask IoU, log metric
scale error, teacher-relative depth AbsRel/RMSE/δ1, FOV error, invalid prediction
fraction, and optional teacher/student depth metrics against ground truth.
Metric-depth scores do **not** align away scale errors. Student masks cannot hide
bad depth pixels. Metrics average per-image scores and report their sample counts.
Focal-recovery failures are reported explicitly and disqualify best-checkpoint
selection. The comparison teacher uses the same target token grid as the student.

Timing reports adapter + decoder separately from CPU/GPU postprocessing; it
excludes the shared RADIO forward and teacher. These are diagnostic timings, not
a standalone RADIO-vs-MoGe benchmark. Training logs report optimizer-step time,
images/second and peak allocated GPU memory.

### Depth before and after training

Every new training stage saves its actual starting weights to `initial.pt`.
Render RGB, initial adapter depth, trained adapter depth, and MoGe-2 depth:

```bash
.venv/bin/python -m radio_moge.compare \
  --checkpoint /shared/radio-moge/chongxi/mlp-pilot/latest.pt \
  --initial /shared/radio-moge/chongxi/mlp-pilot/initial.pt \
  --manifest data/val.jsonl \
  --output /shared/radio-moge/chongxi/mlp-pilot/before-after \
  --count 8
```

This runs the **frozen MoGe decoder and official focal/metric-scale postprocessing
for all three branches**, reusing one RADIO pass per image for both adapters.
It selects the first image of distinct scene types, then distinct scenes, in
manifest order, before inference. Each comparison uses one shared depth color
range (the teacher's 2nd–98th percentiles, labeled in meters); values outside the
range are clipped. No scale alignment is performed. Predicted validity masks
do not hide errors; nonpositive/nonfinite depths are magenta. Raw decoder point-z
is saved separately and never substituted for postprocessed metric depth.

Outputs include labeled PNGs, `overview.png`, full-resolution `.npz` depth arrays,
per-image metrics in `results.json`, selection metadata, and baseline provenance.
These selected examples are diagnostics, not a replacement for the full held-out
evaluation. Comparing existing checkpoints does not retrain or select weights.

For the first legacy run, which did not save `initial.pt`, replace `--initial`
with `--reconstruct-from-log /path/to/pilot/train.jsonl`. This rebuilds the seeded
initialization using the pinned models and **requires reproducing all first-step
feature losses on the original training batches** before rendering. It aborts if
the losses fail to match. This mode supports feature-only training; it is not a
general way to recover arbitrary checkpoints. The first run's replay matched
all four logged losses exactly with PyTorch 2.8.0+cu128.

## Export and reuse your existing RADIO features

### Interactive held-out viewer

The viewer lets you browse every held-out image, filter by scene, compare depth
maps side by side or with a swipe slider, inspect depth in meters under the
cursor, and compare the trained head's error against MoGe-2 or measured depth.
The default **3D point clouds** tab compares the trained head with MoGe-2 using
linked orbit, zoom and pan controls. Switch between RGB, depth coloring, and a
two-color overlay (orange: trained head, teal: MoGe-2). Density and point size are
adjustable; each visible cloud can be downloaded as a binary PLY file.

Each model uses its **own recovered intrinsics**, official metric depth, and
predicted validity mask. No scale alignment, independent centering or geometric
registration is applied. The optional shared MoGe-2 camera mode reprojects both
depth maps with the teacher's intrinsics to isolate depth differences; it is a
diagnostic view. Coordinates in the browser and PLY are x right, y up, z backward,
in meters. PLY exports use the currently selected projection and density.
Single-view clouds contain observed surfaces only, not unseen room geometry.

All depth panels share one range. The top summary always covers the full test
set; the table below the images covers the selected frame.

For new runs, pass the saved `initial.pt` directly to export the full held-out
manifest once on the GPU. The exporter checks that it is step zero from the
same training stage, and checks test separation from both training and validation:

```bash
.venv/bin/python -m radio_moge.viewer \
  --checkpoint /path/to/train/best.pt --initial /path/to/train/initial.pt \
  --manifest /path/to/dataset/test.jsonl --output /path/to/heldout-viewer
```

This computes full-test depth metrics and the viewable predictions together;
a separate evaluation pass is unnecessary. Camera metadata and validity masks
are saved in `geometry/`, with a check that depth unprojection matches the
official MoGe point map. The manifest records the selected
checkpoint step, training budget, and internal validation size. For legacy runs
without `initial.pt`, first use `radio_moge.compare` to create verified
`initial-adapter.pt`, then export:

```bash
.venv/bin/python -m radio_moge.viewer \
  --checkpoint /path/to/pilot/latest.pt \
  --initial-adapter /path/to/before-after/initial-adapter.pt \
  --manifest /path/to/dataset/test.jsonl \
  --original-metrics /path/to/test-final/per_image.json \
  --output /shared/radio-moge/chongxi/heldout-viewer
```

The optional original metrics argument checks that teacher/trained scores are
unchanged. The exporter checks train/test separation, verifies the initial and
trained adapter fingerprints, and writes into a temporary directory before
publishing the completed export. RGB images, compressed float32 depth planes,
and metrics stay outside git. Allow about 1 GiB for all 290 NYUv2 frames.

Copy the exported directory to the workspace and serve it with the lightweight
server (no GPU or PyTorch needed to browse):

```bash
python3 scripts/serve_viewer.py /shared/radio-moge/chongxi/heldout-viewer --port 7860
```

Use this server: it supplies the HTTP `Content-Encoding: gzip` header needed for
the browser to decode the lossless depth planes. It serves only the generated
directory and disables directory listings. BuildHub's workspace preview URL
provides the authenticated browser entry point. Source HTML, CSS and JavaScript
are in `web/`; there is no frontend build or external asset dependency.

An optional Playwright check is in `scripts/check_viewer.cjs`. With Playwright
and Chromium installed, run `node scripts/check_viewer.cjs http://localhost:7860`.
It checks navigation, filtering, depth decoding, swipe controls, pixel values,
metrics, empty search results, and the mobile layout.

`scripts/check_clouds.cjs` additionally verifies actual WebGL rendering, linked
cameras, overlay, projection switching, PLY downloads, scene navigation, and
mobile layout. `node scripts/test_cloud_math.mjs` checks pixel-center projection,
axis conversion, masking and invalid depth handling.

For an existing export that predates camera metadata, add it without replacing
the saved depth maps or recomputing test scores:

```bash
.venv/bin/python -m radio_moge.pointcloud \
  --checkpoint /path/to/train/best.pt --manifest /path/to/dataset/test.jsonl \
  --viewer /path/to/heldout-viewer
```

This repeat inference pass verifies the checkpoint fingerprint, held-out split,
image order, and parity with every existing depth plane before publishing camera
metadata. It does not train or select a checkpoint. Copy the updated `web/`
contents into the viewer to enable the UI. Three.js 0.180.0 is bundled locally
with its MIT license; viewing does not contact a third-party CDN.

### Longer runs with Weights & Biases

Install the optional tracker with `uv pip install --python .venv/bin/python -e '.[tracking]'`.
Authenticate through the W&B SDK's normal environment or credential store; never
put an API key in YAML, source code, or a run directory.

Create an internal validation split from the original training manifest:

```bash
.venv/bin/python scripts/make_validation_split.py \
  --train-manifest /shared/radio-moge/datasets/nyuv2-labeled-s42/train.jsonl \
  --output /shared/radio-moge/datasets/nyuv2-internal-val-s43
.venv/bin/python -m radio_moge.train --config configs/nyuv2-long.yaml
```

This reserves 116 images in 39 scenes for validation, leaving 1,043 images in
336 training scenes. The original 290-image test split is unchanged. Start
fresh: the pilot checkpoint has already seen the newly reserved validation
scenes, so initializing from it would contaminate validation.

The 4,000-step configuration trains only the adapter, with physical batch 16
and accumulation 2 (effective batch 32). W&B logs losses, learning rate, gradient
norm, GPU memory, throughput, decoded depth metrics, and four fixed validation
image comparisons every 250 steps, including a step-zero baseline. All 116
validation frames are evaluated through the frozen MoGe decoder and camera
postprocessing. Checkpoint selection uses validation teacher-depth error.
Feature loss and decoded geometry can move differently; inspect both curves.

The trainer prints the run URL and saves it in `wandb-run.json`. `--resume`
requires the same config/manifests and run directory, and reconnects to the same
online run. `--init` starts a separate optimization stage. Use `tracking.mode:
offline` to collect SDK logs without uploading, then `wandb sync` separately;
offline sessions do not support seamless online resume. JSONL metrics,
TensorBoard, checkpoints and predictions remain available locally regardless
of the dashboard. The optional `tracking.entity` selects a W&B workspace;
otherwise the connected account's default is used.

### Deployment artifact

```bash
.venv/bin/python -m radio_moge.export \
  --checkpoint /shared/radio-moge/chongxi/mlp-refine/best.pt \
  --output /shared/radio-moge/chongxi/mlp-refine/geometry-head.pt
```

```python
from radio_moge.export import GeometryHead
from radio_moge.adapters import RadioFeatures

head = GeometryHead.load("geometry-head.pt", device="cuda")
# Outputs from the RADIO forward your pipeline already performed:
# spatial: B×1280×h×w, summary: B×Ds (all backbone summary tokens).
features = RadioFeatures(spatial=radio_spatial, summary=radio_summary)
geometry = head.infer(features, image_rgb_0_to_1)
# geometry contains metric depth, points, intrinsics, validity mask,
# and normals if that teacher has a normal head.
```

For fusion, request intermediate features **during that same backbone pass**:

```python
from radio_moge.adapters import RadioFeatures

final, intermediate = radio.forward_intermediates(
    radio_image, indices=[7, 15, 23, 31], output_fmt="NCHW",
    norm=True, aggregation="sparse", return_prefix_tokens=False,
    intermediates_only=False,
)
backbone = final["backbone"] if isinstance(final, dict) else final
features = RadioFeatures(backbone.features, backbone.summary, tuple(intermediate))
geometry = head.infer(features, image_rgb_0_to_1)
# final also contains the existing teacher adapter outputs when enabled.
```

For final MLP adapters use the normalized **backbone** spatial output, not a
SigLIP/DINO adapter output. Do not externally normalize images again: RADIO owns
its conditioner. Input images/features must describe the same crop, orientation
and aspect ratio used at training. The deployment artifact stores adapter and
decoder weights; loading it does not construct a MoGe encoder or download weights.
The pinned upstream source and utils3d remain required for the decoder and camera
postprocessing. See [THIRD_PARTY.md](THIRD_PARTY.md) for upstream sources/licenses.
