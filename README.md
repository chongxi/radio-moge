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

### Larger indoor RGB preparation: ARKitScenes and Hypersim

Download a scene-balanced RGB collection without starting training:

```bash
.venv/bin/python scripts/indoor_download.py \
  --metadata /shared/radio-moge/chongxi/indoor-data-planning-20261021 \
  --output /shared/radio-moge/datasets/indoor-mix-v1-20261021 \
  --arkit-target 200000 --workers 4
```

The downloader reads selected ZIP members from the official Apple servers over
HTTP Range, verifies ZIP CRCs and image decoding, and writes images and scene
receipts atomically. Re-run the same command to resume: completed receipts are
checked against image SHA256 hashes. A directory lock prevents duplicate jobs.
Use `--pilot` with a **different output directory** to check one scene per dataset.

ARKitScenes uses native 640×480 `vga_wide` RGB, samples across captures within
each visit, separates frames by at least 0.5 seconds, and rejects exact/near
duplicates and very low-detail frames. Frame selection is deterministic and
quality filtering can leave a shortfall, which each scene receipt reports.
Known validation visits are excluded from training, as are records without a
visit ID. Camera intrinsics, nearby trajectory poses and `sky_direction` are
retained. Files preserve native orientation; **audit and apply upright image
orientation before using these manifests for training**. Camera metadata is
also in the native coordinate system.

Hypersim uses only publicly released official train/validation frames and the
provided tonemapped RGB JPEGs (59,543 train and 7,386 validation in the inspected
v1 split). ARKit validation defaults to up to 24 frames per visit. Ground-truth
depth and test images are not part of this RGB download. The completed `train.jsonl`
and `val.jsonl` contain relative paths; duplicate hashes and cross-split groups
are checked before publication. `status.json` distinguishes downloading,
complete, and incomplete/retryable states. This preparation never starts models
or changes the GPU machine's lifecycle.

Sources and licenses are retained in `metadata/`; see the
[ARKitScenes documentation](https://github.com/apple-aiml-research/ARKitScenes/blob/main/DATA.md)
and [Hypersim documentation](https://github.com/apple-aiml-research/ml-hypersim).

After the downloader reports `complete`, prepare the training manifests:

```bash
.venv/bin/python scripts/prepare_indoor.py \
  --source /shared/radio-moge/datasets/indoor-mix-v1-20261021 \
  --output /shared/radio-moge/datasets/indoor-mix-ready-v1-20261021
```

This verifies every saved RGB hash, checks scene/hash isolation, and derives
quarter-turn orientation from the world-Z-up trajectory. Near-vertical or
ambiguous views use a strong capture-level pose consensus, or are excluded.
The original `sky_direction` label is retained for auditing; it is not blindly
trusted. RGB and optional depth are rotated consistently by the shared data
loader. Original files stay intact, and prepared camera metadata is transformed
to the upright camera frame. `val.jsonl` samples 256 images per dataset across
scenes; `val-full.jsonl` preserves the larger independent validation collection.
Inspect the orientation audit before the full run.

The authorized larger run uses `configs/indoor-fusion-features.json` (30,000
updates), followed by `configs/indoor-fusion-geometry.json` (10,000 updates from
the selected feature checkpoint). Both use intermediate fusion and channel
cleaning, frozen RADIO/MoGe, and effective batch 32. The supervisor
`scripts/run_indoor_training.py` requires a completed prepared dataset and the
recorded training authorization. Each stage tests physical batch 32 first,
falling back to batch 16 with two accumulation steps only on CUDA OOM. Preflight
verifies finite adapter gradients, frozen model parameters, and one RADIO pass
per microbatch. W&B records each training stage separately. A failed stage stops
the queue and leaves logs/checkpoints for inspection; the supervisor never stops
the GPU machine.

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
- Optional `adapter.channel_cleaning: zero-channel` applies the
  [toromi_slam rule](https://github.com/chongxi/toromi_slam/blob/73d5489c39c087fe5a8b2466e0afd06e44a985fc/perception/radio_model.py#L375-L394)
  before each spatial input projection. For each image and layer independently,
  zero channels whose spatial mean absolute activation exceeds 8 times the median
  across channels (`adapter.channel_threshold`, default 8). Fusion cleans each
  selected intermediate map; MLP/linear clean the final map. Summary features and
  shared RADIO outputs remain unchanged. The option is saved in checkpoints and
  exported heads, so training, evaluation and deployment use the same rule.
  Legacy configs default to `none`. `configs/ablations/fusion-zero-channel.json`
  runs a fresh 4000-step comparison against `fusion-features.json` with the same
  seed, data, batch size, optimizer and feature losses; only cleaning changes.
- Input/output channel dimensions are read from the loaded models, not guessed.
- Teacher targets are the **fused projected spatial map and final CLS token from
  the trained MoGe checkpoint**, not raw DINOv2 features.
- Teacher spatial/class losses use fixed per-channel RMS statistics fitted to
  training images. A small cosine term supplements absolute feature matching.
- Geometry refinement adds robust point-map, mask, log-scale and boundary-gradient
  losses. Normal loss is supported if a compatible normal teacher is configured.
- Encoders/adapters use BF16 autocast; decoder and losses use FP32. Frozen model
  parameters are never passed to the optimizer.

The current encoder-output objective is
`MSE((spatial_student - spatial_teacher) / spatial_RMS)` +
`MSE((CLS_student - CLS_teacher) / CLS_RMS)` +
`0.1 * mean(1 - cosine(spatial_student, spatial_teacher))`.
The per-channel RMS scales are fixed after calibration on training images.
Cosine is computed across channels at each spatial position; the CLS branch
uses MSE only. Absolute adapter outputs, not normalized features, enter the
frozen decoder. This retains amplitude information that cosine alone cannot supervise.

### NYUv2 fusion and geometry experiments

`configs/ablations/` records the concrete one-H200 follow-up to the 4,000-step
MLP run, using the same 1,043 training and 116 internal-validation images:

| Stage | Initialization | Budget | Objective |
|---|---|---:|---|
| `fusion-features` | Fresh four-layer adapter | 4,000 steps | Same feature loss as MLP |
| `mlp-geometry` | Existing MLP validation-best checkpoint | 1,000 steps | Feature + decoder geometry |
| `fusion-geometry` | Fusion validation-best checkpoint | 1,000 steps | Feature + decoder geometry |

Fusion reads RADIO blocks 8/16/24/32 in one backbone pass, projects each from
1,280 to 512 channels, averages them, then applies a 512→2,048→1,024 MLP.
The summary branch is unchanged. This adapter has 9,444,352 parameters versus
8,393,728 for the final-layer MLP, so this comparison also changes head capacity.
Refinement uses LR 3e-5 and adds point-map Smooth L1 (1.0), mask BCE (0.1),
log-scale Smooth L1 (0.1), point-gradient L1 (0.05), and normal cosine (0.1).
These differentiable losses use **raw decoder outputs before focal recovery**;
validation still measures fully postprocessed metric depth. Encoders and decoder
weights remain frozen; geometry gradients pass through the decoder to the head.

All stages use physical batch 16, accumulation 2, and the same 480×640 input
bucket. `scripts/preflight_ablation.py` checks actual intermediate shapes,
single-pass extraction, finite feature/geometry gradients and frozen weights on
training images. `scripts/run_ablations.py` runs the three stages sequentially,
logs separate W&B runs in one group, and stops the queue on failure. It does not
stop the GPU or evaluate the outer test set. Checkpoint selection uses internal
validation teacher-depth AbsRel. Geometry stages receive additional optimizer
steps, so their improvements must not be interpreted as a controlled effect of
geometry losses alone; equal-step feature-only continuations would be needed
for that causal comparison.

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

After the fixed fusion/geometry queue, combine the selected MLP and fusion
exports into one viewer without another model forward pass:

```bash
.venv/bin/python scripts/build_comparison.py \
  --baseline /path/to/mlp-run/heldout-viewer-3d-v1 \
  --fusion /path/to/ablations/fusion-features/heldout-viewer-v1 \
  --ablation-root /path/to/ablations \
  --output /path/to/ablations/comparison-viewer-v1
python scripts/serve_viewer.py /path/to/ablations/comparison-viewer-v1 --port 7862
```

The builder checks geometry provenance, checkpoint selection, matching test
frames, and identical teacher/sensor targets. It computes MoGe-relative depth
metrics from the saved float32 arrays, using the teacher validity mask and no
scale alignment. Original exports remain unchanged. The model selector retains
the current test frame and 2D/3D mode across both models.

This comparison expects both geometry stages to select their step-zero
initialization, with verified `mlp-selected-model-equivalence.json` and
`fusion-selected-model-equivalence.json` audits in the ablation root. It checks
their actual adapter fingerprints against the reused exports. The page labels
that reuse explicitly, and separately shows internal validation before/after
all training stages. A different selected checkpoint requires its own export;
the builder rejects silently reusing the wrong predictions.

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

`head.indices` lists the intermediate RADIO blocks the exported adapter needs
(`[7, 15, 23, 31]` for joint-attention heads; empty for final-layer MLPs), and
`head.buckets` lists the training `(height, width)` resolutions. The export
supports every adapter kind, including `joint_attention`.

#### Depth inference test

`radio_moge.infer` runs this deployment path end to end: one RADIO pass
(standing in for the perception module) feeds the exported head. Each input is
center-cropped to the nearest-aspect training bucket (or `--bucket HxW`), with
bf16 autocast as in training. It writes `depth/<name>.npz` (metric depth,
mask, normalized intrinsics), `vis/<name>.jpg` (RGB | depth | GT depth | signed
pred/GT ratio, blue = too near, red = too far), `results.jsonl` and `summary.json`:

```bash
# Plain images, a manifest with ground truth, or reproduce an earlier export:
.venv/bin/python -m radio_moge.infer --head geometry-head.pt --images photos/ --output out/
.venv/bin/python -m radio_moge.infer --head geometry-head.pt \
  --manifest /home/buildhub/shared/radio-moge/datasets/nyuv2-labeled-s42/test.jsonl --output out/
.venv/bin/python -m radio_moge.infer --head geometry-head.pt --reference <dir of *.npz> --output out/
```

For final MLP adapters use the normalized **backbone** spatial output, not a
SigLIP/DINO adapter output. Do not externally normalize images again: RADIO owns
its conditioner. Input images/features must describe the same crop, orientation
and aspect ratio used at training. The deployment artifact stores adapter and
decoder weights; loading it does not construct a MoGe encoder or download weights.
The pinned upstream source and utils3d remain required for the decoder and camera
postprocessing. See [THIRD_PARTY.md](THIRD_PARTY.md) for upstream sources/licenses.
For the public IRS dataset, see [the resumable IRS downloader](docs/irs-download.md).
It supports listing, small access probes, and serial full-archive downloads without
manually copied OneDrive tokens.

# Spatial adapters and variety screen (2026-10-21)

`GeometryAdapter(kind="dpt")` reassembles four RADIO intermediate grids at
4x, 2x, 1x and 1/2x resolution, fuses them with residual 3x3 convolution blocks,
and returns the original MoGe encoder grid. `kind="attention"` mixes the same
four taps and uses two 1280-wide, 16-head self-attention blocks. Both use a
summary-query attention readout of the adapted spatial features for the global
summary/metric-scale path. Default trainable parameter counts are 29,322,752
(DPT-inspired), 53,400,580 (attention-inspired), and 9,444,352 (existing fusion).
The final feature outputs retain learnable absolute magnitudes.

These are new adapters inspired by [DPT's reassembly/refinement](https://github.com/isl-org/DPT/blob/main/dpt/models.py)
and [RADIO's SigLIP2 attention adaptor](https://github.com/NVlabs/RADIO/blob/4ffaf15d0ff48197c194e306772bdb5fb92db230/radio/adaptor_attn.py),
not pretrained DPT/SigLIP2 heads. RADIO and the MoGe encoder/decoder remain frozen.
Legacy adapter configurations and state-dict names are preserved.

`scripts/download_coco_variety.py` selects 40,000 official COCO train images and
disjoint 256-image validation/test subsets from official val2017, before model
inference. Each image is decoded and checked against the official ZIP CRC and
SHA256; download concurrency defaults to two threads in one process. It uses
the official image bucket via an S3 HTTPS endpoint with valid TLS. It does not
require depth labels. COCO's split is by image, not verified physical scene.

`scripts/prepare_variety_mix.py` combines the existing 256,886 indoor training
images, NYU's existing 1,043 training images, and the new COCO subset. NYU's
116 validation / 290 test images remain separate; its original split files
are untouched. The existing 20-scene NYU test visualization stays fixed.
NYU's decoded-pixel hashes are retained separately and encoded-file hashes
are computed for consistent cross-source duplicate checking. The preparation
checks all paths and scene/hash splits, using the prior indoor integrity audit.

Training samples a source, then a uniformly selected scene, then a frame.
The initial source probabilities are ARKitScenes 35%, Hypersim 30%, NYU 15%,
COCO 20%; for COCO each image is its own group. This avoids weighting long
captures solely by their frame count. Validation uses the fixed 256 ARKit,
256 Hypersim, 116 NYU and 256 COCO images, with an equal-source mean teacher
AbsRel for checkpoint selection. Per-source metrics are logged. No test scores
are used for selection; teacher agreement is not ground-truth depth accuracy.

`scripts/run_variety_architectures.py` waits for the current indoor run and
COCO download to finish, prepares the mixture, and preflights all three heads.
It chooses one common physical batch size (16, else 8, else 4 on CUDA OOM),
always with effective batch 32. All heads start fresh, use identical sample
streams, augmentations, seed, loss weights and learning-rate schedules, and
receive 6,000 feature updates followed by 6,000 geometry updates initialized
from their own selected feature checkpoint. All feature trials precede the
geometry trials. This is a matched-update screening budget, not equal FLOPs
or a claim that larger heads have converged. Runtime and peak memory are logged.
The geometry budget extends past the 3,000–4,000-update delay before RMSE
improvement seen in the preceding experiment; 2,000 updates are not a rejection gate.
The queue uses a separate immutable checkout; it never modifies the active
training process or stops the GPU. A failed prerequisite or unexpected error
stops the queue for inspection. No RADIO LoRA is included in this screen.
The queue first exports the predecessor's selected head and fixed 20-scene
viewer, then exports each candidate after its complete training budget.
`scripts/summarize_variety_screen.py` writes initial/selected/final metrics,
per-source scores, curves, observed step time and memory. It ranks candidates
only when every candidate has completed the corresponding stage. RMSE remains
visible without changing the preregistered macro-AbsRel checkpoint selection.

The three workstreams and decision gates are recorded in
`configs/experiment-plan-20261021.json`. With `--geometry-control-repo` pointing
to the unchanged predecessor checkout, the queue additionally runs a 10,000-step
feature-only continuation from the exact same feature-best checkpoint, using the
predecessor's code, data, seed, learning rate, warmup and optimizer reset. Only
the geometry loss weights are zeroed. Its initial adapter/statistics are checked
against the geometry run, and both selected and final results are reported.
This control distinguishes geometric supervision from additional optimization.

A fourth screening arm, `fusion_indoor`, keeps the fusion architecture, common
validation, fresh initialization and 6,000+6,000 budget, but trains only on the
existing ARKit/Hypersim images with their conditional mixture weights. Comparing
it with `fusion` measures the new data mixture and sampling policy together;
it does not identify a pure dataset-size effect. Architecture rankings still
compare only the three heads on the identical new mixture. Each stage uses a
common physical batch across all four arms. These controls add 22,000 updates
to the already planned 36,000; all run serially after the current experiment.
The repeatedly viewed NYU20 panel is a fixed qualitative audit, not a new blind
test. Follow-up loss modifications, multi-seed confirmation and RADIO LoRA are
conditional decisions, not part of the automatic first-round queue.

`scripts/compare_scene_metrics.py` pairs per-image predictions against an exact
manifest and bootstraps scene groups within each source. It reports both
equal-source and image-weighted differences, per-source intervals and scene
improvement counts. Missing metrics remain explicit instead of dropping failed
images. The intervals describe these checkpoints on this validation sample;
they do not account for training-seed variation or repeated checkpoint selection.
Use the same utility for the completed objective control, not only before/after
snapshots, before attributing changes to geometric supervision.
