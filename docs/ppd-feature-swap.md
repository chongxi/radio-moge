# Frozen PPD feature substitution probe — 2026-11-07

The final Muon adapter can supply the semantic input of the released MoGe2
Pixel-Perfect Depth (PPD) model without training. In this small probe, PPD with
our fitted features has almost the same relative-depth error as PPD with native
MoGe2 features. This demonstrates usable transfer; it does not establish that
PPD is better than the existing decoder on the full validation distribution.

## Results

Twelve distinct NYU validation scenes, equal image weighting. AbsRel is shown
as a percentage, lower is better. Every method receives the same crop and
per-image ground-truth-assisted relative-depth alignment. These are **not**
unaligned metric-depth errors or the published NYU benchmark numbers.

| Input width × height | Native features → MoGe | Our features → MoGe | Native features → PPD | Our features → PPD |
|---|---:|---:|---:|---:|
| 640 × 480 | 3.428% | 3.623% | 4.175% | 4.205% |
| 1024 × 768 | 3.255% | 4.305% | 3.748% | 3.732% |

At 1024 × 768, PPD reduces the adapted MoGe route's mean AbsRel by 13.29%,
improving 11/12 scenes. However, our existing 640 × 480 MoGe route scores 3.623%,
and native MoGe at 1024 × 768 scores 3.255%. The experiment therefore does not
support claiming an overall accuracy upgrade. PPD's log-depth-gradient error
against this NYU sensor ground truth is also higher than MoGe's; sharper-looking
boundaries alone do not prove more accurate geometry.

The fitted/native spatial cosine averages 0.961265 at 640 × 480 and 0.932291
at 1024 × 768. The high-resolution input is outside the adapter's training
buckets. Despite that feature discrepancy, PPD's aggregate depth error changes
only −0.40% relative to native PPD in the first noise condition.

Additional controls, the same 12 NYU scenes at 1024 × 768:

| Condition | Native features → PPD | Our features → PPD |
|---|---:|---:|
| Original noise | 3.748% | 3.732% |
| Second noise, seed offset +100000, CUDA RGB normalization | 3.798% | 3.799% |

Setting PPD's semantic tensor to zero while retaining the original RGB and noise
increases AbsRel to **35.169%**, with δ₁ falling to 51.86%. This supports reliance
on the supplied features. Zero conditioning is out of distribution and is not
an estimate of a separately trained RGB-only model. Two noise conditions do
not establish statistical equivalence or broad generalization.

The point-cloud replay later identified a preprocessing detail: `results-01`
divides RGB by 255 on the CPU before transfer, whereas `controls-01` divides
on CUDA. The latter is internally paired, but comparison with the original
noise condition changes this rounding detail as well as the noise seed.
These two rows must not be interpreted as a strictly noise-only experiment.

Four additional, deterministically selected empty Structured3D scenes were
rendered at both resolutions. They have no paired ground truth in this probe.
Their displays are aligned to native MoGe depth solely for visual comparison;
they cannot certify wall planarity.

## Interface and protocol

- PPD revision: `427a86f5882aa3f7233c1b94fbdccce87f1c8313`.
- Released **MoGe2** PPD weights, SHA256
  `cd21e79583159db3fe9c8addbc0daa2c0c132e290bf6d537e9ebaa04b47cc7b0`.
- Our fixed final adapter: step **272646**, SHA256
  `943534362e85d5336cde00d6277cb550c2bbc0b180b1253054f1609de50639c8`.
- The shared MoGe2 checkpoint is the pinned
  `Ruicheng/moge-2-vitl-normal` revision
  `cb0e8bbd6b1e243589717c78e750b1ba4c093acf`.
- The native PPD semantic encoder is instantiated from PPD's own code and
  loaded strictly from the same pretrained weights. On one input at each
  resolution, its spatial **and** summary tensors are bit-for-bit identical to
  our training teacher implementation. The adapter, PPD encoder and DiT state
  dictionaries are loaded strictly. RADIO uses its existing official loader,
  which reports two unused `_vis_cond` state entries as before.
- PPD consumes projected and fused spatial features with shape
  `B × (H/16 × W/16) × 1024`. We pass the adapter's raw spatial prediction at
  this boundary; PPD applies its own L2 normalization. No RMS-scaled training
  targets, summary vector, feature resizing or new learned projection is used.
  The adapter still computes its joint representation internally.
- PPD also receives RGB and noisy depth. Its released DiT and Euler sampler
  run for **4 steps**, matching `run.py`'s default. Native and fitted branches
  share the **same noise tensor**, not merely an unspecified random seed.
- All weights are frozen; **zero optimizer updates**. FP32 parameters with
  BF16 autocast; existing MoGe heads remain FP32. Batch size 1.
- The first 12 distinct NYU groups and 4 distinct empty Structured3D groups
  sorted by encoded-image SHA256 are fixed before inference. Inputs come from
  the existing internal validation manifest; this is not a new blind test.
- Shared centered 4:3 crop and PIL bilinear RGB resize. This differs from
  PPD CLI's aspect-preserving OpenCV preprocessing, so absolute numbers are
  not presented as a reproduction of the paper's benchmark.
- Predictions are resized to 640 × 480 for scoring, using the same valid NYU
  depth mask (`0.1 < depth < 10 m`). PPD's official RANSAC fits an affine map
  to `log1p(GT)`; metric MoGe predictions enter that same fit as `log1p(depth)`.
  The RANSAC seed is identical across methods and unchanged for the second
  diffusion-noise condition. Ground-truth alignment is evaluation only.

PPD's [semantic encoder boundary](https://github.com/gangweix/pixel-perfect-depth/blob/427a86f5882aa3f7233c1b94fbdccce87f1c8313/ppd/moge/model/v2.py#L139)
and [DiT fusion](https://github.com/gangweix/pixel-perfect-depth/blob/427a86f5882aa3f7233c1b94fbdccce87f1c8313/ppd/models/dit.py#L213)
explain why this substitution is possible. Its
[training objective](https://github.com/gangweix/pixel-perfect-depth/blob/427a86f5882aa3f7233c1b94fbdccce87f1c8313/ppd/models/ppd_train.py)
trains a depth-generating model, unlike our feature matching objective. The
[paper](https://arxiv.org/abs/2510.07316) and current repository report stronger
results in their evaluation settings; that does not imply every decoder swap
must improve our existing internal validation numbers.

## Reproduction and artifacts

Code: `scripts/probe_ppd_feature_swap.py`,
`scripts/check_ppd_semantic_controls.py`, `scripts/render_ppd_probe.py`.
Main probe revision `1bacef3`; supplemental controls revision `b74a38a`.

Local evidence root:
`/shared/radio-moge/chongxi/ppd-feature-swap-20261107-01`.
Remote evidence, weights, pinned upstream source and isolated environment:
`/home/buildhub/shared/radio-moge/chongxi/ppd-feature-swap-20261107-01`.

- `results-01/plan.json`: fixed identities, hashes, preprocessing, sampling.
- `results-01/report.json`: all 32 paired cases and aggregates.
- `controls-01/report.json`: 12 zero-semantic controls and 24 second-seed outputs.
- Both directories contain per-scene JSON and raw/GT-aligned NPZ predictions.
- `.kanna/visualizations/ppd-feature-swap-20261107.html`: all-scene comparison,
  shared color scales, error maps and a movable overlay.

Two setup attempts exited before any model inference: an isolated environment
initially missed the original environment's packages, and the newly cloned
repository needed a scoped Git ownership exception. The third launch completed
all planned cases; the original logs are retained. No training, viewer,
checkpoint, deployment or prior result was modified.

The next useful investigation is broader PPD transfer and task-specific depth
quality, including temporal behavior if used in video. These results lower the
priority of simply extending the same feature-only Muon training to gain a
small additional cosine increment.

## Interactive 3D point clouds

The completed point-cloud export is `pointclouds-02/` under the same evidence
root. It contains 160 full-resolution binary RGB PLY files: 16 scenes, two
resolutions and five routes. The offline page is
`.kanna/visualizations/ppd-pointclouds-20261107.html`; its default comparison is
native features → PPD versus our features → PPD. Both cameras rotate, pan and
zoom together. RGB, geometry lighting, common depth colors, overlay and front/
side views are available. Original MoGe outputs can also be selected.

The saved PPD predictions are reused without another diffusion run. Frozen
MoGe and adapter replay recovers the intrinsics and masks omitted from the
original raw archive. All 32 native and fitted MoGe depth pairs match their
saved predictions bit for bit before those cameras are accepted. A preliminary
export using CUDA RGB division failed that check; it wrote no PLY files.
Matching the original CPU division restored exact equality. The original
experiment artifacts remain unchanged.

No ground truth or GT-aligned prediction enters these point clouds. PPD's
official RANSAC routine fits its relative output to a metric depth reference:

| Route | Scale reference, intrinsics and mask |
|---|---|
| Native features → MoGe | Native MoGe prediction |
| Our features → MoGe | Fitted-feature MoGe prediction |
| Native features → PPD | Native MoGe prediction |
| Our features → PPD | Fitted-feature MoGe prediction |
| Our features → PPD, shared native calibration | Native MoGe prediction; separately labeled diagnostic |

All routes use pixel-center unprojection, then x-right, y-up, z-backward
coordinates in model-estimated metres. No smoothing, outlier filtering or
registration is applied. Camera framing uses robust bounds of the displayed
points; it does not remove points. The preview samples every four pixels along
each axis (up to 49,152 points per high-resolution cloud); preview PLY downloads
retain that density. The ZIP archives contain every valid original-resolution
pixel, up to 786,432 points per cloud at 1024 × 768.

- `our-features-ppd-1024x768.zip`: our PPD route, all 16 scenes, about 162 MB.
- `pointclouds-1024x768.zip`: all five routes, 80 PLY files, about 804 MB.
- `pointclouds-640x480.zip`: all five low-resolution routes, retained remotely.
- `report.json`: individual PLY hashes, counts, calibration replay checks,
  bundle hashes and the explicit no-GT/no-diffusion-update record.
- `browser-check/browser-verification.json`: real Chromium/WebGL verification
  of all 32 cases, five routes, synchronized orbit/zoom, geometry shading,
  overlay, mobile layout and exported PLY coordinates. No external requests.

Code: `scripts/export_ppd_pointclouds.py`, `scripts/build_ppd_cloud_page.py`,
`scripts/check_ppd_clouds.cjs`, `web/ppd-clouds.html`, `web/ppd-clouds.js`.
The export uses the original four-step predictions; the upstream point-cloud
CLI's other sampling or filtering defaults have not been substituted.
