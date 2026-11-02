# Full geometry corpus: acceptance before training

Chongxi authorized offline labels for the frozen 1,637,909-image training manifest
on 2026-11-02, and training **only after** full data, depth, loss and coverage
acceptance. This supersedes the earlier unconfirmed cache-scope proposal. The
small 81-view GT pilot is not sufficient acceptance for the expanded corpus.

`configs/geometry-corpus-20261102.json` freezes the scope and teacher revision.
`scripts/index_geometry_corpus.py` creates a disk-backed inventory and checks
heldout metadata without reading evaluation depth. Its success is **not** depth
acceptance and cannot start training. Native assets remain immutable.

## Source review and unresolved checks

| Source / training images | Label candidate | Required checks before admission |
| --- | --- | --- |
| ScanNet++ DSLR / 841,431 | Laser-derived mesh render, not a direct scanner pixel | Official undistorted camera/pose, visible mesh intersection, anonymity mask, occlusion/edge support; compare against raw scanner geometry where registration permits. Existing 4,504 raw scans are not 841,431 aligned DSLR depth maps. |
| ARKitScenes / 197,542 | iPad LiDAR or available FARO-derived depth | Current RGB is `vga_wide`, not synchronized `lowres_wide`; exact timestamps, camera mapping, confidence and orientation must be checked. No blind resize or nearest-time pairing. |
| Hypersim / 59,344 | Synthetic radial range | Per-scene camera rays, range-to-camera-Z conversion, nonfinite/background mask, world-position cross-check; nonstandard tilt-shift cameras must not use a universal focal length. |
| NYUv2 / 1,043 | Projected Kinect `rawDepths` | Confirm local provenance, unfilled finite positive support, official RGB calibration/pixel origin; do not substitute inpainted `depths`. Existing 116 validation and 290 test images remain excluded. |
| TartanAir V1 / 229,924 | Synthetic left-camera depth | Match left eye/frame, meters, official calibration, finite support; distant sky-sphere values are not reliable finite scene geometry. Preserve environment-family holdouts. |
| Structured3D / 134,749 | Synthetic perspective depth | uint16 millimeters, zero invalid, FOV/camera mapping; independently distinguish radial range from camera-Z using layout planes before bulk conversion. |
| IRS / 133,879 | Synthetic disparity | Verify eye association, EXR channel/encoding, focal length and baseline before metric conversion. Never attach left disparity to right RGB. Missing metric calibration means no metric GT claim. |
| COCO2017 / 39,997 | MoGe-2 pseudo-label only | No native metric-depth annotation in this image collection; pin teacher and preprocessing, retain teacher support/quality flags and estimated-camera provenance. |

Primary documentation and papers (reviewed 2026-11-02; format ambiguities above
remain explicit rather than guessed):

- ScanNet++: [paper](https://arxiv.org/abs/2308.11417),
  [official rendering and panorama documentation](https://github.com/scannetpp/scannetpp),
  [renderer](https://github.com/scannetpp/scannetpp/blob/main/common/render.py).
- ARKitScenes: [paper](https://arxiv.org/abs/2111.08897),
  [raw sensors and synchronization](https://github.com/apple/ARKitScenes/blob/main/raw/README.md),
  [formats](https://github.com/apple/ARKitScenes/blob/main/DATA.md).
- Hypersim: [paper](https://arxiv.org/abs/2011.02523),
  [coordinates and depth](https://github.com/apple/ml-hypersim#coordinate-conventions),
  [per-scene camera matrices](https://github.com/apple/ml-hypersim/tree/main/contrib/mikeroberts3000).
- NYUv2: [dataset and toolbox](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html),
  [paper](https://cs.nyu.edu/~fergus/datasets/indoor_seg_support.pdf).
- TartanAir: [paper](https://arxiv.org/abs/2003.14338),
  [V1 data format](https://github.com/castacks/tartanair_tools/blob/master/data_type.md).
- Structured3D: [paper](https://arxiv.org/abs/1908.00222),
  [format](https://github.com/bertjiazheng/Structured3D/blob/master/data_organization.md),
  [camera utility](https://github.com/bertjiazheng/Structured3D/blob/master/misc/utils.py).
- IRS: [paper](https://arxiv.org/html/1912.09678v1),
  [release/checksums](https://github.com/blackjack2015/IRS),
  [official loader](https://github.com/blackjack2015/IRS/blob/master/dataloader/SIRSLoader.py),
  [author's updated calibration](https://github.com/HKBU-HPML/IRS#overview-of-irs).
- COCO: [paper](https://arxiv.org/abs/1405.0312),
  [official annotation tasks](https://cocodataset.org/#format-data).

Additional IRS review: the author's updated repository documents baseline 0.1 m
and focal length 480 px. This resolves the missing nominal baseline, **not** the
validity of every released label. The author acknowledged incorrect samples in
[issue 3](https://github.com/blackjack2015/IRS/issues/3); image/disparity alignment
must be checked independently. An official training-list entry alone is not a
label-quality certificate. The initial bounded probe intentionally does not admit
IRS to training; follow-up acceptance must bind the actual version and pairing.

ScanNet++ coordinate caution: the official Nerfstudio conversion changes world
axes as well as camera axes. A manifest's Nerfstudio `transform_matrix` must not
be applied directly to the original aligned scan mesh. Cross-check against the
metric COLMAP poses before rendering. `fetch_scannetpp_scanner_poses.py` obtains
the missing registration metadata in a new supplement, without modifying the
existing dataset or printing private authentication material.

Native-format probes use a separate `prep-venv` with
`requirements/geometry-prep.txt`; the existing training/preview environment is
unchanged. The preparation environment inherits the host Torch installation;
model inference, training and loss parity checks must instead use the established
model environment and record its actual version. Mesh raycasting is restricted
to one CPU by process affinity, including native Embree/TBB work.

## Implemented preparation checks (not corpus acceptance)

- The completed inventory has 820 ScanNet++ scenes actually represented by DSLR
  training images. Scanner-pose supplementation covered 822 eligible downloaded
  scenes; these are different counts. ARKit's 1,478 groups are **visits**, covering
  3,789 videos, not 1,478 videos. COCO's groups are image IDs, not verified scenes.
- `audit_scannetpp_corpus_sources.py` inspects every frozen DSLR scene and every
  available raw panorama, with systematic ray samples. Scenes with substantial
  scan/mesh disagreement are retained as `needs_review`, never silently admitted.
- CUDA mesh rasterization has independent CPU ray-intersection comparisons. Very
  rare visibility/triangle-edge disagreements remain even when central error
  quantiles pass; the current rasterizer probe is not a per-pixel quality mask.
- Hypersim tilt-shift matrices are converted to an exact pinhole coordinate frame
  by RQ decomposition, without warping RGB or substituting a universal focal.
  Some released first-frame pose matrices contain uniform asset scale (observed
  0.25). Remove only verified uniform scale, tolerate at most 1e-5 decimal-metadata
  orthogonality residual, and independently check the world geometry. The
  [publisher's exporter](https://github.com/apple/ml-hypersim/blob/main/code/python/tools/generate_hdf5_from_vrimg.py)
  computes range from world distance and stores range/position as FP16; pose scale
  must not rescale metric range. Per-pixel acceptance accounts for actual storage
  quantization rather than incorrectly demanding sub-ULP agreement.
- A full Hypersim camera census covered all 59,344 training frames in 613 camera
  trajectories. It quarantined 32 first-frame matrices with non-rigid or reflected
  axes. The observed ai_001_006/cam_01 matrix has a 0.054 orthogonality residual,
  not numerical roundoff. Native cache attempt 01 stopped after 1,191 entries and
  remains untouched. Census-bound preflight 03 verified 15 good cache entries and
  rejected three regression cases; its 15 depth/mask/K arrays exactly match
  preflight 02. New bulk attempt 02 uses the unchanged geometric tolerance and
  explicitly accounts for quarantined frames. They need separate fallback or
  exclusion, not guessed pose repair.
- Structured3D now binds the [upstream errata](https://github.com/bertjiazheng/Structured3D/blob/master/metadata/errata.txt),
  checks native archive/member hashes and frozen RGB identity, and uses original
  uint16-mm Z, float FOV intrinsics, semantic support, native normals, and matching
  visible structural planes. Window/mirror, saturation, holes, discontinuities,
  and uncertain boundaries are excluded. Normals alone cannot verify units: an
  image lacking at least 100 metric-plane-verified pixels is fully quarantined.
  Preflight 01 inspected both end samples of all 29 represented archives: 56
  cache/roundtrip successes, two upstream-errata room exclusions, no unresolved
  sample review flags. Four stratified empty/full contact sheets were inspected.
  This is a conversion preflight, not full 134,749-image corpus acceptance.
- ARKit native measurements remain at the original 256x192 grid (or its exact
  quarter-turn), never upsampled into extra measurements. The official
  [point-generation code](https://github.com/apple/ARKitScenes/blob/main/threedod/benchmark_scripts/utils/tenFpsDataLoader.py)
  uses integer-pixel camera-Z backprojection. Native and VGA normalized K differ
  by half-pixel terms; store both cameras and their ray-grid transform. Twelve
  exact-timestamp, four-orientation RGB-stream probes gave mean absolute errors
  0.0054--0.0144 on a 0--1 scale; deliberately flipped/shifted controls were worse.
  Four source-space registration sheets were inspected. Native caching requires
  confidence 2, stable depth, exact timestamps, and per-pixel calibrated RGB
  consistency. It labels AppleDepth LiDAR separately from FARO mesh rendering.
  Photometric registration is not an independent physical depth-accuracy test;
  cross-sensor quality and the native-grid loss loader are still acceptance work.
  ARKit cache preflight 01 wrote 24 pack entries but failed final report JSON
  serialization of a NumPy Boolean; preserve it and use a new tested attempt,
  never treat its stale `generating_cache` status as a live job.
  Preflight 03 subsequently verified 24/24 entries across those 12 videos, with
  zero rejections; depth/mask/K and RGB-ray transforms exactly match preflight 02.
  Four upright RGB/depth/mask sheets spanning all quarter turns were inspected.
  The full 197,542-image native cache now runs from a frozen snapshot, using one
  ordered inventory pass and one network request at a time. Each rejected image
  still needs explicit fallback/exclusion in the final corpus accounting.
- COCO pseudo caching uses the pinned teacher at 2,500 tokens, preserves original
  image dimensions, and saves its explicit support. XYZ/Z/K consistency uses
  vector-relative 3D error: coordinate-relative error is ill-conditioned near
  zero X/Y at a large Z. The original seven-image failed cache attempt is kept;
  corrected preflight includes its far-depth regression image.
- NYU source-byte equality is verified, but final geometry acceptance is pending:
  the official toolbox mixes pixel-origin conventions, clips at 10 m, and writes
  the original sensor's Z after RGB projection. Do not claim exact RGB-camera Z
  merely from reading the `rawDepths` field or guess an unverified correction.
  New six-pair raw-file reproduction covers three frozen training office scenes.
  The official MAT explicitly aliases `office_0000` to raw `nyu_office_0/`;
  enforce the same MAT record, not guessed filename equality. The first failed
  alias check is preserved. Reproduction 02 matches five depth/support maps
  within float32 precision; one has two differing support pixels (agreement
  0.99999349) and a maximum 0.393 mm common-depth discrepancy.
  **RGB reproduction does not match**: MAE 6.04--12.71/255 after
  literal toolbox undistortion (raw RGB is worse). Two four-panel sheets were
  inspected. Do not approve RGB/depth registration or invent an image-fitted
  correction; investigate processing/version/raw-frame correspondence first.
- TartanAir V1 now has a native left-camera-Z candidate producer, using exact
  paired right depth and RGB. It requires both-eye local stability, the published
  `x_right=x_left-80/Z` geometry, Z agreement, and per-pixel photometric support.
  Its 0.1--200 m domain is **our explicit task-range selection**, not a publisher
  invalid-value definition. Finite sky-sphere depths are not admitted. Preflight
  covers first/last images in all 24 source bundles: 47 cached and fully reread,
  one rejected for only 2.06% support. Six day/night/interior/vegetated/underwater/
  desert contact sheets were inspected. There are 12 environment names but only
  11 leakage groups: factory day/night variants are not independent environments.
  Bulk generation of all 229,924 inputs is running from `code-tartanair-v1`.
- The GPU-only ScanNet++ visibility filter **failed** the unchanged 2 mm maximum
  CPU-parity requirement on all 32 views in 16 scenes. Keep that failure visible;
  percentile agreement and GPU-only edge erosion were insufficient. The new
  path verifies **every candidate pixel** by independent CPU closest-hit mesh
  intersection, excludes missing/mismatched hits, and stores CPU camera-Z. It
  passed all 32 mapping/visibility checks; this is software consistency, not an
  independent physical accuracy certificate. A separate full pack preflight has
  six verified views and four expected scene-review exclusions across five
  scenes. Three new cache contact sheets were inspected. The producer also binds
  original undistorted K, Nerfstudio/COLMAP poses, source audit and anonymity mask
  hashes. Full generation requires the completed 820-scene source audit; it has
  **not** started yet. These remain `laser_mesh_derived`, not direct DSLR laser
  measurements; the final corpus source/coverage/loss gates are still closed.
- COCO teacher generation is complete: 39,997/39,997 entries were reread and
  checksum-verified, zero rejections. The generate-mode report intentionally
  has `passed:false` and `training_admission:false`; neither means preflight
  failure, nor may it be turned into automatic corpus acceptance.
- IRS has an explicit, separate teacher-candidate routing record. Its unresolved
  native labels are retained for review; teacher caches do not overwrite them,
  fill native holes, claim measured scale, or select final training supervision.
  Preflight covers both eyes and first/last frames across all 13 families:
  52/52 reread, no rejections, four contact sheets inspected. Full 133,879-image
  pseudo-candidate generation now uses the established Torch 2.8/cu128 model
  environment and project GPU lock. Native review and source-aware relative loss
  routing remain mandatory before admission.
- The candidate reader now supports different RGB/depth grids. Calibration-only
  camera rotation recenters the selected RGB viewport, including Hypersim cameras
  with principal points outside the image. RGB is filtered/warped; **depth is
  never interpolated**. Each loss cell selects one original observation, retains
  its native integer index and exact projected UV, and transforms its metric XYZ
  rigidly (or explicitly reflects X for a horizontal flip). The predicted field
  is sampled at that UV. Labels are neither filled nor duplicated. Loss-grid
  resolution also accounts for native angular density after crop/zoom; the first
  visual review exposed unnecessary checkerboard gaps when only the original
  array size was used as a cap. The old probe is preserved and a new one verifies
  the corrected decimation, without changing any running cache producer.
- `corpus-loss-probe-03` verified **213 real cache entries from seven sources,
  each with and without seeded augmentation (426 cases)**. Its 36 CUDA loss
  cases use the full pinned global/local4/local16/edge/scale parameters. Targets
  and predicted masks were perturbed to verify exact loss invariance; gradients
  were finite and nonzero. Teacher-pseudo labels are normalized to relative
  geometry and produce no metric-scale gradient. Native metric scale uses the
  detached upstream alignment scale. Sparse/empty support is explicit, and
  edge pairs exclude holes and depth jumps. Source-review flags fail closed.
  The latest combined unit suite passed 25 tests, including packed reader,
  camera/flip/crop, loss, pack integrity, and FARO diagnostic fixtures.
  **This is a loss-only diagnostic with analytic differentiable predictions,
  not a student/teacher model forward or full-corpus admission.** No optimizer
  was constructed or stepped. The trainer integration, accepted-manifest gate,
  complete coverage review, and actual locked zero-update model preflight remain.
- CUDA probe 01 failed a bitwise-repeat assertion: unchanged inputs also changed
  local16 by 5.82e-11 in a controlled repeat. The same sampled predictions were
  bitwise equal, and deterministic forward algorithms made every masked-label
  comparison exactly equal. The diagnostic now separates deterministic forward
  equality from finite CUDA backward checks; it does **not** claim deterministic
  GPU backward or relax the masking threshold. Failed probe 01 and completed
  pre-density-fix probe 02 remain available. Probe 03's maximum native-to-target
  relative point error was 5.95e-8 (float32 coordinate storage), not an estimate
  of physical depth accuracy.
- A new ARKit/FARO diagnostic scanned 12 training videos from distinct visits,
  covering all four orientations. Only three of those videos had exact frozen
  RGB/highres-depth timestamps; nine frames were compared. Across the frozen
  manifest, 85,601 frames in 1,663 videos are in the publisher's upsampling
  subset, but this is **not** a count of available exact highres pairs. The
  [official downloader](https://github.com/apple/ARKitScenes/blob/main/download_data.py)
  restricts highres depth to that subset. Its
  [raw documentation](https://github.com/apple/ARKitScenes/blob/main/raw/README.md)
  also states that it was filtered using lowres/highres disagreement, so the
  sample is selection-biased. The diagnostic reports two pixel-origin mappings
  without fitting either to depth; exact highres K remains unresolved. Within
  stable reference regions, frame median absolute differences ranged roughly 4--39 mm,
  and some frames still have substantial >5 cm disagreement and large tails.
  High confidence alone therefore does not complete ARKit acceptance. Preserve
  `arkit-faro-quality-probe-01`; review visibility/calibration and broader
  sensor consistency before admitting native labels. It is not permission to
  apply a fitted bias correction or relabel AppleDepth as FARO GT.
- The running Hypersim producer has encountered a further source-consistency
  issue in `ai_009_003`: many frames fail the independent world-position check.
  At the 11:50 UTC snapshot there were 100 review flags and four published sparse
  candidate entries carrying `source_needs_review:true`; most other affected
  frames were rejected for insufficient verified support. These four cached
  entries are **not accepted**: the new reader rejects the flag, and final corpus
  accounting must quarantine/fallback/review all such images. Do not infer
  quality from the running `completed` counter or relax the quantization bound.

Running jobs use separate code snapshots. Do not edit their source or policies
in place. Failed probes/caches stay available; corrections require explicit new
attempts and matching preflights. `geometry_cache.py` writes checksummed atomic
tar packs with per-image source/camera/mask provenance and bounded random reads.
Source assets, candidate caches, and the eventual accepted corpus remain distinct.

## Publication and supervision contract

Every input image needs an outcome: verified native cache, explicitly identified
pseudo cache, or explicit rejection with a reason. Processing completion is not
the same as accepted geometry coverage. Report both counts and the remaining gap.
Unknown/uncertain native pixels stay invalid; pseudo labels do not fill native
holes under a real-GT provenance tag. Separate label kinds: `measured_sensor`,
`laser_mesh_derived`, `synthetic_renderer`, `teacher_pseudo`.

Cache keys must bind image bytes, orientation, source/version, conversion and
camera convention, masks, and teacher revision where relevant. Publish complete
shards and manifests atomically, with hashes and resume receipts. Never overwrite
a prior dataset or make incomplete shards visible to the trainer.

Depth is positive camera-Z in meters, with an explicit Boolean valid mask.
Ray-range/disparity inputs need documented conversions. Estimated teacher cameras
are not calibrated sensor cameras. Keep camera pixel-origin conventions explicit.
No interpolation across holes or depth edges may create new measured support.

The loss gate needs actual tests of masked target/prediction perturbation, finite
gradients, empty/sparse support, edge/normal stencil support, camera/crop/flip
consistency, and source-specific metric versus relative supervision. The previous
official MoGe-2 **core** is not a reproduction of its complete training recipe.

Coverage review must include datasets, independent scenes/visits/environments,
near/mid/far depth, valid-pixel support, indoor/outdoor content, and near-duplicate
sampling. A million repeated crops or adjacent views is not a million independent
scenes. Do not promote a corpus solely because its image count crosses a round
number. Maintain diverse RGB feature replay during subsequent geometry training.

No optimizer updates until all frozen gates have evidence. The final acceptance
record must reference the exact published manifest/cache and test/report hashes;
changing any input invalidates acceptance. The user has authorized training after
these gates pass; no further permission is needed for that scoped transition.
