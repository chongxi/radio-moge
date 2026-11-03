# Full geometry corpus: acceptance before training

Chongxi authorized offline labels for the frozen 1,637,909-image training manifest
on 2026-11-02, and training **only after** full data, depth, loss and coverage
acceptance. This supersedes the earlier unconfirmed cache-scope proposal. The
small 81-view GT pilot is not sufficient acceptance for the expanded corpus.

`configs/geometry-corpus-20261102.json` freezes the scope and teacher revision.
`scripts/index_geometry_corpus.py` creates a disk-backed inventory and checks
heldout metadata without reading evaluation depth. Its success is **not** depth
acceptance and cannot start training. Native assets remain immutable.

## Hypersim combined accounting and live status (23:27 UTC)

`hypersim-review-teacher-cache-01` completed 3,680 pseudo candidates and their
full producer readback, with the one evidenced constant-RGB rejection.
`hypersim-combined-routes-01/report.json` now accounts for all 59,344 frozen
Hypersim inputs exactly once: 55,663 unflagged native candidates, 3,680 teacher
candidates, and one reviewed input exclusion. The teacher candidates replace
3,076 flagged native entries and 604 native rejections; the original native
flags remain recorded. The pure-white exclusion also applies to eventual RGB
replay. This receipt-based reconciliation does not constitute source acceptance
or another full array readback. Its route and producer-report hashes were
rechecked after the interrupted chat resumed.

Snapshot `code-hypersim-routes-v1` at `02293a3` passed 17 focused tests before
the actual audit. At 23:27 UTC, primary candidate counters totalled 651,950 of
1,637,909 (39.80%), including recovered Structured3D and ScanNet++ subsets once
and excluding separately tracked review replacements. The ScanNet++ review
teacher job is running (2,016 of 35,783), and the training RGB census has read
621,312 images with zero failures. Heldout RGB is complete at 190,954 manifest
memberships; full train/heldout comparison still awaits the training census.
Source quality, final coverage and the real model zero-update gate remain
incomplete; production optimizer updates remain zero.

## Bounded TartanAir rejection review and IRS display correction (November 3)

`tartanair-rejection-review-01` binds the 11 rejection records observed while
the producer is still running to the frozen inventory and retained original
left/right depth and RGB bytes. The unchanged producer converter reproduced
every stored statistic exactly. All 11 diagnostic pages were inspected: these
gascola Hard P002/P003 views point upward toward sky and boundary foliage.
Only 11.06--18.23% of each image falls inside the existing 0.1--200 m task range;
local stability and stereo checks leave 2.0221--4.7754% final support. The task
range is not a publisher invalid-depth definition. No new downloads, threshold
changes, hole filling, or teacher inference were used.

`visual-review-01.json` records the decision to exclude these exact 11 images
from geometry supervision under the unchanged 5% support threshold.
`geometry-exclusions.jsonl` contains the hash-bound per-image decisions for
eventual full accounting. This decision does not exclude their RGB from feature
replay; RGB eligibility still needs its own identity, leakage and sampling gates.
Other gascola images are unaffected. This is an explicit bounded review, not
complete TartanAir source acceptance. The converter and evidence-reproduction
tests passed: six tests, including wrong eye/frame, changed source bytes, changed
RGB, mismatched statistics, sky support and original-array preservation.

IRS's original contact sheets used separate p01/p99 color limits for each view.
`shared-palette-flags-02` independently rereads all ten flagged RGB/member
payloads and displays each unchanged depth divided by its valid median using
one fixed logarithmic color range [0.25,16]. Every new full-resolution page was
inspected. The broad wall/floor/shelf colors in Supermarket views 101/102 are
now similar; the earlier global-background inconsistency argument based on
colors is withdrawn. Local far values remain in the visible window behind the
shelf. Neither the shared palette nor visual similarity verifies physical depth
or multiview consistency. Bistro blurred-glazing blobs and Diner framed-panel
depth ambiguity remain quarantined. The new review preserves all masks and
explicitly leaves source acceptance false; no depth-ratio cutoff was applied.
The first renderer attempt stopped on the publisher's opaque RGBA PNGs; v2
verified alpha=255 and matched the producer's RGB conversion. The failed attempt
and original reviews remain intact.

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

### Recovery and Structured3D format review, 2026-11-02

The 16:20 UTC check found two stopped producers: Structured3D at 27,023
candidate entries and ScanNet++ at 49,220. These are counters before a failed
run's final verification, not accepted or fully committed cache counts.
The Structured3D failure is at
`scene_01181/2D_rendering/2163/perspective/empty/0`. Its unit-length look/up
directions have dot product -0.000112038, just beyond our old 1e-4 requirement.
The [publisher parser](https://github.com/bertjiazheng/Structured3D/blob/master/misc/utils.py)
normalizes direction vectors and constructs camera axes with cross products;
the file does not provide a precomputed orthonormal rotation matrix. The new
parser validates finite, nonzero, nonparallel directions and records their
original angular discrepancy. Camera-local depth and layout comparisons do
not apply a world rotation, so this corrects an input assumption without
changing depth values, fitting a pose, or relaxing the 5 mm plane gate.

The [format contract](https://github.com/bertjiazheng/Structured3D/blob/master/data_organization.md)
specifies uint16 millimeters (zero is missing), half-FOV angles in radians,
and normal encoding `128 * (1 + n)`. This corpus contains perspective images;
panorama geometry must not share this conversion. Keep floating-point focal
lengths and the original integer-pixel origin. In
`structured3d-failure-regression-01`, the failed view and two neighboring
training views were checked against structural planes using the same semantic,
normal, and polygon support for each depth convention. Camera-Z/mm gave median
plane residuals 0.325--0.472 mm and maxima 1.191--1.303 mm. Treating the same
numbers as radial range gave medians 113--193 mm. These are synthetic internal
consistency checks, not independent physical measurements. Half-pixel variants
were also retained; the small residuals alone do not uniquely prove pixel
origin. The declared origin follows the documented projection convention.
All three RGB/depth/mask sheets were inspected: window support is absent,
depth boundaries align visually, and holes remain unfilled.

`structured3d-camera-census-01` checks every frozen perspective camera record.
New cache preflight 02 retains both ends of all 29 archives plus two failure
regression views (60 images). Neither the census nor the targeted diagnostic
is full depth-quality acceptance. A new full generation requires both the
complete census and the matching conversion preflight.

ScanNet++ stopped at the transition to `0fe8539661`, whose mesh contains
17,326,972 faces. The new renderer bounds each rasterization pass to 1,048,576
faces, renders every face, and merges by nearest camera Z. CPU verification of
every retained ray and the 2 mm agreement gate remain. The large scene is now
included in visibility probe 03 and the cache preflight; its outcome must be
checked before generating new candidates. This remains a candidate fix until
the real-scene probes pass.

Explicit recovery rereads each member of each closed tar/receipt pair into a
new, separately identified subset. Original failed attempts and uncommitted
tails remain untouched; incomplete tails and previous rejected rows are
reprocessed by the new remainder run. The recovered subset keeps its original
conversion policy and provenance. Final accounting must include both subsets
without duplicates, and still requires source quality review. The new scoped
tests passed 21 cases, including corruption rejection, immutable recovery,
direction-vector regressions, and cross-batch occlusion. No production
optimizer update or corpus admission was performed.

At the 17:05 UTC follow-up, both recovery and restart gates were complete:

- Structured3D's camera census covered all 134,749 frozen images, with zero
  invalid direction records and two records outside the obsolete raw
  orthonormality requirement. Preflight 02 produced and reread 58 entries;
  its other two rows are upstream errata. All 56 old/new common entries have
  bitwise identical depth, masks and K. Eight contact sheets, including both
  failure regressions, were inspected. The recovered 27,008-entry subset and
  the new 107,741-row remainder are distinct artifacts and policies.
  `structured3d-native-cache-02` is running with explicit recovered-row skipping.
- Mesh visibility probe 03 passed all 34 views, including the first/last views
  of the 17,326,972-face scene. ScanNet++ cache preflight 02 produced eight
  verified entries and four expected source-review exclusions. All six old/new
  common entries have identical masks, K and accepted CPU depth. All eight
  sheets were inspected. Recovery reread 49,152 entries; the new 792,279-row
  remainder runs as `scannetpp-native-cache-02`. The 25 source-review scenes
  remain quarantined. Neither GPU/CPU agreement nor visual alignment claims
  independent per-pixel laser accuracy.
- Each preflight directory contains `array-parity-review-01.json` and
  `visual-review-01.json`, binding the comparison and reviewed sheets by hash.
  Reprocessed old rejections appear in the new remainder counters; they are
  not extra images beyond the frozen scope.

`hypersim-source-mismatch-probe-01` separately diagnosed 29 cached images:
25 flagged examples from 13 scenes and four unflagged controls. All native
reconstructions reproduced the cached depth, mask, K and statistics exactly.
Every inconsistent positive-support pixel in these flagged examples still
passed the distance-norm check under the existing storage-quantization bound.
This is possible because the [publisher's generator](https://github.com/apple/ml-hypersim/blob/main/code/python/tools/generate_hdf5_from_vrimg.py)
derives `depth_meters` from world-position distance before independently storing
both arrays in float16. Norm agreement therefore does not independently verify
the pixel's camera ray. The diagnostic separates norm and angular residuals;
it does not change the production gate.

Sixteen mismatch sheets were inspected, covering all 13 selected flagged
scenes and three controls. In 12 flagged scenes the displayed errors are
localized around windows, curtains, glazing or exterior backgrounds. This
visual association does not prove a specific renderer or optical cause.
`ai_009_003` differs: frames 66/67 lose approximately 94.6/95.0 percent of
support, with median angular errors around 1.31 degrees in rejected regions.
Its broad mismatch remains unresolved; no focal length or pose was fitted.
All source-review flags remain in force. This diagnostic only samples closed
candidate receipts and does not account for rejected or uncommitted rows.

Running jobs use separate code snapshots. Do not edit their source or policies
in place. Failed probes/caches stay available; corrections require explicit new
attempts and matching preflights. `geometry_cache.py` writes checksummed atomic
tar packs with per-image source/camera/mask provenance and bounded random reads.
Source assets, candidate caches, and the eventual accepted corpus remain distinct.

## Complete heldout RGB census and IRS outcome tooling (22:05 UTC)

`heldout-rgb-census-01` completed all 190,954 memberships in the 16 frozen
validation/test manifests with zero decode/identity failures. These memberships
refer to 188,686 distinct file paths, not independently established images or
scenes. Its completed SQLite index, per-manifest/source counts, original hash
conventions, policy and manifest hashes were checked in
`heldout-rgb-census-01/identity-review-01.json`. The source conventions comprise
190,838 encoded-file hashes and 116 legacy NYU raw-RGB hashes. Quarter-turn
orientation metadata was retained, and no evaluation depth was read.

The complete training RGB census is still running. No full train/heldout pixel
comparison or heldout-isolation gate has passed; no claims of semantic or
perceptual near-duplicate independence follow from this census.

The IRS producer remains active in full pack readback. New
`audit_irs_teacher_outcomes.py` requires its complete report and verified-member
count before creating an audit output. It binds every frozen IRS RGB exactly
once to a teacher receipt or explicit rejection, verifies original eye/row
identity and pinned nonmetric teacher provenance, and records relative depth
quantile/support statistics per family and eye. It selects the two largest
p99-to-median ratios and two lowest-support candidates per family/eye for later
inspection. These selections do not change masks, depth values or eligibility.

The audit reads receipt metadata, relying on the producer's completed payload
readback. It does not independently reread arrays or rehash every tar, and is not
source-quality acceptance. Final corpus accounting still verifies actual payloads.
Snapshot `code-irs-outcome-audit-v1` (`840e826`) passed 15 combined outcome/scope/
coverage tests; `code-irs-outcome-audit-v2` (`e0f079d`) passed three strengthened
IRS tests that isolate wrong semantic contracts from stale hash failures and
reject an incomplete verified count even when a report says complete. Evidence
is in `irs-outcome-tooling-tests-01.json` and `-02.json`. The actual full IRS
outcome audit has not yet run. Production optimizer updates remain zero.

## Hypersim complete outcomes and reviewed fallback scope (21:20 UTC)

The Hypersim producer has completed all 59,344 inputs and verified all 58,739
cached members by full readback. `hypersim-outcomes-01/report.json` then bound
every frozen input to exactly one receipt or rejection: 55,663 unflagged cache
members, 3,076 flagged cache members, and 605 rejected images. Rejections comprise
32 non-rigid camera poses and 573 insufficient-support cases. The source-review
counter includes 372 already rejected images; the distinct unresolved scope is
3,681 images across 60 scenes. The outcome audit preserves every source flag and
does not itself approve the unflagged members or verify physical depth accuracy.

`hypersim-review-teacher-routing-20261102.json` binds that exact unresolved scope
to separate, whole-image MoGe-V2 teacher candidates. The native arrays, poses,
masks and source artifacts remain unchanged. These candidates use only the
original frozen RGB and carry `teacher_pseudo`, estimated-camera and nonmetric
provenance, including the original native failure reason. They are alternatives
for final per-image routing, not repaired native GT or native-hole filling.

Snapshot `code-hypersim-review-teacher-v1` at commit `8b42fff` selects the first
and last image in each scene/failure-reason stratum: 116 views covering all 60
scenes. All 116 input identities, decoded RGB hashes, dimensions and orientations
passed the bounded input check. The teacher preflight is queued behind the
existing GPU lock; inference and visual review have not yet completed. Bulk
generation requires both a matching numeric pass and a hash-bound visual review
covering every scene. Its 3,076 replacements for flagged native candidates must
never be counted as additional unique training images.

The scope/producer and existing loss regression suite passed 22 tests in 2.33 s
in the established model environment. Evidence is recorded in
`hypersim-review-teacher-tooling-tests-01.json`. This is distinct from the earlier
22-test heldout/outcome suite and is not the actual student-model gate.

The original feature-only initialization is the archived
`model-archive-feature-final232288-20261101/final.pt`, 2,661,040,151 bytes, SHA256
`3d3920bcaf89b294ba8e3d5016797d8121a9dbd4a1d938f050661ad4e23c66ef`.
Its bytes and metadata were rechecked against the archive record: source step
232288, cursor 409480, joint-attention width 2048 with four blocks and qk-norm.
`initial-feature-checkpoint-inspection-01.json` preserves the exact model,
adapter and source-training configuration. No model was constructed, no tensor
values were revalidated, and no forward/backward or optimizer update occurred.
Final training configuration and all actual-model prerequisites remain pending.

IRS has generated all 133,879 teacher candidates and is performing full pack
readback while retaining the GPU lock. Both RGB censuses continue without errors
in the last observation; the full train/heldout comparison has not run. No full
corpus admission or production optimizer update has occurred.

`visualize_teacher_preflight.py` now renders every cached view of a completed,
bounded teacher preflight, with four views per page and hash-bound image/cache
identities. Display colors use per-image log depth percentiles and explicitly
do not claim metric GT. Snapshot `code-teacher-qc-v1` at `9e1e9a8` rendered all
52 IRS preflight views into 13 pages, all of which were inspected. The rendered
index is not itself a visual pass. The explicit review is
`irs-teacher-preflight-01/all-view-qc-01/visual-review-01.json`.

That review found no obvious image flip or scene mismatch at contact-sheet
resolution, while retaining uncertainty for bright windows, reflections, thin
objects and low-light views. All 52 teacher masks have full support; this must
not be interpreted as physical pixel accuracy. In particular, Kitchen views
4/5 and SCLRestaurant views 40/41 have p99-to-median teacher depth ratios around
16--25 in ambiguous window/reflective regions. They remain explicit review cases
for the full normalized-depth distribution and final source-quality decision.
No depth values or masks were modified and no images were automatically excluded.
This stratified preflight does not measure full-corpus error prevalence.

## Heldout RGB census and complete native-outcome accounting (20:33 UTC)

`audit_heldout_rgb.py` now reads the 16 inventory-bound validation/test manifests
under their original checksum and split contracts. The legacy NYU internal-val
manifest stores raw RGB pixel hashes and has no split/dataset fields; its named
schema identifies it as heldout NYU without rewriting those fields. The two
older ScanNet sensor/laser manifests use encoded-file hashes. Their producer
code establishes these conventions. RGB decoding checks EXIF, explicit quarter
turns, and published dimensions. No evaluation depth/geometry is opened.

`heldout-rgb-preflight-01` verified 104 records (78 distinct paths) from all 16
manifests, with no errors: 101 encoded-file hashes, three legacy raw-pixel hashes,
and rotations 0/90/180/270. The numeric identity review is stored beside its
report. Full `heldout-rgb-census-01` is running from `code-heldout-rgb-v1`, one
worker at nice 10. Its scope is 190,954 manifest memberships, including repeated
subsets, **not** 190,954 independently established images/scenes. At 20:33 it had
read 5,888 records without error. The separate full training RGB census had read
178,432 images without error. Neither census was complete.

After both complete error-free censuses, `audit_heldout_rgb.py compare` will join
all raw/oriented canonical hash combinations in a new disk-backed index. It
preserves every heldout membership, records exact pixel and normalized scene
overlaps, and retains the documented IRS official-test shared-family exception.
It refuses partial census inputs and does not modify either source index or
automatically exclude images. Exact identity comparison does not establish the
absence of perceptual near-duplicates or semantic scene independence. The full
train/heldout comparison has **not** run yet.

Hypersim has processed all 59,344 inputs and is still performing the producer's
full pack readback. Receipt metadata shows 55,663 unflagged candidates, 3,076
flagged candidates, and 605 rejections (32 non-rigid camera poses, 573 insufficient
world-verified support). The live `source_review_images=3448` includes 372 of
those already rejected images; adding it to all rejections double-counts them.
The distinct native unresolved scope is 3,681 images. This is a metadata tally,
not a completed source-quality review or approval of the unflagged candidates.

`audit_hypersim_outcomes.py` is prepared to bind each input exactly once to its
receipt or rejection, preserve the flag/rejection distinction, and produce a
separate unresolved-row manifest. It requires a completed producer report with
full array readback before running. Its output is explicitly a receipt-metadata
audit, not an independent second array verification or training admission. No
fallback selection or exclusion is made by this audit.

The combined native-outcome, heldout-RGB, training-RGB and coverage suite passed
22 tests in the bounded preparation environment, including reencoded identical
pixels, legacy hash conventions, heldout split/orientation failures, partial
census rejection, and duplicate native outcomes. Evidence is in
`heldout-rgb-tooling-tests-01.json`. These tests do not exercise the actual
student/teacher model gate. Production optimizer updates remain zero.

## Actual-model preflight tooling (prepared, not yet run)

`scripts/probe_corpus_model.py` now provides the genuine zero-update model gate.
Its plan references the full accounting report, the four completed nonmodel
gates (`source_quality`, `coverage`, `heldout_isolation`, `loss_tests`), the exact
training configuration, and the initial checkpoint by absolute path/SHA256.
`source_step` must match the checkpoint (232288 for this continuation). Missing
or stale prerequisites reject execution before model loading. The command also
requires the existing absolute GPU lock path and acquires it nonblocking; it
does not create a replacement lock or queue a partially approved model run.

The script selects support/depth extremes and different scenes in every selected
dataset/label stratum, reads the actual checksummed cache/RGB pairs, and keeps
their native target grids. It exercises both paired-RGB and diverse replay-RGB
feature slots with the production geometry loss. It separately differentiates
geometry through the frozen decoder to the adapter outputs, so a functioning
feature loss cannot conceal a detached geometry branch. There is no optimizer
construction or update. Parameter and buffer hashes before/after the complete
run must match for RADIO, the MoGe encoder/decoder, adapter, and feature scales.
Teacher/RADIO checkpoint bytes, upstream revisions, training configuration and
implementation hashes are also bound to the evidence.

The structural configuration validator has an explicit observational mode;
the trainer never uses it and still requires the sealed acceptance report.
All four nonmodel gates can now be checked independently without inventing an
accepted training seal. Final acceptance additionally requires the actual-model
report as before. Implementation binding now includes the joint adapter,
feature objective, model loader/configuration and their local dependencies.

Snapshot `code-model-probe-v2` at commit `a783953` passed 57 CPU tests in the
established model environment. These include tiny-model backward checks through
the actual loss functions, buffer mutation and detached decoder rejection,
mandatory prerequisite gates, training and RGB replay regression tests.
The test-only loss fixture reduces alignment resolution/patch count for speed;
the actual-model command uses the pinned production settings. These tests are
tooling validation only: they are not the actual-model gate or corpus admission.
No actual model run has been started, because full corpus/source acceptance is
still incomplete. Running bulk generators retain their original code snapshots.

## Coverage audit and isolated ScanNet teacher candidates

`inventory-coverage-audit-01` scanned all 1,637,909 frozen rows and bound the
inventory/report hashes. It records 1,478 ARKit visits, 365 Hypersim scenes,
336 NYU scenes, 820 ScanNet scenes, 2,997 Structured3D scenes, 13 IRS environment
families and 11 TartanAir environment families. COCO's 39,997 groups are image
identities; their physical scene independence is unknown. IRS's largest family
contains 22.9 percent of its images and TartanAir's largest family contains
22.2 percent. Frame count must not be substituted for scene diversity.

Stored decoded-RGB hashes exist for 499,595 frozen rows (IRS, NYU, Structured3D
and TartanAir); no repeated hash values were found within that subset. Follow-up
code inspection found incompatible conventions: NYU hashes contiguous RGB bytes,
whereas the three archive sources also prepend width, height and RGB mode. Thus
the earlier report's `exact_decoded_duplicates` field cannot establish pixel
uniqueness across those conventions. Future reports use the narrower name
`stored_decoded_hash_repeats` and explicitly state this limitation. The other
1,138,314 rows have no decoded-RGB hash in this inventory. This metadata audit
does not reread RGB, verify perceptual near-duplicates, classify indoor
versus outdoor content, or describe geometry after exclusions. Its stage is
explicitly `coverage_audit_complete_not_reviewed`, not a passed training gate.

The accounting reader now records valid-pixel depth histograms during each
selected cache member's existing integrity readback. Native calibrated depth
uses meter bins. Teacher depth uses ratios to each image's valid-depth median;
its nominal scale never enters a metric coverage claim. Both pixel-weighted
and image-weighted histograms are reported, along with valid support. The final
accounting-mode audit must run after every frozen row has an outcome and must
be reviewed with source/scene balance, near-duplicate sampling, image content
and post-augmentation loss participation. Snapshot `code-coverage-audit-v1`
(`c932840`) passed 62 CPU coverage/model-probe/admission/training/replay/loss
tests. This is tooling validation, not completion of those corpus reviews.

The complete ScanNet source audit identifies 25 unresolved scenes containing
35,783 frozen training images. `scannetpp-review-teacher-routing-20261102.json`
binds that exact subset and its original source reports. The new producer
`scripts/cache_scannetpp_review_teacher.py` prepares separate, nonmetric MoGe-V2
candidates from unchanged RGB with the original anonymity mask. It neither
repairs native geometry nor clears source-review flags. Only mask value 255,
with conservative erosion, can intersect teacher support; it adds no pixels
or interpolated support. Each scene contributes its first and last view to a
50-image preflight. Bulk generation additionally requires a matching numerical
preflight and a hash-bound visual review covering all 25 scenes. All inference
uses the existing GPU lock and pinned model environment. The first producer
snapshot passed 40 CPU source-selection/coverage/model-gate/admission/loss
tests; snapshot `code-scannet-review-teacher-v2` (`8e2ed09`) additionally passed
the three focused tests after its bounded scene-index query correction.

`scripts/audit_corpus_rgb.py` now prepares a separate, read-only RGB census. It
verifies encoded file hashes, checks each supplied decoded hash under its source
convention, applies only the frozen quarter-turn orientation, and computes one
size-bound RGB hash for every source. It records both raw and oriented pixel
hashes, leaving images and frozen manifests unchanged. Its bounded preflight
selects large/small/stable-hash scene groups and consecutive first/middle/last
frame windows plus hash-selected views. A 128-bit horizontal/vertical grayscale
descriptor retrieves visually similar pairs; collisions are candidates for
inspection, never automatic exclusions or proof of duplicate scenes.

The full RGB census requires a matching successful identity preflight and uses
one CPU process with a bounded SQLite cache. It describes exact pixel repeats;
it does not perform an all-pairs perceptual search over the full corpus. Canonical
comparison with held-out images remains a separate isolation check. Snapshot
`code-rgb-audit-v1` (`852cf7a`) passed all 28 RGB/coverage/admission tests in the
established model environment. An initial run in the native preparation
environment passed 25 tests but lacked `cv2` and `huggingface_hub` for three
trainer tests; rerunning in the model environment resolved those dependency
failures without code changes. The actual RGB census uses the native preparation
environment, where the RGB tests passed. No actual-model training gate was run.

RGB preflight 01 stopped before image reading because download task IDs were
mistaken for scene/visit groups. Snapshot `code-rgb-audit-v2` (`191dd62`) derives
those groups from the frozen index and passed 13 focused tests, including that
failure case. `rgb-identity-preflight-02` then read 1,524 selected images across
all eight sources with zero identity/orientation/hash failures. No exact
canonical pixel repeats occurred in this sample. The descriptor retrieved
237 similar pairs, all within existing groups; this is not a prevalence estimate.
All eight source overview sheets and 14 closest-pair sheets were inspected.
The IRS examples were adjacent left-eye frames; two additional exact-directory,
same-frame stereo pairs were inspected separately in `rgb-stereo-spotcheck-02`.
No stereo disparity or calibration claim follows from that RGB comparison.

The inspected sheets show overlapping adjacent views and Structured3D empty/full
variants with similar appearance but different details. ARKit includes repeated
poster designs across visits; visit IDs are not certificates of unique physical
scenes. Six source overviews are indoor, while the COCO/TartanAir samples include
outdoor content, with dark and underwater TartanAir examples. Textureless/very
bright Structured3D walls also remain visible. These are sampled observations,
not corpus-wide content percentages or automatic exclusion decisions.

`rgb-identity-census-01` is now rereading all 1,637,909 frozen training RGB rows in
one low-priority CPU process using the unchanged v2 snapshot. It writes a new
sidecar database, with no GPU, source edits or optimizer updates. It must finish
before claiming full canonical exact-duplicate results. Canonical held-out
comparison and final sampling balance after geometry exclusions still remain.

## Completed teacher preflights and the blank-image rejection

On November 2 at 22:07 UTC, the IRS producer completed all 133,879 candidates
and their full payload readback with no producer rejections. The queued NYU,
ScanNet review-scene and Hypersim review-scene preflights then completed. All
12 NYU sheets (45 views), 13 ScanNet sheets (50 views), and 29 Hypersim sheets
(116 views) were actually inspected. These are candidate-generation reviews,
not full source acceptance or calibrated depth validation.

NYU's 45 views cover 42 scenes and 24 scene types, including all six raw-projection
regressions. Its review permits separate nonmetric teacher candidates while
native registration remains unresolved. `nyuv2-teacher-cache-01` subsequently
completed all 1,043 images and full readback with zero rejections at 22:25 UTC.
The 116 validation and 290 test images remain outside this producer's scope.
ScanNet's review covers all 25 quarantined scenes; original anonymity holes
remain excluded. Its separate 35,783-image teacher job is queued behind the
Hypersim producer on the existing GPU lock. Neither review clears native flags.

The first Hypersim visual review **failed**, despite its numerical preflight
passing: `ai_012_007/scene_cam_01/frame.0000.tonemap.jpg`, corpus ordinal 210812,
is entirely white. Its teacher output had full support and nonconstant depth.
`hypersim-review-rgb-degeneracy-01` verified the original bytes and RGB extrema
for all 3,681 unresolved native inputs. Exactly this one input is spatially
constant; the other 3,680 are nonconstant. This finding applies to that subset,
not to every image in the corpus. The explicit exclusion retains its frozen
identity, original native failure, diagnostic evidence and reason. It must also
be honored when preparing the eventual RGB replay manifest.

Snapshot `code-hypersim-review-teacher-v2` (`74b7ba3`) binds the complete RGB
diagnostic and explicit review, verifies the constant image again, and rejects
it before inference. Ten focused input-exclusion, source-scope and outcome
tests passed. A one-pixel variation cannot be called exactly constant;
unreviewed, unexpected or duplicate preflight rejections cannot pass.
`hypersim-review-teacher-preflight-02` then completed 115 candidates with one
reviewed input exclusion. All 115 retained RGB rows and actual depth, support
and intrinsics arrays are bitwise identical to the previously inspected
preflight. The second visual review explicitly references that parity and the
original sheets; it does not claim newly rendered sheets were viewed.
`hypersim-review-teacher-cache-01` is generating the full 3,681-input scope with
that exclusion. Its outputs remain separate nonmetric candidates; in particular,
the original `ai_009_003` ray-direction discrepancy is still unresolved.

## IRS complete outcome distributions and selected extremes

`irs-teacher-outcomes-01` accounts for all 133,879 frozen inputs once, across
13 environment families and both eyes. It validates bound receipt metadata
after completed producer readback; it does not independently reread every
payload. There are 132,863 fully supported teacher maps. Relative p99 depth
exceeds eight times its image's valid-depth median in 9,548 images, sixteen
times in 2,579, thirty-two times in 426, and sixty-four times in four. The
largest ratio is 77.717. These are per-image relative quantities, not meters,
confidence scores or proof that a sample is wrong.

`selected-view-qc-01` independently reread 104 actual RGB/member payloads:
the two largest relative p99 tails and two least-supported images per family
and eye. All 26 sheets were inspected. This deterministic extreme selection
does not estimate prevalence; ties in full support can select adjacent views.
Nearby furniture, plates, partitions and doors sometimes dominate the median,
leaving narrow distant backgrounds. Kitchen, SCLRestaurant and ConvenienceStore
tails also occur at glazing or exterior regions. Dark interiors, fine plants,
mirrors and displays remain uncertain; a full teacher mask is not confidence.

The saved review flags ten specific images for further source-quality decisions:
three blurred Bistro window close-ups, five Diner dark/framed-panel views,
and two nearby Supermarket shelving views with differing predicted background
depth. These are observed ambiguities, not automatic exclusions or verified
physical errors. No far-depth threshold has been used to repair labels or
declare all high-ratio samples invalid. Original native disparity concerns,
source-quality acceptance, canonical heldout comparison and corpus admission
remain open. Pseudo replacements must not inflate unique-image coverage.

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
