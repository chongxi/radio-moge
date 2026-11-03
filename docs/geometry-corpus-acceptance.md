# Geometry corpus: acceptance before training

Chongxi authorized offline labels for the frozen 1,637,909-image training manifest
on 2026-11-02, and training **only after** full data, depth, loss and coverage
acceptance. This supersedes the earlier unconfirmed cache-scope proposal. The
small 81-view GT pilot is not sufficient acceptance for the expanded corpus.

On November 3, Chongxi proposed starting with the roughly 840k already generated
candidates if preparation did not require the GPU. Live inspection found two
GPU preparation jobs: ScanNet++ mesh rendering and MoGe review-teacher inference.
The first response therefore does not assume an idle GPU or preempt these jobs.
Preparation now also prioritizes an immutable first-stage candidate snapshot so
that a verified subset can be used without waiting for all 1,637,909 labels.
The proposed subset still needs its own payload, source, coverage, heldout, loss
and actual-model checks. No previous failed or unfinished check becomes a pass.

### Priority start using already reviewed sources (November 3, 06:33 UTC)

Chongxi subsequently asked why training had still not started. Waiting for the
entire 870k snapshot had unnecessarily held back completed reviewed sources.
The first optimization stage is now explicitly narrowed to those sources;
preparation and review of the remaining sources continue separately.

`reviewed-stage1-data-02` accounts for all 234,263 original inputs in COCO,
Hypersim, IRS and NYUv2, selecting **234,036** images after 227 reviewed
exclusions. Selected supervision is **55,462 native Hypersim synthetic GT** and
**178,574 relative teacher labels**. The five source policies already completed
their producer's full payload readback and source review. Accounting reuses
those completed proofs, checks bound receipts and unchanged pack size/mtime,
and preserves every native/external quarantine flag. Every actual training read
still verifies RGB and geometry-member hashes and decodes the arrays. Receipt
quantiles are reported as quantiles, not invented pixel histograms. Neither
the unfinished snapshot census nor unfinished source reviews are prerequisites
for this narrower stage. The broader snapshot remains unchanged.

The canonical comparison covers all **796,477** replay RGB images from the
seven completed RGB-census sources, after excluding the known white image.
It checks all 190,954 heldout memberships, with zero exact raw/oriented pixel
overlap and zero forbidden shared-scene rows. The incomplete ScanNet RGB census
is not claimed complete or included in this first replay pool. All four geometry
sources are contained within this checked RGB scope.

Source/group sampling weights for geometry are COCO .35, Hypersim .35, IRS .25
and NYU .05. Half the feature slots use the seven-source RGB replay mixture.
The initial segment retains the prepared 400x640, adapter-only, archived-step
232288 initialization, learning rate 1e-5 and batch1/accumulation16 settings.
It logs every update, saves every50 and is bounded to1000 updates/two hours.
The initial full validation sweep is deferred to the scheduled validation
interval so it does not delay the first update.

The actual loader checked50 seeded extreme/scene views. One COCO crop has no
retained points; this is an explicit zero geometry contribution with feature
supervision retained, as in the production loss. It is not filled or treated
as measured geometry. The model preflight now exercises that path while still
requiring positive geometry gradients in every source/label stratum. This
targeted1/50 observation is not a corpus-wide prevalence estimate. Sixty-seven
remote tests passed in6.95s, including the sparse path and a negative test that
rejects a source with no geometry gradient.

The old waiting eight-source launcher was stopped before any model ran.
`reviewed-stage1-launch-01` uses the existing GPU lock and the four completed
nonmodel gates, then runs the actual zero-update model preflight and seals only
on success. To prioritize the requested training, ScanNet review-teacher
generation was interrupted after110 closed packs/28,160 images. Its pause
receipt preserves their exact hashes; remaining7,623 images must resume in a
new attempt after recovery verification, without regenerating the entire set.
Other native generation and the full RGB/payload censuses continue.

The first launcher was stopped during input loading, before models or an
optimizer were constructed. Its replay export had retained NYU depth fields,
and the legacy reader resolved/statted every replay file at startup.
`reviewed-stage1-replay-02` strips all non-RGB fields while proving the same
796,477 ordered image identities, groups and orientations. Its manifest SHA256
is `1ff48734105a0ca14532f2a511916d7db9cc057ece37d2d74210bc860ea81aaa`.
Code `95f1ba3` checks this complete frozen manifest once and checks image bytes
on each actual replay read. Changed manifests and changed sampled images fail
tests. The canonical isolation result remains bound to the same RGB inputs.
Sixty-eight remote tests pass in 6.95s. The replacement launcher is
`reviewed-stage1-launch-02`, using immutable `code-reviewed-training-v4` and
`reviewed-stage1-readiness-02`; the old attempt and evidence remain preserved.

The actual archived-model probe passed all 50 paired/replay forward/backward
cases. All five source/label strata have nonzero geometry gradients; the one
empty crop follows the explicit feature-only path. Parameter and buffer hashes
prove RADIO, MoGe, adapter and feature statistics unchanged during this probe.
Peak allocated GPU memory was 10,531,868,160 bytes. The scoped acceptance seal
is `reviewed-stage1-launch-02/acceptance.json`.

The first trainer process then stopped before any optimizer update because its
remote environment lacked the workspace W&B authentication. Its three metadata
files are preserved in `reviewed-stage1-training01-preupdate-failure-01`.
`reviewed-stage1-training-restart-01` revalidates the unchanged acceptance seal
and reuses the completed model probe, supplying the existing connected account
only through the process environment. No credential is recorded in evidence.
Actual optimizer updates began at approximately 06:52 UTC. The active output is
`reviewed-stage1-training-01`, initialized from archived feature step 232288;
its new-stage step counter starts at 1. Online tracking:
https://forge.coreweave.com/wandb/brainsim/radio-moge/runs/f6e8d335 .
This acceptance covers the reviewed four-source stage, not the unfinished full
corpus or the unrelated 870k candidate snapshot.

`configs/geometry-stage1-candidates-20261103.json` lists all current primary,
recovered and review-teacher sources, completed source reviews and the 238
explicit geometry exclusions known at capture. `scripts/snapshot_geometry_candidates.py`
captures only closed tar/receipt pairs at a fixed cutoff while producers keep
running. It preserves original RGB identities, native and external quality flags,
the distinction between native and pseudo labels, and separate candidate counts
versus unique RGB counts. The snapshot cannot authorize training and does not
change the original full-corpus accounting gate. The active attempt is
`stage1-candidate-snapshot-01`, from `code-stage1-snapshot-v1` / `2182cc0`.
Four focused tests passed locally and remotely, including incomplete tails,
late receipts, changed original identities, duplicates and wrong exclusion
ordinals. Previously source-reviewed eligible images total 234,036; this is not
yet a training-ready subset or a claim that all reported candidates are GT.

The snapshot completed at 04:41 UTC on November 3: **4,098 published packs,
873,626 candidate entries and 870,550 unique RGB identities**, with all eight
sources represented. Native/teacher alternatives are counted once in the RGB
total. Its database SHA256 is
`6ce23b197f2c510c0996e65057e7b5d79d56e295091dc16b6f9c915619b92831`.
This completes metadata capture, not full payload verification or admission.

### Requested three-way point-cloud sanity viewer

Chongxi explicitly requested selectable GT/cache versus MoGe versus Adapter
point clouds from these packs. The prior cache and image-alignment checks did
not include this exact eight-source three-way comparison. It is now exported
as `stage1-sanity-clouds-01`, using `scripts/export_stage1_sanity.py` (`44fb97f`):
six distinct scene groups per source, with each available label kind represented,
plus three explicitly marked flagged/excluded diagnostics, **51 images total**.
Selection was fixed before model inference; no outputs were used to choose images.

Every selected member payload and original RGB hash was verified. Reference
observations retain native depth and camera geometry without hole filling.
The same prepared RGB was sent to the resident MoGe/Adapter service, with exact
pixel parity asserted on returned inference arrays; Adapter is the archived
feature checkpoint at step 232288, before geometry training. Reference projection
and both models' point/depth/intrinsics consistency checks passed on all 51.
Actual arrays and per-image provenance remain in the remote `arrays/` directory.
The public manifest SHA256 is
`ab813ef10caecf58446642889f4d17126e1b0ff9f31fccde426067ee7b8aca85`.

Viewer: <https://buildhub-nano-test--p7873.pub.astera.work/> (BuildHub sign-in).
It offers dataset/scene selection, synchronized orbit/zoom/pan, overlay, raw
versus explicitly normalized display, fixed-reference versus own-mask support,
per-image numerical differences, and raw PLY export. Display/PLY are uniformly
decimated to at most 30,000 points per branch; numerical comparisons use all
reference observations. Pseudo-reference rows are clearly identified and do
not provide independent GT accuracy. Normalization does not change saved data.
The page only serves precomputed public assets; viewing needs no server GPU.

Three export-unit tests passed. The real browser check loaded all 51 samples,
tested source filters, search, previous/next, synchronized controls, support
switching, overlay and mobile layout, with no page errors or failed requests.
PLY bytes reproduce the original displayed coordinates/colors even when the
view is normalized. A follow-up test checked hash links, empty-search races and
fixed gallery sizing. Screenshots from one image in each of eight sources were
visually inspected: no obvious shared coordinate flip or RGB mismatch was seen;
surface/detail and scale differences remain visible. This limited inspection
does not certify all samples or all packs. This forward-only sanity check is
also **not** the required training forward/backward zero-update model gate.
Production optimizer updates remain zero.

### First-stage payload verification and one additional Tartan rejection

`scripts/verify_snapshot_payloads.py` (`7f9bec2`) verifies the fixed snapshot's
791,996,323,840 bytes with one CPU worker, without using the GPU. It decodes every
declared NPZ member, checks the exact RGB/policy identity, depth/support/camera
contract, receipt statistics, sequential USTAR headers/padding, and whole-file
SHA256. A pack verification is published only after the final whole-file check,
not after its last member happens to decode. Candidate-level metric/relative
coverage is recorded separately and never clears source or external flags.
The full run requires a code- and snapshot-bound preflight on both endpoint
packs of all 12 primary/recovered/replacement sources (24 packs). Fourteen
storage/snapshot tests passed remotely (1.05 s); the broader local set including
geometry sampling and admission passed 41 tests (5.52 s). No optimizer ran.

The actual `stage1-payload-preflight-01` then completed all **24 packs / 4,762
members / 7,435,008,000 bytes**, with zero errors, covering every source attempt.
`stage1-payload-census-01` is now running from immutable `code-snapshot-payload-v1`
with the same reader, one low-priority CPU process and one BLAS thread. Its
individual pack reports can support later scoped accounting, but the snapshot
payload gate remains incomplete until the full run publishes its final report.

The additional TartanAir rejection at exact ordinal **417529**, neighborhood
Easy/P000 frame003273, was separately reproduced from retained left/right
depth and RGB in `tartanair-rejection-review-02`. All source hashes and published
statistics agree. Its sky-dominated view has 27.0462% task-range support and
4.9863% final stereo support, below the unchanged 5% per-image threshold. The
six-panel evidence was inspected. Only this image's geometry is explicitly
excluded; its RGB remains in replay. Earlier 11 decisions were reused, not
re-audited or silently extended. This input is absent from the already frozen
stage-one candidate snapshot, which remains unchanged.

The merged exclusion ledger now has **239** images, at
`tartanair-rejection-review-02/known-geometry-exclusions.jsonl`, SHA256
`648dfdeeb8247247f941208c0ec8f8646aaecabba481094e4ccd80835a6b10c2`.
Only white RGB210812 is excluded from RGB replay; the other 238 are geometry-only.

### First-stage scope and automatic launch preparation (2026-11-03)

After viewing the three-way point clouds, Chongxi explicitly asked to start
training. `stage1-training-scope-01` (`64ee942`) now materializes every frozen
RGB identity and closed pack without waiting for later producer outputs. It
contains 870,550 unique images: 226 known geometry exclusions, 234,036 candidates
with completed source reviews, and 636,288 candidates awaiting source review.
The proposed geometry count is 870,324; this is not an accepted training count.
No additional image lacks an unflagged candidate in this particular snapshot.
The stage inventory SHA256 is
`aed3b3f3f83506128c356e2c1fc8a52829a7d59be3e065c1d4b4f8aab21cf5e9`.
The parent manifest remains the RGB identity anchor; the eventual corpus index
restricts training to this exact stage scope. Original full-corpus preparation
and the separate replay pool continue unchanged.

Of the 234,036 source-eligible images, 55,462 are Hypersim native synthetic GT
and 178,574 are relative teacher pseudo labels. They are not all measured GT,
and their source reviews do not replace final payload, coverage, isolation,
loss, or actual-model checks. Stage-only deferral preserves the original
quality flags and RGB eligibility. Eight scope/snapshot tests passed locally
and remotely, including changed identities and ambiguous alternative candidates.

`configs/geometry-stage1-training-20261103.json` prepares the first segment from
archived feature step 232288: adapter-only optimization, 400x640, one-image
microbatches accumulated over 16, constant learning rate 1e-5, pinned MoGe-2 core
geometry weights and 0.1 feature weights. Half the feature slots use the diverse
RGB replay pool. Both streams use explicit eight-source weights and scene/group
sampling. The first segment is bounded to 1,000 updates or two hours, saves
every 100 updates and evaluates every 250. These are prepared settings; no
training admission or optimizer update has occurred.

`stage1-launch-01` runs the CPU-only watcher from `35eed29`, waiting for
`stage1-readiness-01/plan.json`. Readiness must bind the exact requested snapshot,
configuration, checkpoint and four nonmodel gates. The watcher then waits on
the existing GPU lock, runs the actual zero-update forward/backward probe,
seals the verified evidence and starts training. It does not infer a pass from
a progress counter or a visual page. The separately archived execution directory
`code-stage1-training-v1` must be created from the final tested implementation
before readiness is published. Missing readiness is still an active dependency,
not a claim that training is already running. Twelve launcher/scope/snapshot
tests passed in the remote preparation environment.

`finalize_snapshot_source_readbacks.py` and `snapshot_readback.py` (`0b6569c`)
allow accounting to reuse the completed per-pack CPU census instead of reading
the approximately 792 GB of payloads again. Every captured member, receipt,
whole-tar proof, statistic and quality flag remains bound to its original
identity; changed packs or removed external flags fail. This route requires the
complete snapshot census, not its small preflight. The finalizer creates new
sidecar pack indexes without changing producer indexes, restricts earlier
source reviews to the captured scope, and leaves missing reviews pending.
No new source review or final gate is invented. The original full-corpus
accounting path retains its own array readback behavior.
The 33 accounting, payload-proof, coverage and model-probe unit tests passed
locally (7.13 s) and in the remote model environment (6.65 s). These include
incomplete census, changed pack, removed quarantine and missing source-review
failures; they are not the actual archived-model forward/backward preflight.

`configs/geometry-corpus-20261102.json` freezes the scope and teacher revision.
`scripts/index_geometry_corpus.py` creates a disk-backed inventory and checks
heldout metadata without reading evaluation depth. Its success is **not** depth
acceptance and cannot start training. Native assets remain immutable.

## Completed source-review milestone (2026-11-03)

COCO, IRS, NYU and both Hypersim label routes now have evidence-bound source
reviews. Eligible image counts are 39,984 COCO relative pseudo, 133,869 IRS
relative pseudo, 1,043 NYU relative pseudo, 55,462 Hypersim native and 3,678
Hypersim relative pseudo. The whole 59,344-image Hypersim scope reconciles once
to its two eligible label routes plus 204 explicit geometry exclusions.
The source reviews above used the 238-row merged corpus exclusion set. The later
Tartan rejection review adds one geometry-only exclusion (239 total); only
pure-white RGB210812 is also excluded from RGB replay. All native flags remain recorded.

These are source eligibility decisions, not corpus admission. The other four
sources are still generating; final full payload accounting, actual training
view coverage, canonical heldout isolation, loss evidence and the archived-model
zero-update probe remain required. The RGB replay sampler is still a candidate
policy, not a frozen training configuration. Production optimizer updates are 0.

## Hypersim combined accounting and historical status (2026-11-02 23:27 UTC)

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

## RGB replay candidate inventory (November 3)

`rgb-replay-candidates-01` streams every frozen inventory row into a separate
RGB-only candidate manifest. Exactly 1,637,908 images remain: the reviewed white
Hypersim image at ordinal 210812 is excluded, while all 11 TartanAir geometry-only
exclusions (381511--381521) remain in the RGB pool. The new
`rgb-replay-input-decisions-01.jsonl` makes both scopes explicit and binds the
original decisions and visual evidence. The blank image's bytes and constant
pixels were checked again; the complete corpus was not decoded again by this
metadata preparation tool. Original image identity, hash protocol, orientation,
source and scene are preserved; depth, masks and camera labels are not copied.

The candidate manifest SHA256 is
`4de531643528b46f72fc8f1317a5c20af5aadf234d50946ad063aaba26f25f27`.
Its 46,017 recorded groups are not a claim of physical scene independence.
Uniform image sampling would allocate 51.37% to ScanNet++ and only 0.064% to NYU;
the largest IRS and TartanAir families contain 22.92% and 22.20% of their
respective candidate images. Source/scene sampling and perceptual redundancy
therefore still need review. No sampling policy or training configuration was
issued. The canonical train/heldout pixel comparison also remains pending.

Snapshot `code-rgb-replay-v1` (`c7d1915`) passed eight focused tests locally and
remotely. They cover scope mistakes, missing RGB-specific evidence, repeated or
changed identities, stale input hashes, actual constant-pixel verification,
preservation of orientation/hash conventions and geometry-only RGB retention.
The output remains `rgb_replay_candidates_prepared_not_accepted` and cannot
substitute for the final data, coverage, loss and zero-update model gates.

## Hypersim full teacher extremes (November 3)

The completed route metadata contains 3,680 teacher candidates: 3,674 have full
support, the minimum support is 89.59%, and six have p99/median depth above 32
(maximum 110.68). These ratios are diagnostic values, not exclusion criteria;
full support is not calibrated confidence. `hypersim-teacher-extremes-02`
independently rereads the actual RGB and cached payloads for the six largest
ratios and four lowest-support views, verifying their statistics against the
completed routes. Entire tar files were not rehashed by this bounded display.

All ten full-resolution RGB/depth/mask pages were inspected. Close walls or
columns occupy much of several high-ratio images, with distant room structure
still visible; these ranks do not justify additional whole-image exclusions.
The first display's upper color limit of 16 saturated those distant regions.
The reviewed v2 uses one shared range [0.25,128]; selected payload hashes and
statistics match v1, and no depth or support was modified. Per-image median
normalization is not metric or multiview calibration.

The left glass/reflective enclosure in ai_041_008 views at ordinals 238221 and
238222 retains a specific surface-versus-background ambiguity. Both candidates
remain explicitly quarantined for a source decision in
`hypersim-teacher-extremes-02/visual-review-01.json`; their incomplete support
is preserved. No new exclusions or source-quality acceptance were issued.
All existing native failure flags remain recorded, including those on other
Hypersim scenes. This diagnostic is additional evidence, not the full quality
gate or a random estimate of error prevalence.

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
  orthogonality residual, and cross-check the separately stored world geometry. The
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
  without fitting either to depth; exact highres K had not been retrieved at
  that point. The later exact-camera review below corrects the assumption that
  it was unavailable: the publisher does release `wide_intrinsics.zip`. Within
  stable reference regions, frame median absolute differences ranged roughly 4--39 mm,
  and some frames still have substantial >5 cm disagreement and large tails.
  High confidence alone therefore does not complete ARKit acceptance. Preserve
  `arkit-faro-quality-probe-01`; review visibility/calibration and broader
  sensor consistency before admitting native labels. It is not permission to
  apply a fitted bias correction or relabel AppleDepth as FARO GT.
- The running Hypersim producer has encountered a further source-consistency
  issue in `ai_009_003`: many frames fail the stored world-position consistency check.
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
and two nearby Supermarket shelving views with unresolved local window-depth
interpretation. The later common-palette review withdrew the global background
color-difference argument for those two Supermarket views. These are observed
ambiguities, not automatic exclusions or verified
physical errors. No far-depth threshold has been used to repair labels or
declare all high-ratio samples invalid. Original native disparity concerns,
source-quality acceptance, canonical heldout comparison and corpus admission
remain open. Pseudo replacements must not inflate unique-image coverage.

## Later candidate quality findings (2026-11-03)

`corpus_admission.py` now requires every completed source review to bind a
`candidate_quarantine` file reference, including an explicit empty register
when no additional candidate flags apply. The register binds the manifest,
source policy and exact pack index. Each flag binds the frozen RGB identity,
ordinal, receipt, geometry key and member payload hash, with a reason and hashed
review evidence. Missing registers, stale evidence, wrong members and duplicate
flags fail closed. This adds a required field to source reviews; old review
records cannot silently imply that later findings have been considered.

Accounting rereads all original geometry payloads before applying these flags.
It stores both native and external flags in `quarantined_candidates`, counts
their union once, and never clears a native flag. A flag applies to one exact
candidate, so a separately reviewed alternative can still serve the same RGB.
Without an alternative or explicit evidenced exclusion, the image remains
unresolved and the accounting report cannot complete. The training gate checks
the registers and their evidence again. RGB replay is unaffected by candidate
quarantine; excluding an RGB input still requires its own scoped decision.

Code snapshot `code-candidate-quarantine-v1` is commit `06be9c0`. Fifty focused
tests passed locally (38 accounting/quarantine tests plus 12 model-probe/RGB
tests) and on the remote model environment (50 in 6.57 seconds). They include
unresolved visual flags, explicit geometry exclusion, candidate-specific
replacement in either input order, overlapping native/external flags, modified
evidence, and the existing tiny-model wiring checks. These are tooling tests,
not the actual-model zero-update gate or a source-quality pass.

`candidate-quality-quarantine-01/report.json` records 12 actual flagged teacher
candidates after matching the complete frozen RGB rows and immutable receipts:
10 IRS candidates and Hypersim ordinals 238221 and 238222. Its SHA256 is
`431109743087606add9c6f25ddcca7c291dfa65435c0afe68afe77fef12ed35e`.
The IRS source review must bind `candidate-quality-quarantine-01/irs.json`
(`428e5627e25e2633c12b4a7ebf0c3e93193771cefb11c35a93a31ac29239a3ee`);
the Hypersim review-teacher source review must bind `hypersim.json` in that
directory (`747b607cf23cc8d7acd27ca3a24d671ec1e40009bb417be4ca50e00762662dc1`).
These registers preserve the latest common-palette review findings, including
the withdrawal of the old Supermarket global-color argument. They create no
geometry or RGB exclusions, clear no native flags, and grant no source or
training acceptance.

The subsequent `teacher-geometry-quality-decisions-01` records a conservative
geometry-only exclusion decision for exactly these 12 images. Existing reviews
cannot settle the local surface-versus-background target, and no verified
confidence/surface mask isolates reliable supervision in the uncertain regions.
The decision omits those geometry targets without claiming proven physical
errors, applying a depth-ratio cutoff, or reviving the withdrawn Supermarket
color argument. Both affected Hypersim native outcomes were checked to be
flagged or rejected; this does not discard an eligible native alternative.

`teacher-geometry-quality-decisions-01/geometry-exclusions.jsonl` has SHA256
`c2d5f19b38eefe86f5eed4743672d7278c109eea05bd61714e513ac598d82486`.
The decision report is
`ef53a751852c2a545ad2e700a0a62dc145f8cf2e39ee7e710c48895c917777c1`.
All 12 exact RGB identities were found in the existing replay candidate pool;
the pool was neither rewritten nor rehashed in full by this decision step.
The exclusion rows explicitly set `scope=geometry_supervision` and
`rgb_replay_excluded=false`, with the original review and candidate evidence.
Final accounting must combine these 12 decisions with the existing one white
RGB exclusion and 11 Tartan geometry-only decisions, while retaining the
candidate quarantine registers. These 12 now have explicit geometry dispositions;
full source reviews, whole-corpus accounting and all training gates remain open.

## Completed teacher source reviews (2026-11-03)

`teacher-source-reviews-02` completes the source eligibility reviews for the
IRS teacher policy and the Hypersim review-teacher policy. IRS has 133,869
eligible relative-pseudo candidates and 10 quarantined candidates. The
Hypersim alternative policy has 3,678 eligible relative-pseudo candidates,
two quarantined candidates and one previously reviewed white-RGB rejection.
These are policy-specific source decisions; the native Hypersim source and
whole-corpus admission have not been accepted by these records.

The publisher compares all 137,559 cached teacher identities with the frozen
inventory, checks receipt hashes, support statistics, teacher provenance and
relative-loss routing, and requires every external flag to match its exact
reviewed geometry exclusion. It relies on the producer's completed full array
readback and the earlier independent selected-payload reviews; it does not
claim another full array readback or physical metric accuracy. Existing native
failure flags remain intact. The five source checks bind the documentation,
camera/unit policy, support, full receipt distributions, stratified visuals,
later common-palette reviews and explicit dispositions.

The documentation review also records that Hypersim's authors recommend
lossless HDR imagery for downstream learning. This frozen corpus uses its
existing lossy tonemapped RGB previews, without claiming HDR equivalence.
This review accepts the bounded relative-teacher use of those exact RGB inputs;
it does not alter the frozen images, certify native camera repair, or clear the
ai_009_003 native discrepancies. See the
[official Hypersim formats](https://github.com/apple-aiml-research/ml-hypersim#lossy-preview-images).

The source reviews bind their required candidate quarantine registers:

- `teacher-source-reviews-02/irs-source-review.json`:
  `5b5b16efbcb098e321ec31bfed14bad865816959dec02f4f2e059c72b6e10d46`.
- `teacher-source-reviews-02/hypersim-source-review.json`:
  `436ef7bc6d4d3a0bcbababc6ea116b999a17974739fe537e92df8672f538333a`.

`known-geometry-exclusions.jsonl` in the same directory combines the existing
24 exact decisions, with SHA256
`4bfdf2b5e64cf7510994fb1926e572f5bcba73737eade36c75ce175c2f50225c`.
It preserves all original evidence and scope fields: white RGB 210812 is
excluded from geometry and replay; the 11 Tartan and 12 teacher-quality
exclusions apply only to geometry. Future findings require additional explicit
decisions. Full accounting must still apply the quarantine registers before
these exclusions, reread every selected array, and resolve every other source.

Code snapshot `code-teacher-source-review-v2` is commit `fe7981c`. The first
attempt was stopped before any source-review publication because random
SQLite reads over the shared filesystem were slow. The second attempt streams
each bounded source into encoded-row lookup memory with the same identity
checks; it completed successfully. This does not modify any producer process,
source labels, or prior evidence.

## RGB replay source and scene sampling (2026-11-03)

`feature_replay.mixture_weights` now selects the existing deterministic
source → frozen group → frame sampler for RGB replay. The weights must be
finite, positive, and name every replay dataset exactly once. Omitting them
preserves the older uniform stream. Replay still replaces half of feature
supervision slots; geometry sees every original microbatch. The sampler uses
the replay cursor, so odd and even interruption points resume exactly without
advancing the geometry random stream.

The initial policy in `rgb-replay-sampling-candidate-01/policy.json` assigns
15% each to ARKitScenes, COCO, Hypersim, ScanNet++ and TartanAir; 10% each to
IRS and Structured3D; and 5% to NYUv2. This gives equal aggregate weight to
real-image and synthetic-image source families. These are design weights for
the RGB replay slots, not fractions of the geometry stream or an optimized
performance result. Canonical heldout isolation and final coverage acceptance
remain prerequisites to freezing the training configuration.

The actual `MixtureBatches` implementation was exercised for 100,000 draws
using the candidate pool's complete 46,017 group counts. All eight observed
source shares were within 0.5 percentage points of the configured weights:
ScanNet++ was 14.985% (uniform-frame sampling would be 51.372%), and NYU was
4.874% (uniform-frame sampling would be 0.064%). Uniform within-source groups
reduce the largest IRS family's expected share from 22.918% to 7.692%, and
the largest Tartan environment's share from 22.198% to 9.091%.

Sampling uses replacement: the NYU simulation drew 4,874 times from 1,000
distinct group/frame slots out of 1,043 available images. This expected
repetition is recorded, not hidden as independent image coverage. The audit
uses reconstructed group/frame slot IDs, not actual RGB payloads or the final
training batch schedule. Frozen groups still do not prove physical scene
independence, and this sampling change does not perform near-duplicate removal.
The original candidate pool remains unchanged, including its single white-RGB
exclusion and retained geometry-only exclusions.

Sampling report SHA256:
`53e0caf83d705dab0ddf9b31bfe57e10c42aa82cc6daf340761a916f3501a3f8`.
Code snapshot `code-replay-mixture-v1` is commit `bb2adc6`. Seven replay tests
and 45 related admission/model-probe tests passed locally; all 52 passed in
the remote model environment in 7.08 seconds, with CUDA disabled. The tests
cover source/scene imbalance, invalid weights, missing sources, exact resume,
preserved geometry batches, unchanged random streams, and gate enforcement.
They use tiny CPU models and do not replace the actual archived-model probe.
The tooling record SHA256 is
`afb1ea6d87401a619c76aaba3bd0f49b27e072a24455e7031d61024a5073b7d9`.

## Completed COCO and NYU teacher source reviews (2026-11-03)

`coco-nyu-source-reviews-01` records source eligibility for **39,984 COCO** and
**1,043 NYU** relative-pseudo candidates. It does not open corpus admission.
All 41,040 original teacher receipt identities, statistics, provenance and
loss routes were checked against the frozen source audits; completed producer
array readback remains bound. Legacy NYU rows lack width/height fields, so the
audit decodes and checks the original RGB to obtain dimensions without changing
those rows. Eleven local/remote audit tests passed, covering changed RGB,
incomplete readback, duplicate/outside identities and incorrect dimensions.

COCO review inspected 39 full-resolution preflight/aspect/extreme views, then
all 101 candidates with less than 5% original valid support. The latter number
is a diagnostic selection, not an exclusion threshold. In total, 131 distinct
images were inspected. Many sparse masks legitimately cover small birds,
planes or kites against unsupported sky. Thirteen exact candidates were
excluded from geometry for visible text/graphics, letterbox support, or
specific unresolved sky/surface ambiguity. This is not a calibrated physical
error claim or an automatic tail-ratio cutoff. All thirteen RGB images were
verified in the existing replay pool and remain available there.

The sparse-support diagnostic exercised the actual camera-aware loader for
101 images, two output shapes and center/two seeded augmented views: 606 views.
Eighteen views from nine images had fewer than 32 selected observations.
Two of those images are now excluded for separately reviewed watermark support;
seven remain candidates requiring explicit treatment in final crop/coverage
review. The loss already reports zero geometry contribution below 32 supported
observations. Original support ranges down to 71 pixels and must not be filled
to make a crop pass. This source decision is not final training-view coverage.

NYU's ten full-cache extreme/support views and the previously inspected 45
preflight views cover the existing 24 scene types. Minimum original support is
99.604%; no new specific image exclusion was found. Native RGB/raw-depth
registration remains unresolved, and these teacher labels do not repair or
authorize the native route. Intrinsics are teacher estimates and supervision
remains relative.

The source reviews bind the required thirteen-entry COCO and explicitly empty
NYU quarantine registers. Report SHA256:
`df77a02f510a0cbde0f693658fd52e912566b86d7607bde25f776ae6ca0abc5e`.
COCO review SHA256:
`e5b8c319ec71939405bb4d9f83501579907cf0ee3a2991cb4158e478b4408698`.
NYU review SHA256:
`1e00bcb87f33ba1a411601018aa02d892ed0486bc91e2343a4dc727794c18d2a`.
The then-current 37 geometry exclusions remain an immutable intermediate record;
the later merged native-review register below is the latest known set.

## Hypersim native scale and precision review (2026-11-03)

`hypersim-native-quality-01` checks all 58,739 completed native receipt
identities against the existing native outcome audit: 55,663 initially
unflagged candidates and 3,076 original flagged candidates. Its ten selected
actual payloads cover the lowest support, farthest depth and largest storage
uncertainty. `hypersim-native-near-depth-01` adds the two nearest median depths.
All twelve pages were inspected. Lowest-support views show a facade with
invalid background and retain 29.7--32.3% support; no new exclusion follows
from that support fraction.

The far-depth extremes belong to `ai_013_001`. The complete 200-frame scope
has median camera-Z ranges 277.17--314.80 m in cam_00 and 170.45--173.10 m in
cam_01. The corresponding first/middle/last RGB images show a colonnade and
downward views of tiled floor and column bases. The source's
`meters_per_asset_unit` is already 0.009999999776482582.
`hypersim-native-scale-probe-01` checks the retained source receipts and
reproduces cached depth/masks exactly for six frames across both cameras and
the two near-depth cases. These large distances already occur in the original
radial/world data; this is not a missing extra centimeter conversion in our
loader. No guessed scale correction is applied.

The [publisher's exporter](https://raw.githubusercontent.com/apple/ml-hypersim/main/code/python/tools/generate_hdf5_from_vrimg.py)
derives radial distance from world position, camera position and scene scale,
then stores position and distance as FP16. These arrays share renderer
provenance. Their consistency checks storage/conversion, not independent
physical calibration or realistic scene scale. Earlier references to an
"independent world-position check" must be read with this limitation.
Raw radial and finite-world quantiles in the probe use different finite-support
sets; common-support relative-error statistics are the appropriate comparison.

The inspected near frame 251659 (`ai_052_002/cam_01/frame.0021`) has almost
featureless RGB, median Z 0.000917 m and a median world-norm/radial relative
discrepancy of 1.4075. The original absolute quantization gate still passes all
pixels because world-coordinate storage uncertainty exceeds the tiny range.
That is insufficient verification for metric supervision of this exact image,
not proof that its original radial distance is physically wrong. By contrast,
textured near frame 210303 has median Z 0.02254 m and p99 relative discrepancy
about 0.0166; it remains a candidate. Near/far depth alone is not a rejection rule.

`hypersim-native-extreme-review-01` therefore records exact geometry-only
exclusions for all 200 `ai_013_001` images (ordinals 211112--211311) due to the
scene-scale plausibility concern, plus frame 251659 due to its precision and
visual-support limitation. All 201 candidate identities bind the native policy,
pack receipt, member, RGB and visual/source evidence. All 201 RGB images were
verified in the unchanged replay pool. The 201 dispositions are complete;
they must not be repeatedly treated as undecided findings. Original 3,076
native flags remain intact. This exclusion record itself is not a source pass
or a training gate. The subsequent source review below binds the remaining
55,462 native candidates; final corpus coverage is still required.

At this milestone, known geometry exclusions were **238 unique inputs**, in
`hypersim-native-extreme-review-01/known-geometry-exclusions.jsonl`, SHA256
`d334f481ad592d79ba0af1a86ca92569454f39e605befd0f9f5550a94158916d`.
Only the existing pure-white RGB 210812 is also excluded from replay; the other
237 exclusions are geometry-only. Native quarantine register SHA256:
`f2d9ef08f80c6c20ddc820df64feefccdb958b728aa6fa4a8cceff9be85a9229`.
Review report SHA256:
`a2a50a21d27d736fde7fa3236ed5854c4885831b0aea531c58b35e9adf8060e5`.
No geometry arrays, native producer policy, existing combined-route evidence,
or prior source reviews were overwritten.

`hypersim-native-source-review-02` subsequently completed the native source
review and a fresh reconciliation of all 59,344 original Hypersim identities.
It rechecks all 58,739 native receipt identities and their unchanged flags,
binds the producer's full array readback, camera census, RQ/tilt verification,
earlier source-mismatch diagnostics and the new extreme-scale review. The
native source review uses its mandatory 201-entry external quarantine register
in addition to the original 3,076 flagged native candidates. It accepts 55,462
native candidates from 355 frozen scene groups. This is synthetic source-scale
supervision with documented limitations, not independent real-sensor accuracy.

Its new source-route ledger contains exactly 55,462 native-eligible, 3,678
relative-teacher-eligible and 204 excluded images. No original image is missing
or selected twice. The old combined-candidate ledger remains unchanged.
Native source-review SHA256:
`1be522b9b26d9aaae5644510967c82b48f637e0c4e43720b92af585d07bd5296`.
Source-route ledger SHA256:
`580806095875a4826a772fc3b5b0c6a46ae24801e68cb6149f6f82504f846c48`.
Completed report SHA256:
`a937b494336bf12c8c197d8ef332d7e438efbae6b0c7a115ac38daef7caee237`.

Source-review attempt 01 stopped before publishing a pass because the legacy
white-image exclusion omits a redundant scene-group field. Attempt 02 binds
its exact original ordinal/path/hash/dataset and checks group whenever present,
without rewriting the legacy record. Nine compatibility and negative identity
tests passed locally (0.10 s) and remotely (0.80 s). Changed paths, hashes,
datasets, explicit groups, ordinals and decisions still fail. No actual model
forward/backward probe or optimizer update occurred.

## Active scoped training and first validation (2026-11-03)

The separately accepted first stage is running from immutable execution commit
`95f1ba3`: 234,036 geometry images (55,462 native Hypersim synthetic targets and
178,574 relative teacher targets), with 796,477 RGB replay images. Its exact
configuration, acceptance and 50-case actual model backward probe are bound in
`reviewed-stage1-launch-02`; the active wrapper is
`reviewed-stage1-training-restart-01`. The initial launch's pre-update W&B error
does not describe the active run. The unfinished full corpus remains unaccepted
and is prepared independently; its new decisions do not change active inputs.

`reviewed-stage1-validation-250-review-01/report.json` compares the first
250-step validation with the archived feature-step-232288 initialization using
identical images, order, manifests and preprocessing. Frozen teacher GT metrics
are unchanged per image. NYU AbsRel improves from 0.079412 to 0.074031, while
ScanNet++ LiDAR AbsRel worsens from 0.092118 to 0.096241. Equal-source macro
spatial cosine decreases from 0.961692 to 0.959636, and summary cosine from
0.932301 to 0.931899. This is a mixed first evaluation, not evidence of overall
improvement; training loss alone does not establish validation improvement.

At step 500, the same paired checks and per-image macro recomputation pass in
`reviewed-stage1-validation-500-review-01`. Spatial cosine recovers to 0.959927,
still below initialization; summary cosine is 0.931840. NYU AbsRel is 0.075632
(4.8% below initialization, worse than step 250), and ScanNet++ LiDAR AbsRel is
0.095354 (3.5% above initialization, better than step 250). IRS accounts for the
largest spatial-feature decrease: 0.964346 to 0.956972. The configured metric
checkpoints retain the best observed stage result separately: NYU/summary at
250 and spatial/LiDAR at 500. No active loss, sampling or acceptance changes
were made from these intermediate results.

## Completed Structured3D outcomes and candidate review (2026-11-03)

`structured3d-outcomes-01` accounts for all 134,749 original inputs: 133,572
native candidates (27,008 recovered and 106,564 newly generated), and 1,177
rejections requiring disposition. Rejections comprise 467 upstream errata,
686 insufficient layout references and 24 insufficient valid support. All 686
source flags belong to rejected images; no cached candidate carries that flag.
The audit binds completed producer readback and checks every receipt identity;
it does not claim a second full payload audit. Sixteen relevant tests passed
locally and remotely.

All 24 selected extremes were read back and visually reviewed on 12 pages in
`structured3d-extreme-visuals-01`. Original RGB, a common depth palette, relative
depth and unchanged support were inspected. The follow-up covered all nine
original images in scene 00103: ordinals 532493 through 532501 have nearly
uniform high-intensity RGB despite detailed geometry. Exact geometry-only
exclusions and both policy-specific quarantine registers are published in
`structured3d-scene00103-decisions-01`. RGB replay is retained; these are not
exactly constant images or a general brightness cutoff. The merged known
geometry-exclusion ledger now has 250 rows, SHA256
`0926ea364696e21e139fc5662ec6c9444888f247e02e5455f9f7b0f5c4f8a9f0`.
Only the previously excluded Hypersim white image is also excluded from RGB.
Structured3D source acceptance, rejection dispositions and crop coverage remain
pending. These artifacts do not modify the active training index or seal.

The subsequent `structured3d-rejection-census-01` reproduced all 710 non-errata
rejections from retained original RGB/depth/normal/semantic/camera/layout ZIP
members. Every member SHA/CRC and original RGB identity was checked; every
computed statistic matches the producer exactly. The other 467 original inputs
match the bound upstream errata. No whole archive rehash, label inference or
threshold change was performed. A four-image real preflight passed, and 22
tooling tests passed locally and remotely.

All 24 census-selected extreme pages were inspected after the four preflight
pages. Window/exterior or blinds dominate the sparse-support views. Some views
have smooth, extensive native surfaces but no matching structural reference;
others retain fewer than 100 verified layout pixels amid large conflicts. These
observations do not prove that every excluded depth is physically wrong. They
do support retaining the unchanged native eligibility rules without guessed
scale correction, mask filling or clearing source flags.

`structured3d-rejection-decisions-01` records exact geometry-only exclusions for
all 1,177 inputs and retains RGB replay. Together with the previous nine external
candidate exclusions, Structured3D now has 133,563 remaining native candidates
and 1,186 explicit geometry exclusions, with no undecided rejection. Its source
review and crop coverage remain separate unfinished steps. The latest global
ledger has 1,427 geometry exclusions, SHA256
`dc8c6d04ec55d5e61e6fefac7ec8156adeffaf9bd36ba15965ec520e52605cae`.
Only the prior Hypersim constant-white input is excluded from RGB. The active
training corpus, configuration, optimizer and seal are unchanged.

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
### Paired step-750 validation and Structured3D eligibility, 2026-11-03

`reviewed-stage1-validation-750-review-01` reuses completed evaluations and
checks the same image order, manifests, preprocessing and frozen teacher GT
values against the archived initial adapter and steps 250/500. Original-view
equal-source spatial cosine is 0.9599162064, essentially unchanged from step
500's 0.9599270676 and below the initial 0.9616920552. Summary cosine is
0.9318284911 versus initial 0.9323009271. NYU AbsRel improves to 0.0732686892
(7.74% below initial), while ScanNet LiDAR AbsRel worsens to 0.0975475884
(5.89% above initial). This is mixed performance, not overall improvement.
The report SHA is `b399ea8c01c09f8280e5f072a690f402fa512885da8f6d29fb2d9b3b0e325d32`.
The active immutable training configuration and optimizer remain unchanged.

`structured3d-source-reviews-01` completes qualified source eligibility for
both preserved native policies using the existing full outcome audit,
producer/recovery payload readback, camera census, conversion regression,
array parity, actual visual inspections and explicit rejection decisions.
All 134,749 original images receive one route: 133,563 native eligible and
1,186 geometry exclusions. The recovered policy contributes 27,004 eligible
and four quarantined candidates; the new policy contributes 106,559 eligible,
five quarantined candidates and 1,177 source rejections. These exclusions
retain RGB and the original source flags. Eligible images cover 2,987 groups.
The mandatory policy-specific quarantine registers remain bound. This does
not complete crop coverage, canonical heldout isolation or the actual model
gate for adding Structured3D to a later training stage. It does not change
the currently accepted 234,036-image geometry training set.

The publisher's source-route/identity and existing outcome/rejection tests
passed 23 cases locally and remotely (remote 1.68 seconds). Completed array
audits and visual inspections were reused rather than counted as new work.

ScanNet review-teacher continuation now supports explicit verified recovery
and exact remaining-scope filtering, rejecting foreign/duplicate identities
or changed inference/mask policies. Eleven relevant tests passed remotely.
The runnable continuation snapshot is `code-scannet-review-teacher-v4`, commit
`5228f12`, applying this change to original snapshot `8e2ed09`. This preserves
the exact old inference, geometry helper and pack-format bytes. The main-tree
snapshot v3 is not used: its newer sequential cache reader changes the
pack-format file hash. Preflight 02 is queued on the existing GPU lock for
all 50 original probes. It needs numeric success and actual depth/mask/K
parity with the reviewed preflight before a new visual review can reuse that
evidence. Closed-pack recovery and the 7,623-image remainder have not yet
started. Never rerun all 35,783 inputs or overwrite the 110 closed old packs.

### First geometry segment completed, 2026-11-03 08:37 UTC

The first segment stopped normally at its 1,000-update limit, completed final
validation and released the existing GPU lock. W&B run `f6e8d335` is `finished`,
with matching final validation values. Its logging index 1,003 includes
validation events; the actual optimizer count is exactly 1,000.
`reviewed-stage1-finish-01` retains an immutable hardlink of the final
`latest.pt`, SHA `047253cc38aa012def90405e879dbddbc843c9a08560b5e679388742309e499f`.
CPU readback confirmed step 1,000/cursor 16,000, all 100 Adam states at step
1,000, finite adapter/optimizer tensors, 100 changed adapter tensors, unchanged
feature statistics and the same accepted configuration and stream identities.
All 1,000 training records are finite. This is not an independent resume test.

`reviewed-stage1-validation-1000-review-01` repeats only the paired metadata
comparison, with SHA `9d19e02a7baedd35b0c1ad28517d7502f09869d4f3574a0fc7d6bad2c9cd09ad`.
Original-view spatial cosine is 0.9599341052 and summary cosine 0.9320206924;
both recover slightly but remain below the archived initial adapter. NYU
AbsRel is 0.0720507646, 9.27% better than initial; ScanNet LiDAR is 0.0969569659,
5.25% worse. Under the separate 400x640 crop protocol, summary cosine rises
from 0.9296161041 to 0.9298317738, while spatial remains below initial. Do not
mix those protocols or claim an overall gain. Best NYU/spatial/summary are now
step 1,000, best LiDAR step 500, and best known-camera laser step 250.

ScanNet review-teacher preflight 02 subsequently completed all 50 views with
zero rejections. `review_teacher_preflight_parity.py` checked exact RGB,
provenance, depth/mask/K bytes and all 25 reviewed scenes against preflight 01.
The new visual gate explicitly reuses the earlier actual inspection through
this exact parity; no new visual inspection is claimed. Five parity/recovery
tests passed remotely. New review SHA:
`dcdc4dc1660590a4fb9559e7552f758b8c8682e92ab78d3f6345eaea59b9a437`.
`scannetpp-review-teacher-recovered-01` is rereading only the 110 closed old
packs using one CPU worker. Wait for its complete 28,160-image proof before
launching the 7,623-image remainder from execution snapshot v4. The main-tree
v3 snapshot remains unused for generation because its pack-format file hash
differs; no hash guard was relaxed.

### Reviewed five-source expansion, 2026-11-03

`reviewed-stage2-data-01` accounts for all 369,012 inputs in the five completed
sources: **367,599 selected and 1,413 excluded**. The selected labels are
189,025 native synthetic observations and 178,574 relative teacher candidates.
Structured3D adds 133,563 images; prior source selections and all native/external
quality flags remain bound. Completed producer payload readbacks are reused
explicitly, with per-read hashes retained in the training loader. Preparation
report SHA: `ed6a258bb7cf4328f0ce84d5588b1d242e5d45cadcefd853dc67b004dac65280`.

Every selected ordinal, dataset, scene and RGB hash was matched to the unchanged
seven-source canonical RGB isolation index, which already contains Structured3D
and was compared against 190,954 heldout memberships. There are zero exact pixel
overlaps or forbidden shared scene rows. This reuses the completed comparison;
it does not claim that the unfinished eight-source RGB census has completed.
The 796,477-image RGB-only replay manifest remains unchanged.

`structured3d-training-crops-01` uses the production 400x640 loader on 224 images
from 171 groups, with 112 images per preserved policy. Selection includes low
support, nearest/farthest median depth and deterministic scene samples. Three
views per image produce 672 views. All selected native observations are unique,
supported and correctly transformed; maximum relative point error is
5.95e-8. Three views from ordinals 529248 and 570412 are empty. Both full pages
in `structured3d-sparse-crop-visuals-01` were inspected: accepted opaque wall
support lies at the left edge, which the augmentation crops out while retaining
unsupported window/exterior RGB. Center views still contain 26,944 and 32,941
observations; one augmented view retains 67. No mask filling or whole-image
exclusion follows. The existing <32-observation rule supplies explicit zero
geometry and preserves the scheduled paired/RGB-replay feature branch. This
deliberately extreme selection is not a corpus prevalence estimate.
Crop report SHA: `d3bcf9d98c0ecb876e322951bf9e3be6cd913102472fb39ab6e9fa12c1b9f0ce`.

The proposed next segment preserves the step-1,000 adapter, feature statistics,
Adam moments and optimizer count. It starts a new sample stream at cursor zero,
keeps the existing LR, loss weights, replay and evaluation protocols, and has
an additional 1,000-update/7,200-second limit. Geometry draws allocate 20% to
Structured3D and multiply previous source probabilities by 0.8: COCO/Hypersim
28% each, IRS 20%, NYU 4%. A 100,000-draw probe of the actual source/scene/frame
sampler matches all source probabilities within 0.00316. Replay sampling is
unchanged. Final step-1,000 validation values become checkpoint-selection
baselines, while all prior best checkpoints remain available.

Execution snapshot `code-reviewed-continuation-v1` is commit `8332c8c`. It allows
only an explicit corpus data fork with a distinct new acceptance seal bound
to the exact checkpoint and real model probe. Objective, optimizer settings,
replay and heldout identity changes are rejected. Tiny CPU tests verified exact
Adam/adapter/statistics restoration and a fresh cursor in a new trainer
invocation. Nineteen crop/identity tests passed locally and remotely; 36
continuation tests passed locally, 73 training/continuation/replay/model tests
passed remotely, and nine pinned geometry-loss tests passed remotely. These
tests do not substitute for the actual model preflight.

`reviewed-stage2-readiness-01/plan.json` binds all four nonmodel gates, SHA
`a9828e67bc1bc13ff987ce237644a530602f0bfbde2b7d25963d41ac39e25753`.
The supervisor `reviewed-stage2-launch-01` must finish the real zero-update
model gate before sealing or optimizing. At 09:24 UTC it was running that
preflight; no second-stage optimizer update was yet claimed. The ScanNet
remainder supervisor is separately waiting for complete 28,160-member recovery,
then will submit only the remaining 7,623 images from code v4 under the same
existing GPU lock. Neither supervisor bypasses that lock.

The second-stage actual model preflight subsequently completed **62 forward/
backward cases**. All six dataset/label strata produced nonzero geometry
gradients: twelve Structured3D native cases, ten cases in each Hypersim native,
Hypersim pseudo, IRS and NYU stratum, and nine active COCO cases plus one
explicit sparse feature-only case. Adapter, feature statistics and frozen
backbone states were unchanged, with zero optimizer updates. Report SHA:
`81328693eb9630a4b649a45f99b204293d2e31eaba8c325db6c167f85e6d193a`.

The second segment is now actually optimizing in `reviewed-stage2-training-01`.
At 09:35 UTC its latest optimizer step was 1,057, with finite losses and
gradients. `reviewed-stage2-startup-01/report.json` inspected the new real
trainer's initial state: all 100 Adam states, adapter tensors, feature statistics
and RNG state exactly match the step-1,000 parent, while the new data cursor is
zero. Replay and validation digests are unchanged; the geometry digest changes
to the newly accepted five-source index. Its first-update proof records 26
contiguous new updates and nonzero parameter-update norms. Runtime config SHA:
`079bf5f6875fcd049334930fe8710e12eab5f6b42a3f5c4c02d38654f7be27ba`;
acceptance SHA: `34f2413ea42b68970110805447762a581358ea90f656268d7bbae4745c54f954`.
W&B run is `df4196ed`. The next paired validation is at step 1,250; no benefit
from the expanded data is claimed before that measurement.
Online SDK verification observed step 1,083 with exact provenance and matching
configuration apart from a recorded one-ULP serialization difference in the
summary checkpoint-selection baseline (0.9320206923933305 versus the service's
0.9320206923933304). The immutable runtime file and acceptance hash are unchanged.
Live local logs reached step 1,085 at 09:38 UTC; all 85 new records were finite.

ScanNet review-teacher recovery also finished all 28,160 members in 110 closed
packs with zero rejections. The existing remainder controller has submitted
only the 7,623 remaining inputs from execution v4/`5228f12`; its child is waiting
for the original GPU lock held by the active trainer. Do not relaunch either
the recovery or the queued remainder, or run teacher inference alongside this
training segment by bypassing that lock.

### Additional TartanAir rejection review, 2026-11-03

`tartanair-rejection-review-04` reproduces only five new rejected inputs,
ordinals **451241–451245**, neighborhood/Hard/P006 frames 157–161. The previous
14 decisions remain bound to their exact RGB and rejection receipts and are
not rerendered. Both the left/right source identities and published conversion
statistics match exactly. All five rendered pages were inspected: the upward
views contain extensive sky, utility wires and fragmented foliage/roof edges.
Final stereo support is 4.016%–4.684%, below the unchanged 5% minimum. These five
inputs receive explicit geometry-only exclusions; all five original RGB records
were found unchanged in the active replay manifest. No active training input,
mask, threshold or optimizer state was modified.

The incremental review helper accepts both existing spellings for a geometry-only
decision while retaining exact image/SHA/rejection/evidence binding. Eight tests
passed locally and remotely. The initial attempt rejected the older spelling
before creating review output; the corrected execution snapshot is
`code-tartanair-incremental-review-v2`, commit `c886075`.

The merged known geometry exclusions now number **1,432**:
`tartanair-rejection-review-04/known-geometry-exclusions.jsonl`, SHA
`ac6d3c783b1f12a9b17c5a5c0e5785a0228bb5b0183043a382b30014fc4f7699`.
Only the previously reviewed Hypersim white image 210812 is excluded from RGB
replay. These dispositions do not constitute full TartanAir source acceptance.

### Second-stage validation at step 1,250

`reviewed-stage2-validation-1250-review-01/report.json`, SHA
`40ceeff2e949dd5d17e947d377e49f49358239e50ba4e4cf1ff2943cd367dc1e`,
compares recorded outputs without repeating model inference. Original and native
400x640 views each contain the same 1,268 images, alongside the same 120 LiDAR,
128 DSLR and 18 known-camera laser views. Image order, preprocessing protocols,
heldout digests, frozen model revisions and fixed teacher/GT quantities match.

Against the step-1,000 parent, original-view spatial cosine falls from
0.9599341052 to 0.9597938078 and summary cosine from 0.9320206924 to 0.9318046097.
NYU AbsRel rises from 0.0720507646 to 0.0738057821 (2.44% worse), ScanNet LiDAR
from 0.0969569659 to 0.0983683786 (1.46% worse), and known-camera laser AbsRel
from 0.1333986210 to 0.1335096824 (0.083% worse). NYU remains 7.06% better than
the original feature-only adapter, while LiDAR is 6.78% worse. There is no
overall improvement claim.

The largest original-view spatial decline is Structured3D
(0.9679181371 to 0.9655191442); IRS improves from 0.9578006924 to 0.9587802906.
This describes where the change occurs, not a causal attribution. Under the
separate native crop protocol, spatial cosine changes by only +1.24e-6 and
summary cosine declines. The active stage's five checkpoint selections remain
at its step-1,000 parent, while earlier stage-specific bests are retained.
Training continues within its existing bounded segment; compare step 1,500
before drawing a trend conclusion or changing the accepted objective.

### Second-stage validation at step 1,500

`reviewed-stage2-validation-1500-review-01/report.json`, SHA
`fbc4138d17ab4eb468615c8130c8639035539443f3abc33fde0957d34ee8a4c8`,
checks the completed recorded validation against the same initial, step-1,000
and step-1,250 images. All four preprocessing views, fixed teacher/GT values,
five heldout digests, frozen model revisions and the 18 known-camera laser views
match. No model inference or optimizer change was made for this comparison.

Original-view macro spatial cosine is **0.9597261074**, down from 0.9597938078
at 1,250; summary cosine is **0.9318716183**, up from 0.9318046097. Both remain
below the step-1,000 parent and the initial feature-only adapter. NYU AbsRel is
**0.0727026400**, 1.49% better than 1,250 but 0.90% worse than 1,000, and 8.45%
better than the original adapter. ScanNet LiDAR AbsRel is **0.0990181056**,
0.66% worse than 1,250 and 7.49% worse than the original adapter. Known-camera
laser AbsRel is 0.1342309032, 0.54% worse than 1,250. The native 400x640 protocol
remains separate: spatial 0.9593450126 and summary 0.9296253680.

Structured3D original-view spatial cosine falls further to 0.9650159772;
COCO improves from 0.9423794388 to 0.9425661252. These are descriptive changes,
not a causal attribution or evidence of overall improvement. The active stage's
five best-checkpoint selections still refer to its step-1,000 parent. Training
continues within the existing 2,000-global-step / 7,200-new-second bound.

### Retained ARKit/FARO visual diagnostic

`arkit-faro-visuals-01` reproduces all nine existing exact-frame comparisons
from retained publisher members. SHA256, CRC, byte sizes, receipts, native
conversion statistics and both unfitted high-resolution camera hypotheses match
the original probe exactly. Seven tests passed locally and remotely. All nine
pages were actually viewed with common depth and signed-error palettes.

The successful comparisons cover three videos and only rotations **0 and 90**;
the earlier 12-video availability scan does not imply completed comparisons for
180 or 270. The near-wall view at ordinal 27177 has a broad -0.039 m median
residual under both mappings, rather than only a boundary tail. Views 27185 and
41546 also show residuals on wall/panel surfaces. Mirror, sink, lamp and fabric
views expose different holes and boundary support between the two sources.

The visual review, SHA
`68c40ad1c876423b8ae0e20ec9c4b241aa2d4cd1239c90b29e65a92f3e94ceb0`,
does not identify which source is physically correct, fit a correction, issue
exclusions or accept ARKit. Publisher-filtered sampling remains a limitation.
Exact high-resolution intrinsics were subsequently obtained in
`arkit-exact-faro-camera-01`; the earlier availability assumption is corrected
below. Cross-sensor visibility and physical depth residuals remain. Reuse these
localized observations in the eventual complete native-source review; do not
rerender these nine pages merely to postpone disposition. Active accepted
training inputs and all source-quality flags remain unchanged.

### Five further TartanAir dispositions

`tartanair-rejection-review-05` reproduces only the five newly published
rejections at ordinals **454273–454277**, neighborhood/Hard/P010 frames 643–647.
The previous 19 reviewed rejections are skipped using their exact evidence.
All five pages were viewed: sky dominates the upward tilted views, while wires,
foliage and roof edges leave fragmented support. The unchanged final support is
**4.624%–4.892%**, below the existing 5% requirement. These exact inputs receive
geometry-only exclusions; their RGB remains eligible for replay. No depth,
mask, threshold or active training artifact is changed. The cumulative known
geometry exclusion count is now 1,437, including 24 TartanAir inputs; only the
existing white Hypersim image 210812 is excluded from RGB replay.

### Second-stage validation at step 1,750

`reviewed-stage2-validation-1750-review-01/report.json`, SHA
`ec35d39d7c50157b098410a03f1623cbe477d4156fe96c64948f08533496fafa`,
compares the completed recorded outputs against identical initial, step-1,000
and step-1,500 samples. The original and native 400x640 views each contain
1,268 images; LiDAR, DSLR and known-camera laser views contain 120, 128 and 18.
Image order, protocols, fixed teacher/GT values, five heldout digests and frozen
model revisions match. All five main validation metrics also match W&B.

NYU AbsRel reaches **0.0699425375**, a new best: 3.80% better than step 1,500,
2.93% better than step 1,000 and 11.92% better than the initial feature-only
adapter. Original-view spatial cosine recovers to **0.9599099911** from
0.9597261074, but remains slightly below step 1,000 and the initial adapter.
Summary cosine declines to **0.9318509872** from 0.9318716183. ScanNet LiDAR
AbsRel rises to **0.1003067840**, 1.30% worse than step 1,500 and 8.89% worse
than the initial adapter. Known-camera laser AbsRel rises to **0.1385882757**,
3.25% worse than step 1,500. There is no overall improvement or causal claim.

The separate native 400x640 protocol reports spatial cosine 0.9594301366 and
summary cosine 0.9296619351. IRS original-view spatial cosine improves by
0.0011598 since 1,500, while Structured3D improves by 0.0002749. These protocol
and dataset results remain separate from the main macro comparison.

The NYU-selected step-1,750 checkpoint is preserved by immutable hardlink in
`reviewed-stage2-best-nyu-1750-01`, SHA
`ca25803f5ad3d17bb9a5abd16ffd14e85694f1f6b1d0ed94fa858a36d989f3a9`.
CPU readback verifies finite adapter/optimizer tensors, all 100 Adam states at
step 1,750, data cursor 12,000 and exact runtime configuration, stream digests
and model revisions. This is checkpoint readback, not a separate resume test.
Other active-stage best selections remain at step 1,000; earlier stage-specific
bests are retained. The running optimizer and accepted inputs are unchanged.

### ARKit outcome tooling and one further TartanAir disposition

`audit_arkit_outcomes.py`, execution snapshot `code-arkit-outcomes-v1` at
`c7474f9`, is ready for the completed producer output. Ten tests pass locally
and remotely. It requires full producer readback, binds every frozen RGB input
to one native candidate or rejection, checks native units/grid/timestamp proofs,
and retains both native flags and the unresolved FARO calibration/visibility
diagnostic. Its SQLite working index uses local temporary storage. These tests
are not an actual full-source audit or a source-quality acceptance.

`tartanair-rejection-review-06` reproduces only ordinal **455036**,
neighborhood/Hard/P010 frame 1406. The actual page was viewed: the upward view
is mostly sky with trees, a house, pole and wires near the lower edge. Final
support is **4.91146%**, below the unchanged 5% minimum. The exact input receives
a geometry-only exclusion; its unchanged RGB was found in the active replay
manifest. The prior 24 TartanAir decisions were not rerendered. Known geometry
exclusions now number **1,438**, SHA
`e94b103829bf70d7237329a7db8c9111b062c75c6de8b783a79714c03e108206`.
Only white Hypersim image 210812 is excluded from RGB. Further producer
rejections after this snapshot require their own disposition.

### Second segment completed at step 2,000

The second training segment ended normally at its 2,000-global-step limit.
All 1,000 new update records are finite, final validation is complete and the
trainer has released its GPU lock. The existing ScanNet review-teacher remainder
controller acquired that lock and started its already queued 7,623 inputs;
the 28,160 recovered members were not regenerated.

`reviewed-stage2-validation-2000-review-01/report.json`, SHA
`f9dfae799e5a5334da9433ffb69b60896ee54e22c209a1b3dee1386f535c8878`,
checks the recorded final outputs against the same initial, step-1,000 and
step-1,750 images. Four views, protocols, teacher/GT values, five heldout digests
and the 18 known-camera laser cases match. The macro cosines were also recomputed
from the per-image dataset means. No inference was repeated for this comparison.
The five main validation metrics match the finished online W&B run `df4196ed`.

Original-view spatial cosine improves from **0.9599099911** at 1,750 to
**0.9601634316**; summary cosine improves from 0.9318509872 to **0.9319317282**.
Both remain below their initial feature-only values, 0.9616920552 and 0.9323009271.
NYU AbsRel is **0.0711039240**, 10.46% better than the initial adapter but 1.66%
worse than the preserved 1,750 best. ScanNet LiDAR AbsRel recovers to
**0.0967663059**, 3.53% better than 1,750 and 0.20% better than the stage-1,000
parent, but still 5.05% worse than the initial adapter. Known-camera laser
AbsRel worsens to **0.1424284187**, 2.77% worse than 1,750 and 6.77% worse than
1,000. These are mixed task-specific changes, not an overall improvement.

The final checkpoint is preserved in `reviewed-stage2-finish-01`, SHA
`3c1a51d839e2535fa849aba03e14e241de7408bef61ae678dd0d94a7f502ce50`.
CPU readback verifies all 100 changed adapter tensors and 100 Adam states are
finite, Adam steps are 2,000, the new data cursor is 16,000, feature statistics
are bitwise unchanged from the parent, and configuration/stream identities match.
This is a final-checkpoint readback, not an independent resume experiment.
Active-stage spatial and LiDAR selections now use 2,000; NYU remains at 1,750,
summary and laser at 1,000. Earlier stage-specific bests remain preserved.

### Completed snapshot payload census and ARKit accounting

`stage1-payload-census-01` has completed **4,098 packs and 873,626 candidate
members**, covering **870,550 distinct RGB inputs**, with zero failures. The
producer reread and verified whole tar hashes, every declared member, array
statistics, cameras and source identities. The separate closure review checks
the frozen pack membership, source totals, report identities and policy hashes;
it does not repeat the payload census. All candidate flags are retained and
the snapshot remains unaccepted as a whole. Its report SHA is
`c5307f15eb0a5e2278bbf13bbadf637a7b758ba80a25ca97bf07b2958ce07a40`.

`arkit-outcomes-01/report.json`, SHA
`c815c418169c87445cdd76ac07bbb8b5a51af5eca6474d168997cc3ade10f27a`,
accounts for all **197,542** frozen inputs from 3,789 videos and 1,478 visit
groups after the producer's complete array readback. There are 194,924 native
candidates and 2,618 rejected inputs: 1,458 failed image registration, 977 have
insufficient registered support and 183 lack an exact-timestamp native member.
The metadata audit issues no exclusions or source acceptance. Its 36 distinct
review candidates cover all four rotations; these candidates have not yet been
visually reviewed. The existing nine FARO diagnostics and their unresolved
calibration/visibility limitations remain attached and are not rerendered.

### Nineteen additional TartanAir dispositions

`tartanair-rejection-review-07` reproduces only 19 new neighborhood/Hard/P012
and P014 rejections; the prior 25 were not rerendered. All 19 complete six-panel
pages were viewed in ten contacts with uniform 50% display resizing. Extensive
sky, sunlight, foliage and roof/wire edges leave **4.559%–4.995%** final support,
below the unchanged 5% requirement. Exact source arrays and statistics match.
Only the listed original identities are excluded from geometry supervision;
intervening frames and whole families are not excluded by extension. All 19
unchanged RGB records were found in the accepted 796,477-image replay manifest.

There are now 44 explicitly disposed TartanAir rejections and **1,457** known
geometry exclusions overall. The merged manifest SHA is
`3dae6b787721a92e78c4828cbf1c1bf8724c78b5bd1fd45e3d16eea2fa812191`.
Only white Hypersim image 210812 is excluded from RGB. The completed training
configuration, index and checkpoint selection are unchanged by these decisions.

### ARKit four-orientation native outcome review

`arkit-outcome-visuals-01` reproduces all 36 previously selected native
candidates from the five retained publisher members per input. Member SHA256,
source SHA256/CRC/receipts, depth, valid mask, both cameras, camera transform
and producer statistics agree exactly. Original conversion code and camera
helpers match the producer policy. Sampling upright VGA RGB onto the native
depth grid also agrees with sampling before rotation for 0/90/180/270 degrees;
no depth resizing or filling is performed. Twelve tests passed locally and
remotely, including shifted-camera and changed-array/provenance failures.

All 36 six-panel pages were inspected through 18 contacts at two-thirds display
scale, with three uncertain cases additionally inspected at full resolution.
The fixed depth palette is 0.125–8 m. Low-support examples include glass,
reflective floors, dark trees and nearby trim. High photometric-error cases
often show carpet, stripes or printed texture; corresponding pixel holes are
retained. No broad quarter-turn mismatch was observed in this selection. This
is an extreme selection, not a prevalence estimate or independent physical
calibration. The review SHA is
`01726c9355f3d456a75f14277cf7620ecdab7fdc365213e0001c427ce450add0`.

`arkit-candidate-decisions-01` explicitly excludes three exact images from
geometry: **182561** (blurred reflected camera/hand), **153819** (mirrored
wardrobe with sparse ambiguous surface support), and **167294** (fingers
occluding the lens with uncertain foreground/background support). These are
conservative quality dispositions, not claims of a measured physical bias.
No depth-ratio, range or support threshold was added. A policy-specific
quarantine register binds each exact RGB, receipt and cache member; the actual
admission validator checked all three flags. All three unchanged RGB rows
were found in the existing 796,477-image replay stream and remain there.

The merged known geometry exclusions now total **1,460**, SHA
`62e76a8189c8cfd53ad05c29645b1849796cd1114d6a57557f6aeeb299871aac`.
Only white Hypersim image 210812 remains excluded from RGB. The nine existing
FARO pages were not rerendered; their calibration/visibility limitations and
ARKit's 2,618 producer rejection dispositions remain pending. This review does
not admit the ARKit source or change the completed training artifacts.

### Complete ARKit rejected-input dispositions

`arkit-rejection-census-01` reproduces all **2,435** numeric failures from the
five retained source members per image, using the original converter and
unchanged thresholds. Source SHA256/CRC/receipts and every stored statistic
match. These comprise 1,458 image-level RGB registration failures and 977
images with insufficient final registered support. The latter range from zero
to 4.9357% support, below the existing 5% minimum; failed image registration
leaves zero final support.

The other **183** exact native timestamps were independently found absent from
the central directories of **106 publisher ZIP archives**. Exact directory
bytes, member lists, offsets and hashes are retained. Missing local files were
not treated as absence evidence. No native depth payload was downloaded or
substituted, and no nearest timestamp was used. This verifies the current
publisher directory snapshot, not a full archive hash or historical publisher
state. Thirteen relevant tests passed locally and remotely; the 11-record
preflight covered eight numeric cases and three publisher absence checks.

All eight preflight pages and all **18 full-census extreme pages** were
actually inspected, covering four rotations and both numeric failure types.
Sparse cases contain no support or small disconnected patches. Several RGB
gate failures concentrate on mosaics, carpet, stripes and printed texture,
including borderline p90 failures. These are conservative correspondence
failures under the existing policy, not proof that all pre-gate depths are
physically wrong. Pre-gate depth remains diagnostic only. Census report SHA:
`05a68aad07ecae8cff5bcfa613c34c6f3d14b88582e3a1a909dd47cc7c7c043e`;
visual review SHA:
`24355983a173568a77affe42a04962433bba59fc8448e383c1a2221919bfb3b4`.

`arkit-rejection-decisions-01` explicitly excludes these 2,618 exact images
from geometry only. Publication checked every frozen rejection identity,
reparsed all 106 retained directory proofs, and found every original RGB in
the unchanged accepted 796,477-image replay stream. Together with the three
previous candidate exclusions, ARKit's 197,542 inputs now have **194,921
remaining native candidates and 2,621 explicit geometry exclusions**. No
rejected-input disposition remains unresolved, and neither source calibration
nor overall source acceptance is inferred from this accounting.

The latest global exclusions total **4,078**, SHA
`a80a5da641bf914e0f590a9e9314e269bdae01e1ca1c54b1e68e7fce431c99c5`.
Only white Hypersim image 210812 is excluded from RGB; all 4,077 geometry-only
images retain RGB eligibility. The three mandatory ARKit candidate flags and
existing FARO calibration/visibility limitations remain. The accepted 367,599
geometry inputs, completed 2,000-update training segment and best checkpoints
are unchanged. Reuse the finished census and visual reviews rather than
repeating them during the remaining source-quality and crop-coverage work.

### ARKit actual crop support and ScanNet remainder audit preparation

`arkit-training-crops-01` exercises the unchanged production loader on **127
candidate images, 99 visit groups and 381 actual 400x640 views**. The selection
covers all four rotations, low support and depth extremes, deterministic visit
samples, centered crops, random crops and an actual horizontal reflection.
The three previously excluded candidates are omitted. Diagnostic seed 13145
did not exercise a flip; the new probe uses seed 0 and asserts actual flip
coverage. Production augmentation is unchanged. Twenty tests passed locally
and remotely. These tests and the crop probe are not a real-model gate.

Every selected target remains one unique original native observation and
projects onto its prediction query. Maximum relative native-to-target error
was 5.941e-8 from float32 storage; this is not physical depth accuracy. There
are **six sparse views from two original images**. Both complete pages in
`arkit-sparse-crop-visuals-01` were inspected after reproducing all six view
metrics. For ordinal 36153, most of the 2,651 original valid points lie above
or below the horizontal viewport, leaving 15/11/11 points. For ordinal 168525,
all 3,218 original supported points lie below the three tested viewports,
leaving zero points. Its blurred close door/panel RGB does not independently
establish native depth accuracy.

The existing fewer-than-32-observations rule skips geometry for these views;
RGB feature eligibility remains. No depth, mask, crop policy or exclusion was
changed. This extreme selection does not estimate whole-source sparse-view
prevalence. ARKit source quality and final crop coverage remain unaccepted,
including the retained FARO calibration/visibility limitations. Visual review
SHA: `0caa3c0dc2919854f5f0a6ef066c35e966ecd95ec3296154348b43a0a0f8b82e`.

`audit_scannetpp_teacher_outcomes.py` is ready in immutable
`code-scannet-teacher-outcomes-v1` (commit `bcaaf79`). Eighteen tests passed
locally and remotely. After the current remainder producer completes its full
readback, the auditor will bind the 28,160 recovered and 7,623 remaining
original RGB identities exactly once, check both unchanged inference/mask
policies and retain all native source-review flags. It uses a local temporary
SQLite index, publishes relative receipt statistics and bounded all-scene
review candidates, and cannot admit labels or repeat inference. The actual
combined audit has not run yet; the existing producer continues unchanged.


### ScanNet teacher completion and six-source continuation preparation

The existing remainder producer completed all **7,623** images and its full
array readback with zero rejections. `scannetpp-review-teacher-outcomes-01`
binds these to the **28,160** previously recovered images exactly once:
**35,783 distinct training RGB images, 25 scenes and two unchanged policies**.
The combined audit retains native source flags and reports relative teacher
geometry only. Audit SHA:
`172b7996c09609c1fef77d8507293c2caa93ef248476edfeea11aec549151ef2`.

All **100 selected payloads and RGB images** were read and checked against
receipt statistics and original anonymization masks. All **34 four-panel
pages** were actually inspected. Individual depth palettes and a shared
log(Z / image median) range of [0.25, 128] separate display scaling from
geometry. Broad room/furniture ordering is plausible; glass, mirrors, fine
shelves and machinery remain locally uncertain pseudo supervision. No new
exclusions or physical calibration claims were made. Review SHA:
`ade0c7c57e9379afbfe4f154313404b0309fab86cc2fd88b2d9675ec8e5e4d18`.

`scannetpp-teacher-source-reviews-01` completed both policy-specific source
reviews with explicit empty additional quarantine registers. All native flags
remain. The actual production loader then checked **300 views from those 100
images**, all 25 scenes, both policies, crops and real flips. Minimum support
was 213,306 original observations; no selected view was sparse. Maximum
native-to-query relative error was 5.926e-8, a float32 consistency check rather
than physical accuracy. The source review and crop report SHAs are
`a0ca499ffe2c1b0f7c68ab5918ce607a5c10055a25adf6b05c1b87ca2cddb530`
and `584ca1b9228b7e059222a8262c6b3d110e6e55062a146219162bb03246efa8f0`.

`reviewed-stage3-data-01` has reconciled **404,795 original images** into
**403,382 selected geometry labels and the same 1,413 exclusions**. The selected
labels comprise **189,025 native synthetic and 214,357 relative teacher**.
Only the reviewed 35,783-image ScanNet subset was added. Accounting SHA:
`4800c6b8aaf2965e78ec3d0f769f80ba87a918eab4bb95831f49f2cfaa8fcf68`.

Whole-corpus canonical RGB completion and heldout comparison are still required
for the new subset. The comparison uses a checked local SQLite work copy to
avoid shared-filesystem random writes; the full comparison logic is unchanged.
Sixteen comparison tests, twelve source/crop checks and nine exact subset and
isolation checks passed remotely; local focused checks also passed. The frozen
continuation builder preserves the completed step-2000 adapter/Adam state,
unchanged RGB replay and losses, and proposes a bounded 1,000 further updates.
ScanNet receives 20% of geometry draws and previous weights scale by 0.8.
The new stage remains unadmitted until canonical isolation and the actual
archived-model zero-update backward gate pass. No optimizer has restarted at
this preparation point. Prior final and best checkpoints remain preserved.


### Complete canonical RGB and six-source nonmodel gates

`rgb-identity-census-01` completed **all 1,637,909 original training RGBs**,
with zero identity/orientation/hash errors and zero exact duplicate clusters
under either the raw or explicit-orientation canonical RGB hash. This finally
resolves the legacy cross-source hash-convention limitation. It does not
measure perceptual near-duplicates or independent scene counts. Report SHA:
`9046e72d8ebde10e32f2a7574ef0c6ab30fdaa25d2ee0660dc58ec16d2dd294f`.

`heldout-rgb-full-comparison-01` compared the full training census against
**190,954 memberships in all 16 heldout manifests**. Exact train/heldout pixel
overlaps and forbidden shared-scene rows are both zero. The official IRS test
protocol has 22,320 memberships in shared environment families by design;
its independent IRS validation/test protocols have no shared families. These
memberships include repeated subset manifests and are not unique image counts.
No evaluation depth was read, no automatic exclusion was made, and no
near-duplicate claim is inferred. All 16 manifest results were reviewed.
Comparison SHA: `07dca135c4dbadf3aaee920e2785818a924a70b01b25d04e7fcfeaec22cacc2b`;
review SHA: `2f107f36ee8ea59e3afdb920446774f19601f493105680629adfe189e2405584`.

The new stage isolation binds **all 404,795 scoped originals**, including
geometry exclusions, to this completed census and comparison. Its 403,382
selected labels and unchanged 796,477-image replay are now isolated; replay
reuses its exact prior canonical proof. Isolation SHA:
`7d58beffd139cea2d7cf47b8ae11edbfe1be98f60768705c474e9ff4dd3dfb17`.

The actual source/scene/frame sampler made 100,000 metadata draws. Observed
shares were COCO 22.033%, Hypersim 22.548%, IRS 16.221%, NYU 3.216%, ScanNet
19.949% and Structured3D 16.033%, all within 0.5 percentage points of the
frozen target shares. All 25 added ScanNet scenes were sampled. Sampling is
with replacement, and source or scene counts do not imply independent images.

`reviewed-stage3-request-01` and `reviewed-stage3-readiness-01` now bind the
completed source, coverage, isolation and unchanged-loss checks to the exact
new accounting. The prior loss/continuation tests are reused only after all
implementation hashes match immutable execution `8332c8c`. The new request
starts from the verified step-2000 checkpoint, preserves Adam and unchanged
objective/replay/evaluation protocols, resets the new sample stream to zero,
and ends at global step 3000 or 7,200 new seconds. Request SHA:
`265f75d3701031129ca4f60d23b1ad53ab7d61e11be9632b6301361bf0b8b205`;
nonmodel plan SHA:
`35512867b54243873dfe879354b46cdd22d8073368e7f3cfa34aa384529b170b`.
The launch wrapper is running the actual archived-model gate under the original
GPU lock; optimizer admission still requires that gate to pass.


### Six-source continuation is actually running

The new actual-model gate completed **72 forward/backward cases across seven
source/label strata**. All 72 produced finite nonzero adapter gradients; 71
produced geometry gradients. The previously understood empty COCO crop 287969
kept its paired feature branch and skipped geometry. Every stratum, including
all ten ScanNet cases, produced nonzero geometry gradients. Adapter, feature
statistics and all frozen backbones were unchanged by the zero-update probe.
Model report SHA:
`bd7b7c359aeea840d4c616ea203311e75fc3c0f359d24fe5344e8dbdf227dd57`.

A new seal and immutable runtime config were issued:
`59218a6c3e152f4c5258d9e8b8ab9ee58987d288ff708ed5e1c78826dbea50b8`
and `47d57f659f0c08551e987f14589214885957f3e20d7c42c6110e5bde61e18f29`.
The independent trainer's actual initial checkpoint was read on CPU: adapter,
all **100 Adam states at step 2000**, RNG and feature statistics are bitwise
equal to the parent. The new stream cursor is zero; replay and every validation
digest are unchanged. Initial-state proof SHA:
`1bfd897618ff53b31884cbf935ac5bcfacb3b7ff6d34bedb0debf7e4defe3308`.

`reviewed-stage3-training-01` has produced real optimizer updates. The first
20 records (steps **2001–2020**) are finite, with positive gradients and
nonzero update norms at steps 2010 and 2020. Its W&B run is
[4edbd06f](https://forge.coreweave.com/wandb/brainsim/radio-moge/runs/4edbd06f),
verified online as running with exact provenance. Three selection baselines
have one-ULP API serialization differences, recorded explicitly; the immutable
runtime config is unchanged. First-update proof SHA:
`0fedf60d5b580a3e2eb342e0ace237cd398a3ed0ea01eca28a104057e35776e9`.

At 14:14 UTC the actual optimizer reached **step 2028**. The next paired
validation is step 2250 against the unchanged initial/step-2000 protocols;
no new-stage accuracy or cosine improvement is claimed before that result.
The previous final and all best checkpoints, including NYU step 1750, remain
preserved. The active wrapper/trainer hold the original GPU lock and must not
be relaunched. Native TartanAir and ScanNet preparation continues; the full
geometry corpus is not admitted by this six-source stage.


### Paired step-2250 validation and preserved depth checkpoint

The first stage-3 validation completed at global **2250**. The comparison checks
the same 1,268 original-resolution images, 1,268 fixed-bucket images, 128 DSLR
images, 120 LiDAR images and 18 known-camera laser cases against the archived
feature model and the completed step-2000 results. Image order, preprocessing,
frozen-teacher outputs, fixed laser support, model revisions and all five
heldout digests match. Macro scores were recomputed from the recorded per-image
results; no inference was repeated. Four focused negative/aggregation tests
passed locally and remotely. Comparison SHA:
`23d84905bd39a484051925efaca57573bccd8aa891b41fc42b84e3246facfd29`.

| Original evaluation metric | Step 2000 | Step 2250 | Interpretation |
| --- | ---: | ---: | --- |
| Macro spatial cosine | 0.96016343 | 0.95980578 | Slightly lower |
| Macro summary cosine | 0.93193173 | 0.93185091 | Slightly lower |
| NYU AbsRel | 0.07110392 | 0.06986465 | 1.74% lower error; new NYU best |
| ScanNet LiDAR AbsRel | 0.09676631 | 0.09596369 | 0.83% lower error |
| Known-camera laser AbsRel | 0.14242842 | 0.12761458 | 10.40% lower error |

NYU is 12.02% better than the initial archived feature model, but LiDAR remains
4.17% worse than that initial model. Both original cosine scores remain below
their initial values. The fixed 400x640 protocol is separate: its NYU error
increased 0.18% from step 2000, and its summary cosine remains slightly above
its own initial baseline. IRS and Structured3D contribute the largest original
spatial-cosine drops in this interval. These are task-specific observations,
not an overall improvement or a causal attribution to the added source.

The W&B **history record at optimizer step 2250**, rather than its rounded
summary response, exactly matches all five displayed primary metrics and the
immutable provenance. Online evidence SHA:
`1bd861ece1921de1256244dda8688636b93693415291526fc9e3172171b21bd0`.

`reviewed-stage3-best-depth-2250-01/checkpoint-step2250.pt` preserves the common
checkpoint selected by this stage's NYU, LiDAR and laser metrics. CPU readback
confirmed finite adapter/Adam tensors, all 100 Adam states at step 2250,
cursor 4000, and exact config/stream/revision identity. Checkpoint SHA:
`e1dd4917ff97b02e3bce524daf0f75fe4612d05915dcd5e82a3cfd33e5d00878`;
verification SHA:
`68b0bee06bc3a4fc6dcf443a1af09d691c0799c4ee17f3585fd029d95541446f`.
This is not another resume-process test. Earlier checkpoints remain preserved:
the historical LiDAR step-500 and laser step-250 minima still outperform this
new stage's respective selections. The optimizer continues unchanged toward
the existing global-3000/7200-new-second bound; the next validation is 2500.


### Correction: exact ARKit high-resolution intrinsics are released

The earlier assumption that exact high-resolution K was unavailable was
incorrect. The [official raw documentation](https://github.com/apple/ARKitScenes/blob/main/raw/README.md)
names `wide_intrinsics`; direct publisher URLs for `wide_intrinsics.zip` and
`wide.zip` work, despite their omission from the default downloader's asset
list. The new diagnostic retrieved the **exact timestamp camera and wide RGB
for all nine previously reviewed training frames**, with retained member
SHA256/CRC proofs. No broad archive download, fitted calibration, native mask
change or guessed camera was used. Eight new local tests and thirteen remote
camera/display tests passed.

`arkit-exact-faro-camera-01` reproduced the original retained depth, confidence,
camera and support before comparing FARO with the newly released camera.
All nine new six-panel pages were actually inspected. Released focal/principal
values differ from the integer-scaled VGA hypothesis by at most **0.004 pixels**
(consistent with textual precision), while the half-center hypothesis has
approximately a one-pixel principal-point difference. Exact-camera wide/native
RGB mean absolute differences span **0.003705–0.010926** on a 0–1 scale.
Diagnostic SHA:
`caddc9793dccb335e41a737c67df599952a6588e44c311068030d7db3a5a5c3a`;
visual review SHA:
`0002e9d0a069aa74435bf885ef99fe4cfff66d7eac9fac2b61234f770f3297c2`.

The camera-availability ambiguity is now resolved for these nine frames.
It does **not** remove the broad -0.039 m near-wall residual at 27177, the
wall/baseboard residual at 27185 or the panel/bowl disagreement at 41546
(87.59% of compared pixels within 5 cm). Mirror and reference-hole limitations
also persist. These comparisons still cover only three videos and rotations
0/90, in a publisher-filtered subset. They neither identify a physically
correct source at every pixel nor establish full-source accuracy. No new
exclusion or ARKit source acceptance is issued. Use these exact camera proofs
in the remaining qualified source review; do not repeat the availability
search or rerender the same nine cases simply to defer disposition.


### Paired step-2500 validation and preserved NYU/laser checkpoint

The completed step-2500 comparison reuses the same four evaluation views,
image order, protocols, frozen teacher, five heldout digests and 18 known-camera
laser cases. Aggregates were recomputed from recorded per-image results; no
model inference was repeated. Comparison SHA:
`50e1cd6648faad031361f3c4bf3b832014e84c9db07ca3a5944f8e2fba55d692`.

| Original evaluation metric | Step 2250 | Step 2500 | Interpretation |
| --- | ---: | ---: | --- |
| Macro spatial cosine | 0.95980578 | 0.95991014 | Slight recovery |
| Macro summary cosine | 0.93185091 | 0.93192420 | Slight recovery |
| NYU AbsRel | 0.06986465 | 0.06906248 | 1.15% lower error; new NYU best |
| ScanNet LiDAR AbsRel | 0.09596369 | 0.09701413 | 1.09% higher error |
| Known-camera laser AbsRel | 0.12761458 | 0.12574232 | 1.47% lower error |

Original-resolution NYU improves 13.03% from the archived feature baseline.
LiDAR remains 5.31% worse than that baseline, and both original cosine scores
remain below the initial values and step 2000. The fixed 400x640 view is separate:
its NYU AbsRel is 0.06985523, and its summary cosine 0.92977160 remains slightly
above its own initial 0.92961610. These results do not establish overall gains
or attribute changes causally to a source.

W&B's optimizer-step-2500 history exactly matches all five primary metrics and
the immutable provenance. Its three stored selection baselines retain the
previously observed one-ULP API serialization differences; the runtime config
is unchanged. Online evidence SHA:
`a58e0784659e83564c117c69a0e50685444d6c8c68dd6afb0b469f5f25b19589`.

`reviewed-stage3-best-nyu-laser-2500-01/checkpoint-step2500.pt` is an immutable
hard link to the common NYU/laser selection. CPU readback found finite adapter
and Adam tensors, all 100 Adam states at 2500, cursor 8000, and exact runtime
config, stream digests and model revisions. Checkpoint SHA:
`b497d63fb6999c66f21aac2966332b38e65301fc14ebdeb122b84ea81ac0c791`;
verification SHA:
`b2030a9f90e4e3e41fc919768cca255f28346102ec5e95723eff67ff9c53942b`.
This does not repeat the independent initial-resume proof. Earlier selections
remain preserved, including the lower historical LiDAR/laser minima. Active
stage-3 training continues under its existing step/time bound and data seal.


### Qualified ARKit source eligibility and explicit FARO dispositions

`arkit-source-review-02` reuses the completed full producer readback, all
197,542 frozen outcomes, source camera/units checks, 36 candidate reviews,
2,618 reproduced rejection dispositions, nine exact-camera FARO pages and
381 actual production-loader crop views. It records **194,916 eligible native
sensor candidates across 1,478 visit groups and 2,626 geometry exclusions**.
Source review SHA:
`c7939c09f59947b45bf3158f604d2067a08ea2e9b099da385ea64093150c7941`;
complete routes/report SHA:
`bf8a5db072eeedef6691616cc0cea03e5d06378220c8d84d3724d6287f95f7c9`.

The exact reviewed frames 27177, 27185, 41530, 41531 and 41546 receive explicit
geometry-only exclusions for persistent broad residual or reflected/occluded
surface ambiguity. These are conservative individual decisions, not proof
that one sensor is physically wrong, a numeric error threshold, fitted bias
correction, or a rule extended to neighboring frames. Frames 4591, 4600, 4606
and 27155 remain eligible under the unchanged confidence/stability/registration
masks, with their smaller residual and reference-hole limitations retained.
All five newly excluded frames were found in the unchanged accepted RGB replay.

The mandatory policy-specific register now contains eight candidates (the
three earlier flags plus these five); SHA:
`dcf838252f128a18b886ef40f6e56660ed155a838c1ea92d7000bf0dbf278314`.
Global known geometry exclusions now total **4,083**, SHA
`02ba17535280f30ddc57443f62bcc4f5db6042558268adc251d7fce58beae150`.
Only the previously excluded white Hypersim image 210812 is excluded from RGB.

Eligibility is for **noisy native mobile-sensor metric supervision**, distinct
from rendered synthetic depth, FARO ground truth and relative teacher labels.
Metric units and a known camera do not guarantee physical accuracy. The nine
FARO comparisons remain limited to three videos in a publisher-filtered subset;
visibility, timing and reflective-surface uncertainty are explicitly retained.
Sparse crop cases keep their existing fewer-than-32-observations zero-geometry
route. No source-wide accuracy or crop-sparsity prevalence is inferred.

Sixteen source publication/outcome tests passed locally and remotely. Attempt
01 stopped before source-review publication because rejected rows legitimately
have NULL candidate statistics; attempt 02 handles that representation and
reconciles every route. Preserve attempt 01 as failed evidence and use 02.

`prepare_arkit_reviewed_training.py` prepares a separate seven-source scope:
602,337 original images, expected 598,298 selected geometry and 4,039 stage
exclusions. Expected labels are 194,916 measured sensor, 189,025 synthetic and
214,357 relative teacher. It copies the exact previous inventory, preserving
the 35,783-image ScanNet teacher subset rather than widening it to the entire
unfinished source. Thirty scope/readback/admission tests passed locally and
remotely. The CPU job `reviewed-stage4-data-01` runs from immutable
`code-arkit-stage4-v1` / `fc3e77c`, with one worker and explicit producer-readback
reuse. It will bind the expanded identities to the completed full canonical
comparison. This launch is not completed accounting or a training seal;
coverage, loss and an actual archived-model gate remain required. The active
403,382-geometry / 796,477-RGB stage remains unchanged.

The same CPU job subsequently **completed** all 602,337 source routes:
598,298 selected, 4,039 excluded and zero unresolved. It preserved the exact
35,783-image ScanNet subset and all native/external quarantine records.
Accounting SHA:
`a033a256479c68e3d7d9ba8485ea56b75df7db5d0bfed5096d712ace97a108e6`;
index SHA:
`c5ec2e5cf58f078519be7ccee54bc804baf1abce877e269f49bb63b146870b40`.
The separate final isolation report binds all scoped identities to the completed
canonical census, with zero exact overlap or forbidden shared-scene rows and
the same 796,477 RGB replay; SHA
`a4b7563743f32696146392aa94ded45ee7a3c6c20e1a8bc46f2d9c85fe72a207`.
The intermediate data report retains its earlier `isolation_pending` stage;
use the completed `isolation-report.json` as the later isolation result.
Neither report opens the optimizer gate. No need to rerun this accounting.

### Stage 3 completed at its time limit; paired 2750 and final 2895

The existing trainer stopped normally after **895 new updates, global step
2895**, at the two-hour wall-clock budget. Its final evaluation completed once;
the additional evaluation time does not represent further optimizer updates.
All 895 training records are finite. The final checkpoint has 100 finite Adam
states at 2895, stream cursor 14320, unchanged feature statistics from parent
2000, and exact config, stream digests and revisions. Final checkpoint SHA:
`33c69a8189a53580d483f10677415baf299f498a2f0925c8b470405f6963d2f9`;
finish proof SHA:
`3bfd3a8f5387258d906d7d7772b74415d79da178e2b46a924697a23151552531`.

The existing reviewer checked the same four views, image order, protocols,
frozen teacher, five heldout identities and 18 known-camera laser cases.
No inference was repeated. Paired review SHAs are
`bb561e5f9bf66da347981618b3db913b75ecc1e6cbc6bab1b99044aec6a6ce04`
(2750) and
`cc41341429cb31e83608ef2a970029f6954ebc54c96737e8f788f7b55a3d9df4`
(2895). W&B run `4edbd06f` is finished; its five primary validation metrics
match both recorded steps exactly.

| Original evaluation metric | Step 2750 | Final 2895 |
| --- | ---: | ---: |
| Macro spatial cosine | 0.95975035 | 0.96011045 |
| Macro summary cosine | 0.93188013 | 0.93197556 |
| NYU AbsRel | 0.06847355 | 0.06907859 |
| ScanNet LiDAR AbsRel | 0.09686707 | 0.09651619 |
| Known-camera laser AbsRel | 0.12533590 | 0.12954709 |

Both final cosine scores recover from 2750 but remain below the archived
feature baseline. Final NYU error is 13.01% lower than that baseline; the new
best at 2750 is 13.77% lower. Final LiDAR remains 4.77% worse than the initial
value. Laser worsens 3.36% from 2750. These are mixed results, without an
overall improvement or source-causality claim. The fixed 400x640 protocol
remains separate. All prior bests are preserved; the immutable common NYU/laser
2750 checkpoint SHA is
`93d39a7247b6920ee4e0874f1426d5593f38a00074e11fe8b721a3722f6f2389`.

`freeze_arkit_training_stage.py` binds the next seven-source stage to the actual
completed parent step, including a time-limit stop between validation intervals.
It proposes 20% ARKit geometry draws and scales the previous six source shares
by 0.8, keeping RGB replay, losses, LR and Adam settings unchanged. Nineteen
local and remote tests cover incomplete final evaluations, changed source or
isolation evidence, and exact final selection baselines. A fresh actual model
backward probe and seal are still required before that stage can update weights.

### Twelve additional TartanAir seaside rejections

`tartanair-rejection-review-08` reproduced twelve new exact source pairs and
statistics while leaving the earlier 44 decisions unchanged. All twelve
six-panel pages were inspected in six paired half-size contacts. The views
are dominated by sky and distant water, with narrow building, rail and pole
support near the image edges. Final stereo-supported fractions are 2.56–4.77%,
below the unchanged 5% threshold. Original arrays and thresholds are unchanged;
this is a geometry-task support decision, not an independent physical-error
or invalid-RGB claim.

All twelve receive explicit geometry-only exclusions, and all twelve exact
RGB identities were found in the unchanged 796,477-image replay. There are now
56 disposed TartanAir rejections in this snapshot and 4,095 known global geometry
exclusions. The merged exclusion SHA is
`2ab2ee38c8c789b5cc1f86b2890fed3227103cbfa5b5c9254ab952b44067cf7f`.
Only the earlier white Hypersim image remains excluded from RGB; no neighboring
frames or active training artifacts were changed.

### Stage 4 admitted and actually continuing from 2895

`reviewed-stage4-request-01` and `reviewed-stage4-readiness-01` bind the completed
598,298-image selection to final parent 2895. The 100,000-draw production sampler
observed ARKit at 19.69% for a requested 20%, within the unchanged 0.5 percentage
point tolerance; it covered all 1,478 ARKit visit groups and 25 ScanNet teacher
scenes. Previous geometry source shares are scaled by 0.8. The 796,477-image
RGB replay, losses, constant LR 1e-5, batch 1/accumulation 16 and 400x640 grid
remain unchanged. Request SHA:
`aa5b415cab16000d9a18bc1f73f2f15efc97c9df220e8dfa6ff7e45a964a50d4`;
nonmodel plan SHA:
`db2aebe4da67a240af313d2620349673f61da8041b726826799cec0db2e4bae0`.

The actual parent-model probe passed **84 forward/backward cases across eight
source/label strata**, including twelve ARKit measured-sensor cases. Each stratum
has finite nonzero geometry gradients. The known sparse COCO ordinal 287969 at
probe cursor 50 has zero observations, zero geometry and active feature loss;
the other 83 cases exercise geometry. The frozen backbones, adapter and feature
statistics remain unchanged and no optimizer was constructed. Model proof SHA:
`fa8e5418c248587b2f862dd1e528ae75ec8abec5080d66582415e364f4f661b1`.

The new seal SHA is
`5d238e12ab80c2979f63496850b051aa66d72d409dba616ece195d65ad654be9`;
runtime config SHA:
`9baea9769b89a940204ed870f4f49337ab83c7f6c2470ecdf4181583046cd28f`.
The existing immutable training implementation `8332c8c` launched a separate
trainer, and its actual initial checkpoint was independently read back on CPU:
all 100 Adam states remain at 2895; adapter, Adam, RNG, feature statistics and
revisions are bitwise equal to the parent, with new stream cursor zero. Replay
and heldout digests are unchanged. Startup proof SHA:
`ca8112f3ac6824c9f0eefedc95cac43943e6f9b992b296283d702bd35213ec4b`.

Actual optimizer updates have resumed in `reviewed-stage4-training-01`, online
W&B run `6bddb43a`. Online provenance matches exactly. The API serializes only
the spatial selection baseline one ULP above the immutable file; all remaining
config values match. This is recorded without changing the runtime file.
The stage allows at most 1,000 new updates (global 3895) or 7,200 new seconds.
Cadence is relative to parent 2895: first save 2945 and first validation 3145.
All earlier bests and the native-source jobs are preserved. Whole-corpus
acceptance is still separate from this accepted seven-source training scope.

The first 20 actual updates (2896–2915) were preserved and checked: all losses
and gradients are finite, with measured nonzero parameter-update norms at 2900
and 2910. First-updates proof SHA:
`44ec3469155a0e504071e5bdeb0f66dd8a07b4f274687d15702bac7f4f0fb87f`.
At 16:33 UTC the stage had reached 2927, with 32 finite new updates and no OOM.
Stage-4 evaluation gains have not yet been measured; compare 3145 to the same
parent/initial images and protocols when that validation completes.

### Stage 4 paired validation at 3145

`reviewed-stage4-validation-3145-review-01` verifies the same original and
400x640 image sets (1,268 each), LiDAR (120), DSLR (128), and known-camera laser
cases (18), with unchanged protocols, frozen-teacher results and five heldout
digests. Aggregates were recomputed from the saved per-image results; no model
inference was repeated. Comparison SHA:
`cc0f4c85925806978cdc717a42c4cccce62456847d85edb663e09f5a87183f79`.
The five online validation metrics match their saved values exactly in W&B
run `6bddb43a`; online proof SHA:
`f4d4d62e278327eb6a113bbd72fe1372953ba7f3b0435e115176a2188d6a6366`.

Original-resolution macro spatial cosine moves from 0.96011045 at parent
2895 to 0.96003750, and summary cosine from 0.93197556 to 0.93192506; both
remain below the archived feature baseline. NYU AbsRel improves 1.29% from
the parent to **0.06819089**, a new observed best and 14.13% below the initial
0.07941177. LiDAR AbsRel rises to **0.09990830**, 3.51% worse than the parent
and 8.46% worse than initial. Known-camera laser AbsRel falls 5.33% from the
parent to **0.12264333**. A separate exact protocol/image/support comparison
with the earlier step-250 result, 0.12278013, confirms a narrowly lower mean
(0.11%); this is not a significance or overall-improvement claim. Historical
laser comparison SHA:
`9c44b73f39097c5f222ac624b1b8f75aa00de55c184ebc5643891be9666c0ac4`.
The 400x640 NYU result improves 0.19% from the parent; that protocol remains
separate from original resolution.

The selected NYU/laser checkpoint was hard-linked into
`reviewed-stage4-best-nyu-laser-3145-01` before later selection could replace
it. CPU readback confirms finite values in all 100 adapter tensors and 300
optimizer tensors, all 100 Adam states at 3145, cursor 4000, and exact runtime
config, stream digests and revisions. Feature statistics and an independent
resume process were not re-tested in this checkpoint review. Checkpoint SHA:
`2481da224014c0f3752fefaf2fc1876965c2b4196e98043d4b99286cd84ac845`;
proof SHA:
`bf75330542655f92b2cebc36e2808a1fbde413e5efb9f9565f86e60d27cfa607`.
The active trainer, accepted inputs and all earlier best checkpoints continue
unchanged. Next scheduled validation is 3395.

### Four additional TartanAir Hard/P000 rejections

`tartanair-rejection-review-09` reproduces only the four new exact inputs
496469–496472 (seasidetown/Hard/P000, frames 000000–000003), preserving the
earlier 56 dispositions. Retained stereo arrays, source receipts and rejection
statistics match exactly. All four original six-panel pages were inspected
individually. Sky and distant sea dominate the views; the task-range mask
retains 14.89–19.63%, while final stereo support contracts to a narrow lower
corner and covers only 2.06–4.11%, below the unchanged 5% threshold.

These four exact inputs receive geometry-only exclusions. A full scan of the
unchanged 796,477-image replay found each original RGB identity, hash and group
exactly once. No arrays, thresholds, active training inputs or neighboring
frames were changed. This is a task-support decision, not independent proof
of physical depth error. Decision report SHA:
`25be4c4244d752ab74483944d88653331a39063105a950a5cb47d97f7da7e347`.
The snapshot now has 60 disposed TartanAir rejections and 4,099 known global
geometry exclusions; only white Hypersim ordinal 210812 is excluded from RGB.
Merged exclusion SHA:
`8da1d75e2f5adf96ffd3c78290207011d1fda7f449f4c51acda0317a8d5a817a`.

### Stage 4 paired validation at 3395

`reviewed-stage4-validation-3395-review-01` verifies unchanged four-view image
orders, protocols, frozen-teacher results, five heldout digests and 18 known-camera
laser cases against 3145. The five W&B validation metrics match their saved
records exactly; no inference was repeated. Comparison SHA:
`f9844a2bf4f1c6ba6d253a81ae751a03cd49a7342ea05560a80ef6709c71822a`;
online proof SHA:
`e215d9f10ec398e32da07add93cd425f543f15d0d21cf9bbec6b183c7dd45016`.

Original macro spatial/summary cosine decline slightly to 0.95997293/0.93192033,
remaining below both the stage parent and initial feature checkpoint. NYU AbsRel
is 0.06962723, 2.11% worse than 3145 but 12.32% better than initial. LiDAR AbsRel
improves 3.46% from 3145 to 0.09645290, still 4.71% worse than initial; the earlier
step-500 result remains better. Known-camera laser AbsRel improves another 2.70%
to **0.11932770**, a new observed low. These remain task-specific changes without
an overall gain or causal claim. In the separate 400x640 protocol, NYU worsens
1.15% from 3145 while summary cosine increases slightly.

The stage-selected laser/LiDAR checkpoint was preserved in
`reviewed-stage4-best-laser-lidar-3395-01`. CPU readback confirms finite adapter
and optimizer tensors, all 100 Adam states at 3395, cursor 8000, and exact config,
stream digests and revisions. Checkpoint SHA:
`710dff5cf6259a31cf5577b591c922791cf407f1837f9de03a1757f5da489bda`;
proof SHA:
`6a38a21cabae47359f0114bf81900a0c2ae5e3c777f610acdc258ff685b0703b`.
The best NYU checkpoint remains 3145 and all older bests are retained. Training
continues on the unchanged 598,298 geometry inputs and 796,477 RGB replay inputs;
the next validation is 3645.

### Two TartanAir Hard/P002 rejections and full-outcome tooling

`tartanair-rejection-review-10` exactly reproduces the two new inputs 497095 and
497096, seaside-town Hard/P002 frames 000000/000001. Both original six-panel
pages were inspected. Sky and distant sea dominate, with a narrow building edge,
rail and posts; final support is 2.87–3.84%, below the unchanged 5% minimum.
Both receive exact geometry-only exclusions and both original RGB identities
were found once in the unchanged replay. The earlier 60 decisions were reused.
The snapshot now contains 62 disposed TartanAir rejections and 4,101 known global
geometry exclusions, still with only white Hypersim ordinal 210812 excluded from
RGB. Decision report SHA:
`f13b4c95c5ad4feadd72acf02e8f6341e10b56baacdb360f4ca7ce455e1bf317`;
merged exclusion SHA:
`81b99a558ea9abe5335e7f8adf5c884a384f4e7170ee5c20c0ae920c1c30a13b`.

`audit_tartanair_outcomes.py` is ready in `code-tartanair-outcomes-v2` at
`94a410f`. It requires completed producer payload readback, reconciles every
frozen RGB once with a candidate or the exact persisted rejection, checks stereo
source identities, support nesting, units and unchanged thresholds, and uses
a local temporary SQLite index. It selects bounded per-environment/difficulty
extremes for later review; it does not accept labels or repeat payload readback.
The original family/split helper preserves day/night grouping and heldout
variants, while environment, family and trajectory counts remain distinct.

All 22 targeted tests pass locally and remotely. Metadata compatibility was
checked against the existing 47 preflight candidates and one rejection, covering
12 environment names and 11 families; source generation and the original
preflight were not rerun. Tooling receipt SHA:
`97ffd034acaf3b01ae90938b4e7e06c77ce83e44bb475d40ead386a505082898`.
The full TartanAir outcome audit has **not run**: its producer is still generating
the remaining images. Run it only after its final generation report and full
payload readback complete.
