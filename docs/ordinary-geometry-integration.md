# Geometry attached to the restored feature stream

## November 5 priority change: pure-feature continuation for deployment

Chongxi explicitly stopped further ScanNet++ geometry/source validation and
requested renewed MoGe-v2 feature alignment on the projection-filtered RGB pool.
The pending geometry wake-up was cancelled. Do not restart that investigation,
additional dense images or bulk label generation under the old handoffs.

Current follow-up: Chongxi explicitly authorized adding generated empty-room RGB.
The first2,000 corrected ProcTHOR RGB are now in actual bounded feature training
from checkpoint233538,12.5% synthetic/87.5% real, all F weights1 and no geometry
loss. Original unreplaced real keys, crop/resolution conventions, batch32 and
Adam/RMS/RNG are preserved. See `docs/procthor-empty-rgb.md` for current jobs,
frozen manifests, independent official-house validation and completion checks.

`projection-filtered-feature-continuation-01` launched at 06:41:17 UTC, wrapper
PID266999, using immutable `code-filtered-feature-continuation-v1` / `002a58a`.
It starts from the completed pure-feature A checkpoint at232538, SHA256
`99b315124ae08905980f500245e828299f9586bc58deb673b2ba0c4b6c6c2375`.
The new bounded segment allows1,000 further updates to233538 or3,600 seconds,
with save50/eval250 and no automatic retry or additional segment. It has a new
output and W&B identity; the completed250-step A artifacts stay immutable.

All trainer `src` bytes are identical to the previously exercised A execution
83efb10. The additive launcher reuses its four-thread ordered metadata loader
and per-consumed-image SHA checks. It verifies adapter/RMS/full Adam/RNG/cursor,
model revisions, data digests and performance at the actual initial save before
the first optimizer update. A normal constant-LR fork preserves cursor2000 and
Adam state; only constant scheduler bookkeeping, stage budget, selection and
tracking are new. Four feature coefficients stay1, LR1e-5, three resolutions,
effective batch32 and the original permutation/crop pipeline are preserved.
There is no geometry loss, sidecar or replay mixture. Thirteen targeted tests
passed locally and remotely, including actual tiny-model fork preservation and
metadata equivalence; eleven new launcher tests passed again after adding CPU
copies for cross-device initial-state comparison.

The segment completed normally at233538 with1,000 new finite updates. Actual
initial adapter/RMS/all100 Adam states/RNG/cursor matched the parent. Final
CPU readback verified all100 Adam steps233538, cursor10000, finite state and
unchanged RMS/model/data identities. W&B88bd627a finished; all475 final metrics
match saved evaluation. Two API config baseline values differ by exactly one
floating-point ULP; runtime config is unchanged. The plan SHA
is `25baa4bcda2ad70679bc1b43d17fa2f9719fcad8bada8471cc7a7fa795b7b0d5`,
config SHA `000d128c1842b4df47342268d40816fc1a90e82b71df3d07e77ef2b4b2d23125`.

The retained1,597,911 RGB rows exclude all39,997 unscreened COCO images and
white210812. This is a documented source hold, not individual projection
certification of every remaining image. ZInD panoramas were never admitted.
The contamination is confirmed but its role as the main historical regression
cause is unproven. The cleaned250-step A result itself had slightly lower
primary cosines and2.18% worse iPhone LiDAR AbsRel than archived232288.
Keep that archive and A as deployment candidates; evaluate the new checkpoints
on the same six-source/four-view protocols before choosing a deployment model.
No live model replacement is implied by a later training step.

Final six-source spatial/summary cosine: .9646449821/.9357640150; independent
LiDAR AbsRel .0931902844. Relative to filtered A232538 both cosines improve
slightly and LiDAR improves0.9991%. Relative to archived232288 spatial cosine
remains lower and LiDAR remains1.1635% worse; the original joint target is not
met. All four saved validation endpoints were paired without new inference.
Preserve the final checkpoint SHA
`d4ae22dc3b6f8b6a88dbcf2129b5a0f37337a10f87ce26a6096e1d2d6a717ec5`.

Current authorized work is clean empty-room RGB generation using ProcTHOR,
with no depth labels or geometry loss. See [the generation record](procthor-empty-rgb.md).

November4 projection correction: the historical1,637,908-row stream below is
superseded for future launches by the COCO hold. Materialized candidates contain
1,597,911 RGB rows and558,314 geometry targets; primary validation retains1012
non-COCO images. `train()` rejects old COCO inputs before model/Adam construction.
The new sidecar scope, sampler plan, config and acceptance proofs require
their own bindings; do not relabel the candidate index as accepted or reuse old
seals. Historical pixel/loss parity proofs remain evidence of their original
scope only. See the projection correction in `geometry-corpus-acceptance.md`.

This is an implementation contract for later geometry integration. It does not
admit another optimizer run. The completed original-feature reference remains
the comparison: all four feature weights are 1, the original ordered RGB pool
has 1,637,908 entries, the three resolutions and effective batch 32 are retained.
Its two primary cosines fell slightly and LiDAR AbsRel improved slightly; the
joint target failed. No historical regression cause is established.

## Implemented CPU foundation

`data.ordinary_crop_parameters` exposes the original integer crop box and mirror
decision. `TrainingImages` still performs PIL crop, mirror and bilinear resize.
The seed, Python rounding, both randint draws (even for zero ranges), flip draw,
orientation and uint8/float transport are unchanged. Independent synthetic tests
compare exact pixel tensors against the literal earlier implementation, across
all four rotations, all three training resolutions and both transport modes.

`ordinary_geometry.ordinary_view` consumes the already-oriented RGB camera and
the exact crop descriptor. For source dimensions W,H and crop l,t,cw,ch:

```
A = [[W/cw, 0, -l/cw], [0, H/ch, -t/ch], [0, 0, 1]]
F = [[-1, 0, 1], [0, 1, 0], [0, 0, 1]]   # mirror, else identity
R = diag(-1, 1, 1)                        # camera X reflection, else identity
H_view = F A
K_view = H_view K_rgb R
```

This gives `K_view R inv(K_rgb) == H_view`. The normalized convention uses pixel
centers `(x+.5)/W,(y+.5)/H`; no extra half-pixel translation belongs in A. For
output column j, the unmirrored source boundary coordinate is
`l + cw*(j+.5)/out_width`. Mirroring replaces normalized u with `1-u`.
Reflection preserves length and Z, and changes skew's sign as required.
Off-center/out-of-image principal points and anisotropy from integer crop
rounding remain represented. Calibration is not fitted or physically certified.

The existing observation selector can project these views without resampling
native depth. Synthetic tests verify exact native IDs/XYZ/query UV, multigrid
projection, invalid-storage invariance, unique observations and empty support.
Selection excludes rays outside the output pixel-center rectangle to avoid
prediction extrapolation. Neither the source masks nor missing depths are filled.

`verify_ordinary_rgb_pixels.py` is a CPU-only check of this newly changed RGB
implementation against archived source code and saved real-input hashes. It
selects the bounded rows in one sequential, hash-checked manifest pass. It does
not repeat the completed model gate or establish geometry-loss correctness.

## Remaining integration and admission requirements

1. Keep `DeterministicBatches` and the full ordered RGB records. Join optional
   accepted geometry by normalized image path plus encoded RGB SHA after sample
   selection. Bind the sidecar index to acceptance evidence; do not modify RGB
   records/digest or sample only the labeled subset. Missing, rejected or
   unaccepted targets contribute no geometry and keep their feature image.
2. Carry sidecar targets, including missing slots, through the exact existing
   resolution grouping in `grouped_window`, CPU normalization and device
   prefetch. Preserve tensor operations, ordering and group sizes. Bind each
   target to its original key and geometry receipt. Do not group by coverage.
3. Keep frozen encoding, adapter prediction and `FeatureStatistics.loss`
   unchanged. Geometry uses the raw adapter prediction through the frozen
   decoder; it must not feed RMS-normalized features to that decoder. When G is
   off, bypass geometry decoding/backward entirely and preserve global RNG.
4. For a compute group of n RGB images, geometry returns the sum of per-image
   losses divided by n, including zeros for missing/sparse targets. The existing
   outer `n/32` weight then produces the whole-update sum divided by 32. Do not
   normalize by the number of labeled images or divide by 32 twice. Report
   missing and sparse counts separately, as actual integer counts.
5. Propagate and validate an explicit camera contract from the sidecar through
   the selected observations into the loss. `select_observations` now carries the new view's contract field. The legacy
   loss checks centered K and rejects any explicitly ordinary target.
   An ordinary crop that happens to yield centered K could pass that numeric
   check. Therefore absence of an off-center error is not admission. The ordinary trainer is separately gated; no old loss guard is removed.
6. Define and test skew/anisotropy treatment in local losses. Their existing
   scalar focal radius cannot silently stand for a full arbitrary camera.
   Preserve physical XYZ and unchanged RGB; do not shear XYZ, recenter K or
   replace ordinary crops with ray warps to satisfy a validator. Unknown-camera
   evaluation retains its existing centered focal/shift recovery and projection;
   known-camera raw-XYZ supervision must not silently change that protocol.
7. Add meaningful paired tests of geometry-off feature pixels, all four losses,
   adapter gradients, RNG and grouping. Add ordinary-view analytic loss/gradient
   cases, native/pseudo routing, no pseudo metric-scale gradients, invalid masks,
   sparse/missing slots and all-image normalization. Then run a new config/code
   bound actual-model zero-update gate with exact initial state and nonzero
   geometry gradients before any future geometry optimizer run.

The old reference/performance validators deliberately reject geometry. Any new
path needs a narrow explicit validator and new evidence; neither the completed
reference seal nor earlier centered-warp corpus seals authorize the extension.

## Optional target references and prepared transport

`ordinary_geometry_data.GeometrySidecar` now binds a historical acceptance file,
its accounting report and the exact SQLite index. It checks disposition and
selected-membership counts, then queries by original RGB path and checks the
encoded SHA, source, group and orientation. It never uses the selected-subset
sample number to drive RGB sampling. Known-path identity mismatches fail even
for excluded rows; genuinely absent or excluded labels return `None`. Pack and
entry references remain bound to that index. This is historical data provenance,
not a new training seal. Payload checks are still required when geometry is read.

Connections are process/thread local, read-only and cleared on serialization.
Index replacement or metadata change after initial hashing fails. The optional
`GeometryAttachedImages` wrapper delegates RGB to the hash-verified compact
`TrainingImages`; it retains every original index and attaches a `GeometrySlot`.
`RGBGeometryBatch` carries all slots, including `None`, through the existing
shape grouping and normalization with the same tensor operations. Device
transport methods retain CPU references; CUDA behavior still needs actual gate
testing. The trainer/observer now accept these containers only through the new ordinary gate.

CPU tests compare both paths across worker and background preparation modes,
resumed cursors and mixed target availability. A tiny frozen-backbone model test
uses the existing four feature losses, nontrivial RMS, all weights 1 and the
existing per-group n/32 weighting: losses, adapter gradients and Torch RNG match
bitwise. This is a synthetic regression test, not the actual RADIO/MoGe gate.
`verify_geometry_sidecar.py` additionally checks bounded saved real input keys
through complete groups of 32, against saved compact pixel hashes and an ordinary
prepared RGB loader. It reads RGB and index metadata only, not geometry arrays.

Still pending: actual training admission for the new ordinary target/loss core,
CUDA transport verification and
actual-model paired gradient tests. Do not infer
loss admission from an ordinary camera whose principal point happens to be centered.

The first remote sidecar test exposed coarse filesystem timestamps: a rapid
SQLite change did not always change the stat fingerprint. The unresolved row
still failed the per-read check, but stat-only detection was insufficient. The
reader now also checks the SQLite header/commit counter before and after queries,
and only accepts a closed non-WAL index. A regression test holds all timestamps
constant while changing a selected row to excluded; the lookup must fail.

## Local-loss camera convention (optional ordinary core)

In pinned upstream `moge/train/losses.py`, the scalar focal enters the local loss
as `radius_3d = Z / (2 * level * focal)`. It influences anchor density, patch
membership by physical XYZ distance, alignment weighting and error diagnostics.
It does not perform ray projection. The existing training value is
`focal = (fx**-2 + fy**-2)**-.5`, for normalized intrinsics. With square physical
pixels it equals pixel focal divided by the full image diagonal. Inference
focal uses the half-diagonal convention and is twice that value; do not replace
the training scalar with inference focal without correcting that factor.

For `K = [[A, c], [0, 1]]`, lateral normalized rays are `A^-1 (uv-c)` and their
Jacobian is `J=A^-1`. Principal point c disappears from that derivative. Thus an
off-center or outside-image principal point alone needs no local-radius change.
The current inverse focal equals `||J||_F` when off-diagonal entries vanish.
The explicit ordinary core uses `focal=1/||J||_F`, which reduces exactly
to the existing scalar for zero skew and preserves reflection/skew-sign symmetry.
For upper-triangular skew s its inverse square is
`1/fx**2 + 1/fy**2 + s**2/(fx**2*fy**2)`.

This declares an RMS diagonal-radius convention: the mean squared lengths
of transformed diagonals `(1,1)` and `(1,-1)` equals `||J||_F**2`. A single scalar
does not reproduce every directional neighborhood of an anisotropic/skewed
camera. Exact directional neighborhoods would require a larger upstream patch
selection change. Any scalar extension must be opt-in to the ordinary contract,
tested for the zero-skew reduction, center independence, reflection symmetry,
physical ray displacement and finite gradients. Do not change XYZ or the fixed
unknown-camera evaluation to conceal these distinctions. This analysis does not
establish a historical regression cause or admit a geometry optimizer run.


## Ordinary target construction and loss core

`ordinary_geometry_targets.ordinary_target` now reads each bound member with its
payload SHA and metadata checks, rechecks original RGB bytes/oriented dimensions,
and requires the explicit native-to-RGB camera mapping for unequal grids. It
reproduces `ordinary_crop_parameters` from the original key. `augment` is a
required argument and must come from the actual original `TrainingImages`.
Source metadata/flags are retained; a missing reference returns `None`. No RGB
image, measured depth or support is warped/interpolated by this constructor.

`ordinary_geometry_losses` is a separate opt-in core. It requires the ordinary
camera, half-center pixel and local-radius contracts, verifies supported XYZ
project back to their query UVs, and rejects invalid shapes, IDs, cameras or
out-of-bounds queries even for sparse targets. The zero-skew special case uses
the exact legacy focal arithmetic. The legacy entry rejects ordinary targets
even if their principal point happens to be centered. Unknown-camera evaluation
is unchanged. The new trainer/config gate is implemented; no accepted runtime seal has been issued.

The loss mean counts all RGB slots, including missing and <32-observation slots.
Coverage counters are separate integer counts. Native metric scale retains the
upstream detached target; pseudo labels are median-normalized before clamps and
never use predicted metric scale. Tests cover full32 normalization across compute
groups. Upstream tests use96x128 geometry and assert finite nonzero gradients for
each component, including local16 (the old32x32 fixture could never meet its
32-point patch minimum). Zero-skew legacy/ordinary component losses and gradients
and their sum are checked bitwise using an explicit single-thread CPU fixture.
The suite's `conftest.py` sets two PyTorch threads, overriding environment thread
settings. At two threads even repeated legacy/legacy local4 backward differed
by3.64e-12 (and its summed gradient by7.28e-12); legacy/ordinary showed the same
effect. Forty single-thread comparisons were exact throughout. The arithmetic
test therefore explicitly selects one thread and restores the previous setting;
it retains zero tolerance. This observation does not explain historical model
regression or establish deterministic GPU backward behavior.
This CPU core evidence is not an actual RADIO/MoGe gradient gate.


## Trainer integration boundaries

Keep original `make_loader`/`DeterministicBatches` and `TrainingImages`; the new
path must not route through `CorpusImages`, mixture sampling or replay. The wrapper now binds the actual `TrainingImages.augment` Boolean in every
`GeometrySlot`. The target builder receives that setting rather than guessing it from K.

`RGBInputAudit` now carries annotated samples through collation and prepared
windows while hashing only RGB pixels. It separately binds each slot key, RGB
identity, augmentation setting and full reference hash. Publication remains commit-only. Prepared transport already supports the wrapper, but
CUDA behavior has not been checked on this path.

In the trainer, unwrap `.image` before the unchanged encode/adapter/four-feature
loss sequence. Geometry-off must bypass sidecar payloads, decoder and geometry
random work entirely. Geometry-on must decode raw adapter features, never RMS
normalized features. Keep the outer `n/32` gradient weight. Integer coverage
counts now sum in a separate int64 branch; only loss metrics receive the n/32
weight, and coverage is converted to JSON integers after the complete update.

Use a new config/code admission path. Retain existing reference and legacy corpus
validators; do not let historical data acceptance stand in for the new runtime
seal. Tests must exercise the real trainer branch and audit, including no-call
assertions for geometry-off, cursor/Adam/RNG parity, and full32 normalization.
Only then can a new actual-model zero-update gate check CUDA transport, frozen
state and paired feature/geometry gradients. Existing original reference02 can
supply the completed geometry-off comparison; its250-update input/evaluation
proofs should not be repeated without a concrete new discrepancy.


## Separately gated trainer and CPU integration checks

The top-level `ordinary_geometry` configuration keeps geometry separate from the
four feature weights, which must all equal 1. It requires the ordinary camera,
pixel and local-radius contracts, an exact historical sidecar and a NEW admission
reference. It rejects old paired/corpus objectives, replay, mixture sampling,
coverage-epoch switches, baseline inference and automatic tuning. The runtime
requires an explicit original-feature data fork and the prepared effective32
profile. The historical feature-reference runner explicitly rejects this key.

`open_ordinary_training_gate` executes before Accelerator/model/Adam construction.
It binds every runtime Python helper, configuration, performance profile, full
1,637,908-row RGB manifest, archived232288 checkpoint, historical sidecar, complete
8,000 presentation keys, heldout digests and the separate250-update budget. Both
CPU integration and actual CUDA zero-update evidence must match these bindings.
A separate publisher/endpoint wrapper now exists. No successful ordinary runtime seal has been issued; the actual CUDA gate remains pending.
The later actual trainer also reads back its own initial checkpoint before any
update and requires adapter/RMS/Adam/RNG equality, old Adam counters and cursor0.
This hook is CPU readback in the trainer process; it does not claim an independent
verifier process or replace the earlier actual-model gate.

The actual training loop now unwraps `.image` for the unchanged encode, adapter
and feature-loss sequence. G=0 bypasses payload reads, decoder and geometry core
entirely. G=1 reads the exact carried targets and decodes raw adapted features.
It adds the pinned official geometry sum separately from F1. Missing/sparse
images remain in the feature batch and the whole-update denominator.

New tests exercise the production loop with tiny CPU models and real fixture
cache members, including synthetic native and relative teacher targets,
sparse/absent labels and multiple resolutions. G=0 checks bitwise four feature
losses, per-update gradients, adapter/RMS/Adam/scheduler/RNG and cursor against the
ordinary feature path, with geometry payload/decoder/loss calls forbidden. G=1
checks actual payload construction, raw decoder input identity, full32 weighting
and integer counts. Two-worker audit tests verify identical pixel/group hashes,
slot permutation/metadata rejection and exclusion of unused prefetched windows.
Stale/pending config, code, profile, budget, plan or actual-model reports reject
before runtime construction. Synthetic test gates exist only as in-memory test
substitutions; no production admission is written by those tests.


## Actual CUDA observer (implemented; first run pending)

`probe_ordinary_geometry_model.py` now binds a concrete F1/ordinary-geometry
candidate, its original8000 permutation keys, complete runtime code digest,
archive232288, historical sidecar and existing target/pixel evidence. It takes
the existing GPU lock and explicitly forbids optimizer construction. It never
issues a training seal or starts a trainer. The first execution is
`ordinary-model-gate-01`, code `code-ordinary-model-v2` / `83c1803`.

The observer compares plain and annotated actual CUDA pixels, all four feature
losses, adapter gradients and RNG on the first three real32-image windows and
96 separately labeled source/resolution diagnostic keys. A predefined plain vs
plain repetition distinguishes nondeterminism from an annotation-path mismatch;
either failure blocks the gate, without relaxing tolerances. Geometry checks
use raw adapted features and independently differentiate the geometry objective.
Every pseudo/missing/sparse slot has zero scale gradient even in mixed native
batches. Targets are compared with their bound camera-proof identities, ray
arrays and support counts; missing targets cannot pass a dense-case check.

Memory diagnostics keep a full32 dense group and a future32 CUDA group resident
at each of the three resolutions. Exact archived Adam tensor bytes and twice
trainable parameter bytes are additionally reserved for moments/step workspace.
The report checks peak-reservation-adjusted free memory with8GiB headroom;
this conservative capacity observation is not an executed Adam update and
other processes' future allocations cannot be guaranteed. No automatic shrinking
or retry is permitted by the observer.

Actual adapter/RMS initialization is compared against CPU archive tensors using
bounded cross-device copies. Model state, source Adam bytes and RNG must remain
unchanged after discarded backwards. Source Adam is not loaded into an optimizer
by this observer; the separate real trainer initial-checkpoint hook still must
prove its actual Adam restoration before its first update.

The new observer/gate and production-loop regression selection passed28 tests
remotely with Torch/BLAS1; the local combined run plus final added mixed-scale
negative case covers the same28 CPU tests. One CUDA-only unit test was skipped
in the CPU invocation; actual device comparison belongs to the running model
gate. Earlier164 CPU integration checks remain supporting evidence. No actual
CUDA pass or new training admission is claimed here yet.


## First CUDA gate failure and isolated repeatability diagnosis

The first actual gate did not pass. On its first original32-image update,
original-vs-annotated and original-vs-original parameter-gradient equality both
failed. It exited with zero optimizer updates; no admission or training output
was created. The publisher was exercised against that failed evidence and
refused to publish. The new completion wrapper separately rejects incomplete
update logs, changed RMS/Adam and short input accounting (7 tests local/remote).

`ordinary-g0-repeatability-current-01` then repeated only the original feature
path three times, with exact saved reference pixels and archive232288 weights,
RMS and RNG. RADIO/teacher encoded outputs, adapter outputs, four feature losses,
loss gradients at adapter outputs and RNG were all bitwise equal. Parameter
gradients differed in78 of100 tensors: relative L2 differences were
0.00099430565 and0.00097277541, with maximum absolute differences
7.74278e-6 and8.55117e-6. This localizes the observed discrepancy to adapter
backward, not RGB transport or the feature-loss output derivative.

Global deterministic mode stopped at the existing CUDA median-with-indices
operation. That failed diagnostic is not evidence that median values caused
the discrepancy, and the original channel-cleaning formula was not replaced.
A separate `ordinary-g0-repeatability-sdpa-math-01` changed only SDPA backend
selection for diagnosis. All100 parameter-gradient tensors then matched exactly
in both repeated comparisons, as did forwards/losses/RNG. Thus attention-backend
selection explains repeatability in this bounded example. This does NOT establish
a cause of historical cosine/LiDAR regression or a benefit from changing the
training backend. No optimizer was constructed in any diagnostic.

The revised gate (`ordinary-model-gate-02`, execution116e7f6) explicitly separates
runtime-backend exact pixels/forward losses/RNG from exact gradient parity under
the demonstrated repeatable math diagnostic backend. It restores the original
backend before every geometry-on check. Training backend settings remain
unchanged. Evidence fields now name these separate checks; no default-backend
bitwise-gradient claim or relaxed numeric tolerance is emitted. Runtime code,
candidate config and both repeatability reports are bound anew. The previous
failed gate remains failed. Actual geometry gradients and full32 memory admission
are still pending this revised run.

## Completed CUDA gate and the bounded corrected training attempt

`ordinary-model-gate-02` completed with zero optimizer updates. Report SHA
`d09810e27547d469c49b16da672519bba9ce797e0f0f6ab898b44c8652670213`
binds execution116e7f6 and the exact runtime candidate. All27 feature comparisons
passed the separately named runtime-forward and diagnostic-math-backward checks.
All30 geometry cases passed, including actual first96 stream presentations,
source/resolution diagnostics and dense32-plus-future32 at all three resolutions.
Measured-sensor, synthetic-renderer and relative-teacher targets were covered;
pseudo/missing/sparse slots did not acquire a metric-scale gradient. Frozen
models, adapter/RMS, source Adam bytes and restored RNG remained unchanged.

Peak allocated memory was105,486,952,448 bytes and peak reserved memory was
118,459,727,872 bytes. Peak-adjusted free memory was20,956,774,400 bytes,
above the8GiB gate, with source Adam/workspace reservation included. This was
an optimizer-free capacity test, not a completed Adam update or a promise about
future allocations by other processes.

The runner separately binds the recorded runtime attention flags at publication
and launch. It rejects a report that claims default-backend bitwise gradient
equality or uses the math diagnostic backend for training. Its12 local/remote
tests include scope mislabeling and incomplete fixed250 endpoints. Runtime
package bytes remain identical to the actual model gate.

`ordinary-geometry-training-01` was admitted with a new seal
`c2902f84c58cae2cbe00dccc917574f3265c4d2097109e7344cce0b5c0f07e3c`
and launched from immutable runner executione71ab8c. It preserves F1, the full
1,637,908-image RGB permutation, ordinary crops, three resolutions, effective
batch32, and archive232288 adapter/RMS/Adam/RNG. Geometry is joined by exact
RGB identity, with missing/excluded targets retaining their feature slots and
the full RGB denominator. No corpus mixture or extra replay stream is introduced.
The bound is250 updates to232538 or2700 seconds; a short budget endpoint cannot
be reported as fixed250. No automatic follow-up or retry is configured.

The real trainer subsequently passed its saved initial-state comparison:
adapter, RMS, all100 Adam states and RNG matched archive232288 before its first
update. Full metadata startup was included in the time budget. This proof is
separate from the optimizer-free model observer.

The evaluation contract compares fixed final232538 against both archive232288
and `feature-original-reference-02`, verifies all8000 actual RGB pixels/keys/
prepared groups against that reference, and retains all four protocols. Both
original-resolution cosines must rise and independent LiDAR AbsRel must fall.
Known-camera laser remains a separate18-case post-training evaluation. No
baseline inference is repeated. Reviewer177f9f6 passed7 local/remote tests and
an actual saved-reference self-comparison across159/159/77/93 metric fields,
all exactly zero difference. This validates review compatibility, not training
benefit. Historical regression causality and joint improvement remain unresolved.

## Fixed250 result: matched inputs, joint target not met

`ordinary-geometry-training-01` completed normally at232538 with250 finite
updates, followed by its final evaluation. The stop reason was `step_limit`;
total elapsed time was2569.49 seconds within2700. The runner verified the finite
final adapter/Adam and unchanged RMS and preserved the fixed endpoint checkpoint
with SHA `e7f4d236bccf7c13ed6981446a43fa70d8df669069ece7a45cca56679e5d91d9`.
No further optimizer segment was launched.

Reviewer177f9f6 verified all8000 actual CPU RGB pixel hashes, identities and
prepared groups against `feature-original-reference-02`, removing only the
geometry annotation. Both used the restored feature recipe and archive Adam/RNG.
The current run had2910 geometry-supervised presentations (1876 metric and1034
relative teacher),5089 without targets, and1 sparse skip. All8000 retained their
feature branch and every update retained the full32 geometry denominator.

| Fixed comparison | Original spatial cosine | Original summary cosine | Independent LiDAR AbsRel |
| --- | ---: | ---: | ---: |
| Archive232288 | 0.961692055 | 0.932300927 | 0.092118447 |
| Restored feature-only,250 updates | 0.961659603 | 0.932249937 | 0.092032092 |
| Restored feature+geometry,250 updates | 0.961565083 | 0.932253152 | 0.091780616 |

Against archive, both original cosines declined and LiDAR AbsRel improved0.3667%.
Against the matched feature-only endpoint, spatial cosine declined0.000094520,
summary cosine increased0.000003215, and LiDAR AbsRel improved0.2732%.
Neither comparison meets the predeclared joint target. Structured3D had the
largest original spatial-cosine decline versus the feature-only endpoint among
the seven sources in that macro; this identifies a location of the observed
change, not its cause.

All four saved evaluation views retained identical images, protocols and frozen
teacher/model revisions, with all five heldout digests checked. No archive or
reference inference was repeated. A new18-case final-checkpoint known-camera
laser evaluation had AbsRel0.125318168 versus reference02's0.125379018, an
observed0.0485% reduction; this remains separate from independent LiDAR and the
joint criterion. Original-view NYU AbsRel improved7.9259% against archive and
2.3384% against reference02, also a secondary result.

W&B run `1f3ccc07` finished. All497 final validation values matched local records
numerically;32 equivalent integer/float representations had zero numeric
difference. `results-review-01/report.json` preserves the paired metrics and
`results-closure-01.json` closes its pending online check. Closure SHA:
`0db662b1e1c88eef366f6f9582fb5f475f013830f1355bd540a20ab0320804d2`.
This single-seed comparison establishes neither statistical significance nor
the historical regression cause. The documented default-backend gradient
repeatability limitation remains; training did not switch to the diagnostic
math backend. Existing reference, control, model-gate and startup proofs need
not be repeated.

## November 4 input-projection correction

The matched-pixel proofs above establish input equality, not suitability of
those pixels for a perspective-camera task. Subsequent review found a spherical
panorama in historical COCO training and pixel-verified fisheye/multi-photo
inputs in this250-update run. The old selected geometry index also contains
their teacher labels. Refer to the projection correction at the top of
`geometry-corpus-acceptance.md` and the bound evidence under
`projection-audit-20261104-01`.

All39,997 COCO images are held from new RGB/replay candidates pending individual
projection review;39,984 old selected COCO geometry targets are removed from the
new candidate index. `projection-filtered-candidates-01` is a data correction,
not a training seal. No historical config or result is rewritten, and no new
optimizer segment is authorized by an old unfiltered seal. Input contamination
is confirmed; its effect on the observed cosine/LiDAR changes remains untested.

### Projection-filtered paired continuation, 2026-11-04 08:12 UTC

The user authorized starting the corrected, COCO-held comparison. Arm A is the
pure original-feature path (all four feature weights 1, no geometry branch),
independently forked from archived step232288 with its Adam/RMS/RNG. Immutable
execution `code-projection-feature-run-v2` / `83efb10` runs
`projection-filtered-feature1-geometry0-02`, PID236620, W&B `06adaa78`.
At08:11UTC all250 updates reached232538; final evaluation was still running.
The wrapper's loading status remains stale until final completion; actual
`training/train.jsonl` is authoritative for successful update counts.

The new validator permits only the exact ordered removal of COCO from main
validation (1268 to1012), checking both runtime file digests. Other validation
views remain unchanged. The1597911-image permutation and all8000 planned keys
were checked. The actual initial adapter/RMS/all100Adam states/RNG matched the
archive; actual116 image presentations and24 zero-update backward cases passed.
The first20 successful updates/640 actual RGB identities matched the plan with
no COCO. Their online loss/gradient/cursor values matched exactly; metadata had
only equal integer/float representations. Nonzero update norm at232300 was
0.0326720551. Startup evidence SHA:
`ecb9f926f6331571b352132308fcac820606db8a4b914591192a495019bc4339`.

Preparation attempt01 stopped before any process launch because the old
byte-identity guard rejected the optional batch-container wrapper in
`performance.py`. V2 binds the exact two-line wrapper change and helper hash;
plain tensors still use the original `torch.cat`. No feature formula or runtime
backend was changed.21 related tests passed locally and remotely.

Arm B is not yet admitted or launched. Code `1f01a7b` adds a data-only filtered
geometry sidecar (558314 selected;39984 held;4039 previously excluded) and binds
ordinary model/runtime gates to the same comparison, manifest, parent and all
8000 keys. It retains mandatory fresh CPU/CUDA/model evidence and original
camera/loss checks.75 tests passed remotely. Sidecar publication is in progress;
128 saved A pixel presentations have been assembled for the next CPU checks,
without new image/model reads. This preparation does not establish improvement
or identify the historical cause. Compare fixed final endpoints against the
saved six-source baseline and unchanged iPhone/LiDAR view; preserve old results.

At08:13UTC A completed normally at fixed250, including final evaluation and
controller final finite-state/checkpoint readback. The immutable final checkpoint
SHA is `99b315124ae08905980f500245e828299f9586bc58deb673b2ba0c4b6c6c2375`.
All8000 actual keys matched the shared plan, with source counts
Scan4187/Hyp298/ARKit986/Tartan1206/Structured654/IRS663/NYU6 and no COCO.
Full input closure SHA `5e785553dc7b75b777c6475723e2542cbbbb13d4ec1e4e4d94e3f78753e2882d`.
Do not rerun A. Its paired result review and B's fresh actual-model gate remain
pending. The geometry-sidecar publisher PID236974 is still validating the closed
index; it has no GPU or optimizer and must not be relaunched concurrently.

## Filtered pair: A closed, fresh B checks active (November4 08:39 UTC)

A completed fixed250 at232538. Saved four-view review checked the retained
1012 identities, unchanged120 iPhone/LiDAR and128 DSLR identities, teacher GT
scalars where present, revisions and every saved aggregate against the final
validation log. The only permitted original/native protocol change from the
archive was images1268 to1012. Its six-source spatial cosine is
0.9645999481 versus0.9647740929, summary0.9356193124 versus0.9356613590,
and LiDAR AbsRel0.0941307515 versus0.0921184475 (+2.1845%). All three directional
criteria fail. Filtering this source alone did not solve the observed decline
in this250-step screen; no historical causal or significance claim follows.
W&B06adaa78 is finished; all475 final metric values match numerically, with only
equal integer/float representations. No inference or checkpoint proof repeated.
A closure `results-closure-01/report.json` SHA:
`7950a52675bcbedb7c4fbd8d342ae5632b6a0df2a61e0a54bc14dbdf362a48a1`.
Saved review SHA `7f662d234dc297ac2f09d0d91d49fe925eb454377f4130edbf4734bbce3c1b91`.

The existing B CPU job completed128 presentations:80 actual geometry members
and48 feature-only slots. RGB and prepared grouping match saved A pixels;
member hashes, target camera rays and unchanged RNG passed. Six target sources
and all seven RGB sources are represented; these are bounded diagnostics, not
population coverage. Target report SHA:
`14cbb8ff6076985cee3af6bd6ed13fe716c191785eede2ec038ec8810c48c9ad`.

Authorized controller242064 (`projection-filtered-geometry-sequence-01`) now
runs the fresh checks and, only on success, prepares and executes B once using
immutable `code-projection-geometry-v1/1f01a7b`. It preserves archive232288
adapter/RMS/Adam/RNG, A's1597911 RGB rows, all8000 planned keys, F1, three shapes,
effective32 and fixed250/2700s. New config digest starts `efe1022b` and binds
A's actual data digests plus the filtered sidecar. It never retries a failed
stage or extends a budget. Do not launch another B controller.

Fresh runtime and math-only repeatability checks completed with exact forward
losses/components/RNG. Default-backend gradient relative L2 differences were
0.00105924/0.00100742; isolated math gradients were bitwise equal. As before,
math is diagnostic only and the runtime backend is unchanged. The actual
full-scope CUDA geometry/state/memory gate is active; no B optimizer update
was yet verified at08:39. Its separate admission must complete before training.

The new saved-result reviewer can compare B to archive and A without inference;
it requires exact8000 committed RGB pixels/keys/groups after removing only the
new geometry annotation. Fifteen reviewer/input-pairing tests passed. A mirror
mount lookup failed before output publication in reviewer v1; v2 resolves the
same hash-bound metadata at the GPU mount. This did not affect training.

At08:49UTC the B actual model gate completed:27 feature comparisons and30
geometry cases, including all three dense32 plus staged-future32 shapes.
Peak-adjusted free headroom was19.5156GiB. Frozen modules, source Adam and RNG
checks passed; no optimizer was constructed by the observer. Model proof SHA:
`224cd7a08c351ebfe8c2f5d0872b0f5534067e400aec88bedb38364573d385c1`.
The existing controller then published the distinct admission and launched
`projection-filtered-feature1-geometry1-01` at08:49:08. It is loading metadata
and will verify the actual trainer initial checkpoint before updates. Zero B
optimizer updates have been verified at this observation; do not confuse process
launch with a completed update, and do not relaunch during metadata loading.

## Filtered B: actual startup verified (November 4, 09:30 UTC)

The existing B process completed initialization and began optimizer updates at
approximately09:26UTC. Its actual initial checkpoint readback verifies archive
232288 adapter, RMS, all Adam state and RNG exactly, with cursor0. The initial
proof SHA is `aee6bce5698a1c5b62f08a58be3255571cb4bbbce793071bf25880feaa5011a1`.
No checkpoint payload or actual-model gate was repeated by this review.

The first20 committed updates use640 actual RGB presentations whose pixels,
keys and compute groups match A exactly after removing only geometry annotation.
All four first-update feature components are equal to A. Coverage is232
supervised,408 feature-only and0 sparse, including173 metric targets; every
update retains the whole32-image denominator. Finite loss/gradients, weighted
F+G arithmetic and nonzero parameter updates passed. Startup proof SHA:
`f3ab29cf476ccde8486d9f1652f5df65bc5aca34805151e11d6fae78a6c92ef0`.

W&B run `a938045c` is online. Configuration/provenance and all first20 training
values match numerically, including optional update norms. Seven metadata and57
training values differ only in equal integer/float representation. Online proof
SHA `9628cd14010518b8b96ad0938358a79cfcdc331a12b359f97164e2d360036b77`.
The first read-only online wrapper failed before publication because its recursive
metadata helper expects string dictionary keys for optional per-step values.
Wrapper02 converts both step-key maps to strings, preserving exact value checks;
no training code or configuration changed.

At09:30:26UTC there are66 finite updates, globalstep232354. Initialization used
about2268 of the2700 elapsed-second budget, which includes loading and I/O.
Consequently fixed250 completion is not yet established and may be prevented by
the existing budget. The trainer checks that budget between updates, saves and
evaluates its actual endpoint; the fixed250 wrapper rejects a partial endpoint.
No automatic extension, retry or relabeling of partial results is authorized by
this experiment. The first20 proof establishes correct startup, not final quality
or a historical regression cause. A's completed fixed250 result remains unchanged.

## Filtered B stopped at133: partial result preserved (November4 09:48 UTC)

B stopped updating at global232421 after133 updates/4256 presentations. The
stop reason is `wall_clock_budget`: optimizer stopping elapsed2706.566s versus
the2700s limit; final evaluation and saving completed at2903.142s. All133 log
records are finite. The trainer finished and W&B closed normally; the outer
fixed250 wrapper then correctly rejected the incomplete endpoint. There is no
`completed.json`, no fixed250 pair result, and no automatic restart/extension.

Separate reviewer `code-projection-partial-review-v1/c070988` passed two focused
tests locally and remotely and preserved `partial-step232421.pt` by hardlink.
CPU readback checks finite adapter/all100 Adam states at232421, scheduler133,
cursor1064, unchanged RMS/model/data/config/provenance and original runtime
limits. It reuses the already completed initial restoration proof. All4256
actual input pixels/keys/groups match A's corresponding prefix. Saved four-view
identity/protocol/teacher-scalar/aggregate/log checks pass without inference.
The checkpoint SHA is
`8ae81b398dab992d565e8988a1d7b9cdfae9d0f6918b3f1d14e57763b9435398`;
partial review SHA is
`53003deb0829084bd3a825ca20ceba0e701a2137772b86526c180f3e6db4d43d`.

Descriptive133-step metrics versus the same archive baseline are spatial
0.9645058395 versus0.9647740929, summary0.9355873813 versus0.9356613590,
and LiDAR AbsRel0.0923453629 versus0.0921184475 (+0.24633%). These are not an
equal-update comparison against A250 or completion of the predeclared endpoint.
W&B `a938045c` is finished; all475 final values match numerically, with7 metadata
and30 metric equal integer/float representations. Online partial proof SHA:
`92f7438a891d27e075f71d94817228a8afc360e932c4c9ad7d4db9f582e4f15e`.

Future-only source revision2382b57 restores the same bounded four-thread ordered
metadata pass used by A and adds a progress file. Manifest hash/count/digest
checks remain mandatory. It changes neither image-loader workers, sample order,
augmentation, model/loss, runtime budget nor optimizer state. Four focused CPU
checks pass locally and remotely in final snapshot `code-ordinary-metadata-v2`
at6a1f7e2, including workers1/2/4 equality with original records/digest and
failure-before-verified-progress on a wrong digest. The first remote test failed
because its fixture inherited the workspace-specific output directory; the
fixture now uses its temporary directory, with runtime bytes unchanged. Its
test-only local progress file was retained under a separate evidence name.

The new metadata code has not run on the full corpus or been admitted for a new
optimizer run. Its implementation digest differs, so existing training seals
must not be reused. Neither the expected startup speedup nor a historical
model-regression explanation is established by these CPU tests.
