# Geometry attached to the restored feature stream

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
