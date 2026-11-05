# Empty-room RGB for MoGe-v2 feature alignment

Chongxi requested large numbers of clean empty-room, first-person RGB views.
Further ScanNet++ source/geometry investigations were explicitly stopped.
Generation uses official ProcTHOR houses and custom structural rooms rendered
in the same pinned simulator; ZInD and UnrealZoo are not inputs.
No synthetic depth labels or geometric loss are introduced.

Current state, Nov5 22:55 UTC: `room-feature-mix-40890-training-02` completed
its fixed1,000 updates normally at234538, using immutable
`code-room-feature-mix-v2` / `b5fde32`. The user's "start now" instruction
superseded the earlier50,000-image waiting threshold. All1,000 records are finite;
the final adapter/all100 Adam states are finite at234538, cursor8000, and the
fixed RMS, configuration and data identities match. The final checkpoint is
preserved under `finish-verification-01`. All four saved paired reviews are
complete. W&B is finished and all631 final validation scalars match numerically,
including78 official-house and78 structural metrics. Existing one-ULP metadata
display differences are recorded without changing the runtime configuration.
`room-feature-40890-finish-closure-01/report.json` binds the final evidence; the
old controller's `quality_review_pending` label is historical, not outstanding work.

The initial adapter/RMS/full Adam/RNG proof,33 actual model cases/152 RGB
presentations, and first20-update/640-key/online proofs are complete and reused.
Attempt01 stopped before model construction or any update because the archived
code lacked its vendor symlink; attempt02 used the original pinned clean vendor
sources. No package installation, model changes or geometry work was involved.
Do not relaunch either attempt or repeat the completed proofs.

The fixed training pool contains40,890 qualified RGB images:40,000 earlier
ProcTHOR empty-house views and890 new structural views from45 complete scenes.
The old images retain their earlier camera/window policy; only the new structural
source promises closed rooms with no windows or exterior background. Source labels
are preserved:40,800 perspective images plus90 explicitly approved orthographic
supplements. All source file/pixel checks and prior sampled reviews are reused.
The stopped7,180-image old batch, scene53 and growing generation outputs are held
out of this run. The combined manifest preserves every original row byte-for-byte
and has zero exact training/validation pixel, group or layout intersections.

Training starts independently from the existing pure-feature checkpoint233538,
preserving its adapter, RMS, full Adam and RNG. The original real permutation
continues at cursor10000; exactly4/32 slots use a seed43 permutation of the fixed
40,890 pool. All other real keys, crop seeds, three resolutions and effective32
remain unchanged. All four feature weights are1; no geometry targets or loss.
The bound is1,000 new updates or3,600 seconds, with an outer4,200-second timeout.
There is no automatic retry, next segment or deployment replacement.
W&B run: https://forge.coreweave.com/wandb/brainsim/radio-moge/runs/80b05757 .
The full1000-update plan draws3905 official-room and95 structural RGB images
from the pool, plus28000 original RGB; it does not claim a full40890-image epoch.

The completed new gate verified the actual initial state and model backward/pixel
behavior, including both structural projection types. Original holdout protocols and the
saved64-image official-house baseline are reused. The distinct317-image structural
validation population gets its own parent baseline; it is never training data.
Validation runs every250 updates, with saved paired results reviewed against the
same parent and archive. A larger image pool does not establish a quality gain.

At the fixed1,000-update endpoint, the original validation macro spatial cosine
changes0.9646449821 to0.9644499831, summary cosine0.9357640150 to0.9358074926,
and independent LiDAR AbsRel0.0931902844 to0.0954916715 (+2.4696% worse than
the parent, +3.6618% worse than archive232288). The predefined joint target
fails against both parent and archive. These original macro results include
multiple real and synthetic sources; they must not be called an all-real average.

The distinct unseen structural317 population improves spatial cosine
0.9223505710 to0.9510275748 and summary cosine0.7415038003 to0.9381207416;
teacher-depth discrepancy falls65.7567%. Official-house64 cosine improves
0.9614996593 to0.9716912052 and0.8897025390 to0.9366651960, with teacher-depth
discrepancy down41.3044%. These are synthetic heldout alignment results, not
physical depth accuracy, real-wall planarity, a causal attribution, or replication.
The250-update candidate had lower LiDAR error than the final endpoint; all existing
best checkpoints are retained, with no automatic next segment or deployment.

Controller03 / PID316684 continues generating new scenes separately from the
frozen training pool. Batch08 closed all100 scenes54–153 with1,984 accepted RGB
(1,784 perspective and200 orthographic),16 rejected frames and no quarantined
scene. Dedicated freezing checked every encoded/decoded image, scene, mesh,
material, camera and receipt; exact pixel/group/layout intersections with the old
40,000, previous890, official64 and structural317 sets are zero. All64 fixed
sample images on16 pages were actually viewed. No visible exterior/window/agent
or magenta failure was observed in this sample; highlights, repetitive motifs and
shadow aliasing remain limitations, without an all-image visual certification.
`structural-batch08-closure-01/report.json` binds those completed results.

Batch09 also closed all100 scenes154–253 with1,984 RGB in4,069.49 seconds:
1,784 perspective and200 orthographic,16 rejected frames, zero duplicates and
no quarantined scenes. The dedicated freezer checked all new encoded/decoded
images, source assets, cameras and receipts. Exact pixel/group/layout overlaps
with both validation sets, the old40,000, previous890 and batch08's1,984 are zero.
All64 new fixed sample images on16 pages were actually viewed. No windows,
exterior backgrounds, simulator bodies or failed magenta frames were observed
in this sample. Bright walls/floors/highlights, repeated textures and coarse
shadow aliasing remain documented limitations; this is not an all-image visual
certification. No new exclusions were issued.
`structural-batch09-closure-01/report.json` binds the new evidence, with SHA
`fa4d5ff4bfe56ffc6ee6519df48052e41a4fa80210038905a99b442a2f84acba`.
At that closure, qualified images totalled44,858 in separate frozen manifests:40,000 official
ProcTHOR views and4,858 custom structural views. The additional3,968 images
since the fixed40,890 pool have not been trained on. Batch10, scenes254–353,
is already running under the same controller; do not duplicate it or change its
configuration. No optimizer was restarted and the completed training pool is unchanged.

Batch10 has since completed all100 scenes254–353 with1,986 RGB in4,052.50
seconds:1,786 perspective and200 orthographic,14 rejected frames, zero duplicates
and no quarantined scenes. All new files, decoded pixels, assets, cameras and
receipts passed freezing. Six exact pixel/group/layout isolation checks have zero
overlap, including the newly frozen batch09. All64 fixed new sample images on16
pages were actually viewed; no window, exterior, agent or magenta failure was
observed. Bright highlights, muted darker materials, repeated structural motifs
and coarse shadows remain documented limitations; no all-image certification or
new exclusions. `structural-batch10-closure-01/report.json` binds these results,
SHA `f60b65f49dd84c169076ccef4f597b616b1abaa08c8c9ee01b305142340cc617`.
Qualified synthetic RGB now totals46,844 in separate frozen manifests:40,000
official ProcTHOR views and6,844 custom structural views. The additional5,954
images since the completed40,890 pool have not been trained on. The existing
controller has automatically advanced to batch11, scenes354–453. No training
restart, input replacement or geometry work occurred.
Scene53 is held in full. Previously completed890-image and317-validation proofs
and sample reviews must not be repeated. The withdrawn small pilot remains
withdrawn; this new40890 run is separately bound to the latest user instruction.

The user changed the requested future images to
farther wall views, no windows showing outdoors, and visible square/round pillars,
rectangular beams and horizontal round members. The old batch06 was gracefully
stopped at a complete-house boundary: 359 houses / 7,180 RGB, IDs2000–2358.
Its final report is complete with `all_requested_houses_closed=false`; those
images are preserved separately and are not frozen into the40,000-image pool.
No next batch with the old camera/window policy will start automatically.

The bounded three-scene/six-image preview has completed in41.54 seconds using
the existing CPU Unity environment; all six actual images were viewed. It creates
new parameterized closed rooms and actual runtime structural meshes, with a
12m wall distance, square/round pillars, rectangular beams and horizontal round
members. These are custom scenes, not unchanged official ProcTHOR houses.
Both far frontal perspective (32-degree vertical FOV) and explicitly labelled
orthographic comparisons (2m half-height) are available. The user accepted the preview, then explicitly asked for MORE ordinary-camera
views from different angles. This latest instruction supersedes an orthographic-only
interpretation. The new batch uses18 ordinary pinhole views plus2 orthographic
supplemental views per scene, with explicit projection labels. Neither camera
choice changes the pure-feature/no-geometry objective.

Immutable execution `code-structural-preview-v3` / `e7d75c2` passed eight local
CPU tests for closed, outward-oriented primitive meshes, connected structure
placement, and camera origins inside the room. The actual render returned six
unique640x480 RGB images, zero visible magenta-background sentinel pixels,
zero windows/doors and hidden simulator agents. All39 referenced image/scene/
mesh/material files were hash-verified after copying. Actual image inspection
also found no window/exterior scene or simulator agent. Bright material highlights
and synthetic lighting remain limitations; this is a six-image preview, not a
diverse training corpus. Orthographic views crop the floor/ceiling and pillar ends.

Renderer report: `structural-room-camera-preview-03/report.json`, SHA
`c949e8f3a77f4818b5b35b93a1b4e1ed34b0c8874faacc09127600202a562435`.
Visual review: `structural-room-preview-review-01/report.json`, SHA
`bbc9c7a358404fbef67203b36073628dd79e712dfb358ac0a26ec831a907c477`.
Gallery: `.kanna/visualizations/structural-walls-columns-preview.html`.
The first preview attempt failed before RGB on a required `skyboxId`; the second
caught an orthographic frustum extending above/below the room. Both are preserved
as failed development attempts, and neither failure was bypassed or admitted.

The previous mixed-feature run completed its fixed1,000 updates at234538.
Only the earliest2,000 generated images were used. All previous model proofs
and results remain closed. The separately launched40890-image run does not alter
those artifacts; no deployment replacement has been made.
The user's stop on all ScanNet++ geometry/source work remains in force.


## Current multiview structural generation

`render_structural_room_batch.py`, immutable86dcd01 / `code-structural-multiview-v1`,
creates parameterized closed rooms with no windows/doors or visible agents.
Every scene has a grounded square pillar, round pillar, connected rectangular
beam and horizontal round member. Four layout families add rear pillars/depth
beams. Room dimensions, neutral wall colors, plain/textured wall materials,
member sizes and indoor point lights vary deterministically by scene identity.
These are custom procedural scenes rendered by the pinned ProcTHOR environment,
not unchanged official ProcTHOR houses or measured geometry labels.

The20-view camera plan contains18 perspective frames (front, left/right oblique,
high/low, farther/closer;32–58-degree vertical FOV) and2 explicitly tagged
orthographic frames. A smaller four-view preflight selects front, the two most
oblique side positions, and an orthographic view. Its ratio is therefore3:1;
the production ratio remains18:2. No projection type is mislabelled as another.
Scene-level train/validation seeds and layout hashes are disjoint; image-level
exact isolation must still be checked when the new pools are frozen.

The new eight CPU tests passed in0.90s; lint/compile passed. Actual preflight
`structural-room-multiview-preflight-01` completed8 scenes/32 images in128.91s,
with24 perspective and8 orthographic frames, zero duplicate frames and no
background-sentinel failures. Every encoded/decoded RGB hash was checked and
ALL32 images were actually viewed on8 contact pages. Structure, room corners
and perspective changes are visible; no window/exterior or simulator body was
seen. Synthetic highlights and shadow aliasing remain limitations. These32
images are preflight evidence, not the future training pool.

Preflight report SHA`cbcb032a33242c5e3f4939c6b4e9f1f600e2851a3ba6d1e665c9db4f133d52bd`;
visual review SHA`1871e5ac51db6fd4d24ca2e1e2926f1317eb9d4b1345212c340ffdb3cc777dbb`.
New gallery: `.kanna/visualizations/structural-room-multiview.html`.

Historical controller`structural-multiview-generation-controller-01`, PID315236, started
Nov5 17:59:03 UTC. It runs one CPU renderer at a time:16 independent validation
scenes/320 images, then training scene ranges8–257 and258–507, up to5,000 images
each. This is a10,000-training-image target plus320 validation images, not a
claim that those images already exist. Each250-scene job has a7200s inner bound;
the controller stops on any partial job without retry/extension. The first full
20-view validation scenes took about42s each, so a250-scene job may close only
a prefix within its bound. Preserve that prefix and use a fresh bounded range for
unprocessed scenes after inspection; never overwrite/retry completed scenes.
The generation plan SHA is`361d153c821c21994517d4406beee70438ffc159b5e4abaf29ff539a84486c01`.
This controller subsequently stopped on the validation failure described below;
neither of its planned training jobs started. It must not be relaunched.

No optimizer or training-manifest swap occurs in this controller. The old freezer
correctly requires official-house perspective data and must not be weakened to
mislabel this new source. A dedicated new-source freeze/isolation and explicit
feature-only training configuration are required before the approved new RGB is
used. All real-image validation identities/protocols remain unchanged.

## Strict frame rejection and continued structural generation

The original validation run closed scenes0–6 (140 frames), then stopped at
scene7/slot12 on one magenta background-sentinel pixel. A bounded six-image
diagnostic rendered slots12/13 with magenta, cyan and black backgrounds; all six
were viewed. Only pixel(y305,x525) in slot12 changed with the background color.
Slot13 was unchanged. This establishes a visible rendering seam at that pixel,
not a window or panorama; it does not identify a unique floating-point cause.
The incomplete scene7 is held in full, including its12 previously saved frames.
Diagnostic review SHA
`5b738f3f19a6c5822aa511699d743de7004f4a29293decf6aa15068068e10862`.

Renderer`c511d05` / `code-structural-multiview-v2` preserves the original scene,
camera and strict RGB predicates. It now retains a failed whole frame and its
metadata under`rejected/`, omits it from the candidate manifest, and continues
with the next originally planned slot. It does not paint pixels, move the camera
or generate replacements. More than two rejected slots in a20-view scene still
stops the job for inspection. All20 attempted slots must be accounted for in a
closed receipt. Ten targeted CPU tests passed; no image-validity threshold was
weakened. Accepted frames carry`strict-rgb-frame-filter-v2` metadata.

Controller02, PID315828, started18:34:59 UTC. It completed validation scenes8–16,
then sequentially processes five100-scene training ranges:8–107,108–207,
208–307,308–407 and408–507. Each training job has a6,000-second bound; one CPU
renderer runs at a time. Immutable plan SHA
`6b100d9048ff55c2c464d1a2369134f87afb2f038be3ddca59a4136db69a817f`.
Do not duplicate or change the running jobs. Ten thousand planned camera slots
will yield fewer retained images if frames fail. After these jobs close, any
shortfall must use new disjoint scene IDs at or above508, in new bounded outputs.
Never count rejected/validation frames or rerender closed scenes to meet50,000.

## Structural validation frozen and sampled review complete

The dedicated freezer`4833b26` reconstructs expected scene and camera records,
checks meshes/materials/assets and closed receipts, decodes and hashes every
selected image, validates projection labels and accounts for accepted/rejected
slots. Fourteen CPU tests passed, including pose/projection relabelling, missing
slots, source tampering, invalid RGB and validation overlap. The original
official-house freezer remains unchanged.

The actual317-image validation set combines140 frames from old closed scenes0–6
and177 frames from new scenes8–16. It has285 perspective and32 orthographic
frames, zero exact duplicates and zero pixel/group/layout overlap with the old
fixed64-image ProcTHOR validation selection. Three failed new views are retained
outside the manifest: scene8/slot9 and scene11/slots8,9. The whole held scene7 is
excluded. Manifest SHA
`e5213027b3664cbffaf5b6c3a39320c2e0c2f1af0b38b88fd9fa87e553c9dfed`;
freeze report SHA
`4981c01413c7405952de9e4b92e089e2d5b7ddf2355b94849e998b6db09984a9`.

Fixed slots0,10,11,18 from each of16 scenes were selected before viewing. All64
images on16 contact pages were actually inspected. Closed walls, square/round
pillars, connected beams and ordinary-camera angle changes are visible; no
window/exterior, simulator body or failed background frame was observed in this
sample. Bright highlights wash out some surfaces, shadows remain visibly
synthetic, orthographic frames crop structural ends, and motifs repeat. No new
exclusions were issued. This is a64-image sample, not visual certification of
all317 images or proof of model improvement. Visual report SHA
`3f5bb3aedb9e9894b6c043901cbb53b82e5f8f7d8f677bcf0ad6fdc520b6b456`.
No model inference, baseline rerun or optimizer was launched during this review.

## First890 structural training images and whole-scene quarantine

Controller02 stopped when training scene53 produced its third invalid view.
All three actual rejected originals (slots1,2,6) were viewed and their hashes
checked. Each contains exactly one magenta sentinel pixel at a wall/corner
boundary. All seven attempted images from this incomplete scene, including
four otherwise-valid frames, are held and will not be retried. Its remaining
camera slots were never generated. Hold report SHA
`c30d7625c138f1bad0c3dea2169468e031b24e525cfd1b86a8c8b802e678f351`.
The old terminal counter894 included those four unclosed frames; only890
belong to the45 complete scene receipts8–52.

The unchanged dedicated freezer verified all890 images, scene assets, camera
records and complete attempted-slot accounting. Ten invalid frames in the
closed scenes remain omitted. The frozen890 has800 perspective and90
orthographic frames, with zero exact pixel/group/layout overlap against both
validation pools and the old40,000 training images. Freeze report SHA
`08107c9ee86067ee984d0dfd7617b94032e29aa77e21a5d804e8d80477dffa80`;
manifest SHA
`b032da1aa742572c82e248e1f373f8466211053bf9d4bd594a0504f3236ef104`.

All64 fixed-angle images from16 evenly spaced scenes were actually viewed on
16 contact pages. Closed walls and columns/beams are visible, with no observed
window/exterior, simulator body or magenta failure in the sample. Highlights,
blocky shadows and repeated motifs remain. No new exclusions resulted. This
is sampled visual review, not inspection of every890 image. Visual report SHA
`63c4de6f9654da6d58b9fb819b03203d27c0823a20b5ccbfeddae23598c0884b`;
prefix closure SHA
`70d01f5e3e3d07bd9b7209e1708e6f656a67e2a44bad8822d9fd4fb67a5937eb`.

Renderer`6d5b736` / `code-structural-multiview-v3` keeps every frame predicate,
scene, camera and drawing operation unchanged. New explicit production behavior
holds an entire scene after its third invalid RGB view, retaining raw images,
camera records, rejection proofs and unattempted-slot identities. No row from
a held scene is committed to the candidate manifest. The next new scene can
then proceed. There are at most five such holds per100-scene job; exceeding
that bound or encountering other errors stops the batch. Four-view preflight
and default behavior still fail immediately. There is no image repair,
replacement camera or retry of a held scene. Seventeen renderer tests passed
in7.04s, and fourteen existing freezer tests passed in8.86s. These are CPU tests;
no paired rerender or empirical pixel-equality experiment was performed.

Controller03 uses five new100-scene ranges54–153,154–253,254–353,354–453,
454–553 with6,000-second bounds. Its plan SHA is
`213afbf753dc262076248ffca1e155025192acf0b40a246d5be9a1a9ef92e5a6`.
The first actual scene54 closed all20 unchanged planned views,18 perspective
and2 orthographic, with zero rejected frames and feature-only metadata.
The controller accounts separately for admitted candidate frames, invalid
frames, held otherwise-valid frames and unattempted held slots. Completion
of a range does not mean all its scenes were admitted. Continue to freeze only
explicit closed-scene receipts and count actual eligible images toward50,000.
The source/camera invariance flag in the plan describes unchanged code semantics,
not a claim of a measured pixel-parity experiment. No training has started.

## Sources and renderer

The official ProcTHOR-10K training source is pinned to repository revision
`439193522244720b86d8c81cde2e51e3a4d150cf`:10,000 houses, archive SHA
`ee3c4aa14b4d8f0895fecfb5fdaca59395427ca1018b2f9aeeedbc61e5824587`.
For the earlier house-derived batches02–06, only the top-level objects list was
cleared. Structural walls, floors, doors, windows, materials and lighting remained;
columns/railings coverage was not established. The new preview above uses a
separate custom structural-scene policy and does not relabel those old batches.
The eventual200,000-image target is a plan, not an existing dataset.

AI2THOR5.0.0 / Unity build`f0825767cd50d69f666c7f282e54abfe58f1e917`
runs in a separate environment with private Xvfb/OpenGL llvmpipe libraries.
It uses two CPU cores and one math/raster thread, leaving the GPU free.
The Vulkan CloudRendering build rejected the CPU device; Linux64 OpenGL with
the normal display loop works. Unity batch mode produced invalid framebuffers
and is not used. Private Xvfb needed its XKB compiler prefix relocated; the
original and relocated binary hashes and a successful OpenGL4.5 CPU preflight
are retained. No training environment or system package installation was changed.

## Actual samples and correction

`procthor-empty-render-pilot-04` rendered16 RGB views of4 houses in43.13s,
including20.17s startup. All16 were viewed. These are actual640x480 perspective
renders; some initial views were too close to a single wall.

The first distributed-camera batch exposed simulator-agent shadows. Inspection
of the exact pinned Unity source found `makeAgentsVisible=true` by default.
The old pilot and batch01 remain diagnostic artifacts and are held from training.
Batch01 was stopped explicitly; no images were deleted or used for training.
The corrected renderer sets `makeAgentsVisible=False` and verifies that setting
after every house reset. Duplicate camera cycles were also removed before rendering.

## First bounded batch

Remote root:
`/home/buildhub/shared/radio-moge/chongxi/new-architecture-rgb-20261104`.
Local reports mirror under `/shared/radio-moge/chongxi/new-architecture-rgb-20261104`.

`procthor-batch02-controller` runs immutable`code-procthor-empty-batch-v2` /
`774c84d`; output`procthor-empty-rgb-batch-02`. It targets the first500 official
training houses and up to20 distinct RGB views each, with a7200s generation
budget and7500s outer bound. Do not start another copy. Four camera-placement
tests passed locally/remotely, including concave and narrow rooms and60 unique
single-room camera slots. The first16 corrected sample views across8 houses
were actually reviewed; no visible agent silhouette was observed. At07:42 UTC,
640 images across32 houses were closed, with0 duplicate images. Sixteen failed
camera actions were recorded and omitted; all32 houses still yielded20 images.

Cameras have a0.5m floor-boundary margin and distributed positions across rooms;
Unity verifies each teleport. Perspective vertical FOV cycles55/70/85 degrees,
with pitch-10/0/10. Failed teleports, near-uniform images and exact duplicate
pixels are recorded and omitted. Every retained row binds house/source hashes,
actual camera pose, projection, source split and hidden-agent rendering.
All rows remain training candidates until the batch is reviewed.

Read live`status.json` and per-house reports. `STOP_AFTER_HOUSE` requests a
graceful stop. A finished report gives the actual image count and next house;
do not equate requested capacity with completed images.

Chongxi subsequently explicitly requested adding the generated RGB to training.
The first frozen batch and its bounded mixed training are described below.
Improved teacher alignment and flatter real-world predictions remain hypotheses.

The1000 official validation-house descriptions have also been downloaded and
verified against their pinned Git-LFS SHA
`d808540514e26b6726cd2790490e669b572eeb94febb5188a2f403591dd21721`.
`procthor-empty-rgb-val-01` has now rendered320 RGB images from the first16
official validation houses. The split-aware renderer f49072d passed seven local
and seven remote tests. These houses are never training inputs.

## Frozen first batch and actual mixed training

`procthor-feature-train-2000-01` contains2,000 RGB images from100 closed
corrected training houses, manifest SHA
`1a73ad276ed13d6ca7e64b96c91c735a44da993f037b09e3dbd10d10bc564a16`.
`procthor-feature-val-64-01` selects four fixed views of each of16 official
validation houses, manifest SHA
`c7d98b1484a5f11b336a1bf90fcff274efa6f331dd77c26bde105659e532b6ba`.
Every selected RGB was decoded and checked against its encoded and pixel hashes,
closed-house receipt, source split, perspective/hidden-agent metadata and original
house description with only the object list removed. No exact pixel duplicates
were found. Original and empty-house hashes are disjoint across these two sets.
The growing rendering manifest is not a training input.

All64 selected validation images and16 additional training samples were actually
viewed in five contact pages. No agent body/shadow or invalid black frame was
observed in those samples. Low-texture wall-dominated views, dark rooms and
synthetic material/lighting appearance remain explicit limitations. This is
sampled visual review, not individual certification of all training images.
Visual review SHA `5be90e4d9441e4a4d95a1c961ea394a23a2a10f6aa81fa6c29dc6c86be985180`.

`procthor-feature-mix-training-01` launched using immutable116bf4c /
`code-procthor-feature-mix-v1`, wrapper287930. It starts from the completed
pure-feature checkpoint233538, SHA
`d4ae22dc3b6f8b6a88dbcf2129b5a0f37337a10f87ce26a6096e1d2d6a717ec5`.
The bound is1,000 new updates to234538 or3,600 seconds, with save50/eval250,
no automatic retry or next segment. The plan SHA is
`be07d35f138dfdf70ab9fc73d1cdf9296fdb82f111056fe4bb4055df3848b24a`;
runtime config SHA is
`c7b0f799cbd45a7341883506bcddbe5fcdb3366bd4c391cde8a173e5091d32d9`.

Exactly four of32 images per optimizer update come from the frozen synthetic
pool:12.5% synthetic and87.5% real. The real deterministic permutation continues
at the parent's logical cursor10000. Every unreplaced image index, crop seed and
resolution key remains exactly the original one; synthetic images retain the
replaced slot's crop seed and resolution and use a separate frame permutation.
The new mixed stream has an explicitly reset local cursor0. This avoids changing
real sources to a source/scene mixture. All four feature weights remain1, three
resolutions and effective batch32 remain, Adam/RMS/RNG are preserved, and no
geometry supervision, sidecar or geometry loss is enabled.

The integration checks cover sampler resume, actual pixels, prepared grouping,
full32 normalization, successful-update-only auditing, and a tiny real optimizer
loop that rejects any decoder call during training. All42 relevant CPU tests
passed locally/remotely. An initial remote test invocation omitted the fixture
configs directory; copying those unchanged files resolved all13 affected tests.
This was packaging of the test snapshot, not a training or data failure.

Actual CUDA model/pixel/backward checks and the new64-image heldout baseline
completed before optimizer updates. At08:19 UTC,64 real updates were confirmed
finite. The first20-update proof checks640 actual input keys/crops:80 synthetic
and560 real, original real keys preserved, F1 arithmetic and nonzero adapter
update. Proof SHA
`e3fea983ae4f940deeb3c2a9386c09f05e0ca4dbd464b86e444cc141aeee33d8`.
Use actual `training/train.jsonl`; the root status retains optimizer_updates0
until completion. Do not relaunch based on that stale counter.

Saved-only result review is prepared in24e046e /
`code-procthor-feature-review-v1`, with six local/remote comparison tests.
At each completed validation it checks the same four real protocols and the
new official-house validation, exact input-mixture prefixes and teacher/GT
support. Compare fixed endpoints against parent233538 and archived232288;
report synthetic gains separately from real gains. A later checkpoint alone
does not demonstrate flatter real-world predictions or justify deployment.

The first250-update saved comparison is complete, SHA
`e79f857b028d4d819e748c27aead05d3d92e27ceccc9a770b2f784480580a846`.
Against parent233538, real spatial cosine rises .9646449821→.9646814722,
summary .9357640150→.9358417018, and independent iPhone LiDAR AbsRel falls
.0931902844→.0919515565 (-1.329%). Against archived232288, spatial still falls
slightly; the predefined joint archive target is not yet met. On the64 unseen
official-house RGB, spatial .9614996593→.9700959381 and summary
.8897025390→.9254881674; teacher-depth discrepancy falls .1195481062→.0850851858.
These are early paired observations, not a replicated causal result or proof
of real-world planarity. At08:28 UTC500 new updates were reached and the second
validation was in progress. The completed segment is reviewed below.

W&B run `f6b5fe37` is online. First20 training values are numerically exact,
online startup report SHA
`20b3f70d89c923e11ac4b8ae291cd087e57bf4f775cb122ddecbe7b8cc605e69`.
API metadata has exactly one-ULP serialization differences for three baseline
display values (NYU delta1, spatial cosine, teacher depth); runtime config is
unchanged. The first online verifier stopped before publication because its
old two-baseline allowlist omitted NYU delta1; the second explicitly checks all
three observed paths and exact one-ULP differences. This is not an optimizer
or configuration failure. No startup/model proof needs repeating.

## Fixed1,000-update result

The original wrapper and trainer exited normally at234538 after1,000 updates,
with final evaluation complete and step_limit termination. The actual input
audit contains32,000 presentations:4,000 synthetic and28,000 real, with all
planned keys matched. Mixed cursor8000 and all100 Adam counters234538 were
read back from the final checkpoint. All adapter/Adam tensors and1,000 logged
updates are finite; fixed feature statistics and config/model/data identities
are unchanged. Logged F1 totals agree within float32 summation rounding
(maximum3.7253e-8). The fixed checkpoint is preserved by hard link:
`procthor-feature-mix-training-01/finish-verification-01/checkpoint-step234538.pt`,
SHA`80c9d0ec3c0839ecaefab5e3c8edde06e1174fff17bee814a0430ab0a073ea5c`.
Readback report SHA
`8425ca0802ea217cf0bc9837c07cab667bf56acd3190805df6ba0be5604597c0`.

Saved-only paired reviews for500,750 and1,000 updates are complete. All four
real views and the64-image unseen-house view retain exact images, protocols,
teacher identities and support. No inference or baseline evaluation was rerun.
Final review SHA
`db48ee2c24acd330007170ca485ad0c94dd31ee55bfcfa287d81fc59f005866e`.

| Original validation | Parent233538 | +250 | +500 | +750 | Fixed +1000 |
|---|---:|---:|---:|---:|---:|
| Spatial cosine | .964644982 | .964681472 | .964488311 | .964516835 | .964583766 |
| Summary cosine | .935764015 | .935841702 | .935748113 | .935802933 | .935825432 |
| Independent iPhone LiDAR AbsRel | .093190284 | .091951557 | .093302035 | .097170404 | .094354521 |

The original macro cosines combine six existing sources, including synthetic
datasets. The separate iPhone LiDAR metric is from real captures. Original
macro performance and performance on real scenes should not be conflated.

At the fixed endpoint, spatial cosine falls slightly, summary rises slightly,
and LiDAR AbsRel worsens1.2493% versus the parent. Against archived232288,
spatial remains lower and LiDAR is2.4274% worse; summary is higher. Both joint
comparisons therefore fail. The earlier250-update improvement is reported as
an intermediate observation, not substituted for the fixed endpoint. Its
saved spatial/summary/teacher-depth candidates remain available.

On the64 held-out official-house RGB images, spatial cosine rises
.9614996593→.9718591366 and summary .8897025390→.9364838200; teacher-depth
discrepancy falls .1195481062→.0816755904 (-31.6797%). This is agreement with
MoGe-v2 on unseen synthetic houses, not ground-truth accuracy, a causal
isolation of the data addition, or evidence that real walls are flatter.

The fixed2,000-image training manifest remains unchanged. At08:58 UTC the
renderer had closed377 houses/7,540 images,
zero uniform/duplicate frames, and154 failed camera actions omitted. Newly
generated views are not retrospectively counted as training inputs. The final
mixed model is retained for comparison and has not replaced the deployment model.

W&B`f6b5fe37` is finished. All553 emitted final validation scalars, including78
in the new ProcTHOR namespace, match committed results numerically. Config and
provenance retain only the three previously recorded one-ULP display differences
and equal integer/float representations. Online final report SHA
`c35ed3ffd502e6d09ba5022b895800367da9f2ef5c72e26cd929d8eb76233253`;
combined closure SHA
`6364ec0dc2b038f5c6c403c79c68402da638ef4b02413c48122cc6f5186a7a26`.
The first online read-only launch used a bare`argv[0]`and selected the wrong
Python environment before importing W&B; explicitly passing the existing model
Python path fixed the launcher. No training rerun, environment installation or
credential change occurred. Verifiers are immutable`9681dd6`in
`code-procthor-finish-v1`; lint/compilation and the actual closed-run checks passed.

## Completed10,000-image pool and the next generation batch

`procthor-empty-rgb-batch-02` closed all500 houses, IDs0–499, and10,000 RGB
images in6,714.08 seconds. All houses have20 retained views. There were zero
house-creation failures, exact duplicate frames or frames rejected by the
renderer standard-deviation threshold;210 failed camera actions were omitted.
The old renderer controller287194 exited normally. This is generation closure,
not a claim that all images contain useful room structure.

`procthor-feature-train-10000-01` was frozen with the existing116bf4c freezer.
Every image's encoded bytes, decoded pixels, source line, house description,
closed-house receipt, perspective projection and hidden-agent metadata were
checked. Manifest SHA
`8be8d9a6d98ee64a87352dd0fb246c0b68008da986e785fb5a6930df65e3adb8`;
report SHA`50b8a56ea924523033de2214bd1b283e6bc542ea74cb8efcb526f8bfb713e368`.
The prior2,000 training rows are exact members of this larger pool. The8,000
additional images have not been trained on. No original-house, empty-house or
decoded-pixel overlap with the existing official validation selection was found.
FOV55/70/85 image counts are3,340/3,340/3,320 respectively.

A fixed sample of48 views from24 evenly spaced newly added house IDs plus the
four lowest-contrast new frames was rendered in five contact pages. All52 views
from27 houses were actually inspected, covering all three FOVs. No agent
silhouette or failed framebuffer was observed in that sample. The four low-
contrast originals were also inspected at full resolution: train00301 slots13/14,
train00246 slot20 and train00258 slot18. They are wall-dominated, show little room
structure, and are retained with explicit low-context labels. No exclusion or
inferred benefit follows solely from their contrast. This sampled review does
not certify every image. Visual report SHA
`77dc850b117a315cabb8f06bedb819eeea887ac159fdb6e50e38674a69b3d72a`;
review index SHA`a789f78a6a3f501c5fe8070ea8e11f77636932eb7cdcf65b90c9c5e7a40debb1`.

`procthor-batch03-controller` started09:38:11 UTC, wrapper308305, using the same
immutable774c84d renderer and private CPU OpenGL environment. Output is
`procthor-empty-rgb-batch-03`; the only batch changes are output path and explicit
house range500–999. It targets another500 houses/max10,000 images with the same
7,200s generation/7,500s outer bounds, one renderer/two CPU cores/math threads1.
The first house closed20 frames successfully. Do not start another copy or
modify its live config. Official validation houses remain separate.

The freezer now also accepts`--start-house`for later disjoint batches, preserving
publisher house IDs and source-line hashes instead of renumbering them. Its
default start0 behavior is unchanged; six targeted CPU tests cover old train/val
selections, incomplete-tail avoidance, hidden-agent rejection, nonzero house IDs,
and invalid bounds. The first test invocation had two expected-error text
mismatches; correcting the asserted message made all six pass, with runtime
validation unchanged. This future helper has not rewritten the completed10,000
pool or altered any training input.

## Completed20,000-image pool and continued generation

Batch03 closed all500 official training houses500–999 and10,000 RGB images in
6,710.70 seconds, with222 failed camera actions omitted and zero house failures,
uniform-frame rejections or within-batch pixel duplicates. Wrapper308305 and its
controller exited normally. The existing range freezer verified every new
encoded image, decoded pixel hash, closed-house receipt, source line and rendering
contract. The frozen second manifest SHA is
`f71e9bf8280868196dc6f2f476da7140676f537d1b4f3a2e2af5fabdd78ddbf9`;
its report SHA is`4051ce6f3e85c57e92fd1447624ad4e9cceb8e0afdbc461ea062c820c9d3bd3c`.

The new combination helper`4d01426`checks hash-bound frozen reports, complete
source house ranges and all image/house identities before publication. Fifteen
targeted CPU tests passed locally, including corrupt manifests, incomplete
houses, incorrect projection, visible agents, source changes and overlap
rejections. The actual20,000-row combination preserves both inputs byte for
byte. Exact image paths, groups, decoded pixels, original-house hashes and
empty-house hashes are disjoint between batches and from the fixed64-image
official-validation selection. This does not certify near-duplicate absence
or isolation from every unpublished validation view. Manifest SHA
`7475936acbd13149a6b9cf4f87cefa009b96e2ac84036421b46e4db8c2ce2003`;
combination report SHA
`37383b0182f040b49363ce6b00beec600922cb05507c0ad04784735614d04bad`.

All52 new sample images across26 houses and all five contact pages were viewed.
They include48 fixed evenly spaced house/retained-view selections and the four
lowest-contrast frames. All four low-contrast original640x480 PNGs were also
viewed. No visible agent or failed framebuffer was observed in this sample.
The four extrema (00618 slots15/13/14 and00763 slot19) and two sampled wall-only
views (00760 slots00/10) have explicit low-context flags and remain feature-only
RGB. Close walls, dark or glossy surfaces, strong textures and synthetic lighting
remain. No new exclusions were issued, and this is not an all-image visual
certification or physical-depth validation. Visual review SHA
`70d63e9b68e6a5bf99c06feac438e13d165664924f6b7e97b13051112d186605`.
The review launch first failed before reading data because its small immutable
directory omitted an imported helper. Supplying the existing116bf4c dependency
directory via PYTHONPATH fixed the launcher without changing any code or data.

The new20,000-image pool has18,000 images not yet used by the completed mixed
run. The current training inputs and all earlier evaluation proofs remain
unchanged. The completed model's original-validation joint target still fails;
no optimizer restart or deployment replacement follows from this data expansion.

Batch04 wrapper309984 started11:43:44 UTC with the same774c84d renderer and
private CPU OpenGL setup. It targets houses1000–1499, up to10,000 images, under
the same7,200s generation/7,500s outer bounds. Only output path and source range
differ from batch03. Keep one renderer/two CPU affinity cores/math threads1;
do not duplicate the job. After completion, freeze the actual closed range
with`--start-house1000`, compare all three frozen batches and the validation
selection, and inspect only new samples. No ScanNet geometry work resumes.

## Completed30,000-image pool and fourth generation batch

Batch04 completed all500 official training houses1000–1499 and10,000 RGB images
in6,493.08 seconds. Its193 failed camera actions were omitted; there were no
house failures, uniform-frame rejections or within-batch pixel duplicates.
Wrapper309984 exited normally. The unchanged range freezer verified every new
encoded image, decoded pixel hash, closed-house receipt, source line and rendering
contract. Frozen manifest SHA
`fe5f7e32309edd98d20fccef0d9b336d5a99f9a315f0174395bcb0f70caf4d69`;
report SHA`67e3b3cb7f717b8ae2b3876ad92d6646dc8e5b6df568ee0e27210b193d1875c6`.

The existing combiner preserved all three frozen batches' rows byte for byte.
Exact image paths, original-house hashes, empty-house hashes and decoded pixels
are disjoint across the batches and the fixed64 official-validation images.
No near-duplicate or all-validation-image claim is made. The30,000-image,
1,500-house combined manifest SHA is
`30bfb0f81f9024fd9609c3d7156f72411da96c1769778a81e9471b72ad5e08b7`;
report SHA`781a4faead37c18d52e4b056ec36838acaf720c1332d14f8ac574885fe65f7ab`.
Each of FOV55/70/85 now has10,000 images.

All52 new sample views across28 houses and all five contact pages were actually
inspected. Six original640x480 PNGs were also viewed: the four lowest-contrast
frames, a close red-wall view and a border-structure check. No visible agent or
failed framebuffer was observed in this sample. The partial brown surface in
01043-slot10 touches the left image border; it is not a confirmed floating
fragment. Seven exact low-context flags retain the four lowest-contrast views
(01395-slot17,01219-slot13,01221-slot16,01333-slot19) and the sampled wall-dominated
views01086-slot10,01390-slot10,01455-slot11 as feature-only RGB. No exclusions or
all-image visual certification follow. Visual review SHA
`d4f37b49187eaf938ef15c41b30ee4a8bcae2db82a5915462cc037c66bc8008b`.
The additional28,000 images have not been used in training. Existing model
results, training inputs and deployment selection remain unchanged.

Batch05 wrapper311629 started13:49:39 UTC using the unchanged774c84d CPU renderer
and private OpenGL setup. It targets houses1500–1999, up to10,000 RGB images,
with the same7,200s generation/7,500s outer bounds, two CPU affinity cores and
one math thread. Only output path and source range changed from batch04.
Do not duplicate the running job. After all500 houses close, freeze the new
range, combine the four original batch reports, and inspect new samples.
No tests, completed model proofs or earlier image reviews were repeated.

## Completed40,000-image pool and fifth generation batch

Batch05 closed all500 official training houses1500–1999 and10,000 RGB images in
6,379.79 seconds. Its188 failed camera actions were omitted. There were no house
failures, uniform-frame rejections or within-batch pixel duplicates. Wrapper311629
and its children exited normally. The unchanged freezer checked every new encoded
image, decoded pixel hash, source/house identity and perspective/hidden-agent
contract. Freeze report SHA
`4c736c1c6f566a5e74633466fb815132e869575df14101dca8c4fcd7c4aed180`.

The existing combiner preserved all four input manifests' rows byte for byte.
Exact image paths, decoded pixels, original-house hashes and empty-house hashes
have no overlap across batches or the fixed64 official-validation selection.
The40,000-image/2,000-house manifest SHA is
`24d3494b5149734ce3208754a6a9e9e1342feda66a51828c800df085a49d966d`;
combination report SHA
`982e51d3a58be54b43802eab6982cd4ebe46977ad72d6ccb439406b07bf26509`.
This does not certify near-duplicate absence or every unselected validation image.

All52 new sample views across26 houses and all five contact pages were actually
viewed. Six original640x480 PNGs were also inspected. No visible simulator agent
or failed framebuffer was observed in this sample. Six exact low-context flags
retain01521 slots00/10,01800-slot15 and01911 slots13/14/15 as feature-only RGB.
01521-slot00 shows a dark vertical wall corner; the other flagged frames contain
almost only weak wall gradients. Close doors/corners, strong synthetic textures,
glossy materials and lighting remain. No exclusions or all-image certification
were issued. Visual review SHA
`22c1eeed7a9afaea90627fefdc4eda3f92b3d880df3847ed99d4bb5ccc759fc3`.

The additional38,000 images have not been trained on. The completed mixed run,
its failed original-validation joint target, all prior proofs and deployment
selection remain unchanged. No geometry work or optimizer was restarted.

Batch06 wrapper313183 started15:47:09 UTC with the same immutable774c84d renderer,
private CPU OpenGL environment, two affinity cores and one math thread. It targets
houses2000–2499 and up to10,000 RGB images within7,200s generation/7,500s outer
bounds. Only source range and output changed. Do not duplicate or modify the
existing job. Once it closes, freeze that range, combine the five original batch
reports, and inspect only new samples. No previous tests or reviews were repeated.
