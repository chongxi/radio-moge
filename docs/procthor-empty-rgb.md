# Empty-room RGB for MoGe-v2 feature alignment

Chongxi requested large numbers of clean empty-room, first-person RGB views.
Further ScanNet++ source/geometry investigations were explicitly stopped.
This generation task uses ProcTHOR only; ZInD and UnrealZoo are not inputs.
No synthetic depth labels or geometric loss are introduced.

Current state, Nov5 11:44 UTC: the first mixed-feature run has completed its
fixed1,000 updates at234538. It improves agreement on unseen synthetic houses,
but the final original-validation joint target is not met. No deployment replacement or
further optimizer segment has been started. Two500-house generation batches
are complete and their20,000-image pool is combined with exact cross-batch and
validation isolation. A third CPU batch has started on disjoint official
training houses1000–1499. Only the earliest2,000 images participated in training.

## Sources and renderer

The official ProcTHOR-10K training source is pinned to repository revision
`439193522244720b86d8c81cde2e51e3a4d150cf`:10,000 houses, archive SHA
`ee3c4aa14b4d8f0895fecfb5fdaca59395427ca1018b2f9aeeedbc61e5824587`.
Only the top-level objects list is cleared. Structural walls, floors, doors,
windows, materials and lighting remain. Columns/railings coverage is not established.
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
