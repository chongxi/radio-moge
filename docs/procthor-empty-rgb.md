# Empty-room RGB for MoGe-v2 feature alignment

Chongxi requested large numbers of clean empty-room, first-person RGB views.
Further ScanNet++ source/geometry investigations were explicitly stopped.
Generation uses official ProcTHOR houses and custom structural rooms rendered
in the same pinned simulator; ZInD and UnrealZoo are not inputs.
No synthetic depth labels or geometric loss are introduced.

Nov6 23:59 UTC: the user authorized the proposed Muon comparison. The new
bounded design reuses the closed LR1e-5 reference above and adds two independent
forks of262121: reset AdamW state on the16 core attention/FFN matrices, and use
fresh Muon state on exactly those16 matrices. The84 auxiliary tensors retain
their original native PyTorch AdamW moments/counters; all weights, RMS and RNG
start from the same source. Core names are explicit, excluding 2D registers,
input/output projections and gates. The50787 pool,128 actual images per forward,
original cursor314864, crop seeds, bucket assignments, F1/G0 and1000-update
budget remain paired with the saved reference. No data is regenerated.

Muon uses the separately pinned microsoft/dion commit
`7692479288f928a3d61d1cc58137b4dbe4f71000`, with Nesterov momentum.95,
base LR1e-5 and RMS scaling0.2*sqrt(max(matrix dimensions)). The stored matrices
are unsplit. Polar Express, Gram and custom Triton paths are disabled; the exact
pinned five-stage Newton-Schulz coefficient schedule is retained. This scaling
does not assert equality to the current late-training Adam update magnitude.
No PyTorch upgrade or replacement of auxiliary AdamW is needed. New named
optimizer checkpoints separately record auxiliary Adam ages, fresh-core Adam
ages or Muon group ages, and the new update count. Old all100-Adam-age verifiers
are intentionally not used for these arms.

The24 new CPU tests cover named grouping, CPU source immutability, exact save/load
continuation, native AdamW auxiliary/update parity, missing gradients, scoped
configuration rejection, production-loop initialization and counter/result
guards. The61 existing LR/reference tests also pass. Two early fixture mistakes
(an output-layer name and missing tiny-model config) were corrected without
changing runtime behavior. Actual GPU/model proofs are still pending at this
preparation checkpoint. Each arm must verify the first and last actual128
forward/backward, a temporary optimizer update with finite state and measured
core/auxiliary norms, then bitwise restore weights/optimizer/RNG before any
retained training update. Bounds remain5400s/arm and5700s outer; no automatic
retry, extra training or deployment. All six saved validation views and complete
actual input bytes must agree with the matched reference protocol.

Nov7 00:02 UTC: all85 tests pass remotely (19.19s); the24 new tests pass locally
(4.16s), and all61 existing LR/reference tests passed in the preceding local run.
Immutable `code-room-optimizer-pair-v1` / `ea13727` is launched in
`room-feature-optimizer-pair-01`, controller611086, first trainer611088.
At00:02:46 it had read79,872 original metadata rows, with zero retained updates
verified. The actual model, scoped initial state and first20-input/online proofs
remain pending. The next Muon arm is already queued in the same controller;
it starts only after the reset-AdamW arm completes its fixed budget. The pinned
Dion tree is loaded by PYTHONPATH with all17 package Python-file hashes bound;
the training environment has not been upgraded. All prior checkpoints and the
viewer remain intact. No inference or old completed proof was repeated.

Nov7 00:17 UTC: attempt01 stopped before writing `initial.pt` or executing any
backward/optimizer step. Controller611086/trainer611088 exited; Muon never
started. The initial checker copied an OrderedDict model state into a plain
dict, while `equal_finite` intentionally requires identical container types.
A two-element equal-tensor reproduction confirms this false rejection. The
unsaved failed invocation is not claimed to have passed actual tensor equality.
The original artifacts and W&B167be3c6 remain as a failed attempt, closure SHA
`d54a48a5062ed4fd013ec9192ebae86669c2968d76de36d42df368a1b6c603ca`.
The fix preserves the mapping type and PyTorch `_metadata` during CPU copying;
the exact finite/tensor/type comparator is unchanged. The actual runner's
initial checker now runs inside both tiny production-loop cases, and deliberate
weight corruption still fails. All26 targeted tests pass locally (5.07s).
No loss, optimizer rule, source state, input plan or budget changed. A manually
reviewed new immutable execution/output is required; attempt01 is never resumed
or overwritten. Actual GPU initialization and update gates remain pending.

The follow-up restoration audit found a second issue before a new launch:
native optimizer loading can share saved tensor storage, including host Adam
step counters when parameters are on CUDA. Restoring one snapshot for two
temporary gate updates could therefore mutate that snapshot. A new regression
failed for both optimizer modes before the fix. Parent loading and trial-state
restoration now copy all saved buffers before native loading; each temporary
restore also explicitly requires zero new steps,84 unchanged auxiliary ages
and the declared fresh16-core state. The exact equality checks remain in place.
All28 targeted tests pass locally (5.23s), including repeated restore/update
snapshot immutability in both modes. This path was never reached by attempt01;
it is not evidence about any earlier training regression. The intended update
rules, paired inputs and fixed budgets are unchanged. Fresh actual GPU gates
are still required before retained updates in the new execution.

Nov7 00:25 UTC: all28 targeted tests also pass remotely (8.23s). The other653
Python files, including the trainer, match attempt01 exactly. A prelaunch
comparison verifies both candidate configs differ from attempt01 only in output
directory and tracking group; source weights/state, input plan, LR, losses,
budgets and GPU lock are unchanged. Fresh immutable
`code-room-optimizer-pair-v2` / `ffce69c` is running in
`room-feature-optimizer-pair-02`, controller611615, first trainer611617,
launched00:24:52 UTC. At00:25:18 it had read81,920 of1,597,911 original metadata
rows; zero retained updates and no actual initial/model proof are reported yet.
Muon remains queued after the first arm's complete budget. This is a manually
reviewed corrected attempt; the zero-update failure and its artifacts remain
intact. The28 CPU tests do not substitute for actual model/state proofs.

Nov7 00:39 UTC: the corrected reset-AdamW arm is now training, with233 finite
updates/global262354 at00:38:48 and W&B`0c169f21`. Actual initial adapter/RMS/RNG,
configuration and model revisions match262121 bitwise; all84 auxiliary AdamW
states retain age262121, and the16 core states are fresh. The two actual128
model cases (256 RGB) passed, including nonzero core/auxiliary temporary updates,
finite state and complete restoration before retained training. First20 updates
use2,560 actual RGB presentations (260 ProcTHOR,60 structural,2,240 original),
with keys/crops/buckets and complete recorded input bytes identical to the
continued-AdamW reference. Measured update norm0.038504 is nonzero. All442
online training scalar values match numerically; two saved-baseline metadata
fields differ by one display ULP, and seven metadata/twenty training fields
only differ in equal integer/float representation. Configuration is unchanged.
Startup closure SHA`574f5cf2f46ef23085957f6e29f82f67b12f572fd024c0f925e945faaca3a2d7`
is complete. Do not repeat its initial/model/input/online checks. Muon remains
queued; these are correctness checks, with no optimizer quality conclusion yet.

Nov7 01:03 UTC: the reset-AdamW arm reached750 finite updates/global262871;
its third validation is in progress. Saved250/500 reviews are complete for all
six views, identical image/protocol/teacher/support identities and32,000/64,000
actual RGB presentations. Their4,000/8,000 synthetic identities are unique,
and the complete input prefixes match the continued-AdamW reference exactly.
All1,262 online validation scalar values match numerically, with37 equal
integer/float representations at each endpoint. Closure SHA
`6120d3bcc68e25d1a67eaeb6fef36970624f76e18df615d4fac4462ad5e09c57`
contains every same-step metric contrast across all six views; these proofs
must not be repeated.

Both endpoints pass the immediate-parent joint criterion and fail the original
archive criterion. At500, primary spatial/summary cosines are
0.964727895/0.936308916; LiDAR reference AbsRel0.096069170 is0.2493% lower than
the common262121 parent but4.2887% higher than archive232288. Compared with
continued AdamW at the same500 updates, cosine differences are only
+0.000000933/+0.000003151, while LiDAR error is0.0427% higher. This is not an
established reset advantage. Synthetic outcomes are mixed: structural317
spatial rises, summary is slightly below its parent, and teacher-depth
discrepancy falls3.3415%; ProcTHOR64 spatial falls, summary rises, and that
discrepancy increases5.4489%. Muon has not started; fixed final optimizer
comparison, significance and real-world planarity remain unresolved.

Nov7 01:29 UTC: reset-AdamW completed its fixed1,000 updates/global263121;
trainer611617 exited normally and the existing controller611615 started Muon
trainer633903 at01:15:27. The remaining750/final saved reviews and final CPU
readback are complete. All four six-view validations retain identical image,
protocol, teacher and support identities. All2,524 online validation values
match numerically, and W&B`0c169f21` is finished. The128,000 actual RGB input
bytes match continued AdamW exactly, including16,000 unique synthetic images
(12,688 ProcTHOR and3,312 structural) plus112,000 original presentations.
This is the fixed comparison budget, not a full50,787-image epoch. The final
checkpoint is finite, preserves RMS/config/data identities and cursor32000,
and has84 auxiliary Adam states at263121 plus16 core Adam states at age1000.
Checkpoint SHA`a306f33d0459c17b21da2364d26fbe3680260b3ea1efbef56fab7d7e480c950c`;
final closure SHA`64d2038d10abdde248967a0bd9580a4733e23298a226cef222fb1fb3219d5b62`.
The old controller receipt saying review pending is historical; the separate
closure is complete. These finished proofs must not be repeated.

Reset-AdamW's fixed final primary spatial/summary cosines are
0.964664730/0.936330270, and LiDAR reference AbsRel is0.095762393. Both cosines
improve over the common262121 parent, with LiDAR error0.5679% lower; all four
endpoints pass that parent joint criterion. All four fail archive232288:
final spatial is lower and LiDAR error is3.9557% higher. Against continued
AdamW at the same final1,000 updates, cosine differences are only
+0.000000812/+0.000002453, with LiDAR error0.02624% higher. There is no
established reset advantage. Structural317 spatial rises but summary is
slightly lower than the parent, and teacher-depth discrepancy increases0.9737%.
ProcTHOR64 spatial falls, summary rises, and its discrepancy falls2.5380%.
All six views and every fixed-final contrast are retained in the closure;
no best endpoint was selected and no new inference was performed.

Muon is now actually training:199 finite retained updates/global262320 were
observed at01:29:15, with one actual128 tensor/forward per update, peak GPU
95.46GiB and zero OOMs. Its initial adapter/RMS/RNG match the same262121 source;
all84 auxiliary Adam moments/counters remain at262121 and16 fresh Muon states
start at group step0. Both actual128 model cases pass, including finite
temporary core/auxiliary updates and exact restoration before retained updates.
The first/tail input keys, pixels, pre-update losses and model digests equal the
reset control. The first temporary Muon optimizer call took52.15s including
compilation; the warmed tail call took0.0203s. This is not a per-step52s cost.

The first20 retained updates use exactly the same2,560 actual RGB input bytes
as both controls (260 ProcTHOR,60 structural,2,240 original), with a measured
nonzero update norm0.026568871. All442 online training values match numerically.
Two saved-baseline metadata fields and the start timestamp have display-ULP
differences; seven metadata/twenty training fields only differ in equal
integer/float representation. No configuration changed. W&B`569aa52e` is
running; startup closure SHA
`a5063978cadacfa9c874fcc7837ed8fcb298d1ffef2ffcd0689db5de3277e0c1` binds all
actual initial/model/input/online proofs. Do not repeat them. Muon's first
250-update validation and the fixed-final optimizer comparison are pending.
The existing bounds and pure-feature policy remain unchanged; no extra run,
generation, geometry work or deployment was started.

Nov7 01:48 UTC: the existing Muon arm reached629 finite updates/global262750,
cursor20128, with its250/500 validations committed. Both saved six-view
reviews and1,262 online scalar checks are complete; all online values match
numerically, with37 equal integer/float representations at each endpoint.
The32,000/64,000 actual RGB prefixes match both controls byte for byte, with
4,000/8,000 distinct synthetic identities and no repeats. All images,
protocols, teacher outputs and support identities remain fixed. Closure SHA
`df8fc05e969791b0e6f93c7689ac88baa62536f5838f0f46ea6fb4baefd03023`
binds every same-step metric contrast across all six views against both reset
and continued AdamW. Do not repeat these reviews or the completed startup proofs.

At500, Muon's primary spatial/summary cosines are0.964667375/0.936324510;
LiDAR reference AbsRel0.096013977 is0.30665% lower than the common262121 parent
but4.22883% higher than archive232288. The500 endpoint passes the parent joint
criterion;250 fails because its LiDAR error rises0.41553%. Both fail the archive
criterion. Against continued AdamW at500, spatial is lower by0.000059587,
summary higher by0.000018746, and LiDAR error lower by only0.01482%. Against
reset AdamW, the respective differences are-0.000060520/+0.000015595 and
-0.05745%. This is mixed interim evidence, not an overall Muon advantage or a
statistically established optimizer effect.

Structural317 cosines at500 are0.966176231/0.982952673, both above the parent,
with teacher-depth discrepancy1.83062% lower. ProcTHOR64 has spatial0.975866191
below the parent and summary0.962245921 above it; its teacher-depth discrepancy
is1.70171% higher. These synthetic feature/depth results do not establish
real-world planarity. The750 and fixed-final1,000 comparisons remain pending;
the unchanged existing training continues, with no retry, extra epoch or
deployment. Viewer3462 is preserved and OOM counters remain zero.

**Completed optimizer comparison, Nov7 02:09 UTC:** both new arms finished their
fixed1,000 actual128 updates at263121. Controller611615 and trainers611617/633903
have exited; W&B`0c169f21` and `569aa52e` are finished. The new arms' eight
six-view validations and5,048 online validation scalar values are verified;
the continued-AdamW reference's four validations/2,524 online values are reused.
All three arms have identical128,000 actual RGB input bytes, including16,000
distinct synthetic and112,000 original presentations. This remains a bounded
comparison, not a full50,787-image epoch. No new segment or deployment started.

Muon final CPU verification confirms84 auxiliary Adam states at263121 plus16
Muon momentum states/group step1000, cursor32000, finite adapter/optimizer/RNG,
and unchanged RMS/config/data/model identities. Its preserved fixed checkpoint
SHA is`84b36960bed9ac74efb0536e13d9a3f8b188fe0157a1dffdb28cb5a760e8299a`.
Muon arm closure SHA
`a5cc64fdfe164e4de91e20f809025629a5e267812ba1444808084d8ad37b4c0f`;
fixed-final three-arm comparison SHA
`88189ada1ffe6e89b6cf03068c64e0b305d177fbdd04da4792f1e577121385f7`;
complete pair closure SHA
`3270370bd37f278e4a976001dd5969840e1ae4c915d1f0b376b02149866dde20`.
The controller's preserved review-pending receipt is historical; these separate
closures are complete. No model inference or earlier checkpoint verification
was repeated. All original best checkpoints and viewer3462 remain intact.

| Fixed endpoint | Primary spatial cos | Primary summary cos | LiDAR reference AbsRel |
| --- | ---: | ---: | ---: |
| Common262121 start | 0.9646276428 | 0.9362591955 | 0.0963093091 |
| Continued AdamW, +1,000 | 0.9646639170 | 0.9363278162 | 0.0957372693 |
| Core-reset AdamW, +1,000 | 0.9646647295 | 0.9363302697 | 0.0957623925 |
| Core Muon, +1,000 | 0.9646817986 | 0.9363510735 | 0.0957353114 |

At the predefined final endpoint, Muon passes the numerical joint contrast
against both controls: versus continued AdamW, cosines rise by0.000017882 and
0.000023257, while LiDAR reference error falls only0.002045%. Against reset
AdamW the cosine differences are0.000017069/0.000020804, with LiDAR error
0.028279% lower. These small fixed-final differences are not replicated or
statistically established. Intermediate Muon results were mixed: parent-joint
outcomes at250/500/750/1,000 are fail/pass/fail/pass. All four fail the original
archive232288 criterion; final spatial is still below that archive and LiDAR
error is3.926319% higher. The same nominal LR1e-5 and this one pinned Muon policy
do not establish its optimal LR or equal update magnitude to AdamW.

Separate structural317 final cosines are0.966276258/0.983085058, compared with
continued AdamW0.965877677/0.982565414; teacher-depth discrepancy is4.5130% lower
than continued AdamW (3.5221% below the common parent). ProcTHOR64 final cosines
are0.976096902/0.962449190 versus0.975938929/0.961967275; teacher discrepancy is
1.3685% lower than continued AdamW. Both cosines also improve over that control
on the native-resolution primary view and DSLR holdout. However, the separate
iPhone-LiDAR holdout has slightly lower feature cosines and0.2169% higher
teacher-depth discrepancy; its independent LiDAR reference error is nearly
unchanged. Thus the positive primary result is not an all-metric/all-view win,
and synthetic alignment gains do not establish physical plane accuracy.
All six views and every metric, plus all twelve predefined endpoints including
the historical control, are retained. No best endpoint was selected.

**Completed LR comparison, Nov6:** both matched arms finished1,000 actual128
updates at263121. All128,000 input presentations per arm have identical bytes,
including16,000 distinct synthetic and112,000 original presentations. Both
started independently from262121 with preserved Adam/RMS/RNG; only the declared
LR fields differ, with the same constant-scheduler reset. This fixed comparison
budget is not a full50,787-image epoch. All eight saved six-view validations and
all5,048 online validation scalar values are verified. Both final checkpoints
are preserved, finite, with100 Adam states at263121/cursor32000 and unchanged
RMS/data/config identities. Both W&B runs finished and all training processes
exited; no extra training or deployment started.

| Fixed endpoint | Primary spatial cos | Primary summary cos | LiDAR reference AbsRel |
| --- | ---: | ---: | ---: |
| Common262121 start | 0.9646276428 | 0.9362591955 | 0.0963093091 |
| LR1e-5, +1,000 | 0.9646639170 | 0.9363278162 | 0.0957372693 |
| LR2e-5, +1,000 | 0.9645971339 | 0.9362241694 | 0.0957647715 |

The primary cosines are the fixed1,012-image, six-dataset equal macro average,
not all training images. At the predefined final endpoint,2e-5 is lower on
both primary cosines and has0.0287% higher LiDAR reference error than1e-5.
The differences are small; this one pair establishes neither statistical
significance nor a universal LR rule. It supports retaining1e-5 for a future
authorized run. The1e-5 arm passes the parent joint criterion;2e-5 fails it.
Neither passes the archive232288 joint criterion. No best endpoint was selected.

The separate synthetic final comparisons also favor1e-5: both cosines are
lower at2e-5 on structural317 and ProcTHOR64, with teacher-depth discrepancy
7.5532% and5.0823% higher, respectively. All six views and every metric remain
in `room-feature-lr-pair-01/results-review-01/report.json`, SHA
`656e2587dc033ba869dee29329718f49ac66220e14fbfbade10161be4d002846`.
Full closure is `results-closure-01/report.json`; the controller's original
`training_complete_review_pending` receipt is historical and is preserved.
No inference, earlier proof or first-arm checkpoint readback was repeated.

Nov6 19:12 UTC: the user authorized a matched LR comparison after the completed
actual128 epoch. The new frozen arms start independently from checkpoint262121,
preserve all100 Adam states, RMS and RNG, and compare constant1e-5 against2e-5.
Each arm has1,000 updates/128,000 RGB presentations (16,000 distinct synthetic
plus112,000 original), with saved six-view validation every250 updates. This is
a fixed comparison budget, not a full50,787-image epoch. Both use the same
continued original cursor314864, synthetic permutation, augmentation seeds and
one-resolution-per128 grouping; the full actual input bytes must match.
The initial-state guard allows only Adam group `lr`/`initial_lr` changes and the
same new constant-scheduler position; betas, epsilon, weight decay and moment
history remain. Both arms retain F1/G0. Wall bounds are5,400s per arm, with a
5,700s outer guard, sequential execution under the original GPU lock, no retry
or deployment. New current-checkpoint first/last actual128 model checks precede
updates; earlier three-resolution throughput evidence is explicitly historical.
The46 CPU contract, sampler, initial-state, tiny actual-loop and result-checking
tests pass locally and remotely. Immutable `code-room-lr-pair-v1` / `39e1851`
launched at19:14:33 UTC in `room-feature-lr-pair-01`, wrapper563782,
controller563783, first-arm trainer563785. At19:15:41 the first arm had read
262,144 original metadata rows; zero actual updates were verified yet. The
controller starts the second arm only after the first completes its fixed
budget, and compares both complete actual input hashes. New actual model,
initial-state and startup proofs are pending, not inferred from CPU tests.
At19:28 UTC the first arm was actively training, with243 finite updates/global
262364 and W&B `5a310516`. Its new actual first/last128-image model checks passed
on256 RGB presentations, with unchanged frozen/model state. Actual initial
adapter/RMS/all100 Adam states/RNG matched262121; the constant scheduler reset
was explicit. The first20 updates matched2,560 planned RGB keys/crops/groups:
260 ProcTHOR,60 structural and2,240 original images; measured parameter update
norm0.03169997 was nonzero. All442 online training scalar values were numerically
exact. Two saved-baseline metadata fields differed by exactly one display ULP;
seven metadata and20 training integer/float representations were numerically
equal. No configuration or training behavior was changed for this check.
These first-arm startup proofs are complete; the second arm and paired quality
comparison are pending. Fixed final results and all six views will be compared;
no best-checkpoint selection or higher-LR benefit is presumed.

Nov6 19:45 UTC: the1e-5 reference arm passed its saved250/500-update six-view
reviews, including32,000/64,000 actual RGB keys and4,000/8,000 unique synthetic
presentations with no repeats. All1,262 online validation values matched
numerically (37 equal integer/float representations per endpoint). At500,
primary macro spatial/summary cosines were0.9647269617/0.9363057642 versus
0.9646276428/0.9362591955 at262121; LiDAR reference AbsRel0.0960282073 was
0.2919% lower than that parent, but worse than the250-update value. Both observed
endpoints pass the parent joint criterion and fail the archive232288 criterion.
Separate synthetic outcomes remain mixed: structural317 spatial rose, summary
was slightly below its parent, and teacher-depth discrepancy fell3.2617%;
ProcTHOR64 spatial fell, summary rose, and teacher-depth discrepancy rose5.1780%.
These are reference-arm observations only. The2e-5 arm remains queued, so there
is no higher-LR comparison result or real-world planarity conclusion yet.

Nov6 20:05 UTC: the1e-5 reference arm finished all1,000 updates at263121 and
all four saved six-view validations. Actual128,000 RGB presentations comprise
12,688 ProcTHOR,3,312 structural and112,000 original images; the16,000 synthetic
identities are unique and match the frozen plan. This is the comparison budget,
not a full synthetic epoch. Final adapter/all100 Adam states are finite at
263121, cursor32000, with unchanged RMS/config/data identities. The final
checkpoint is preserved by a hardlink with SHA`beee224c45488aef53d476617c5d3d609a34e9cfc4b59815aaaea0ed2f8bd2de`.
The complete actual input record SHA is
`b4fd4a309ae3750f36fe97818bf490df2fc805f2a98de817e0382f69627adcab`;
the second arm must match it. W&B`5a310516` is finished and all2,524 validation
scalar values across the four endpoints are numerically exact. The two known
baseline metadata display differences remain one ULP; training was unchanged.

Final primary spatial/summary cosines are0.9646639170/0.9363278162, both above
the common262121 parent. LiDAR reference AbsRel0.0957372693 is0.5940% below
that parent and3.9284% above archive232288. All four reference endpoints pass
the parent joint criterion and fail the archive criterion; changes are not
monotonic. Separate structural317 spatial rises but summary falls slightly,
with teacher-depth discrepancy1.0378% higher. ProcTHOR64 spatial falls slightly,
summary rises, and teacher-depth discrepancy falls2.7155%. No physical
planarity or higher-LR benefit is established from this reference arm.

The original controller automatically started the2e-5 arm, trainer586053,
at20:04:55 UTC, independently from the same262121 checkpoint. At20:05:30 it
was reading original metadata (135,168 rows), with no verified optimizer
updates yet. Its actual model/initial-state/startup proofs and the fixed final
paired comparison remain pending. The first arm's full closure is recorded in
`room-feature-lr-pair-01/lr1e-5/final-closure-01/report.json`; those proofs must
be reused, with no repeated checkpoint readback or inference.

Nov6 20:21 UTC: the2e-5 arm is actively training as W&B`3fcc0f3f`, with250
finite actual128 updates and its first validation in progress. Its actual
initial adapter/RMS/all100 Adam moments and counters/RNG match the same262121
source; only the declared LR fields differ from the reference arm, with the
same constant-scheduler position. Both first/last128-image model cases preserve
frozen state and have identical input keys, pixels and initial loss to1e-5.
The actual first20 updates contain2,560 RGB presentations (260 ProcTHOR,60
structural,2,240 original); their pixel/key/metadata prefix SHA is exactly the
reference's`b3fa53f651770b1debff50ea5c88f565813a8f5b990e9e72bc50763aec6c81e9`.
Measured update norm0.06278372 is nonzero. All442 online training scalars are
numerically exact. Two baseline metadata values and the start timestamp have
one-ULP display differences; seven other metadata and20 training integer/float
representations are equal. These startup checks are complete in
`lr2e-5/startup-closure-01/report.json`. Complete128,000-input equality and the
fixed final LR comparison remain pending; no training configuration changed.

At20:23 UTC the second arm's250-update six-view review completed, with all631
online validation values numerically exact (37 equal integer/float forms).
Its complete32,000-RGB input prefix, including4,000 unique synthetic images,
has exactly the same bytes/hash as the reference arm at250. Primary spatial
0.9646430589 is above the common parent, but summary0.9361995972 is below;
LiDAR reference AbsRel0.0957755232 is0.5542% lower than the parent. This endpoint
fails both parent and archive joint criteria. Against1e-5 at the same250 steps,
both cosines are slightly lower; LiDAR is only0.01747% lower. No overall higher
LR advantage is established. Structural317 spatial rises from the parent but
summary falls, with teacher-depth discrepancy25.4395% higher; ProcTHOR64 both
cosines fall and teacher-depth discrepancy rises6.6546%. These are interim,
separate synthetic results, not a final LR decision or physical planarity
measurement. The unchanged authorized1,000-update comparison continues.

Nov6 20:40 UTC: the second arm's500-update review and all631 online values
are complete, again with37 equal integer/float forms. Its64,000 actual RGB
inputs/8,000 unique synthetic presentations match the reference prefix bytes
exactly. Primary spatial0.9647363316 rises from the parent, summary0.9362376372
remains below it, and LiDAR reference AbsRel0.0957458814 falls0.5850%. Both
parent and archive joint criteria still fail. Compared with1e-5 at500, spatial
is slightly higher and LiDAR is0.2940% lower, but summary is lower: this is
a mixed interim result. Structural317 spatial rises and summary falls versus
the parent, with teacher-depth discrepancy0.9590% lower; ProcTHOR64 both
cosines remain below the parent and teacher-depth discrepancy is5.9535% higher.
No final LR choice, physical planarity benefit or monotonic improvement is
established. The existing trainer is proceeding toward the fixed1,000 updates.

Completed run, Nov6 08:20 UTC: the user explicitly requested larger **actual**
model batches (128 preferred, at least64), fast loading and batched inference,
and authorized continuing training. `room-feature-truebatch128-50787-training-01`
launched at05:55:17 UTC, wrapper483459/trainer483460, from the verified258946
checkpoint. Immutable execution `code-room-true-batch-epoch-v1` / `3180ddf`
kept all four feature weights1, geometry off, LR1e-5, all three resolutions,
Adam moments/RMS/RNG and the same50,787 synthetic plus1,597,911 original pool.
It finished normally at the planned3,175 updates/global262121. Both training
processes exited. All50,787 synthetic images were presented exactly once;
406,400 total RGB presentations include355,613 original images. The fixed final
checkpoint, all100 Adam states and cursor101600 passed CPU readback. No extra
epoch or deployment was started.

Each update now makes **one actual128-image forward/backward**, with16 synthetic
and112 original images normally. The last update uses3 synthetic and125
original images. A complete pass requires3,175 updates to262121,406,400 RGB
presentations, and101,600 logical four-image transport groups. The original
sampler resumes at213264; the new mixed cursor is explicitly0. Four-image
DataLoader transport groups protect the64MiB shared-memory limit and are
concatenated before the model call; they are not gradient accumulation.

To make a true128 batch possible without padding/warping, one resolution is
chosen per complete update instead of independently per four-image transport
group. The three resolutions and uniform draw rule remain; original image
identities and crop seeds remain, but per-image resolution assignments can
change. This additional grouping change is explicit, so later results cannot
isolate a pure effective-batch causal effect. No LR scaling was introduced.
Four decode workers, compact transport, pinned prepared windows and a separate
CUDA copy stream keep the next batch ready. Actual batch size, forwards per
update and input-wait time are recorded in every committed training row.

The bounded zero-update performance probe completed on the actual258946 model
and real saved RGB inputs at all three sizes. In `room-true-batch-probe-02`,
actual128 forward/loss/backward/clipping throughput is71.85--89.10images/s,
peak allocated memory81.10--95.91GiB, and warmed consumer queue wait0.10--0.14ms.
All measured future windows were already copied before consumption. This
measurement excludes Adam.step, final evaluation and output writing; production
training speed is reported separately. Actual64 was72.11--87.73images/s, so128
increases images per update, without establishing a throughput improvement.
No model state changed and no optimizer step was taken by the probe.

For student inference (RADIO, adapter and raw decoder), actual64 passed all
three sizes at85.04--102.97images/s with inputs resident on the GPU. These figures
exclude focal recovery/postprocessing and disk I/O. Actual128 raw inference
passed400x640 but hit upstream padding's32-bit indexing limit at480x640 and
640x480; this is separate from the successful128 feature-training path. The
universal tested depth-inference batch is64. The existing public viewer and
its chosen checkpoint remain unchanged. Probe01 stopped on that decoder error;
probe02 explicitly records it and reuses the three completed64 measurements.

The new narrow batch-fork guard rejects unplanned model/loss/LR/data changes,
and requires a single full compute tensor before every optimizer update.
87 local sampler/core regression checks and22 saved-review checks passed;
55 relevant tests also passed remotely, including actual tiny trainer loops
at64 and128, inherited initial Adam/RMS/RNG and one encoder call per update.
The runtime additionally checks the actual first and final128-image batches
before updates. Results will be reviewed every250 updates (32,000 presented
images) and at the fixed final endpoint, using the six saved parent baselines.
The wall bound is14,400s, with no automatic retry, extra epoch or deployment.

At06:04 UTC the actual run is active as W&B`55dca03c`, with76 finite
updates at259022. All76 recorded updates use exactly one128-image model
forward/backward. Initial adapter,RMS,all100 Adam states and RNG match the
258946 parent bitwise. The actual first and last128-image model gates both
passed without optimizer updates, and restore RNG/frozen state exactly.
The earlier all-three-resolution performance probe is reused.

The first20 committed updates and2,560 actual RGB inputs match the new plan:
260 ProcTHOR,60 structural and2,240 original images. Image identities/crop
seeds, one128-image compute group, all feature coefficients1 and geometry off
are checked. The measured nonzero update norm is0.018689. All442 online startup
scalars match numerically; only one baseline display value differs by one ULP,
with seven other metadata and20 training values using equal integer/float
forms. The first76 updates average0.288ms waiting for the prefetched input,
with maximum1.867ms and peak allocated memory96.21GiB. These are pipeline
measurements, not validation gains. First saved validation is259196 (+250);
all startup proofs are complete and must be reused.

All thirteen actual128 saved validations are complete. All six views retain
the same images, protocols, frozen teacher and support. Every reviewed committed
RGB prefix matches the plan. The final (+3175) covers406,400 presentations,
including all50,787 synthetic images with no omissions or repetitions. All8,203
online validation values match numerically and W&B`55dca03c` is finished. Each
endpoint has37 equal integer/float representation differences only. The one
previously bound baseline display ULP and seven metadata integer/float forms
remain unchanged; actual final validation values have no numerical difference.

| Actual128 comparison | Primary spatial cosine | Primary summary cosine | LiDAR-reference AbsRel |
| --- | ---: | ---: | ---: |
| Immediate parent258946 |0.9646147596|0.9360840607|0.0972530157|
| +250 /259196 |0.9647335855|0.9361621222|0.0952842571|
| +500 /259446 |0.9646708565|0.9362329607|0.0943515258|
| +750 /259696 |0.9647070075|0.9362420658|0.0968443200|
| +1000 /259946 |0.9646628006|0.9363029492|0.0955888441|
| +1250 /260196 |0.9647164502|0.9362204487|0.0959826544|
| +1500 /260446 |0.9646592495|0.9362024370|0.0971774676|
| +1750 /260696 |0.9647315295|0.9362519977|0.0943507718|
| +2000 /260946 |0.9645660295|0.9363477593|0.0953588753|
| +2250 /261196 |0.9646356767|0.9364488603|0.0948713978|
| +2500 /261446 |0.9645059323|0.9362618706|0.0974016903|
| +2750 /261696 |0.9647176330|0.9363110939|0.0961006664|
| +3000 /261946 |0.9646713266|0.9363431548|0.0960200150|
| Fixed final+3175 /262121 |0.9646276428|0.9362591955|0.0963093091|
| Same-population archive232288 |0.9647740929|0.9356613590|0.0921184475|

Eleven of these thirteen endpoints meet the numerical three-metric criterion against
their immediate parent. The exceptions are+2000 (spatial below parent) and+2500
(spatial below and LiDAR-reference error above parent). None meets the criterion
against the archive. At+500 LiDAR-reference AbsRel
is2.9834% lower than the parent and2.4241% higher than the archive. Spatial
cosine declines between+250 and+500, while summary and the reference-depth metric
improve; gains are not uniformly monotonic. This is not proof of a causal batch
effect, statistical significance or real-wall planarity improvement.

Structural317 at+500 has cosines0.9653114603/0.9819599388 and teacher-depth
discrepancy3.6621% below the parent. ProcTHOR64 has cosines0.9756405717/0.9608373558;
its teacher-depth discrepancy is essentially unchanged (-0.0086%). Synthetic
populations remain separate from the primary six-dataset macro. At06:24:59 UTC
the existing run is healthy at554 updates/259500; no input, loss, optimizer,
inference protocol or training bound changed during these reviews.

The new+750/+1000 reviews reuse the completed startup and earlier reviews.
At+1000 both primary cosines remain above the parent; LiDAR-reference AbsRel
is1.7112% below the parent and3.7673% above the archive. Its improvement over
the parent was only0.4202% at+750, so depth gains are not monotonic. Summary
cosine has risen at each of these four endpoints, while spatial fluctuates.
Structural317 at+1000 reaches0.9655911464/0.9820425886 with teacher-depth
discrepancy5.5595% below the parent. ProcTHOR64 reaches0.9758126494/0.9610970700
with discrepancy5.4161% below the parent. These remain separate synthetic
results, not evidence of physical wall-flatness or a unique batch-size cause.
The06:49:01 UTC snapshot has1,140 finite updates at260086, all actual128 and
one forward each; viewer3462 remains alive and OOM counters stay zero.

The+1250/+1500 reviews add another1,262 numerically exact online values and
verified160,000/192,000 RGB prefixes, without repeating inference or earlier
proofs. At+1500 primary cosines still exceed the parent, but LiDAR-reference
AbsRel is only0.07768% lower, essentially back to its starting value; it is
5.4919% above the archive. The numerical joint criterion still passes against
the parent, but the earlier depth improvement has mostly disappeared. Summary
cosine also declines at these latest two endpoints, so its first-four-point
upward sequence did not continue. The fixed single pass proceeds unchanged.

Structural317 at+1500 reaches0.9656539779/0.9822966693, yet its decoded
teacher-depth discrepancy is5.5597% worse than the parent. ProcTHOR64 reaches
0.9758656919/0.9611066617 with teacher-depth discrepancy1.6328% worse. These
mixed secondary results are retained; stronger feature cosines do not by
themselves establish better decoded depth or physical wall-flatness. At07:13:14
UTC the existing trainer has1,731 finite updates at260677; actual batch128,
one forward per update, preserved viewer and zero OOM remain verified.

The+1750/+2000 reviews preserve all six evaluation identities and verify
224,000/256,000 actual RGB prefixes plus another1,262 exact online values.
At+1750 all three primary metrics improve versus the parent, with LiDAR-reference
AbsRel2.9842% lower. At+2000 spatial cosine falls to0.9645660295, below the
parent0.9646147596; summary rises to0.9363477593 and LiDAR-reference AbsRel
remains1.9476% lower than the parent (3.5177% above the archive). This is the
first current-run endpoint without all three simultaneous improvements.
Its score remains evidence of fluctuation, not a unique diagnosed cause.

At+2000 structural317 cosines are0.9659929341/0.9824190100 with teacher-depth
discrepancy8.4000% below the parent; ProcTHOR64 cosines are0.9758166699/
0.9612071076 with discrepancy3.4263% below the parent. ProcTHOR discrepancy
had been9.1419% worse at+1750, another nonmonotonic secondary result. At07:38:31
UTC the trainer has2,250 finite actual128 updates at261196 and is evaluating
that endpoint. No optimizer, data, loss or bound was changed; all old proofs
are reused and the fixed single pass continues without an extra epoch.

The+2250/+2500/+2750 reviews add verified288,000/320,000/352,000 actual RGB
prefixes and1,893 numerically exact online values. Both+2250 and+2750 again
meet the parent joint criterion; +2500 does not. At+2750 the primary cosines
are0.9647176330/0.9363110939 and LiDAR-reference AbsRel is1.1849% below the
parent but4.3229% above the archive. At+2500 the reference error was0.1529%
above the parent. These reversals do not establish stable improvement or a
unique batch-size cause.

Structural317 at+2750 has cosines0.9656235367/0.9825545927 and teacher-depth
discrepancy5.1693% below the parent. ProcTHOR64 has cosines0.9758289559/
0.9616708448 and discrepancy1.1123% below the parent. All other secondary
metrics remain in the paired reports. At08:05:18 UTC the existing trainer
has2,916 finite actual128 updates at261862. Only the predefined+3000 and
fixed final+3175 reviews remain; complete synthetic coverage and final state
verification were still pending at that snapshot. No new epoch or deployment
was started.

The run subsequently reached its fixed endpoint and completed final evaluation
in8,666.11 seconds (about2h24m, including startup and validation). All3,175
committed updates were finite, used one actual128-image model forward, and
preserved the feature-only contract. The measured update time sums to5,337.60s,
or76.14 images/s excluding startup, validation and checkpoint work. Mean input
queue wait is0.260ms, p951.170ms and maximum1.867ms; this is consumer wait with
prefetch, not individual disk-decode latency. Peak allocated GPU memory is
96.25GiB and OOM counters remain zero. Viewer3462 is preserved.

At fixed final262121, spatial cosine rises by0.0000128831 and summary cosine by
0.0001751348 versus the immediate parent258946. LiDAR-reference AbsRel is0.97036%
lower. The spatial increase is very small, and the trajectory includes two
joint failures, so this is a numerical endpoint improvement rather than an
established stable or causal gain. Relative to archive232288, spatial remains
0.0001464501 lower and LiDAR-reference error4.54943% higher; summary is higher.

Final structural317 cosines are0.9656564223/0.9825944254 and teacher-depth
discrepancy is5.04759% below the parent. ProcTHOR64 cosines are0.9760322012/
0.9618501803, but its teacher-depth discrepancy is0.50359% higher. These synthetic
feature gains do not establish physical wall-flatness or universal decoded-depth
improvement. All other metrics remain in the six-view reports.

The final checkpoint is preserved as
`finish-verification-01/checkpoint-step262121.pt`, SHA
`510ed857872d99e10b18bf60edcc86d96da9b55097e7ffead94c08fdd8ac7781`.
The final readback verifies all100 finite Adam states at262121, cursor101600,
unchanged RMS/statistics, config/model/data revisions and complete input coverage.
Finish report SHA`25295bd5631018fc9681bebd0d9353dd03488bfa97d48c84803b1d677c3201a6`;
fixed final six-view review SHA
`3cb0d42c1a6fa69463dcce96bf828a58f3bef98f838560ef1a504649eb755bf4`.
Historical bests, including235538/246249/258946, remain preserved. All generation
is complete, and the user's stop on ScanNet geometry work remains in force.

The original final-online checker stopped before publication because the API
returned two rows with final optimizer_step262121: internal3186 is the last
training row with every validation field null; internal3187 is the complete
validation row. Cached and uncached bounded history tails are identical, so
the observed mismatch is not resolved by bypassing cache. The diagnostic first
used an unavailable SDK attribute and stopped before history reads; the actual
SDK property is`lastHistoryStep`.

Verifier-only revision`4bf3c9a`, preserved separately in
`code-room-online-final-v2`, excludes only records whose requested validation
fields are all null or absent. Partial records, duplicate complete results,
missing results, numerical mismatches and nonfinite values still fail. Eleven
local and eleven remote CPU checks passed, and the saved actual final row
matches all631 values exactly numerically (37 equal integer/float forms).
The training execution remains immutable3180ddf; no model, input, weight or
optimizer changed. The corrected final online query passed and published SHA
`5b9b372bf78b6417f3a272ea0f18af1b8bbebb8fa53b8035c7abcd08aa099c5a`.
It records exactly one ignored null-only row and one complete final validation.
All13 endpoints are now closed. The runner's old`finished_review_pending` status
is historical; the finish/readback, paired reviews and online closure are complete.

Previous completed epoch, Nov6 05:04 UTC: the expanded training run and all final checks
are complete. The user-authorized frozen pool contains
**50,787 images**:40,000 official ProcTHOR and10,787 custom structural RGB,
including3,943 images generated after the previous frozen epoch. Its50,787 rows
preserve all earlier input bytes. Cross-batch image/group/pixel/layout checks and
both fixed synthetic validation populations have zero exact overlap. Prior
payload verification and actual sampled visual reviews are reused; quarantined
scene493 and rejected diagnostic507-slot11 remain outside training. Projection
labels remain explicit:49,699 perspective and1,088 approved orthographic views.
Pool manifest SHA`1173ca7e84f2939ae15de47988ec42e402a4228f1e817fbb25b954e6f1a26637`.

`room-feature-full-epoch-50787-training-01` launched at02:04:08 UTC under the
original GPU lock, wrapper409645/trainer409646. Immutable execution is
`code-room-full-epoch-v2` / `00f95a8`. It finished normally at04:46 UTC after
12,697 finite updates, final step258946. Both processes exited and W&B
run`20d7e04c` is finished. All50,787 synthetic images were presented exactly
once, with zero omissions or repetitions. All13 saved validations and8,203
online scalar comparisons are complete and numerically exact. Geometry stayed
off and all four feature weights stayed1. At that closure no optimizer or generator remained active. The newly authorized
true-batch continuation is described above; no deployment was launched.

The new parent is the verified final246249 checkpoint, preserving its full Adam,
adapter,RMS and RNG state. Original logical cursor111688 continues from
18000+93688; the explicitly new mixed cursor begins at0. The three resolutions,
effective batch32 and LR1e-5 remained. A single synthetic permutation took12,697
updates to258946. The last update uses3 synthetic and29 original images, so all
50,787 synthetic images can appear exactly once without padding by repetition.
The actual406,304 presentations include355,517 original RGB presentations.
This is one complete pass over the synthetic pool, not the full original RGB
pool. It reached the step bound within9,709.19 seconds including startup and
final evaluation; the unchanged time bound was28,800s, outer29,400s.

The scoped single-epoch sampler leaves all other original keys unchanged, also
on resume and unused prefetch. The core training, losses, models, data transforms
and original sampler files are byte-identical to the previous execution. The
modified wrapper is independently hash-bound; old model gates are not relabeled
for the new configuration.63 local and63 remote CPU tests pass, including exact
50,787 coverage, production-loader selection, tail/continuation semantics,
commit-only audit, final review and exclusion bindings. Two earlier fixture-only
failures lacked tiny-loader batch_size/seed; no runtime constraint was weakened.
The actual model probe additionally includes the final32 planned presentations.

The new actual CUDA/model gate passed41 cases/184 RGB presentations, including
all sources/resolutions, the approved orthographic supplements and final32 keys.
Original pixel parity, all frozen module state and restored RNG are exact. The
actual independent trainer initial checkpoint preserves adapter,RMS,all100 Adam
states,RNG and model revisions bitwise at246249, with new cursor0. Metadata loading
finished in363.54s. The first20 committed updates contain640 images:60 official
room,20 structural and560 original RGB. Original keys/crop seeds/buckets are
exact, all feature terms retain weight1, losses/gradients are finite and the
measured update norm is0.034979. Six saved parent views are reused without model
inference. All382 first20 online training values are numerically exact. Only
three allowed checkpoint-baseline display values differ by exactly one ULP;
seven metadata and20 schedule values use equal integer/float forms. Config bytes
remain unchanged. Actual startup closure SHA
`481eee68312db590989929fe803d9cdf3f3955f3e773e810e3703c737d571c87`.
No model, checkpoint or startup proof needs to be repeated.

All six saved parent views are reused without baseline inference. Saved reviews
and online checks are scheduled every1,000 updates and at final258946; comparison
remains both original cosines up and iPhone-LiDAR-reference AbsRel down, against
both the immediate parent and the same-population archive. Full final coverage
and all100 Adam counters at258946 are verified. The sensor metric is not an
independent physical calibration or a wall-flatness measurement; see the new
metric audit below.
No generation or ScanNet geometry work resumes.

The first saved six-view review at247249 (+1,000 updates) retains all image,
protocol, frozen-teacher and support identities. All32,000 actual inputs match
planned keys:3,198 official-room,802 structural and28,000 original RGB. The4,000
synthetic identities are unique. All631 online validation values match
numerically;37 values differ only by equal integer/float representation.

The primary cosine columns below are `original` view macro averages: first
average within each of six datasets, then average those six values equally.
The fixed 1,012-image population is ARKitScenes256, Hypersim256, IRS128,
NYUv2116, Structured3D128 and TartanAir128. It contains real and synthetic
sources. ProcTHOR64, structural317, ScanNet++ DSLR128 and LiDAR120 are separate
views and do not enter these primary cosine averages. Cosines compare student
and frozen MoGe-v2 teacher features; the primary LiDAR column separately uses
student depth AbsRel against the ScanNet++ LiDAR reference.

| Comparison | Spatial cosine | Summary cosine | LiDAR AbsRel |
| --- | ---: | ---: | ---: |
| Immediate parent246249 |0.9646110087|0.9358942373|0.0937995260|
| Expanded epoch +1,000 /247249 |0.9645510395|0.9359580217|0.0950027198|
| Expanded epoch +2,000 /248249 |0.9645484637|0.9361284375|0.0950587437|
| Expanded epoch +3,000 /249249 |0.9645448126|0.9360928243|0.0964023908|
| Expanded epoch +4,000 /250249 |0.9645538756|0.9361691227|0.0969707513|
| Expanded epoch +5,000 /251249 |0.9645651862|0.9359653671|0.0980188375|
| Expanded epoch +6,000 /252249 |0.9646610397|0.9361102704|0.0977118384|
| Expanded epoch +7,000 /253249 |0.9646985648|0.9360524568|0.0959907461|
| Expanded epoch +8,000 /254249 |0.9645417578|0.9361976697|0.0951506437|
| Expanded epoch +9,000 /255249 |0.9646997786|0.9363011115|0.0949404479|
| Expanded epoch +10,000 /256249 |0.9646211579|0.9363219419|0.0954678102|
| Expanded epoch +11,000 /257249 |0.9645506524|0.9361307469|0.0948617947|
| Expanded epoch +12,000 /258249 |0.9644747915|0.9360612218|0.0972581755|
| Expanded epoch final +12,697 /258946 |0.9646147596|0.9360840607|0.0972530157|
| Same-population archive232288 |0.9647740929|0.9356613590|0.0921184475|

At the first validation, summary rises, but spatial falls and LiDAR is1.2827% worse than the parent,
3.1310% worse than the archive. Both joint targets fail at this endpoint. The
user-authorized complete image pass continues without changing inputs, learning
rate or loss. Earlier candidates and the prior final checkpoint remain preserved;
this result does not justify deploying the latest weights or claiming that
longer training improves every metric.

Structural317 cosines rise slightly to0.9629938352/0.9766264086, while decoded
teacher-depth discrepancy worsens14.2320% to0.1002726873. Official-house64 cosines
both decline and its teacher-depth discrepancy worsens7.9811%. Native400x640
original cosines improve slightly; it remains a secondary view, not a substitute
for the fixed primary criterion. No unique cause, significance or real-wall
planarity gain is established. First-review closure SHA
`80e9231a8ff36a008134ec7deba6762418f6f741a2e02d7acc49a592d905527a`.

The new +2,000 and +3,000 reviews retain all six populations, image identities,
protocols, frozen teacher values and support. Their 64,000 and 96,000 actual
input keys match the plan, with 8,000 and 12,000 unique synthetic images and no
repetitions. All 1,262 online validation values match numerically; each endpoint
has 37 equal integer/float representations. No inference, model gate, initial
state proof or checkpoint readback was repeated.

Both endpoints fail the joint criterion against both the parent and archive.
At +3,000, summary cosine is higher than the parent, spatial cosine is slightly
lower, and LiDAR error is 2.7749% worse than the parent and 4.6505% worse than the
archive. These observations do not support an overall benefit from continuing
this expanded run. The authorized complete image pass continues unchanged;
earlier best checkpoints remain preserved and no additional epoch is automatic.

Structural317 cosines reach 0.9639728492/0.9784087148 at +3,000 and its teacher-depth
discrepancy is 15.8162% lower than the parent. Official-house64 remains mixed:
spatial cosine falls, summary rises, and teacher-depth discrepancy is 3.0778%
higher than the parent. These synthetic results do not establish improved
real-wall planarity, causality or statistical significance. New review closure
SHA `a80dfa8289cc39e76cb710a9847fb91c5e388f7fad6552bac8c1b094494fb7fa`.

The +4,000 and +5,000 saved reviews also preserve all six views and image,
protocol, teacher and support identities. Their 128,000 and 160,000 actual input
keys match the plan, with 16,000 and 20,000 unique synthetic images and no
repetitions. All 1,262 new online scalar values are numerically exact, with 37
equal integer/float representations per endpoint. No new inference or checkpoint
read was performed. Closure SHA
`2184d2a6aa1c8a76b99f412d2c3879b989d89e0fce0ed60eac492b12c954b55c`.

Both endpoints again fail the parent and archive joint criteria. At +5,000,
LiDAR error is 4.4982% worse than the parent and 6.4052% worse than the archive;
it has increased at each of this run's five periodic evaluations. Spatial cosine
recovers from +3,000 but stays below both baselines, while summary stays above
both. Structural317 cosines reach 0.9645697570/0.9791387201 and teacher-depth
discrepancy improves 16.4503% versus the parent. Official-house64 cosines also
improve, with teacher-depth discrepancy 5.4123% lower than the parent. These
synthetic gains do not establish improved original-data performance or real-wall
planarity. The user-authorized single pass continues unchanged; earlier best
weights remain preserved, with no automatic additional epoch or deployment.

The +6,000 and +7,000 saved reviews preserve all six views and exact image,
protocol, teacher and support identities. All 192,000 and 224,000 actual input
keys match the plan, with 24,000 and 28,000 unique synthetic images, zero repeats
and unused prefetch excluded. All 1,262 new online values match numerically,
with 37 equal integer/float representations per endpoint. No new inference,
model gate or checkpoint read was needed. Closure SHA
`a00cc17c32448bf1ac61f6d7c97054becb68d307f9236368c7245cc458da2e02`.

Both original cosines now exceed the immediate parent at +6,000 and +7,000.
LiDAR recovers from +5,000, ending at 2.3361% worse than the parent and 4.2036%
worse than the archive at +7,000. Thus neither joint criterion passes; the
earlier five-endpoint LiDAR deterioration does not continue monotonically.
Archive spatial cosine remains higher than this endpoint. Structural317
cosines are 0.9645291533/0.9791871856 and teacher-depth discrepancy is 3.0256%
lower than the parent, less favorable than at +5,000. Official-house64 cosines
both exceed the parent while teacher-depth discrepancy is 1.6086% higher.
These mixed changes do not establish causality or improved real-wall planarity.
The fixed single pass continues without input, loss or learning-rate changes.

The +8,000 and +9,000 saved reviews preserve all six view identities, protocols,
teacher values and support. All 256,000 and 288,000 actual input keys match
the plan, with 32,000 and 36,000 unique synthetic images and zero repetitions.
All 1,262 new online values are numerically exact, with 37 equal integer/float
representations per endpoint. No inference or checkpoint read was repeated.
Closure SHA `9b08ece4823a093353215f5de4de374aa8e9c48feae7f98d9a526b6ee1a96af2`.

At +9,000 both primary cosines exceed the immediate parent; spatial had dipped
below it at +8,000. LiDAR continues recovering but remains 1.2163% worse than
the parent and 3.0634% worse than the archive. Both joint criteria still fail.
Structural317 cosines reach 0.9646387918/0.9798911650 and teacher-depth discrepancy
is 10.0452% lower than the parent. ProcTHOR64 cosines reach
0.9758128310/0.9594018944, with teacher-depth discrepancy 7.8963% lower. These
synthetic results do not establish real-wall planarity or a causal explanation.
Earlier best weights remain preserved; no automatic extra epoch or deployment.

The +10,000 and +11,000 saved reviews preserve all six views and exact image,
protocol, teacher and support identities. All 320,000 and 352,000 actual keys
match the plan; 40,000 and 44,000 synthetic identities are unique with no
repetitions. All 1,262 new online scalar values match numerically, with 37 equal
integer/float forms per endpoint. No inference or checkpoint read was repeated.
Closure SHA `d6813e3b0980de5cfca0f059d39e3c74ad211820fd913d7858d98853a16ceb6b`.

Both original cosines exceed the parent at +10,000, but spatial falls below it
at +11,000. LiDAR worsens at +10,000 and then recovers to 1.1325% worse than the
parent and 2.9781% worse than the archive at +11,000. Neither joint criterion
passes. Structural317 cosines are 0.9645956499/0.9802469369, while teacher-depth
discrepancy is now 7.4092% worse than the parent. ProcTHOR64 cosines are
0.9751851531/0.9597080657 and its teacher-depth discrepancy is 6.8707% lower.
Feature and decoded-depth metrics remain distinct; these fluctuations establish
neither a cause nor improved real-wall planarity. The existing run continues to
its fixed final step258946, followed by full coverage and final-state checks.

Those final checks are now complete. The last two saved reviews retain all six
populations, image identities, protocols, teacher values and support. Their
384,000 and406,304 actual input presentations match the plan. Final coverage is
40,000 official-room and10,787 structural identities, each once. Final CPU
readback verifies all100 finite Adam states at258946, cursor101576, finite
adapter/RNG state, unchanged RMS statistics, config, data and model revisions.
The closed final checkpoint is preserved by hardlink, SHA
`c5d1e6b2b98a54fd0031906d27d5c623a1448de62c9ba1dd09080c33a73d0840`.
All13 endpoints have631 numerically exact online scalars each,8,203 total.
The three known baseline display differences are exactly one ULP; no runtime
configuration was changed. Final closure SHA
`5d9cf4d6f30871d14791d495b43bc8b89ca6ebaef335052dd366c1116b1ce711`.
The original controller's `quality_review_pending` field is historical and is
superseded by that closure. Earlier235538 and246249 checkpoints remain intact.

At the fixed final endpoint, primary spatial cosine rises only0.0000037509 and
summary rises0.0001898234 versus the immediate parent. Published iPhone LiDAR
AbsRel is3.6818% higher than the parent and5.5739% higher than the archive;
archive spatial cosine also remains higher. The predefined numerical joint
criterion therefore fails against both baselines. This is not a claim that real
walls became less planar. Structural317 cosines reach0.9649753879/0.9811997077
and its teacher-depth discrepancy falls15.5410% versus the parent. Official
ProcTHOR64 reaches0.9756197520/0.9602337917 with discrepancy7.5941% lower.
Synthetic agreement and real-scene physical accuracy remain separate claims.

The user then questioned whether the LiDAR score itself was computed correctly.
The new read-only `scannetpp-lidar-evaluation-audit-20261106-01` checks that
question separately from the already completed training closure. Seven CPU
fixtures pass. All120 prepared RGB/depth payload hashes match, and an independent
mean of the120 stored image scores exactly reproduces both published scores.
A bounded optimizer-free inference audit saves actual arrays for the same120
images and both checkpoints. All240 re-evaluated student AbsRel values exactly
match their original logs. Independent float64 NumPy AbsRel differs from the
tensor implementation by at most2.711e-8; no arithmetic error was found.

An independent native decoder and analytic distortion map reproduce every
supported prepared value for eight frames in two scenes covering both actual
compressed formats: uint16 millimeters and float32 meters. This is bounded unit
and mapping evidence, not physical calibration of all120 views. The
[publisher documents RGB-aligned iPhone depth at256x192](https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation)
and provides the [two decoding paths](https://github.com/scannetpp/scannetpp/blob/main/iphone/prepare_iphone_data.py).
The iPhone reference is separate from the previously investigated mesh labels
and panocam images. None of the120 manifest records contains a confidence map.

The prepared768x576 depth is resized to480x640. Independent nearest-neighbor
indexing initially disagrees at2,379,719 depth pixels across120 images, but all
coordinate differences occur at exactly equidistant nearest-neighbor ties.
Using the alternative tie rule yields parent/final AbsRel0.0938107574/
0.0972726925, a3.6903% increase, so this convention does not explain the observed
3.6818% increase. Neither the input nor the published metric was replaced.

The primary score uses absolute metric depth, without GT scale fitting, and
MoGe infers focal/shift rather than receiving the known sensor intrinsics.
A separately labeled diagnostic fits one median scale per image: parent/final
AbsRel becomes0.0497093317/0.0502954050, a1.1790% increase. This shows sensitivity
to absolute scale; it is neither a replacement benchmark nor a proof of the
cause. The unchanged MoGe-v2 teacher itself scores0.1213605191 on this same
sensor protocol, higher than the final student's0.0972530157. Closer teacher
agreement therefore does not guarantee a lower sensor-reference error.
Twelve of30 scenes improve and18 worsen on the published score. These
reused validation scenes do not establish statistical significance or a unique
cause. Sensor calibration, reference noise, confidence, and direct wall-flatness
measurements remain unproven. The earlier interpretation is narrowed to an
increase on this particular iPhone-reference protocol, not demonstrated real
geometry degradation. CPU/model/robustness reports are immutable; no training,
source labels, thresholds or existing scores were changed.


Previous completed state, Nov6 01:41 UTC: the user explicitly requested starting with the
qualified data and passing **every new image through training**. A separate
`room-feature-full-epoch-46844-training-01` has completed its11,711 updates and
final evaluation normally at246249. Wrapper342237/trainer342238 exited; W&B run
`b73f7bb0` is finished. Immutable training execution remains
`code-room-full-epoch-v1` / `71c9d2b`. All46,844 synthetic images were presented
exactly once, with zero omissions or repeats. Final state, all twelve saved paired
reviews and online checks are complete. No optimizer or generator remains active.

The new frozen pool appends qualified batches08–10 to the earlier40,890 rows,
without altering those rows:46,844 RGB images =40,000 official ProcTHOR views
+6,844 custom structural views, including690 explicitly labeled orthographic
supplements. Exact image/group/pixel/layout and both fixed validation isolation
checks are reused and combined. Growing batch11/12 images are not inserted into
this run. The prior old-house camera/window limitations remain; the closed-room,
no-window promise applies to the custom structural source.

The run continued from the verified234538 checkpoint with bitwise-preserved
adapter, RMS, all100 Adam states and RNG. The original RGB cursor continues at
18000 (previous base10000 plus8000 mixed logical batches); the new mixed cursor
starts at0 explicitly. Four synthetic images per effective32 update require
11,711 updates to present each of46,844 synthetic identities exactly once. The
plan contains374,752 presentations:46,844 synthetic and327,908 original RGB.
Original crop seeds, three resolutions, learning rate1e-5 and all four feature
weights1 remain. Geometry targets and loss are absent. Actual committed encoded
image identities, not prefetched rows or step counts alone, determine coverage.
The fixed endpoint is246249, with an8-hour training bound and29,400s outer bound;
partial coverage must remain explicit if interrupted. No automatic retry, next
epoch or deployment is authorized by this run.

New CPU tests passed18 local/18 remote. The actual new model gate passed33
cases/152 presentations, including orthographic supplements, with original pixel
parity, frozen-state identity and restored RNG. The first20 actual updates contain
640 RGB:72 official-room,8 structural and560 original images; every untouched
original key/crop/bucket matches its continued stream. Losses/gradients are finite
and the measured adapter update is nonzero. Metadata loading completed in360.81s.
All six saved parent validation views are bound and reused without baseline
inference. Validation occurs every1,000 updates and at the final711-update tail;
the original both-cosines-up and LiDAR-error-down criterion remains unchanged.
The first20 online updates contain382 scalar values, all numerically identical
to the committed local records. Only two checkpoint-baseline API displays differ
by exactly one ULP; seven metadata and20 schedule values use equal integer/float
representations. Configuration bytes are unchanged. Startup closure
`room-feature-full-epoch-startup-closure-01/report.json` has SHA
`256d6147f6bd3921dd5e54e0ddefdcd609f98d336d50431889e46b1f44b672e5`.
Saved-result/full-coverage/final-online reviewers in `code-room-epoch-review-v1`
(`47758c0`) passed22 local and22 remote CPU tests. All twelve scheduled validation
reviews, full coverage and final checkpoint reviews are complete. All11,711
training records are finite. The complete actual-input audit contains374,752
presentations:40,000 official-room,6,844 structural and327,908 original RGB,
with all planned keys matched and uncommitted prefetch excluded. Every synthetic
count is exactly1. The controller's `quality_review_pending` flag reflects its
publication before review; the final closure below supersedes it.

The first1,000- and2,000-update evaluations retain all six parent populations,
per-image identities, protocols, teacher values and support. All actual planned
RGB keys and synthetic identities match, with4,000/8,000 unique synthetic images
respectively. The original-validation comparison is:

| Endpoint | Spatial cosine | Summary cosine | LiDAR AbsRel |
| --- | ---: | ---: | ---: |
| Same-population archive232288 |0.9647740929|0.9356613590|0.0921184475|
| Parent234538 |0.9644499831|0.9358074926|0.0954916715|
| +1,000 /235538 |0.9646070817|0.9358768267|0.0927934218|
| +2,000 /236538 |0.9645692867|0.9360401285|0.0949768736|
| +3,000 /237538 |0.9645070946|0.9359933445|0.0945354724|
| +4,000 /238538 |0.9646226233|0.9358286189|0.0947856377|
| +5,000 /239538 |0.9645942229|0.9359031434|0.0935841831|
| +6,000 /240538 |0.9645106678|0.9358969830|0.0942630424|
| +7,000 /241538 |0.9644845004|0.9358905418|0.0952495001|
| +8,000 /242538 |0.9644182869|0.9358918334|0.0962238367|
| +9,000 /243538 |0.9646211579|0.9360376001|0.0948682129|
| +10,000 /244538 |0.9645768229|0.9360804793|0.0935048023|
| +11,000 /245538 |0.9645637924|0.9360956301|0.0934422740|
| Final +11,711 /246249 |0.9646110087|0.9358942373|0.0937995260|

Both endpoints jointly improve the three primary metrics against this epoch's
parent. LiDAR error is2.8256% lower at1,000 and0.5391% lower at2,000. The2,000
endpoint has weaker spatial/LiDAR results than1,000 and higher summary cosine;
improvement is not monotonic. Against the same-population archive232288 baseline,
both still have lower spatial cosine and higher LiDAR error (+0.7327%/+3.1030%).
Therefore the archived-baseline joint target still fails. No significance,
causal explanation or real-world wall-planarity improvement is established.

The3,000/4,000 reviews retain all six populations, exact per-image identities,
protocols, teacher values and support. Their actual12,000/16,000 synthetic
presentations are unique and match the planned keys. Both endpoints also jointly
improve against the current parent; LiDAR is1.0013%/0.7394% lower. Against
archive232288, spatial remains lower and LiDAR is2.6238%/2.8954% higher. At4,000,
spatial is this epoch's best observed value, summary has fallen from2,000, and
the1,000 checkpoint still has the lowest observed LiDAR error. The first four
endpoints do not establish monotonic generalization gains.

The5,000/6,000 reviews also pass all six population, image, protocol, teacher and
support comparisons. Their20,000/24,000 committed synthetic images are unique and
all actual planned RGB keys match. Both endpoints jointly improve against the
parent, with LiDAR1.9975%/1.2866% lower. Against archive232288, both still have
lower spatial cosine and higher LiDAR (+1.5911%/+2.3281%). At6,000 all three
primary values are slightly worse than5,000; the1,000 checkpoint remains the best
observed LiDAR endpoint in this epoch. No monotonic or causal gain is established.

The7,000/8,000 saved reviews retain all six populations, image identities,
protocols, teacher values and support. All28,000/32,000 committed synthetic
identities are unique and match the plan. At7,000 the parent joint criterion still
passes, with LiDAR0.2536% lower. At8,000 it fails for the first time in this epoch:
spatial cosine is slightly below the parent and LiDAR is0.7667% higher, while
summary cosine remains above the parent. LiDAR is3.3989%/4.4566% higher than
archive232288 at the two endpoints. The archive joint criterion fails at both.
The requested full-image pass continues unchanged, retaining the earlier1,000-step
candidate; the latest weights are not assumed to be the best model.

The9,000/10,000/11,000 reviews preserve all six populations, image identities,
protocols, teacher values and support. Their36,000/40,000/44,000 committed
synthetic identities are unique and match all planned RGB keys. All three
endpoints recover the parent joint criterion: LiDAR is0.6529%/2.0807%/2.1462%
lower, with both original cosines higher. All still fail the archive criterion:
spatial remains lower and LiDAR is2.9850%/1.5050%/1.4371% higher. At11,000 summary
reaches its highest observed value in this epoch, while spatial falls slightly
from10,000. The1,000 checkpoint still has the lowest observed LiDAR error.
Recovery after8,000 does not establish monotonic gains or the best final model.

The fixed final endpoint passes the current-parent joint criterion: both original
cosines rise and LiDAR is1.7720% lower. Against archive232288, spatial is still
lower and LiDAR is1.8249% higher, so the archive joint target fails. Compared
with11,000, final spatial recovers but summary and LiDAR worsen. The best observed
original spatial, summary and LiDAR endpoints differ:4,000,11,000 and1,000
respectively. The final weights are preserved without assuming they are the best
deployment candidate or starting another epoch.

Unseen structural317 cosine reaches0.9581754851/0.9645105841 at2,000, compared
with parent0.9510275748/0.9381207416; teacher-depth discrepancy is16.8250% lower.
Official-house64 also improves both cosines. These remain synthetic validation
results and are reported separately from the original macro/independent LiDAR.
At4,000, structural317 cosine reaches0.9602283230/0.9691518064, with teacher-depth
discrepancy28.1575% below the parent. Both structural cosines have improved at
each of the four scheduled evaluations. This supports completing the requested
one-pass coverage, while preserving earlier validation candidates; it does not
establish that additional epochs or the final checkpoint will be better.
At5,000, structural spatial cosine dips slightly to0.9600520215 while summary
rises to0.9712064710. At6,000 both reach new observed highs,
0.9614731865/0.9730498338; teacher-depth discrepancy is33.7790% below the parent.
Official-house64 cosine reaches0.9742144151/0.9517380241. Its teacher-depth
discrepancy worsens from5,000 to6,000 while remaining15.2131% below the parent,
another reason to retain each metric's trajectory rather than claim uniform gains.
At7,000 structural cosines dip to0.9611236335/0.9726681536. At8,000 they rise to
0.9616520758/0.9740716027, but teacher-depth discrepancy rebounds from0.0822227726
to0.1104664014, leaving only10.3780% improvement over the parent. Both saved
endpoints have zero postprocessing failures and zero invalid depth fraction on
the317 structural cases; this is not evidence of a unique cause. Higher feature
cosine does not by itself establish improved decoded depth or flatter real walls.
Official-house64 cosine at8,000 reaches0.9746606052/0.9530785996, with teacher-depth
discrepancy0.0543011085. The source-specific results remain separate.
At9,000 and10,000 structural cosines reach0.9617647566/0.9748744953 and
0.9625065603/0.9754517086; teacher discrepancy improves to0.0979396034 and
0.0790812359. At11,000 spatial dips to0.9621463897, summary rises to0.9759054400,
and discrepancy is0.0787834318. Official-house64 at11,000 reaches
0.9746892648/0.9550570790, but its discrepancy0.0575116036 is worse than10,000's
0.0534912956. These are separate synthetic alignment results, not evidence of
flatter real-world walls or a unique explanation for earlier regression.
At the fixed final endpoint, structural317 cosines are0.9627747684/0.9765969056,
both new observed highs, while teacher-depth discrepancy rebounds to0.0877798609
(28.7837% below parent). Official-house64 cosines reach0.9750142004/0.9561273092,
with discrepancy0.0561751809 (19.9436% below parent). This confirms synthetic
feature-alignment gains in these fixed validation populations, not real-world
planarity, significance or causality.

The first parent-joint-improving checkpoint235538 is preserved by immutable
hardlink under `checkpoint-235538-parent-joint-candidate-01`, SHA
`0cf772f480da2567518f0f8fbcb5605ed408e75f7afa577e533455ae3a9fd655`.
CPU readback verified finite adapter/all100 Adam states at235538, cursor8000,
unchanged RMS, configuration, stream identities and revisions. It is a candidate,
not a deployed model. Do not repeat this checkpoint readback.

All631 online validation scalars at each of the two endpoints match the local
saved records numerically;37 equal integer/float representations per endpoint
are recorded. The first read-only online check timed out before publication:
the installed SDK pages over internal log-step ranges, so page_size4 made
hundreds of mostly empty filtered requests. Its orphaned read-only verifier was
stopped, and `code-room-epoch-review-v2` / `09bfc31` uses the SDK's1,000-step
page size. Both actual checks passed. The final online helper uses the same
paging correction; training code/configuration and metric tolerances are unchanged.
The3,000/4,000 online checks also match all631 saved scalars per endpoint exactly
in numeric value, with37 equal integer/float representations each. These proofs
are complete; no inference or repeated earlier checkpoint readback was required.
The5,000/6,000 online checks likewise match all631 scalars per endpoint, with37
equal integer/float representations each. New closure
`room-feature-epoch-five-six-reviews-closure-01/report.json` binds these saved
reviews and online checks. Earlier startup, model and checkpoint proofs were not
repeated, and the running training configuration was not changed.
The7,000/8,000 online checks also match all631 scalars per endpoint numerically,
with37 equal integer/float representations each. The new saved-review closure is
`room-feature-epoch-seven-eight-reviews-closure-01/report.json`. No model inference,
checkpoint readback, test rerun or training mutation was needed for these reviews.
The9,000/10,000/11,000 online checks match all631 scalars per endpoint numerically,
with37 equal integer/float representations each. Closure
`room-feature-epoch-nine-ten-eleven-reviews-closure-01/report.json`, SHA
`1464a974581389a8121d664900819080dd089de62aeec5140bb5bf500d0cae8f`,
binds the new saved reviews and online checks. No inference, old proof rerun or
training mutation was performed.

Final CPU readback verified finite adapter/RNG and all100 Adam states at246249,
cursor93688, unchanged RMS, configuration, model revisions and data identities.
The fixed final checkpoint is hardlinked under `finish-verification-01`, SHA
`10c12511ea3f50798ec893b99f471eb0a39516707fefd301d186cb873dffe788`.
No earlier model or checkpoint proof was repeated. All631 final online scalars
match numerically, with37 equal integer/float forms. The same two one-ULP API
baseline displays and seven equal metadata representations remain; config bytes
are unchanged. All12 endpoints therefore have7,572 verified online validation
scalars in total. Final closure
`room-feature-full-epoch-46844-finish-closure-01/report.json` has SHA
`1eca8f0a3ca9f5d1abbf934b2a32edb043bec45fcc975d0e55af465cb7eedb66`.
The existing1,000-step LiDAR candidate and all older bests remain preserved.

Existing structural batch11 has closed100 scenes354–453 with1,981 RGB,
19 rejected frames, zero duplicates and no held scenes. All1,981 encoded/pixel,
source-scene/camera/receipt checks are complete; all seven prior train/validation
isolation comparisons have zero exact overlaps. All64 selected images from16
fixed evenly spaced scenes were actually viewed on16 pages. No windows, outside
background, simulator bodies or failed magenta frames were observed in the
sample; fine beam-junction seams, coarse shadows and bright highlights remain
synthetic appearance limitations. This is a sampled review, not visual
certification of all1,981 images. The batch has1,781 perspective and200 explicitly
labelled orthographic views. Qualified available data totals48,825 images, while
the active epoch's46,844-image manifest remains unchanged. Batch11 closure is bound by
`structural-feature-train-scene354-453-closure-01/report.json`.

Batch12 and controller316684 have now exited normally. All100 requested scenes
454–553 are accounted for:99 closed scenes contain1,962 eligible RGB, while scene493
is quarantined by the existing third-invalid-frame rule. The2,000 requested slots
partition exactly into1,962 eligible,18 closed-scene rejections,14 held candidate
frames,3 quarantined-scene rejections and3 unattempted held slots. The producer's
21 total rejections include the3 in scene493; no new manual exclusion was issued.

The unchanged4833b26 freezer checks only the exact99 closed scene identities,
all1,962 encoded/pixel/source/camera/receipt identities, and eight prior
train/validation manifests. All exact pixel/group/layout overlaps are zero.
The new batch has1,764 perspective and198 labelled orthographic views. Frozen
manifest SHA`e48cec5d89d9b0d43d221cb96928ce366552e00ba8df8139f1f9cdf1d33b2e9f`.
Qualified available data now totals50,787 images;3,943 are outside this active
46,844-image epoch. No manifest was swapped and no further generation was launched.

The fixed16-scene/four-slot review selected scene507-slot11, which was already
rejected for exterior/background visibility. The original reviewer correctly
refused it before rendering any page. The new explicit diagnostic option in
`code-structural-review-v2` / `cf0ed2f` retains this exact position, checks its
frozen-report-bound rejection receipt and original image hashes, and marks it
red as rejected/not training. No substitute image is selected or admitted.
Six local and six remote tests pass, including tampered receipt/image, wrong
identity, admission-flag and default-refusal checks. Generation and training code
remain unchanged.

All16 new contact pages and all64 fixed positions were actually viewed:63 eligible
samples and one rejected diagnostic. A tiny pink leak is visible in the rejected
507-slot11; its source rejection remains unchanged. No windows, outside background,
simulator bodies or magenta failed views were observed in the63 eligible samples.
Coarse shadows, fine beam junction seams, repeated motifs and bright highlights
remain limitations; this does not certify every1,962 image or physical depth.
Scene493's held frames were not visually reviewed or rerendered. Visual review SHA
`7a014f2c57ea86ddf1ec759de6b3dab80de52f42f2a1a2d085afd1ec02db1d2f`;
batch closure SHA`a9d3e7d43cbb209d37d139b2fa23645b0f055ae8dc517aea09365c860e2cbcae`.

The preceding run, `room-feature-mix-40890-training-02`, completed
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

Nov7, authorized longer Muon continuation: the user requested a longer run from
its successful fixed-final checkpoint and asked whether papers support a larger
Muon learning rate. The literature check distinguishes shape normalization from
base LR. Moonshot's `Muon is Scalable for LLM Training`, section2.2 and appendixA
(https://arxiv.org/html/2502.16982v1), uses0.2*sqrt(max(shape)) to permit AdamW LR
reuse; this is the exact pinned Dion `rms_norm` rule already used here. It is not
an exact guarantee of matching this late-stage AdamW update magnitude. A ViT
study (https://arxiv.org/html/2605.24770v1, appendixB) selected Muon1e-3 versus
AdamW3e-4 in its main classification recipe, while also reporting successful
MAE/segmentation swaps at unchanged AdamW learning rates. A fine-tuning study
(https://arxiv.org/html/2605.10468v1, appendixE.3/table11) selected Muon5e-5 for
two CLIP datasets and1e-5 for four, versus Adam1e-5 for all six. These results
support task-specific tuning, not a universal multiplier. Raw0.02 examples use
a different scaling convention and should not be transplanted. The earlier
2e-5 experiment here used AdamW; it is not a Muon LR result. A future core-only
Muon LR sweep could compare1e-5/2e-5/5e-5 while holding auxiliary AdamW fixed.
No new LR sweep was launched or silently inserted into this continuation.

The concrete longer run keeps base LR1e-5, actual128/one forward, F1/G0 and the
same50,787 synthetic plus1,597,911 original pool. It starts from the completed
Muon263121 checkpoint, retaining all16 Muon momentum/group states (age1000)
and84 auxiliary Adam states (age263121), adapter, fixed RMS and RNG. Three NEW
complete synthetic passes use permutation seeds43/44/45 with continuing original
crop seeds and original logical cursor346864. The previous comparison's partial
16,000-image pass is preserved as history, not counted as one of these new full
passes. Each new pass takes3,175 updates with a3-synthetic/125-original tail.
Total9,525 updates end at272646, with1,219,200 actual RGB presentations:
152,361 synthetic and1,066,839 original. Each synthetic identity must occur
exactly once per pass. Normal updates retain16synthetic/112original images;
all passes keep a single resolution per actual128 forward. Data sampling is
explicitly a new three-pass stream, not an exact continuation of the unfinished
previous synthetic permutation. The constant scheduler counter resets, while
all optimizer buffers/counters are preserved. Final mixed cursor304800, expected
Muonage10525 and auxiliaryAdamage272646. Bounds are36,000s and36,600s outer;
no automatic retry, extra pass or deployment. All existing checkpoints remain.

Implementation introduces a dedicated bounded sampler and per-pass committed
coverage; reuses the existing first/tail actual128 finite-update gate with a
counter-validation hook; and adds a strict whole-hybrid-state initial checker.
The trainer change only records truthful preservation of core Muon state in fork
provenance; optimizer and feature update rules remain byte-identical. New saved
six-view reviewers, startup/online and final-checkpoint helpers use the new
budget and separate Muon/Adam ages; old1000-budget helpers are not reused.
The initial CPU fixture omitted tracking metadata; it was corrected without
weakening runtime checks. All29 new CPU tests pass locally (3.10s);28 existing
optimizer tests also pass (46 combined with the original18new tests,4.72s).
Actual CUDA/model proof is still pending at this preparation point. All prior
training/source/generation proofs are reused without rerunning them.

Nov7 06:38 UTC: all57 relevant CPU tests pass remotely (7.80s), including29new
continuation cases. Immutable `code-room-muon-continuation-v1` / `a01ea60` was
launched at06:38:11 under the original GPU lock, wrapper660849/trainer660850,
in `room-feature-muon-three-epochs-50787-training-01`. At06:38:46 it had read
122,880 of1,597,911 original metadata rows; zero retained optimizer updates were
yet verified. Actual initial full-hybrid-state and first/tail128 model proofs
remain pending and must succeed before retained training. CPU tests do not
substitute for them. Viewer3462 and all original checkpoints are preserved;
OOM counters remain zero. The user’s research question is addressed above;
the long run retains1e-5, and no higher-LR experiment was automatically launched.

Nov7 06:55 UTC: the existing longer Muon run is active at 317 retained updates
(global263438), with finite losses/gradients and one actual128 forward per
update. W&B run `f4567959` is online. The new actual initial proof now confirms
bitwise preservation of all16 Muon momentum/group states at age1000, all84
auxiliary Adam states at age263121, adapter, RMS and RNG. The two first/tail128
model cases passed finite temporary core/auxiliary updates and fully restored
weights, optimizer state and RNG before retained training. The first compilation
took31.98s; the warmed temporary optimizer update took0.01993s. These are gate
timings, not per-training-step measurements.

The new startup verifier checked the first20 successful updates and all2,560
actual RGB presentations:260 ProcTHOR,60 structural and2,240 original images.
Planned keys, original identity/crop seeds, single-bucket actual128 grouping and
F1/G0 accounting passed; measured adapter update norm was0.0272188000. All442
online training scalar values match local records numerically. Three saved
baseline metadata values differ by one floating-point ULP; seven other metadata
values and20 training values differ only as equal integer/float representations.
No runtime configuration was changed. Startup closure SHA
`5d19c328de212dd735542b8806148b23539366e4cf8c7c9c9876bccfde135905`
binds these new proofs; do not repeat them. No fixed validation was committed at
this snapshot, so there is no new quality conclusion. Next is the saved six-view
review at global263621 (+500) and its online verification. The run continues
under its original bounds; no additional run or learning-rate sweep was started.

Nov7 07:14 UTC: the first fixed continuation endpoint, +500/global263621, is
fully reviewed using the six saved validation views. Image identities, protocols,
frozen teacher values and support match the immediate Muon263121 parent. All
64,000 actual RGB keys match the planned prefix:8,000 unique synthetic images
(6,332 ProcTHOR/1,668 structural),56,000 original images and zero within-pass
repeats. All631 online validation scalar values match numerically;37 differ only
as equal integer/float representations. No model inference, checkpoint payload
read, startup proof or old result was repeated.

The primary1,012-image equal-six-dataset macro spatial cosine rises from
0.964681798627 to0.964729495150, but summary cosine falls from0.936351073495
to0.936329222366. Independent LiDAR AbsRel rises from0.095735311400
to0.096153052819 (+0.436350%); both the immediate-parent and archive joint
criteria fail at this endpoint. Against archive232288, spatial remains lower,
summary higher and LiDAR AbsRel4.379802% higher. These are fixed heldout feature
cosines, not averages over all training images; the primary macro includes both
real and synthetic datasets.

Structural317 cosine values improve to0.966693474667/0.983271918086 and its
teacher-depth discrepancy falls1.053819% versus parent. ProcTHOR64 cosine values
also rise to0.976228285581/0.962584451772, but its teacher-depth discrepancy
increases2.718799%. Outcomes are mixed; no overall, statistically significant,
causal or real-world-planarity improvement is established. Closure SHA
`9249b4f7e82bd31fc895a6c8b3ffbbbf62589d476e442fe2afc9040910861c59`
binds the complete six-view review and online check. The existing three-pass
run continues at unchanged actual128/F1/G0/LR1e-5; next fixed review is
global264121 (+1,000). Do not repeat the completed +500 or startup proofs.
