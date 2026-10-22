# One-day Attention data expansion

The authorized run waits for **full TartanAir, Structured3D perspective and IRS**
to finish downloading, verifying and preparing RGB. It combines those three
sources with the existing ARKitScenes/Hypersim/NYU/COCO mixture on the current
single H200. ScanNet++ approval and the unqueued ARKitScenes expansion do not
block this run. There is no automatic follow-up geometry, LoRA or architecture run.

## Readiness and data

Structured3D/IRS download completion only verifies archives. Before merging,
extract and decode all eligible RGB, record encoded/decoded hashes, freeze
scene-level partitions and write `train.jsonl`, `val.jsonl`, `test.jsonl` plus a
`dataset_info.json` with `stage: ready`, matching `counts`, and
`verification.rgb_decode_all: true`, `scope: all_eligible_rgb`. Full TartanAir
must have `selection_mode: all_left_rgb_frames`; pilot/subset data is refused.
Source extraction and its actual-archive
validation remain outstanding while downloads are queued; the merge script is
not an archive extractor.

Missing Structured3D `perspective_full_09` stays explicit; other ZIPs are
independent. Full/empty variants of a scene share one partition. IRS stereo
views, lighting and texture variants must share a scene family. Its official
frame-level test list is retained outside training and is not described as an
unseen-scene benchmark. Do not equate 76 IRS directory names with 76 independent
environments: suffix variants and related layouts require grouping first.

Run `scripts/prepare_full_attention_mix.py` with `--base`, `--tartanair`,
`--structured3d`, `--irs`, `--output`. Output must be a sibling of the base so
unchanged old held-out manifest bytes keep their relative-path meanings.
The script requires all three verified new sources, checks every path and split,
excludes exact duplicates across sources (preserved base first, then new held-out
before training), and rehashes bounded samples. It writes references to existing
RGB, rather than another full image copy. Shared assets/perceptual duplicates
are not guaranteed absent; pixel-hash checks apply where those hashes exist.

The old 884 validation images and NYU/COCO test manifests remain unchanged.
`val-selection.jsonl` adds 128 fixed, scene-balanced validation RGB from each
new source, for 1,268 images. Select on macro teacher-depth AbsRel, report all
source metrics, original-884 metrics, and NYU sensor-depth metrics separately.
Tests never select checkpoints. Final 20-image point clouds are a previously
seen qualitative audit, not a new blind test.

The predeclared mixture retains half of presentations for the old four-source
mixture in its successful 35/30/15/20 ratio; TartanAir/Structured3D/IRS get
20/15/15 percent. All eligible images enter the pools, with no new frame quota.
The weighted sampler samples with replacement, so report actual unique image
coverage separately from total presentations and nominal dataset size.

## Training and wall time

`configs/attention-full-day.json` retains the Attention architecture, frozen
RADIO/MoGe weights, RGB feature objective and 480x640/640x480 resolution. Start
from the verified Attention6000 checkpoint with `--init`, preserving adapter and
feature statistics but resetting optimizer state. Changed data is not a resume
of the old experiment. The effective batch is 32 (16 x 2 accumulation), BF16,
learning rate 3e-5, warmup 1,000 updates and a 120,000-update cosine horizon/cap.

`train.max_seconds: 86400` limits cumulative **live trainer wall time**. It includes
model/data loading, validation and checkpoint I/O; it excludes download,
preparation, preflight and downtime between resumed invocations. The trainer
finishes its current optimizer step, saves, evaluates the final checkpoint once
and saves again. Final evaluation/save may extend past the limit; no further
optimizer steps are taken. A 120,000-update finish can end the trial earlier.
The checkpoint records consumed time, so same-run `--resume` cannot reset the
budget. `--init` begins a new budget. This never shuts down the GPU machine.

Save every 1,000 updates and validate every 2,000. Read `stop-report.json` for
the actual stopping reason, elapsed time and final-evaluation status. An
interrupted checkpoint is not called a completed trial. A repeated resume after
budget exhaustion does not append duplicate validation at the same step.

`scripts/run_attention_day.py --plan PLAN --config CONFIG --training-repo REPO
--revision REVISION` checks source completion/locks, prepared-manifest digests,
preserved holdouts and the exact initialization checkpoint hash. It runs the
single-GPU preflight before starting a new online W&B run. The shared plan is
`/shared/radio-moge/chongxi/data-expansion-attention-20261022/attention-24h-plan.json`;
the GPU has the corresponding file beneath `/home/buildhub/shared`.
The runner records the new W&B URL in its status as soon as initialization creates
it. After launch, verify real optimizer progress and send that URL to the user.

The runner ends at `training_complete_export_pending`. Monitoring must then
verify/backup selected and last checkpoints, compare per-source metrics and
coverage, and publish the fixed 20 point clouds under a new path/port. Preserve
existing 7860–7866 viewers and keep the GPU running.
