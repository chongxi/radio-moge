# H200 batch size and input preparation

This investigation keeps the completed 91,367-step model and its backups immutable.
Every benchmark starts from the same adapter, Adam moments, feature statistics,
RNG and logical sample cursor. Updates are disposable; no new training checkpoint,
evaluation or W&B run is produced.

## What is being measured

- Effective batch is the number of images per Adam update.
- Compute batch is an upper bound per forward/backward pass. The existing sampler
  chooses 480×640 or 640×480 for each logical batch of four. These shapes must be
  grouped separately, so a configured compute limit of 128 does not mean every
  forward receives 128 images.
- Prepared input uses four workers, compact uint8 DataLoader transport and a
  bounded background queue of complete, pinned update windows.
- Training throughput includes consuming the prepared window, transfer, both
  frozen encoders, adapter forward, feature losses, backward, clipping and Adam.
  It excludes one-time model loading, evaluation and checkpoint writes.
- Preparation wall time includes waiting for workers, normalization, grouping and
  pinning. Producer thread CPU time excludes workers' decoding CPU time. Consumer
  wait measures how long the training thread actually waits for a ready window.
  CUDA events are stream timings, including gaps; they are not a kernel profiler.

The question is whether increasing batch size exposes an input bottleneck. Total
preparation work can grow while GPU-visible waiting stays near zero because
preparation overlaps the preceding update. A separate resident-GPU control is
needed to test that explanation, rather than relying on queue timings alone.

## Initial batch sweep

One H200, BF16, the same real augmented image stream in every profile. Two rounds
in opposite order; 256 warmup and 1,024 measured images per profile per round.
The result below combines both rounds by total images divided by total time.

| Effective batch | Compute limit | Images/s | Gain over 32/16 | Peak allocated GiB |
|---:|---:|---:|---:|---:|
| 32 | 16 | 64.87 | — | 18.61 |
| 32 | 32 | 67.08 | 3.41% | 28.88 |
| 64 | 32 | 68.82 | 6.08% | 29.71 |
| 64 | 64 | 69.95 | 7.83% | 43.57 |
| 128 | 64 | 71.04 | 9.50% | 51.89 |
| 128 | 128 | 71.77 | 10.64% | 71.30 |
| 256 | 128 | 70.49 | 8.66% | 96.30 |

Mean consumer wait was 16–28 microseconds; measured host-to-device transfer was
about 0.55–0.67% of update time. Peak reserved memory differs from allocated
memory: for 128/128 the allocator reserved up to 111.89 GiB, of which 71.30 GiB
was allocated. Neither figure establishes GPU arithmetic utilization.

## Longer confirmation

Two opposite-order rounds, 512 warmup and 4,096 measured images each. The
pre-existing preparation implementation was retained for all three configurations.

| Effective / compute | Round 1 images/s | Round 2 images/s | Combined images/s |
|---|---:|---:|---:|
| 32 / 16 | 65.11 | 64.95 | 65.03 |
| 64 / 64 | 70.10 | 69.96 | 70.03 |
| 128 / 128 | 71.48 | 71.45 | 71.47 |

In the first round, preparing a complete window took about 74, 331 and 831 ms
respectively. The corresponding update times were 491, 913 and 1,791 ms. The
training thread still waited only 15–17 microseconds on average. Preparation is
not constant-cost, but these observations show that most of its cost is hidden.

## Input control and preparation change

The new preparation path concatenates uint8 inputs before converting each grouped
tensor into float32 directly in its pinned allocation. Previously each logical
batch was normalized first, followed by float32 concatenation and another pinning
copy. Grouping compact inputs reduces intermediate traffic without changing the
sampler, crop/flip seeds, normalization, compute shapes or optimization grouping.

CPU tests compare exact augmented pixels, model/Adam state after updates, worker
shutdown and error propagation. Timing instrumentation is opt-in, so ordinary
training does not accumulate an unbounded timing history.

The final control compares:

1. `prepared_legacy`: original normalization and concatenation order.
2. `prepared`: uint8 grouping followed by one float32 normalization/pinning step.
3. `resident_gpu`: every warmup and measured batch is fully prepared and copied
   to the GPU before the timer starts. There is no input reading, augmentation,
   grouping or host-to-device transfer in the measured loop. Cache setup time
   is reported separately. This is an input-free control for this implementation, not a production
   throughput claim; its memory figure includes the resident image cache.

All 16 profiles completed successfully. Each mode used two opposite-order rounds,
256 warmup and 1,024 measured images per round:

| Effective / compute | Original CPU preparation | Compact CPU preparation | All inputs already on GPU |
|---|---:|---:|---:|
| 32 / 16 | 64.94 images/s | 64.88 images/s | 65.30 images/s |
| 128 / 128 | 70.33 images/s | 71.69 images/s | 72.24 images/s |
| 256 / 128 | — | 70.33 images/s | 71.76 images/s |

At 128/128, removing all timed input work improved throughput by 0.76% relative to
compact preparation. Thus input waiting accounts for little of this measured
runtime. This does not establish an H200 hardware ceiling; the model's kernels,
memory traffic and execution structure may still be improved.

Compact grouping reduced mean producer work at 32/16 from 78 to 24 ms. At 128/128,
wall time fell from 956 to 601 ms and producer-thread CPU from 570 to 157 ms in
this control. Wall time includes decode-worker waits and thread scheduling;
these figures are not GPU stalls. In the compact path, consumer waits averaged
15 and 18 microseconds respectively. The first original 128/128 round was slower
than the second (69.04 versus 71.67 images/s), so the more stable longer
confirmation above is the better baseline for the batch-size comparison.

The resident control cached all 1,280 actual augmented images (4,718,592,000 bytes)
before measurement. Setup took approximately 5.6–7.1 seconds. This setup cost is
excluded only in the resident control, and the cache is included in peak memory.

## GPU double-buffer prefetch

Production training now accepts `gpu_prefetch: true` alongside
`data_pipeline: "prepared"` and `compact_transport: true` in its performance
configuration through the existing supported performance-resume route; the option
is disabled by default. No production resume was started for this investigation.
A dedicated CUDA stream copies one complete future update window
into VRAM while the current window computes. A bounded semaphore reserves the
future slot before allocation. The consumer waits on its ready event, and
`record_stream` protects tensors from allocator reuse while computation reads
them. This follows the explicit dependency and allocation-lifetime requirements
in the [PyTorch CUDA stream documentation](https://docs.pytorch.org/docs/2.9/notes/cuda.html#cuda-streams).
Update timing synchronizes the compute stream, allowing future copies to overlap. This is bounded staging, not caching the entire dataset.

Before timing, the GPU check verifies exact pixels for 128 real augmented images,
exact probe Adam parameters and moments, early cancellation and producer-error
propagation. All producer threads must stop. Historical training checkpoints are
never updated. The paired production-path measurement is recorded in
`h200-device-prefetch-control-20261029`.

The run finished with exit code zero: four configurations, two rounds in opposite
order, 256 warmup and 2,048 measured images per round. Combined results:

| Effective / compute | CPU-prepared images/s | GPU-prefetched images/s | Prefetch gain | GPU-prefetched peak allocated GiB |
|---|---:|---:|---:|---:|
| 32 / 16 | 64.94 | 65.15 | 0.32% | 18.77 |
| 128 / 128 | 71.43 | 71.88 | 0.62% | 71.86 |

All **160/160 measured GPU-prefetch windows** had completed their copy before the
consumer requested them. The exact-input/probe-Adam checks, early-close checks
and producer-error checks passed. The completed source checkpoint's file identity
was unchanged; the benchmark adapter, optimizer and RNG were restored, and no
production update was retained. Local validation comprised 24 CPU tests plus Ruff.

Exactness above applies to input pixels and the small Adam stream test. The full
BF16 model is not bitwise reproducible across these runs: comparing input modes
gave maximum absolute differences of 5.18e-5 in loss and 1.74e-4 in gradient norm.
Repeating an unchanged mode also gave differences up to 5.22e-5 and 1.69e-4,
respectively. All updates remained finite. Full-model bitwise equivalence and
accuracy improvements are not claimed.

For a next separately authorized training run, 128 effective / 128 compute limit,
four workers, compact grouping and GPU prefetch is the throughput-oriented choice
among these tests. Its measured 71.88 images/s is 10.68% above the 32/16 CPU-prepared
baseline in the same run. A 64/64 effective/compute setting is an intermediate
option: about 70.03 images/s in the longer confirmation, with lower memory and a
smaller change to optimization. Increasing effective batch to 256 did not improve
throughput in the resident control.

The next batch is now ready in VRAM. CPU preparation still performs work
proportional to the images, but that work overlapped current GPU computation in
these measurements. The small resident-control gap rules out large input stalls
in this tested window. Further gains require profiling and improving model
execution; these measurements do not show that all H200 compute resources are
fully used. No new long training run was started.

## Reproduction and artifacts

`python -m radio_moge.batch_benchmark --checkpoint CHECKPOINT --plan PLAN.json
--output NEW_RESULT_DIRECTORY` runs the measurement only. For a bounded process
with one-second GPU telemetry, use `scripts/run_batch_benchmark.py` with an
existing root containing `plan.json`, a code snapshot, Python path and checkpoint.
The controller refuses an occupied GPU or an already-started output, limits the
child's runtime, and the benchmark limits allocations to 85% of device memory.

Plans, reports, telemetry and code bundles are under
`/shared/radio-moge/chongxi/data-expansion-attention-20261022/`:

- `h200-prepared-batch-throughput-20261029` — seven-configuration sweep.
- `h200-prepared-batch-confirm-20261029` — longer three-configuration confirmation.
- `h200-prepared-input-control-20261029` — input-path and resident-GPU controls.
- `h200-device-prefetch-control-20261029` — bounded production GPU prefetch.

Changing effective batch changes the optimization trajectory. These benchmarks
establish throughput and finite updates, not convergence or generalization.
