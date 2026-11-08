# PPD speedup at 640 × 480

Measured on November 8, 2026, on the same NVIDIA H200 and the same frozen
RADIO-v4-H → step-272646 adapter → released MoGe2-conditioned PPD pipeline.
Four-step compilation plus CUDA graph replay is the recommended tested option.
It retains all four sampling steps and closely matches the measured depth
accuracy. An uncompiled graph preserves the original outputs exactly.

## Measured latency

| Mode | PPD mean / P95 | RGB → relative depth mean | RGB → calibrated depth + camera mean |
|---|---:|---:|---:|
| Original, 4 steps | 128.25 / 128.53 ms | 140.24 ms | 174.01 ms |
| Cached operations + graph, 4 steps, exact output | 91.70 / 91.73 ms | 106.01 ms | 136.41 ms |
| Compiled + graph, 4 steps | 44.03 / 44.11 ms | 58.43 ms | 88.61 ms |
| Compiled + graph, 2 steps | 22.04 / 22.09 ms | 36.19 ms | 66.78 ms |
| Compiled + graph, 1 step | 11.03 / 11.06 ms | 25.23 ms | 55.21 ms |

PPD measurements begin with GPU-resident RGB, fitted features and noise, and
end at GPU relative depth. They include graph input-buffer copies and use
64 synchronized wall-time samples per mode, in two reversed-order rounds.
The previous 126.6 ms estimate was a separate stage profile that also included
noise preparation; the original implementation is remeasured above using the
same boundary as the optimized implementations.

End-to-end measurements start with decoded CPU uint8 RGB and end with CPU
depth arrays. Every request runs RADIO and the adapter again, creates its own
noise, and includes preprocessing and transfers. Each mode/output has 32
samples across 16 scenes. The calibrated branch also runs the fitted-feature
MoGe decoder, camera recovery and official CPU RANSAC; it uses no GT.
Point-cloud array generation, PLY writing, network transfer and rendering are
excluded. The raw-depth path has no camera or metric calibration.

The final compiled run warms all 16 full metric pipelines before timing.
Earlier runs graphs-02 and compiled-01 warmed model sampling but included a
cold camera initialization in their first original-metric sample; their
original-metric means are not used in the table. Their optimized modes ran
after this initialization. The final rerun preserves all of those artifacts.

## Accuracy and identity

The graph without compilation reproduces the original sampler bit for bit on
all 16 scenes, at each of 4/2/1 steps. Four-step original predictions also match
the earlier saved point-cloud experiment exactly. All compiled graph outputs
match their compiled eager outputs exactly, but compilation changes arithmetic
rounding and does not match the original outputs bit for bit.

The quality table uses the same 12 NYU scenes and noise seeds. Relative AbsRel
uses the official RANSAC alignment to GT and is a measure of relative depth
structure, not deployable metric accuracy. Gradient error is the mean absolute
horizontal/vertical difference error of log1p depth on valid adjacent pixels.

| Mode | Relative AbsRel | Gradient error change vs original 4 steps | Own-model calibrated AbsRel |
|---|---:|---:|---:|
| Original 4 steps | 0.042048 | reference | 0.075470 |
| Compiled 4 steps | 0.042045 | +1.28% | 0.075503 |
| Compiled 2 steps | 0.042342 | +49.77% | 0.075583 |
| Compiled 1 step | 0.046956 | +269.11% | 0.077681 |

The compiled four-step scene-mean raw-depth RMSE versus the original is
0.002361; the maximum pixel difference is
0.391772 in uncalibrated PPD units.
Thus near-identical aggregate quality does not establish identical pixels or
identical point clouds. Two steps preserve average AbsRel reasonably closely
in this small sample but increase gradient error by about 50%. One step is the
fastest tested path and has substantially worse local depth structure. These
12 scenes do not establish equivalence on other scenes, resolutions or GPUs.

## Changes and reproducibility

- Cache fixed-grid RoPE embeddings and timestep embeddings; pass the known
  next timestep directly. Original Q/K RoPE calls read a GPU maximum back to
  Python 192 times per four-step image. The cached path removes those reads.
- Preserve FP32 RoPE tables after autocast Q/K LayerNorm. An initial BF16
  table candidate failed parity and was rejected; graphs-01 retains that log.
- Compile the fixed-shape velocity function with PyTorch Inductor, fullgraph,
  dynamic=False, max_autotune=False; capture the fixed 4/2/1-step samplers in
  separate CUDA graphs with owned input/output buffers.
- All weights remain unchanged: FP32 parameters with BF16 autocast, MoGe
  decoder FP32. No quantization, TensorRT, retraining or inter-frame overlap.
- PyTorch 2.8.0+cu128, CUDA 12.8, driver 580.173.02; batch 1; two PyTorch CPU
  threads, one interop thread, one OpenBLAS thread, two compiler workers.
- Initial compilation took 34.94 seconds; cache reuse took
  6.85 seconds. Both are excluded from steady-state latency.
- Benchmark GPU allocations include two copies of PPD plus simultaneous
  4/2/1-step graphs and saved features; they are not the minimum serving memory.
- The existing viewer remained resident and idle. Each benchmark completed
  and exited; no existing checkpoint or viewer was changed.

Code: `src/radio_moge/ppd_fast.py`, `scripts/benchmark_ppd_speedups.py`.
The graph output buffer is overwritten by the next call and is not thread
safe. It is fixed to the constructed shape, precision and checkpoint; validate
parity again after changing any of them. The licensed upstream adaptation is
documented in `third_party/licenses/pixel-perfect-depth-APACHE-2.0.txt`.

The implementation follows the fixed-buffer constraints in the official
[PyTorch 2.8 CUDA graph documentation](https://docs.pytorch.org/docs/2.8/notes/cuda.html#cuda-graphs).
The upstream operations are pinned to
[PPD revision 427a86f](https://github.com/gangweix/pixel-perfect-depth/tree/427a86f5882aa3f7233c1b94fbdccce87f1c8313).

Evidence root:
`/shared/radio-moge/chongxi/ppd-speedup-640x480-20261108-01`.
`graphs-02/` records exact-output caching/graphs; `compiled-02/` is the final
fully warmed compiled benchmark; `summary-01.json` binds both. Raw timings,
per-scene quality, parity and predictions are retained. Remote execution uses
the same suffix under `/home/buildhub/shared/`.

These are the fastest tested configurations in this experiment, not a lower
bound on achievable latency. Quantized engines and distilled samplers were
not evaluated.
