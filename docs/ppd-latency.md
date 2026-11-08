# RADIO feature transfer to PPD: 640 × 480 latency

This is the original implementation baseline. The subsequent
[PPD speedup experiment](ppd-speedup.md) measures cached operations, CUDA graphs
and compilation with separate output-parity and quality checks.

Measured November 8, 2026 on the existing NVIDIA H200, batch 1, using the same
fixed adapter and four-step PPD predictions as the 3D comparison. The main
pipeline averages **140.21 ms per image** to relative depth. Including the
metric reference, recovered camera and RANSAC calibration used by the point
clouds averages **177.94 ms**.

## End-to-end wall time

Each row uses 64 individually synchronized measurements across 16 scenes,
after all 16 scenes have passed output parity and warmup. Two 32-frame rounds
use reversed profile order. The input is decoded, cropped 640 × 480 RGB in CPU
memory; the output is a CPU depth array. These runs contain no profiling events.

| Output | Mean | P50 | P95 | Serial frames/s from mean |
|---|---:|---:|---:|---:|
| RADIO → adapter → PPD relative depth | 140.21 ms | 140.11 ms | 141.35 ms | 7.13 |
| Same, plus own MoGe metric reference/camera and RANSAC | 177.94 ms | 174.58 ms | 206.57 ms | 5.62 |

These numbers include RGB normalization, host/device transfers, per-frame noise
creation and the four Euler updates. They exclude image file decoding/cropping,
weights loading, network transfer, file serialization and display rendering.
They describe serial batch-1 inference, not batched throughput or other GPUs.

## Breakdown

Separate CUDA-event profiles use 64 frames per output and add no barriers
between stages. Values below are means. Event intervals reflect the execution
timeline, including host-induced gaps; they are not a sum of kernel-only times.

| Relative-depth stage | Mean | Fraction of profiled total |
|---|---:|---:|
| C-RADIOv4-H, including its input grid preparation | 8.187 ms | 5.85% |
| Our joint feature adapter | 4.383 ms | 3.13% |
| PPD, four DiT calls plus noise/conditioning and Euler updates | 126.575 ms | 90.45% |
| Remaining CPU preparation, transfers and dispatch | 0.789 ms | 0.56% |
| Profiled relative-depth total | 139.934 ms | 100% |

The four DiT calls take 31.085, 31.068, 31.035 and 31.035 ms respectively.
Their total is 124.223 ms. The four Euler updates together take 1.214 ms;
PPD setup/output account for the remaining 1.138 ms.

For the metric profile, the additional MoGe decoder takes 7.162 ms, camera
recovery/postprocessing takes 12.065 ms on average (5.667 ms median), and CPU
RANSAC takes 19.626 ms. Camera recovery has a long tail: its profiled P95 is
46.139 ms. It includes the existing upstream focal/shift solve, CPU/GPU copies,
projection and source checks. These are unoptimized current implementation
costs. The original normal head is also retained in the decoder.

Metric profiling averages 180.955 ms versus 177.945 ms in the independent plain
run; its camera stage varies across repeated calls. Relative-depth profiling
differs from the plain run by only −0.195%. Do not force independently measured
stage averages to sum to the uninstrumented end-to-end number.

Generating the full-resolution RGB point-cloud arrays on the CPU after metric
depth takes another 18.084 ms on average, measured outside depth inference.
Thus the current path through CPU point-cloud arrays averages 196.028 ms
(approximately 5.10 serial frames/s), excluding PLY packing/writing and rendering.

## Configuration and parity

- H200; NVIDIA driver 580.173.02; PyTorch 2.8.0+cu128, CUDA 12.8.
- Intel Xeon Platinum 8470 host, two PyTorch CPU threads, one interop thread,
  one OpenBLAS thread. The container has a 12-core quota.
- FP32 parameters, BF16 autocast, MoGe decoder FP32. No `torch.compile`,
  TensorRT, CUDA graphs, quantization or inter-frame overlap.
- C-RADIOv4-H: 651,645,440 parameters. Adapter: 221,740,868 parameters.
  PPD DiT: 499,842,112 parameters. MoGe decoder: 22,334,504 parameters.
- RADIO is called once per frame. The MoGe teacher encoder is absent from the
  timed model; the metric branch reuses the adapter's spatial and summary output.
- Both output paths use the same released PPD weights, fixed step-272646 adapter,
  preprocessing, noise seeds and four-step sampler as the original experiment.
- Before measurement, all 16 scenes reproduce their saved PPD relative depth,
  fitted-feature MoGe depth, calibrated PPD depth, intrinsics and masks bit for bit.
- Models are frozen; zero optimizer updates. Existing checkpoints, result
  artifacts and the viewer process were preserved. The viewer was resident but
  idle at the initial GPU check; the GPU was not dedicated by stopping it.
- Model loading took 9.83 seconds and the first warmup frame 759 ms; neither
  is included in the steady-state numbers above.

## Evidence

Code: `scripts/benchmark_ppd_latency.py`.
Local evidence:
`/shared/radio-moge/chongxi/ppd-latency-640x480-20261108-01/measurements-01/`.
Remote execution root uses the same suffix under `/home/buildhub/shared/`.

- `plan.json`: fixed protocol, hashes, profiles and timing boundaries.
- `loaded.json`: hardware, precision flags and parameter counts.
- `parity.json`: 16-scene, five-output exact comparison and warmup times.
- `measurements.jsonl`: all 256 measured frames, including the two instrumented
  profiles, stage timings and separate CPU point-cloud timings.
- `report.json`: aggregate distributions, per-round results and process/GPU
  snapshots. SHA256:
  `14741ae7429f8148e3f3e175dba47705218f052bd6446f2205cd0dc730fd35fd`.

The measured bottleneck is PPD's repeated DiT inference. RADIO and the adapter
together account for about 12.57 ms of the relative-depth pipeline.
