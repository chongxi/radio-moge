# MoGe point clouds with the official depth-edge rule: latency

On the existing NVIDIA H200, **640×480, batch 1**, the complete
RADIO-v4-H → fitted MoGe-v2 features → MoGe decoder → camera recovery →
depth-edge filter → point-cloud pipeline averages **23.87 ms** with organized
GPU output. Compacting on GPU and delivering the final XYZ/RGB arrays to CPU
averages **25.58 ms**. The filter alone takes **0.132 ms** in the GPU implementation.

The GPU function applies the same 3×3 relative-depth rule at threshold 0.04
that was accepted in the interactive comparison. Its FP32 masks match all
256 saved case/branch/threshold combinations exactly. No PPD inference is
included in this MoGe route.

## Complete point-cloud pipeline

The final delivery comparison (`measurements-02`) has 128 individually
synchronized requests per output, four 32-frame rounds, reversed mode order
on alternate rounds, and all 16 original scenes. Outputs are ready before
the wall clock stops.

| Output after GPU edge filtering | Mean | Median | P95 | 1000 / mean |
| --- | ---: | ---: | ---: | ---: |
| Organized GPU XYZ/RGB + valid mask | 23.87 ms | 23.68 ms | 25.19 ms | 41.90 fps |
| Compact GPU XYZ/RGB | 24.25 ms | 23.81 ms | 25.05 ms | 41.24 fps |
| Compact CPU XYZ/RGB, filtered and compacted on GPU first | 25.58 ms | 24.59 ms | 26.36 ms | 39.10 fps |

Inputs are already decoded 640×480 uint8 RGB on CPU. Timings include CPU RGB
normalization, H2D transfer, RADIO, the adapter, the entire original MoGe
decoder including its normal head, per-frame focal/shift recovery, projection,
edge filtering, OpenGL coordinate conversion, and the requested compaction
and output transfers. The camera stage still includes its upstream CPU solver;
GPU output does not mean every internal operation is on GPU.

Model loading, warmup, image decoding/resizing, network transport, PLY writing
and rendering are excluded. There is no batching, frame overlap, compilation,
CUDA graph, camera caching, quantization or training. Weights are FP32, with
BF16 autocast for RADIO/adapter and FP32 decoder/geometry/filtering. Serial
fps is the reciprocal of measured mean latency, not a rendered video benchmark.

All samples are retained. The delivery comparison includes occasional long
tails: the compact-GPU maximum is 63.21 ms and CPU-delivery maximum is 107.70 ms.
P95 and median are therefore reported alongside means.

## Filtering implementation matters

The initial experiment (`measurements-01`) measures filtering alone with
already-resident depth and validity inputs. Each implementation has 256 samples.

| Filtering only | Mean | Median | P95 |
| --- | ---: | ---: | ---: |
| Official `utils3d-moge` NumPy implementation, CPU | 19.730 ms | 19.623 ms | 20.142 ms |
| SciPy implementation used for the saved visual comparison, CPU | 8.027 ms | 7.722 ms | 9.669 ms |
| Equivalent PyTorch max-pooling implementation, GPU | 0.132 ms | 0.128 ms | 0.138 ms |

The same initial experiment measures complete CPU-output pipelines:
official NumPy filtering **55.01 ms** mean / **57.14 ms** P95; SciPy filtering
**44.13 ms** mean / **46.41 ms** P95. These include CPU filtering and CPU
compaction. They are not measurements of GPU-to-CPU transfer cost alone.
The final 25.58 ms route instead filters and compacts on GPU, then copies
the final XYZ and uint8 RGB arrays.

Initial organized-GPU means were 24.32 ms without filtering and 23.62 ms with
filtering. Their medians were 23.66 and 23.57 ms. The small negative difference
is run-to-run scheduling/camera variability, not evidence that filtering
accelerates inference. The standalone filter cost and stage profiles establish
its small positive overhead. The final organized-GPU rerun is 23.87 ms.

## Breakdown

Separate 64-frame instrumented samples use consecutive CUDA events without
interstage synchronization. Intervals include CPU-induced stream idle time,
especially in camera recovery. They are not kernel-only measurements.

| Stage | Initial profile mean | Delivery-run profile mean |
| --- | ---: | ---: |
| RADIO input grid + backbone | 8.305 ms | 8.419 ms |
| Adapter | 4.341 ms | 4.418 ms |
| MoGe decoder | 6.800 ms | 6.867 ms |
| Camera recovery and projection | 3.398 ms | 4.650 ms |
| GPU edge filter | 0.206 ms | 0.210 ms |
| Validity/RGB preparation + XYZ convention | 0.110 ms | 0.112 ms |
| H2D + remaining CPU preparation/dispatch | 0.633 ms | 0.621 ms |
| Instrumented total | 23.794 ms | 25.296 ms |

Profiling records more host dispatch and camera timing variance than plain
timing; the delivery-run profile includes a 68.21 ms sample. Do not force these
independent stage means to sum to the 23.87 ms uninstrumented result. The
filter's standalone 0.132 ms and in-pipeline profile of about 0.21 ms use
different timing boundaries.

## Numerical checks and artifacts

- Both runs compare all 32 saved cases, two depth branches, and four filter
  thresholds to the previously verified masks: zero differing pixels.
- All 16 benchmark scenes reproduce saved MoGe depth, intrinsics and masks
  exactly. The same step-272646 adapter and pinned upstream models are used.
- CPU-delivered RGB is byte-exact against the saved image's retained pixels.
  FP32 GPU XYZ differs from the previous float64 NumPy unprojection by at most
  `1.7680150206089706e-6` predicted coordinate units. The depth and filter mask
  themselves are unchanged.
- Flat maps, sharp steps, borders, NaN/Inf/zero inputs, and 2D/batched masks are
  checked. Filtering never reads a tensor back to CPU or changes depth values.
- H200, driver 580.173.02, PyTorch 2.8.0+cu128. Host is Xeon Platinum 8470,
  two PyTorch threads, one interop thread and one BLAS thread. The existing
  viewer stayed resident; both benchmark processes exited and no OOM occurred.

Evidence:
`/shared/radio-moge/chongxi/moge-edge-latency-640x480-20261108-01/`.
The same suffix is mirrored under `/home/buildhub/shared/` on the GPU machine.
`measurements-01` contains the six initial routes and filter microbenchmarks;
`measurements-02` adds CPU delivery after GPU filtering with two contemporaneous
GPU-output references. Both preserve plans, raw timing rows, hardware data,
parity reports and immutable executed code snapshots (`code-01`, `code-02`).

Runtime utility: `radio_moge.depth_edges.depth_edge_keep_mask_torch`.
Benchmark: `scripts/benchmark_moge_depth_edges.py`.

```python
keep = depth_edge_keep_mask_torch(prediction["depth"], prediction["mask"], 0.04)
# Organized points stay unchanged; the consumer uses keep as the validity mask.
points = prediction["points"]
```
