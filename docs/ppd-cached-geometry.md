# PPD calibration reuse and GPU point clouds

The 2026-11-08 experiment reduced average CPU-RGB-to-point-cloud latency from
104.51 ms to 60.32 ms by keeping XYZ on the GPU and refreshing calibration every
10 frames. This is a 42.29% latency reduction, but calibration reuse changes
geometry: average depth disagreement with fresh per-frame calibration was 4.04%,
and the worst frame reached 36.48%. Keeping only the first calibration was much
less stable. GPU geometry alone is the change supported without temporal
calibration error; a fixed refresh interval is not yet a quality-safe default.

## Scope and timing boundary

The frozen RADIO-v4-H → adapter → PPD model uses the same compiled four-step
sampler as `ppd-speedup.md`, at 640×480 and batch 1 on the existing H200. No
training, step reduction, model replacement, or viewer deployment occurred.
FP32 weights, BF16 autocast, FP32 MoGe heads and geometry, and CPU RGB division
before H2D are preserved. A single seeded noise map is cached on the GPU and
shared across frames and all compared routes.

The input is already decoded CPU uint8 RGB. The CPU reference returns compact
CPU XYZ and RGB; GPU routes return an organized GPU XYZ map, depth, validity and
GPU RGB. GPU outputs are ready for a GPU consumer. Browser transfer, rendering,
video decoding, file writing, model loading and compilation are excluded. The
input H2D remains necessary. Calibration frames still read back depth for the
existing CPU camera recovery/RANSAC work; ordinary cached frames do not.

All compared routes use the first frame's recovered intrinsics and the same
pixel-center rays. This isolates scale reuse and CPU/GPU geometry from changes
in recovered focal length. These intrinsics are a model estimate, not a measured
camera calibration. Known camera intrinsics would be preferable. MoGe still
recovers its depth-reference camera/shift on calibration frames.

Each timing route processes the first 120 consecutive frames twice, with reversed
route order in the second round: 240 measurements per route. Sixteen full
pipeline warm-up frames precede timing. Frame-zero calibration is included in
each timed clip; the initial camera estimate and graph compilation are excluded.
These measurements replace the earlier depth-only comparison for this experiment:
the old 88.6 ms number ended at CPU depth/K, before CPU XYZ/RGB construction.

| Route | Mean ms | P50 ms | P95 ms | Calibration calls / 240 |
|---|---:|---:|---:|---:|
| Per-frame calibration → CPU XYZ/RGB | 104.510 | 104.398 | 111.219 | 240 |
| Per-frame calibration → GPU XYZ | 88.775 | 88.668 | 95.689 | 240 |
| Calibration every 10 frames → GPU | 60.318 | 57.082 | 88.005 | 24 |
| Calibration every 30 frames → GPU | 58.061 | 57.041 | 57.954 | 8 |
| First-frame calibration only → GPU | 57.295 | 57.009 | 57.219 | 2 |

The every-10 route averages 57.181 ms on ordinary frames and 88.549 ms on update
frames. It provides about 16.58 frames/s serial throughput, not 30 FPS, and has
periodic latency spikes. At every 30 frames the update occurs less often than
5% of frames, so its P95 hides the approximately 87 ms update cost.

A separate 32-frame instrumented normal-path sample gives:

| Stage | CUDA stream elapsed ms |
|---|---:|
| CPU input preparation and H2D gap | 0.555 |
| RADIO + feature adapter | 12.432 |
| Four-step compiled PPD | 44.057 |
| GPU log-depth recovery + XYZ + mask | 0.0165 |

The stream intervals include scheduling gaps; headline timings are synchronized
wall time without per-stage events. A separate 64-call geometry-only benchmark
including graph replay and final synchronization averages 0.0224 ms. Compilation
and graph setup for PPD took 7.01 seconds with the existing Inductor cache.

## Geometry and temporal checks

The replacement fits the same official PPD degree-one RANSAC objective:

```text
log(1 + reference_depth) ≈ a * relative_PPD + b
depth = clamp(exp(a * relative_PPD + b) - 1, 0.001, reference_max)
XYZ = depth * cached_pixel_ray
```

All 16 previously saved scenes reproduce the official CPU calibration result
exactly. With identical parameters, compiled GPU depth and XYZ differ from the
CPU reference by at most 0.0000038147 in predicted depth/coordinate units. Tests
also verify that CUDA graph replay observes new calibration parameters, applies
the supplied current-frame mask, resets to finite-depth validity when no mask is
supplied, and excludes nonfinite current pixels. A previous learned mask is
never silently reused.

For the temporal audit, all 360 frames of upstream's 12-second, 30 FPS
`assets/examples/video/0001.mp4` are resized from 618×464 to 640×480. Fresh and
cached calibrations are applied to exactly the same PPD map at each frame.
Cached parameters include `a`, `b`, and the upper depth clip. Reference-valid
support is shared for error calculation. There is no GT in this video: the
following numbers measure disagreement with the per-frame model-calibrated PPD
reference, not physical accuracy or video reconstruction accuracy.

| Calibration policy | Mean frame AbsRel disagreement | P95 frame disagreement | Worst frame disagreement |
|---|---:|---:|---:|
| Every 10 frames | 4.040% | 13.542% | 36.482% (frame 78) |
| Every 30 frames | 7.703% | 25.344% | 67.479% (frame 149) |
| First frame only | 40.682% | 102.495% | 147.622% (frame 166) |

Every-frame GPU geometry has only the small floating-point differences described
above. Fixed first-frame K cannot establish physical accuracy: fresh inferred
`fx / first_fx` ranges from 0.946 to 1.189 on this clip. This is a model estimate
variation, not proof that the physical lens changed.

The cached runtime uses only current finite-depth validity; it does not run the
MoGe mask head on ordinary frames. The comparison page defaults to shared
reference support so mask differences do not conceal scale error, with a toggle
to inspect the actual cached support. Calibration jumps are recorded by applying
both old and new calibration to the same update frame. The test is one clip;
neither every-10 calibration nor first-frame-only calibration is approved as a
general replacement for fresh depth scaling.

## Code and artifacts

`src/radio_moge/ppd_geometry.py` contains the calibration record, CPU fit/reference
and GPU geometry module/graph. `scripts/benchmark_ppd_cached_geometry.py` runs
the bounded comparison. `scripts/build_ppd_cached_geometry_page.py` packages
saved predictions into an offline Three.js viewer and full-resolution PLY ZIP;
`scripts/check_ppd_cached_geometry.cjs` checks the actual browser output.

Artifacts are under:

```text
/shared/radio-moge/chongxi/ppd-cached-geometry-20261108-01/
  code-01/                         # exact executed source snapshot
  launch-01.json
  console-01.log
  results-01/
    plan.json
    cpu-parity.json
    gpu-parity.json
    quality-records.json           # all 360 frames
    timing-records.json            # 1,200 full-pipeline measurements
    stage-records.json
    report.json
    snapshots/                    # 12 predeclared frames
    pointclouds-01/
      index.html                  # self-contained offline page
      full-resolution-3d.zip       # frames 59/179/359 × four methods
      manifest.json
      browser-check/
```

The compute mirror uses `/home/buildhub/shared/radio-moge/chongxi/` as its base.
The in-chat page is `.kanna/visualizations/ppd-cached-geometry-20261108.html`.
Preview point clouds use stride 4; downloadable full-resolution PLYs use all
640×480 current-reference-valid pixels. Every view shares the same coordinates
and camera; there is no per-method scale correction, smoothing or registration.
The PLY axes are x right, y up, z backward, in the model's predicted metric scale.

The experiment process exited after writing the completed report. The original
viewer PID 3462 remained the sole GPU process, with no OOM events. Existing
checkpoints, viewers and prior experiment results were preserved.

The delivered page passed actual Chromium/WebGL checks for all 12 frames × 3
cached policies, synchronized orbit/zoom, mask toggling, chart selection,
binary PLY coordinates/counts, and mobile overflow. It makes no external network
requests. The browser needed the workspace's existing shared-library and font
configuration paths; no system packages or model dependencies were changed.
