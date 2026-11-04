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
   the selected observations into the loss. `select_observations` currently
   drops the new view's contract field; the old loss checks centered K only.
   An ordinary crop that happens to yield centered K could pass that numeric
   check. Therefore absence of an off-center error is not admission. The current
   helper is not wired into training, and no old loss guard is removed.
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
