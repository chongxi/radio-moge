# MoGe depth-edge filtering beside PPD

Added the official MoGe CLI export filter to the saved RADIO-v4-H → fitted
MoGe-v2 feature experiment. The comparison has three synchronized point clouds:
unfiltered adapter-MoGe, filtered adapter-MoGe, and adapter-PPD. PPD is unfiltered
by default; a checkbox applies the identical rule to PPD's own depth map.

Artifacts: `/shared/radio-moge/chongxi/ppd-depth-edge-filter-20261108-01`.
The offline page is `.kanna/visualizations/ppd-depth-edge-filter-20261108.html`.
The corresponding remote artifact root is under
`/home/buildhub/shared/radio-moge/chongxi/ppd-depth-edge-filter-20261108-01`.

## Rule and preserved conditions

MoGe revision `74fbce054ebed49800de42d0ad0e83495065719a`,
`moge/scripts/infer.py`, masks out
`utils3d.np.depth_map_edge(depth, rtol=threshold)` when exporting point clouds.
The CLI default in that revision is **0.04** (its help string says 0.01, but the
actual parameter default is 0.04). The Web App's separate inverse-depth
Laplacian rule is not used here.

For each pixel, the 3×3 local depth range divided by the center depth is compared
to the threshold. A point is removed if the score is strictly greater. NaN
neighbors are ignored and image boundaries use the upstream padding behavior.
The original validity mask is applied after this test, matching the CLI.
The page offers thresholds 0.01, 0.02, 0.04 and 0.08; smaller values remove more.

Both branches reuse the previously saved adapter MoGe camera and validity mask.
PPD retains its previously recovered depth scale. All 16 original scenes and
both resolutions are retained in the original order. There is no inference,
training, new calibration, registration, smoothing, plane fitting, or use of GT.
Point coordinates and RGB are unchanged for all retained MoGe vertices.

Filtering runs at full resolution before preview sampling. The preview stride
is 2 at 640×480 and 4 at 1024×768; its density selector can halve this density
along each axis again. All cameras rotate, zoom and pan together. Changing the
filter threshold does not reframe the point clouds. Framing uses the unfiltered
geometry of both branches, never independently rescales them, and does not
remove points. The red overlay displays removed points at their original 3D
positions. Downloads in the page export the current preview; separate ZIPs
contain full-resolution filtered MoGe clouds at threshold 0.04.

## Observations

Mean removed fractions across the **12 real NYUv2 scenes**, with threshold 0.04:

| Resolution | Adapter MoGe | Adapter PPD under the same filter |
| --- | ---: | ---: |
| 640×480 | 4.9957% | 5.2084% |
| 1024×768 | 3.4815% | 3.3483% |

These fractions measure the filter's selections, not the fraction of incorrect
points. Sharp legitimate geometry can also be removed. Three of the four
synthetic empty-room scenes have no points removed at 0.04. The remaining room
removes 1.7933% of MoGe versus 21.3392% of PPD at 640×480, illustrating that this
rule's removal count alone cannot establish which reconstruction is better.

Visual inspection of the first scene (`nyuv2:bedroom_0083`) from the side shows
some connecting points removed, while extended radial traces remain. This
filter detects sufficiently sharp local depth changes; it cannot reliably
recognize a smooth erroneous surface. It therefore does not demonstrate that
MoGe filtering can replace PPD. No new accuracy or latency result is claimed.

## Verification and exports

- `upstream-parity.json`: 32 cases × 2 depth branches × 4 thresholds = 256
  full-resolution mask comparisons against `utils3d-moge` revision
  `62f09d58509485564e24d5d9f6aac9ee9ebc0c37`, with **zero differing pixels**
  over 139,984,896 pixel comparisons. Flat maps, sharp depth steps, NaN/Inf/zero
  neighbors and image boundaries are checked separately.
- `report.json`: all 32 prior unfiltered MoGe PLY files are reconstructed from
  saved depth/camera/RGB and match their previously published SHA256 values.
  Every retained output vertex is byte-for-byte identical to its source vertex.
- `preview-verification.json`: all packed masks exactly match the full-resolution
  masks at preview pixels.
- `browser-check/browser-verification.json`: real WebGL in all three panels;
  all 32 cases × 4 thresholds checked for counts and retained coordinates;
  synchronized rotation/zoom, removed overlay, PPD filtering, PLY download
  coordinates/counts, mobile layout, and no external requests or page errors.
- `moge-filtered-640x480.zip`: 16 full-resolution PLYs, 60,681,667 bytes.
- `moge-filtered-1024x768.zip`: 16 full-resolution PLYs, 154,754,276 bytes.

The original, unfiltered MoGe and PPD full-resolution exports remain in the
previous `ppd-feature-swap-20261107-01/pointclouds-02` artifact directory.

To reproduce the CPU export and page in a new output directory:

```sh
PYTHONPATH=src .venv/bin/python scripts/export_ppd_depth_edges.py \
  --source /shared/radio-moge/chongxi/ppd-feature-swap-20261107-01 \
  --output OUTPUT
PYTHONPATH=src .venv/bin/python scripts/build_ppd_depth_edge_page.py \
  OUTPUT .kanna/visualizations/ppd-depth-edge-filter-20261108.html
```

The independent parity script requires the pinned `utils3d-moge` dependency.
`scripts/check_ppd_depth_edges.cjs` runs the offline browser verification.
