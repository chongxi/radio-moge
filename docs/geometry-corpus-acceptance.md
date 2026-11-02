# Full geometry corpus: acceptance before training

Chongxi authorized offline labels for the frozen 1,637,909-image training manifest
on 2026-11-02, and training **only after** full data, depth, loss and coverage
acceptance. This supersedes the earlier unconfirmed cache-scope proposal. The
small 81-view GT pilot is not sufficient acceptance for the expanded corpus.

`configs/geometry-corpus-20261102.json` freezes the scope and teacher revision.
`scripts/index_geometry_corpus.py` creates a disk-backed inventory and checks
heldout metadata without reading evaluation depth. Its success is **not** depth
acceptance and cannot start training. Native assets remain immutable.

## Source review and unresolved checks

| Source / training images | Label candidate | Required checks before admission |
| --- | --- | --- |
| ScanNet++ DSLR / 841,431 | Laser-derived mesh render, not a direct scanner pixel | Official undistorted camera/pose, visible mesh intersection, anonymity mask, occlusion/edge support; compare against raw scanner geometry where registration permits. Existing 4,504 raw scans are not 841,431 aligned DSLR depth maps. |
| ARKitScenes / 197,542 | iPad LiDAR or available FARO-derived depth | Current RGB is `vga_wide`, not synchronized `lowres_wide`; exact timestamps, camera mapping, confidence and orientation must be checked. No blind resize or nearest-time pairing. |
| Hypersim / 59,344 | Synthetic radial range | Per-scene camera rays, range-to-camera-Z conversion, nonfinite/background mask, world-position cross-check; nonstandard tilt-shift cameras must not use a universal focal length. |
| NYUv2 / 1,043 | Projected Kinect `rawDepths` | Confirm local provenance, unfilled finite positive support, official RGB calibration/pixel origin; do not substitute inpainted `depths`. Existing 116 validation and 290 test images remain excluded. |
| TartanAir V1 / 229,924 | Synthetic left-camera depth | Match left eye/frame, meters, official calibration, finite support; distant sky-sphere values are not reliable finite scene geometry. Preserve environment-family holdouts. |
| Structured3D / 134,749 | Synthetic perspective depth | uint16 millimeters, zero invalid, FOV/camera mapping; independently distinguish radial range from camera-Z using layout planes before bulk conversion. |
| IRS / 133,879 | Synthetic disparity | Verify eye association, EXR channel/encoding, focal length and baseline before metric conversion. Never attach left disparity to right RGB. Missing metric calibration means no metric GT claim. |
| COCO2017 / 39,997 | MoGe-2 pseudo-label only | No native metric-depth annotation in this image collection; pin teacher and preprocessing, retain teacher support/quality flags and estimated-camera provenance. |

Primary documentation and papers (reviewed 2026-11-02; format ambiguities above
remain explicit rather than guessed):

- ScanNet++: [paper](https://arxiv.org/abs/2308.11417),
  [official rendering and panorama documentation](https://github.com/scannetpp/scannetpp),
  [renderer](https://github.com/scannetpp/scannetpp/blob/main/common/render.py).
- ARKitScenes: [paper](https://arxiv.org/abs/2111.08897),
  [raw sensors and synchronization](https://github.com/apple/ARKitScenes/blob/main/raw/README.md),
  [formats](https://github.com/apple/ARKitScenes/blob/main/DATA.md).
- Hypersim: [paper](https://arxiv.org/abs/2011.02523),
  [coordinates and depth](https://github.com/apple/ml-hypersim#coordinate-conventions),
  [per-scene camera matrices](https://github.com/apple/ml-hypersim/tree/main/contrib/mikeroberts3000).
- NYUv2: [dataset and toolbox](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html),
  [paper](https://cs.nyu.edu/~fergus/datasets/indoor_seg_support.pdf).
- TartanAir: [paper](https://arxiv.org/abs/2003.14338),
  [V1 data format](https://github.com/castacks/tartanair_tools/blob/master/data_type.md).
- Structured3D: [paper](https://arxiv.org/abs/1908.00222),
  [format](https://github.com/bertjiazheng/Structured3D/blob/master/data_organization.md),
  [camera utility](https://github.com/bertjiazheng/Structured3D/blob/master/misc/utils.py).
- IRS: [paper](https://arxiv.org/html/1912.09678v1),
  [release/checksums](https://github.com/blackjack2015/IRS),
  [official loader](https://github.com/blackjack2015/IRS/blob/master/dataloader/SIRSLoader.py).
- COCO: [paper](https://arxiv.org/abs/1405.0312),
  [official annotation tasks](https://cocodataset.org/#format-data).

## Publication and supervision contract

Every input image needs an outcome: verified native cache, explicitly identified
pseudo cache, or explicit rejection with a reason. Processing completion is not
the same as accepted geometry coverage. Report both counts and the remaining gap.
Unknown/uncertain native pixels stay invalid; pseudo labels do not fill native
holes under a real-GT provenance tag. Separate label kinds: `measured_sensor`,
`laser_mesh_derived`, `synthetic_renderer`, `teacher_pseudo`.

Cache keys must bind image bytes, orientation, source/version, conversion and
camera convention, masks, and teacher revision where relevant. Publish complete
shards and manifests atomically, with hashes and resume receipts. Never overwrite
a prior dataset or make incomplete shards visible to the trainer.

Depth is positive camera-Z in meters, with an explicit Boolean valid mask.
Ray-range/disparity inputs need documented conversions. Estimated teacher cameras
are not calibrated sensor cameras. Keep camera pixel-origin conventions explicit.
No interpolation across holes or depth edges may create new measured support.

The loss gate needs actual tests of masked target/prediction perturbation, finite
gradients, empty/sparse support, edge/normal stencil support, camera/crop/flip
consistency, and source-specific metric versus relative supervision. The previous
official MoGe-2 **core** is not a reproduction of its complete training recipe.

Coverage review must include datasets, independent scenes/visits/environments,
near/mid/far depth, valid-pixel support, indoor/outdoor content, and near-duplicate
sampling. A million repeated crops or adjacent views is not a million independent
scenes. Do not promote a corpus solely because its image count crosses a round
number. Maintain diverse RGB feature replay during subsequent geometry training.

No optimizer updates until all frozen gates have evidence. The final acceptance
record must reference the exact published manifest/cache and test/report hashes;
changing any input invalidates acceptance. The user has authorized training after
these gates pass; no further permission is needed for that scoped transition.
