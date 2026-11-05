# Empty-room RGB for MoGe-v2 feature alignment

Chongxi requested large numbers of clean empty-room, first-person RGB views.
Further ScanNet++ source/geometry investigations were explicitly stopped.
This generation task uses ProcTHOR only; ZInD and UnrealZoo are not inputs.
No synthetic depth labels or geometric loss are introduced.

## Sources and renderer

The official ProcTHOR-10K training source is pinned to repository revision
`439193522244720b86d8c81cde2e51e3a4d150cf`:10,000 houses, archive SHA
`ee3c4aa14b4d8f0895fecfb5fdaca59395427ca1018b2f9aeeedbc61e5824587`.
Only the top-level objects list is cleared. Structural walls, floors, doors,
windows, materials and lighting remain. Columns/railings coverage is not established.
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

## Active bounded batch

Remote root:
`/home/buildhub/shared/radio-moge/chongxi/new-architecture-rgb-20261104`.
Local reports mirror under `/shared/radio-moge/chongxi/new-architecture-rgb-20261104`.

`procthor-batch02-controller` runs immutable`code-procthor-empty-batch-v2` /
`774c84d`; output`procthor-empty-rgb-batch-02`. It targets the first500 official
training houses and up to20 distinct RGB views each, with a7200s generation
budget and7500s outer bound. Do not start another copy. Four camera-placement
tests passed locally/remotely, including concave and narrow rooms and60 unique
single-room camera slots. Actual image review follows as new images appear.

Cameras have a0.5m floor-boundary margin and distributed positions across rooms;
Unity verifies each teleport. Perspective vertical FOV cycles55/70/85 degrees,
with pitch-10/0/10. Failed teleports, near-uniform images and exact duplicate
pixels are recorded and omitted. Every retained row binds house/source hashes,
actual camera pose, projection, source split and hidden-agent rendering.
All rows remain training candidates until the batch is reviewed.

Read live`status.json` and per-house reports. `STOP_AFTER_HOUSE` requests a
graceful stop. A finished report gives the actual image count and next house;
do not equate requested capacity with completed images.

Future work: use distinct official validation houses, inspect MoGe-v2 and adapter
on the generated RGB, and test a declared synthetic share alongside real RGB.
Keep feature weights1 and the restored augmentation/batch/optimizer conventions;
do not silently change multiple settings or the current completed training run.
Improved teacher alignment and flatter real-world predictions remain hypotheses.
