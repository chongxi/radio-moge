---
language:
- en
- zh
tags:
- depth-estimation
- radio
- moge
- adapter
pipeline_tag: depth-estimation
base_model:
- nvidia/C-RADIOv4-H
- Ruicheng/moge-2-vitl-normal
---

# RADIO → MoGe-2 adapter · step 232288

**GT geometry training 前的归档版本。不是 GT 1000-step checkpoint。**

This is the FP32 adapter-only export of the feature-distillation checkpoint at
step **232288**. It maps frozen C-RADIOv4-H features to the MoGe-2 encoder
interface. The frozen MoGe-2 decoder produces metric geometry. ScanNet++ RGB
was used in feature distillation; this checkpoint precedes the later stage
that adds ScanNet++ ground-truth geometry supervision.

`adapter.pt` contains adapter weights and inference metadata only. It excludes
RADIO, the MoGe encoder/decoder, optimizer state, datasets and training paths.
Upstream weights are downloaded separately at pinned revisions on first use.
The adapter alone is not a complete RGB-to-depth model.

## Install and download

Python 3.11+ and Git are required. CUDA is recommended. For a private repository,
first run `hf auth login` with an account that has access; never put a token in code.

```bash
hf download chongxi/radio-moge-adapter-step232288 \
  adapter.pt config.json radio_moge_inference-0.1.0-py3-none-any.whl \
  --local-dir radio-moge-release
pip install ./radio-moge-release/radio_moge_inference-0.1.0-py3-none-any.whl
```

The wheel uses the `radio_moge` namespace. If already developing the full
`radio-moge` project, use its source directly instead of installing both distributions.

## RGB → geometry

```python
from PIL import Image
from radio_moge.deployment import RadioMoGe

model = RadioMoGe.from_pretrained("radio-moge-release/adapter.pt", device="cuda")
# Or: RadioMoGe.from_pretrained("chongxi/radio-moge-adapter-step232288")
image = model.preprocess(Image.open("example.jpg"))  # RGB float32 CHW, 0..1
result = model.infer(image)
depth = result["depth"]            # H,W, meters
points = result["points"]          # H,W,3, camera-frame XYZ in meters
intrinsics = result["intrinsics"]  # 3,3, normalized camera intrinsics
mask = result["mask"]              # H,W bool; honor this validity mask
normal = result["normal"]          # H,W,3, unit surface normals
```

`preprocess` EXIF-transposes, center-crops and bilinearly resizes to the closest
training bucket (480×640, 640×480 or 400×640, **height×width**). Outputs describe
this processed view, not the original uncropped image. `infer` also accepts an
already-prepared BCHW tensor and retains its batch dimension. It does not crop
a tensor input. Pixel intrinsics for the output frame are
`diag([W, H, 1]) @ intrinsics`.

## Perception integration: reuse ONE existing RADIO forward

No second RADIO backbone, no MoGe encoder, and no second RADIO forward are needed:

```python
from radio_moge.deployment import RadioMoGe
from radio_moge.adapters import RadioFeatures
from radio_moge.upstream import extract_radio, radio_input

geometry = RadioMoGe.from_pretrained(
    "radio-moge-release/adapter.pt", device="cuda", load_backbone=False
)
# image: prepared RGB BCHW in [0,1], on the same CUDA device as existing_radio.
# Add these taps to the existing perception pass; do not make a duplicate pass.
import torch
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    features = extract_radio(
        existing_radio, radio_input(image, existing_radio), geometry.indices
    )
result = geometry.infer_from_features(features, image)
# Equivalent: geometry.infer(image, features=features)
```

If perception already exposes tensors, construct `RadioFeatures(spatial, summary,
tuple(intermediates))` directly. Required shapes are B×1280×h×w, B×2560 and four
B×1280×h×w taps from layers **[7, 15, 23, 31]**, in that order. Use RADIO's
`forward_intermediates(..., output_fmt="NCHW", norm=True, aggregation="sparse",
return_prefix_tokens=False, intermediates_only=False)` and the `backbone` output.
All features must come from the same view and forward pass. Do not pre-normalize,
pool, truncate, or independently resize the maps. The adapter handles its own
channel cleaning without modifying shared input features.

Existing perception code must expose those four intermediate maps; a final
spatial map alone is insufficient. This is an integration boundary, not a claim
that a private perception implementation has been modified or verified.

## Reproducibility and limitations

- `config.json` records the exact source/weight revisions and training step.
- The inference loader checks those revisions, uses `weights_only=True` for this
  artifact and constructs only MoGe decoder modules. Upstream RADIO's trusted
  official release loader uses pickle internally.
- The source checkouts are downloaded automatically to `~/.cache/radio-moge`.
  Use `RADIO_MOGE_CACHE`, explicit `radio_source`/`moge_source`, or warm the cache
  before using `local_files_only=True`. Offline use requires upstream weights too.
- CUDA uses the training BF16 adapter/RADIO path and FP32 decoder/postprocessing.
  CPU inference is a functional fallback, not a performance target.
- This is a research model; depth can be incorrect and is not safety-qualified.

Recorded feature-validation cosine similarities (not depth accuracy):

| View | Spatial cosine | Summary cosine |
| --- | ---: | ---: |
| Mixed validation, aspect preserved | 0.961692 | 0.932301 |
| Mixed validation, 640×400 | 0.960182 | 0.929616 |
| ScanNet++ DSLR RGB | 0.988800 | 0.977795 |
| ScanNet++ iPhone RGB | 0.936047 | 0.918007 |

These are the previously recorded archive results, not a new evaluation of this
packaging change. No claim is made that the adapter outperforms MoGe-2.

See `THIRD_PARTY.md`. Upstream model terms remain applicable:
[C-RADIOv4-H](https://huggingface.co/nvidia/C-RADIOv4-H) and
[MoGe-2](https://huggingface.co/Ruicheng/moge-2-vitl-normal).
No additional license grant for the adapter is implied by this private release.
