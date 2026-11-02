---
title: RADIO adapter vs MoGe-2
emoji: 🔬
colorFrom: indigo
colorTo: teal
sdk: gradio
sdk_version: 5.49.1
python_version: '3.12'
app_file: app.py
pinned: false
models:
- chongxi/radio-moge-adapter-step232288
- Ruicheng/moge-2-vitl-normal
---

# RADIO adapter vs MoGe-2

Comparison of the **pre-GT step 232288** adapter with the frozen MoGe-2 teacher.
Both see the same processed RGB view and matched encoder token grid. Depth
visualizations share a metric color scale. No ground truth is supplied and
agreement between the models is not an accuracy metric.

Designed for Gradio + ZeroGPU (no paid hardware is requested automatically).
For a private model, configure an `HF_TOKEN` Space secret with read access to
that model. Configure `ADAPTER_REVISION` with the immutable model commit.
Keep the Space private unless the owner explicitly chooses public visibility.

The release workflow bundles `radio_moge_inference-0.1.0-py3-none-any.whl`
alongside these files. Alternatively run locally on a CUDA GPU, setting
`ADAPTER_PATH` to the local adapter.pt. Optional `RADIO_SOURCE`, `MOGE_SOURCE`,
`RADIO_CHECKPOINT` and `MOGE_CHECKPOINT` reuse existing pinned assets.

Uploaded RGBs and generated NPZ files are temporary demo data, not training data.
Gradio checks its temporary cache hourly and removes files at least one hour old.
The host administrator can
access temporary files while they exist. Do not upload sensitive photographs.
