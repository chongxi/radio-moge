"""Single-image geometry comparison; usable on ZeroGPU or an existing CUDA host."""

import os
import tempfile
from pathlib import Path
import time

# ZeroGPU must be imported before constructing CUDA modules.
try:
    import spaces
except ImportError:
    spaces = None
import gradio as gr
import numpy as np
import torch

from radio_moge.adapters import EncoderFeatures
from radio_moge.config import MOGE_REPO, MOGE_REVISION
from radio_moge.decoder import postprocess
from radio_moge.deployment import DEFAULT_REPO, RadioMoGe
from radio_moge.upstream import extract_radio, load_moge, radio_input


torch.set_num_threads(2)
MODEL = RadioMoGe.from_pretrained(
    os.environ.get("ADAPTER_PATH", DEFAULT_REPO), device="cuda",
    revision=os.environ.get("ADAPTER_REVISION"),
    radio_source=os.environ.get("RADIO_SOURCE"),
    moge_source=os.environ.get("MOGE_SOURCE"),
    radio_checkpoint=os.environ.get("RADIO_CHECKPOINT"),
    moge_checkpoint=os.environ.get("MOGE_CHECKPOINT"),
)
TEACHER, _ = load_moge(dict(moge_repo=MOGE_REPO, moge_revision=MOGE_REVISION,
                           moge_source=str(MODEL.moge_source),
                           moge_checkpoint=os.environ.get("MOGE_CHECKPOINT")), "cuda")


def gpu_function(fn):
    return spaces.GPU(duration=90)(fn) if spaces is not None else fn


def valid_depth(output):
    depth = output["depth"][0].float().cpu().numpy()
    valid = output["mask"][0].cpu().numpy() & np.isfinite(depth) & (depth > 0)
    return depth, valid


def color_depth(depth, valid, lo, hi):
    """A shared log-meter ramp, navy -> cyan -> yellow; invalid pixels are black."""
    x = np.nan_to_num(np.log(np.maximum(depth, lo)), nan=np.log(lo), posinf=np.log(hi))
    x = np.clip((x - np.log(lo)) / np.log(hi / lo), 0, 1)
    stops = np.array([[15, 23, 70], [32, 90, 160], [40, 190, 190],
                      [170, 225, 115], [255, 228, 100]], dtype=float)
    result = np.stack([np.interp(x, np.linspace(0, 1, len(stops)), stops[:, i])
                       for i in range(3)], -1).astype(np.uint8)
    result[~valid] = 0
    return result


def normal_image(output):
    normal = output["normal"][0].float().cpu().numpy()
    rgb = (np.clip(np.nan_to_num(normal) * .5 + .5, 0, 1) * 255).astype(np.uint8)
    rgb[~output["mask"][0].cpu().numpy()] = 0
    return rgb


@gpu_function
@torch.inference_mode()
def compare(photo, resolution):
    if photo is None:
        raise gr.Error("Upload an RGB image first. / 请先上传图片。")
    if photo.width * photo.height > 40_000_000:
        raise gr.Error("Please use an image smaller than 40 megapixels.")
    bucket = None if resolution == "Auto / 自动" else tuple(map(int, resolution.split("×")))
    image = MODEL.preprocess(photo, bucket)[None].to(MODEL.device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with MODEL._autocast():
        features = extract_radio(MODEL.radio, radio_input(image, MODEL.radio), MODEL.indices)
    student = MODEL.infer(image, features=features)
    torch.cuda.synchronize()
    student_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    h, w = features.spatial.shape[-2:]
    # This is the same matched-grid teacher boundary used during distillation.
    # The decoder is frozen and identical in both branches.
    with MODEL._autocast():
        spatial, summary = TEACHER.encoder(image, h, w, return_class_token=True)
        raw = MODEL.decoder(EncoderFeatures(spatial.float(), summary.float()), image.shape[-2:])
    teacher = postprocess(raw, image, MODEL.moge_source)
    torch.cuda.synchronize()
    teacher_ms = (time.perf_counter() - start) * 1000
    sd, sm = valid_depth(student)
    td, tm = valid_depth(teacher)
    values = np.concatenate([sd[sm], td[tm]])
    lo, hi = np.quantile(values, [.02, .98]) if values.size else (.1, 10)
    lo, hi = max(float(lo), 1e-3), max(float(hi), float(lo) * 1.01, 1.01e-3)
    rgb = (image[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    payload = dict(rgb=rgb)
    for prefix, output in (("adapter", student), ("moge2", teacher)):
        for key, value in output.items():
            payload[f"{prefix}_{key}"] = value[0].cpu().numpy()
    # A unique temporary output per request; Gradio moves it to its expiring cache.
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as file:
        np.savez_compressed(file, **payload)
        result_path = file.name
    common = sm & tm
    disagreement = float(np.median(np.abs(sd[common] - td[common]))) if common.any() else None
    info = dict(
        adapter_step=MODEL.metadata["training_step"], stage="before GT geometry training",
        processed_hw=list(image.shape[-2:]), matched_token_grid=[h, w],
        shared_depth_scale_m=[round(lo, 4), round(hi, 4)],
        adapter_end_to_end_ms=round(student_ms, 1), moge2_end_to_end_ms=round(teacher_ms, 1),
        adapter_valid_percent=round(float(sm.mean()) * 100, 2),
        moge2_valid_percent=round(float(tm.mean()) * 100, 2),
        median_model_disagreement_m=disagreement,
        note="No GT: disagreement is NOT accuracy. Timings include postprocessing; first run warms up.",
    )
    # Let Gradio own the output lifetime so its delete_cache policy applies.
    cached = gr.processing_utils.save_file_to_cache(result_path, demo.GRADIO_CACHE)
    Path(result_path).unlink()
    return rgb, (color_depth(sd, sm, lo, hi), color_depth(td, tm, lo, hi)), \
        normal_image(student), normal_image(teacher), info, cached


with gr.Blocks(title="RADIO × MoGe-2 · Geometry Lab", delete_cache=(3600, 3600),
               theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="teal")) as demo:
    gr.Markdown("# RADIO × MoGe-2\n### Same image. Same geometry decoder. Different encoders.\n"
                "比较 **GT 几何训练前 step 232288** 的 RADIO adapter 与 MoGe-2。"
                "上传图片后，两路使用同一视图、同一 token 网格和共享深度色标。")
    with gr.Row():
        with gr.Column(scale=1):
            photo = gr.Image(type="pil", label="RGB input / 输入图片", sources=["upload"])
            resolution = gr.Dropdown(["Auto / 自动", "480×640", "400×640", "640×480"],
                                     value="Auto / 自动", label="Processed height × width")
            run = gr.Button("Compare geometry / 开始对比", variant="primary")
            frame = gr.Image(label="Actual processed view / 实际输入视图", interactive=False)
        with gr.Column(scale=2):
            depth = gr.ImageSlider(label="Depth · adapter (left) ↔ MoGe-2 (right)",
                                   interactive=False)
            gr.Markdown("深度色标：深蓝 = 近，黄色 = 远；黑色 = 无效。两路共用色标，"
                        "具体米数范围见下方结果。拖动滑块比较。")
            with gr.Row():
                sn = gr.Image(label="Adapter normals", interactive=False)
                tn = gr.Image(label="MoGe-2 normals", interactive=False)
    with gr.Accordion("Run details & download / 结果与下载", open=True):
        info = gr.JSON(label="Run details (not a ground-truth accuracy evaluation)")
        download = gr.File(label="Download RGB + both geometry outputs (.npz)")
    gr.Markdown("**注意：** 输出对应中心裁剪后的视图。没有 ground truth，不能据此判断哪一个更准。"
                "上传与结果文件是临时数据，不用于训练；请勿上传敏感图片。")
    run.click(compare, [photo, resolution], [frame, depth, sn, tn, info, download],
              api_name="compare", concurrency_limit=1)


if __name__ == "__main__":
    demo.queue(max_size=8).launch(server_name="0.0.0.0",
                                server_port=int(os.environ.get("PORT", "7860")),
                                show_error=True)
