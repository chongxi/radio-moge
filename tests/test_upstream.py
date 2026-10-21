"""Exercises the REAL upstream modules at tiny width, without downloading weights."""

import sys

import pytest
import torch
from torch import nn

from radio_moge.adapters import EncoderFeatures, GeometryAdapter, RadioFeatures
from radio_moge.config import ROOT
from radio_moge.decoder import MoGeDecoder, postprocess
from radio_moge.export import GeometryHead, export
from radio_moge.upstream import extract_radio, import_moge

pytestmark = pytest.mark.upstream


@pytest.fixture
def moge(monkeypatch):
    if not (ROOT / "vendor/MoGe").exists():
        pytest.skip("Run scripts/setup_upstream.py --install-utils for upstream integration tests")
    pytest.importorskip("utils3d_moge")
    cls = import_moge()
    from moge.model.modules.dinov2.hub import backbones
    from moge.model.modules.dinov2.models.vision_transformer import DinoVisionTransformer

    monkeypatch.setattr(
        backbones,
        "dinov2_vitl14",
        lambda **kwargs: DinoVisionTransformer(
            img_size=28, patch_size=14, embed_dim=32, depth=4, num_heads=4, block_chunks=0
        ),
    )
    config = dict(
        encoder=dict(backbone="dinov2_vitl14", intermediate_layers=[0, 1, 2, 3], dim_out=32),
        neck=dict(
            dim_in=[34, 2, 2, 2, 2],
            dim_res_blocks=[32] * 5,
            dim_out=[32] * 5,
            resamplers=["bilinear"] * 4,
            num_res_blocks=1,
        ),
        scale_head=dict(dims=[32, 16, 1]),
        remap_output="sinh_exp",
    )
    for name, width in (("points_head", 3), ("mask_head", 1), ("normal_head", 3)):
        config[name] = dict(
            dim_in=[32] * 5,
            dim_res_blocks=[32] * 5,
            dim_out=[None] * 4 + [width],
            resamplers=["bilinear"] * 4,
            num_res_blocks=1,
        )
    model = cls(**config).eval().requires_grad_(False)
    # Ensure enough valid pixels for focal recovery with random weights.
    with torch.no_grad():
        model.mask_head.output_blocks[-1].weight.zero_()
        model.mask_head.output_blocks[-1].bias.fill_(5)
    return model, config


def test_upstream_decoder_parity_and_frozen_gradient_flow(moge):
    model, _ = moge
    image = torch.rand(1, 3, 32, 48)
    with torch.no_grad():
        expected = model(image, num_tokens=6)
        spatial, summary = model.encoder(image, 2, 3, return_class_token=True)
    encoded = EncoderFeatures(spatial.detach().requires_grad_(), summary.detach().requires_grad_())
    decoder = MoGeDecoder(model)
    actual = decoder(encoded, image.shape[-2:])
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-6)
    (actual["points"].square().mean() + actual["metric_scale"].mean()).backward()
    assert encoded.spatial.grad.abs().sum() > 0
    assert encoded.summary.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())


def test_upstream_postprocessing_parity_and_no_mutation(moge):
    model, _ = moge
    image = torch.rand(1, 3, 32, 48)
    with torch.no_grad():
        raw = model(image, num_tokens=6)
        original = raw["points"].clone()
        expected = model.infer(image, num_tokens=6, use_fp16=False, apply_mask=False)
        actual = postprocess(raw, image)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-5)
    assert torch.equal(raw["points"], original)


@pytest.mark.parametrize("cleaning", ["none", "zero-channel"])
def test_export_reload_has_no_encoder(moge, tmp_path, monkeypatch, cleaning):
    model, config = moge
    adapter = GeometryAdapter("mlp", 32, 64, 32, 32, hidden_dim=64,
                              channel_cleaning=cleaning)
    head = GeometryHead(adapter, MoGeDecoder(model)).eval()
    state = head.state_dict()
    assert not any("encoder" in key or "backbone" in key for key in state)
    moge_path = tmp_path / "moge.pt"
    torch.save({"model_config": config, "model": model.state_dict()}, moge_path)
    checkpoint = tmp_path / "training.pt"
    torch.save(
        {
            "config": {"models": {"moge_checkpoint": str(moge_path)}},
            "model_config": config,
            "adapter_config": adapter.config,
            "adapter": adapter.state_dict(),
            "revisions": {},
            "step": 10,
        },
        checkpoint,
    )
    path = tmp_path / "head.pt"
    export(checkpoint, path)
    # Loading an exported head must not construct or load a teacher encoder.
    from moge.model.modules.dinov2.hub import backbones

    def forbidden_encoder(**kwargs):
        raise AssertionError("Deployment tried to instantiate an encoder")

    monkeypatch.setattr(backbones, "dinov2_vitl14", forbidden_encoder)
    loaded = GeometryHead.load(path, device="cpu")
    assert loaded.adapter.channel_cleaning == cleaning
    assert not hasattr(loaded.decoder, "encoder")
    for key, value in state.items():
        torch.testing.assert_close(value, loaded.state_dict()[key])
    features = RadioFeatures(torch.rand(1, 32, 2, 3), torch.rand(1, 64))
    features.spatial[:, 3] = 100
    with torch.no_grad():
        expected, actual = head(features, (32, 48)), loaded(features, (32, 48))
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key])


def test_full_evaluation_reports_geometry_and_ground_truth(moge, tmp_path):
    from PIL import Image
    import numpy as np
    from test_training import TinyRadio
    from radio_moge.evaluate import evaluate
    from radio_moge.losses import FeatureStatistics
    from radio_moge.system import FrozenSystem
    from radio_moge.config import load_config

    model, model_config = moge
    image_path = tmp_path / "rgb.png"
    Image.fromarray(np.random.default_rng(3).integers(0, 255, (32, 48, 3), dtype=np.uint8)).save(
        image_path
    )
    depth_path = tmp_path / "depth.npy"
    np.save(depth_path, np.ones((32, 48), dtype=np.float32))
    config = load_config(ROOT / "configs/mlp.yaml")
    config["train"]["precision"] = "no"
    config["evaluation"].update(num_tokens=6, max_side=48)
    system = FrozenSystem(
        TinyRadio().eval().requires_grad_(False), model.encoder, MoGeDecoder(model), model_config
    )
    adapter = GeometryAdapter("mlp", 8, 8, 32, 32, hidden_dim=16)
    stats = FeatureStatistics(32, 32)
    metrics = evaluate(
        system,
        adapter,
        stats,
        [{"image": str(image_path), "depth": str(depth_path)}],
        config,
        tmp_path / "evaluation",
    )
    assert metrics["images"] == 1
    assert metrics["postprocess_failed"] == 0
    assert "gt_student/abs_rel" in metrics
    assert "gt_teacher/abs_rel" in metrics
    assert "teacher_depth/abs_rel" in metrics
    assert (tmp_path / "evaluation/0000.png").is_file()


def test_radio_intermediates_match_final_in_one_pass():
    path = ROOT / "vendor/RADIO"
    if not path.exists():
        pytest.skip("RADIO source not installed")
    sys.path.insert(0, str(path))
    from timm.models.vision_transformer import VisionTransformer
    from radio.enable_cpe_support import enable_cpe
    from radio.radio_model import RADIOModel, Resolution

    model = VisionTransformer(img_size=64, patch_size=16, embed_dim=32, depth=4, num_heads=4)
    enable_cpe(model, max_img_size=128, num_cls_tokens=2, num_registers=0, register_multiple=None)
    radio = RADIOModel(
        model,
        nn.Identity(),
        patch_size=16,
        max_resolution=128,
        preferred_resolution=Resolution(64, 64),
        summary_idxs=torch.tensor([0, 1]),
    )
    radio.eval().requires_grad_(False)
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        direct = extract_radio(radio, image)
    count = [0]

    def count_block(module, args, output):
        count[0] += 1

    handle = model.blocks[0].register_forward_hook(count_block)
    with torch.no_grad():
        multi = extract_radio(radio, image, indices=(0, 1, 2, 3))
    handle.remove()
    assert count[0] == 1
    assert len(multi.intermediates) == 4
    torch.testing.assert_close(direct.spatial, multi.spatial)
    torch.testing.assert_close(direct.summary, multi.summary)
