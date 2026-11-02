"""Offline tests of the public adapter-only inference API."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from radio_moge.adapters import RadioFeatures, build_adapter
from radio_moge.config import ROOT
from radio_moge.decoder import MoGeDecoder, postprocess
from radio_moge import deployment as api

# Real pinned MoGe modules at tiny width, with no downloaded weights.
from test_upstream import moge  # noqa: F401


def tiny_adapter():
    return build_adapter(
        kind="joint_attention", spatial_dim=8, summary_dim=16,
        target_dim=32, target_summary_dim=32, initialization_seed=43,
        indices=[7, 15, 23, 31],
        spec=dict(family="joint", width=32, blocks=2, heads=4, position="rope",
                  gate=True, registers=4, activation="swiglu", local=False,
                  layer_scale=.01, qk_norm=True, summary_cleaning="per-token-zero-channel"),
    ).eval()


def features():
    return RadioFeatures(torch.randn(1, 8, 2, 3), torch.randn(1, 16),
                         tuple(torch.randn(1, 8, 2, 3) for _ in range(4)))


@pytest.fixture
def release(tmp_path, moge, monkeypatch):
    teacher, config = moge
    adapter = tiny_adapter()
    checkpoint = tmp_path / "training.pt"
    torch.save(dict(adapter=adapter.state_dict(), adapter_config=adapter.config,
                    model_config=config, revisions=api.PINS, step=232288,
                    optimizer={"secret-training-state": 123},
                    config=dict(models={"moge_source": "/private/path"},
                                train={"precision": "bf16"},
                                data={"buckets": [[32, 48]], "manifest": "/private/dataset"})),
               checkpoint)
    teacher_path = tmp_path / "teacher.pt"
    torch.save(dict(model=teacher.state_dict(), model_config=config), teacher_path)
    output = tmp_path / "release"
    api.export_adapter(checkpoint, output)
    monkeypatch.setattr(api, "hf_hub_download", lambda *args, **kwargs: teacher_path)
    monkeypatch.setattr(api, "load_radio", lambda *args, **kwargs:
                        pytest.fail("Head-only load constructed a RADIO backbone"))
    loaded = api.RadioMoGe.from_pretrained(output, device="cpu", load_backbone=False,
                                           moge_source=ROOT / "vendor/MoGe")
    return loaded, adapter, teacher, output, checkpoint


def test_adapter_only_export_has_no_training_state_or_paths(release):
    loaded, adapter, teacher, output, checkpoint = release
    state = torch.load(output / "adapter.pt", weights_only=True)
    assert state["format"] == "radio-moge-adapter"
    assert state["training_step"] == 232288
    assert "optimizer" not in state and "config" not in state
    assert "/private/" not in (output / "config.json").read_text()
    assert not any("encoder" in k or "backbone" in k for k in loaded.state_dict())
    assert loaded.radio is None
    assert all(not x.requires_grad for x in loaded.parameters())
    for key, tensor in state["adapter"].items():
        torch.testing.assert_close(tensor, adapter.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        api.export_adapter(checkpoint, output)


def test_inference_exact_parity_shapes_and_shared_feature_nonmutation(release):
    loaded, adapter, teacher, _, _ = release
    image = torch.rand(1, 3, 32, 48)
    feats = features()
    before = [x.clone() for x in (image, feats.spatial, feats.summary, *feats.intermediates)]
    with torch.inference_mode():
        expected = postprocess(MoGeDecoder(teacher)(adapter(feats), (32, 48)), image)
    actual = loaded.infer(image, features=feats)
    single = loaded.infer_from_features(feats, image[0])
    called = loaded(image, features=feats)
    assert set(actual) == {"points", "normal", "depth", "mask", "intrinsics"}
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        torch.testing.assert_close(single[key], expected[key][0], rtol=0, atol=0)
        torch.testing.assert_close(called[key], actual[key], rtol=0, atol=0)
    for tensor, snapshot in zip((image, feats.spatial, feats.summary, *feats.intermediates), before):
        assert torch.equal(tensor, snapshot)


def test_rgb_path_calls_radio_exactly_once_and_shared_features_skip_it(release, monkeypatch):
    loaded, *_ = release
    calls = []
    loaded.radio = torch.nn.Linear(1, 1)
    feats = features()
    monkeypatch.setattr(api, "radio_input", lambda image, radio: image)
    def extract(radio, image, indices):
        calls.append(indices)
        return feats
    monkeypatch.setattr(api, "extract_radio", extract)
    image = torch.rand(1, 3, 32, 48)
    from_rgb = loaded.infer(image)
    from_features = loaded.infer(image, features=feats)
    assert calls == [(7, 15, 23, 31)]
    torch.testing.assert_close(from_rgb["depth"], from_features["depth"], atol=0, rtol=0)


def test_bad_input_contracts_fail_clearly(release):
    loaded, *_ = release
    image, feats = torch.rand(1, 3, 32, 48), features()
    for bad in (image.byte(), image * 2, image[:, :1], image * float("nan")):
        with pytest.raises(ValueError):
            loaded.infer(bad, features=feats)
    with pytest.raises(ValueError, match="Head-only"):
        loaded.infer(image)
    with pytest.raises(ValueError, match="intermediate layers"):
        loaded.infer(image, features=RadioFeatures(feats.spatial, feats.summary))
    with pytest.raises(ValueError, match="summary"):
        loaded.infer(image, features=RadioFeatures(feats.spatial, feats.summary[:, :8],
                                                  feats.intermediates))


def test_preprocessing_matches_training_evaluation(tmp_path):
    from radio_moge.data import evaluation_image
    rng = np.random.default_rng(5)
    for size in ((117, 80), (53, 90), (1, 20)):
        image = Image.fromarray(rng.integers(0, 255, (*size, 3), dtype=np.uint8))
        path = tmp_path / "image.png"
        image.save(path)
        actual = api.prepare_image(image, (32, 48))
        expected, _ = evaluation_image({"image": str(path)}, bucket=(32, 48))
        torch.testing.assert_close(actual, expected[0], rtol=0, atol=0)
    with pytest.raises(ValueError):
        api.prepare_image(image, (7, 11))


def test_missing_local_pt_does_not_attempt_hub_download(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "hf_hub_download", lambda *a, **k: pytest.fail("network called"))
    with pytest.raises(FileNotFoundError):
        api.RadioMoGe.from_pretrained(tmp_path / "missing.pt")


def test_revision_mismatch_rejected_before_fetch(tmp_path, monkeypatch):
    path = tmp_path / "adapter.pt"
    torch.save(dict(format="radio-moge-adapter", format_version=1, revisions={}), path)
    monkeypatch.setattr(api, "ensure_source", lambda *a, **k: pytest.fail("source fetched"))
    with pytest.raises(ValueError, match="revisions"):
        api.RadioMoGe.from_pretrained(path)


def test_source_offline_cache_miss(tmp_path):
    with pytest.raises(FileNotFoundError, match="not cached"):
        api.ensure_source("RADIO", cache_dir=tmp_path, local_files_only=True)
