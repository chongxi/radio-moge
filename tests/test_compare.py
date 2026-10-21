from copy import deepcopy
import json

import numpy as np
from PIL import Image
import pytest
import torch

from radio_moge.compare import color_depth, compare, select_examples, verify_initialization
from radio_moge.data import read_manifest
from radio_moge.train import train
from test_training import config_for, tiny_system


def test_legacy_initialization_replay_rejects_wrong_weights(tmp_path):
    config = config_for(tmp_path)
    config["loss"].update(points=0, mask=0, scale=0, gradient=0, normal=0)
    config["output_dir"] = str(tmp_path / "run")
    state = torch.load(train(config, stop_after=1, system_adapter=tiny_system()), weights_only=True)
    system, adapter = tiny_system()
    log = tmp_path / "run/train.jsonl"
    result = verify_initialization(system, adapter, state, log, "cpu")
    assert result["verified"]
    assert max(result["absolute_difference"].values()) < 1e-7
    with torch.no_grad():
        next(adapter.parameters()).add_(1)
    with pytest.raises(ValueError, match="did not match"):
        verify_initialization(system, adapter, state, log, "cpu")


def test_depth_comparison_decodes_all_three_branches_with_one_encoder_pass(tmp_path, monkeypatch):
    config = config_for(tmp_path)
    system, before = tiny_system()
    after = deepcopy(before)
    counts = {"encode": 0, "decode": 0, "postprocess": 0}
    encode, decode = system.encode, system.decoder.forward

    def counted_encode(*args, **kwargs):
        counts["encode"] += 1
        return encode(*args, **kwargs)

    def counted_decode(*args, **kwargs):
        counts["decode"] += 1
        return decode(*args, **kwargs)

    def postprocess(raw, image, source):
        counts["postprocess"] += 1
        return {"depth": raw["points"][..., 2].exp()}

    monkeypatch.setattr(system, "encode", counted_encode)
    monkeypatch.setattr(system.decoder, "forward", counted_decode)
    monkeypatch.setattr("radio_moge.compare.postprocess", postprocess)
    records = read_manifest(config["data"]["val_manifest"])
    output = tmp_path / "comparison"
    compare(system, before, after, records, config, output, step=12, count=1)
    assert counts == {"encode": 1, "decode": 3, "postprocess": 3}
    arrays = np.load(output / "0000-depths.npz")
    np.testing.assert_array_equal(arrays["before"], arrays["after"])
    assert not np.allclose(arrays["before"], arrays["teacher"])
    assert Image.open(output / "0000.png").width == 1920
    assert len(json.loads((output / "results.json").read_text())["images"]) == 1


def test_scene_selection_and_shared_metric_color_range():
    records = [
        dict(image=str(i), scene_type=t, group=g)
        for i, (t, g) in enumerate([
            ("office", "a"), ("office", "a"), ("office", "b"),
            ("kitchen", "c"), ("bedroom", "d"),
        ])
    ]
    assert select_examples(records, 3) == [0, 3, 4]
    a = color_depth(np.array([[2., 4.], [np.nan, 0.]]), 1, 5)
    b = color_depth(np.array([[2., 5.], [-1., 9.]]), 1, 5)
    np.testing.assert_array_equal(a[0, 0], b[0, 0])
    np.testing.assert_array_equal(a[1, 0], [255, 0, 255])
    np.testing.assert_array_equal(b[1, 1], b[0, 1])
