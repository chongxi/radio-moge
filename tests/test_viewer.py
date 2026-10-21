import gzip
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from radio_moge.viewer import export_viewer, initial_provenance, summarize, write_depths


def test_browser_depth_format_preserves_meters_invalid_values_and_plane_order(tmp_path):
    arrays = [np.array([[1.25, 80.125], [0, np.nan]], dtype=np.float32) + i for i in range(4)]
    path = tmp_path / 'sample.depth.gz'
    write_depths(path, arrays)
    # The browser's Float32Array must recover unscaled metric depths and sensor holes.
    decoded = np.frombuffer(gzip.decompress(path.read_bytes()), dtype='<f4').reshape(4, 2, 2)
    for i in range(4):
        np.testing.assert_array_equal(decoded[i], arrays[i])


def test_summary_retains_failed_predictions_in_metric_counts():
    samples = []
    for value, failed in [(0.1, False), (1.0, True)]:
        samples.append(dict(metrics={k: dict(abs_rel=value, rmse=2, delta1=0, invalid_fraction=int(failed))
                                     for k in ('before', 'after', 'teacher')},
                            failures={'after': 'focal recovery'} if failed else {}))
    result = summarize(samples)
    assert result['after']['abs_rel_count'] == 2
    assert result['after']['failures'] == 1
    assert result['after']['abs_rel'] == 0.55
    assert result['after']['invalid_fraction'] == 0.5


def test_viewer_accepts_saved_initial_weights_only_from_same_stage():
    initial = dict(step=0, cursor=0, adapter={"weight": torch.ones(3)}, adapter_config={"dim": 3},
                   config={"train": {"seed": 42}}, model_config={"encoder": "frozen"},
                   data_digests={"train": "a", "val": "b"}, revisions={"model": "revision"})
    state = deepcopy(initial)
    state.update(step=4000, cursor=8000, adapter={"weight": torch.zeros(3)})
    provenance = initial_provenance(state, initial)
    assert provenance["method"] == "saved stage-start checkpoint"
    assert provenance["trained_sha256"] != provenance["adapter_sha256"]
    artifact = {"adapter_config": initial["adapter_config"], "adapter": initial["adapter"],
                "provenance": provenance}
    assert initial_provenance(state, artifact) == provenance
    for changes in ({"step": 1}, {"data_digests": {"train": "contaminated"}},
                    {"config": {"train": {"seed": 43}}}, {"revisions": {"model": "other"}}):
        with pytest.raises(ValueError, match="same training stage"):
            initial_provenance(state, {**initial, **changes})
    artifact["adapter"]["weight"] = torch.zeros(3)
    with pytest.raises(ValueError, match="provenance"):
        initial_provenance(state, artifact)


def test_test_export_rejects_validation_leakage_before_loading_backbones(tmp_path, monkeypatch):
    paths = {}
    for name in ("train", "val"):
        image = tmp_path / f"{name}.png"
        image.touch()
        paths[name] = str(tmp_path / f"{name}.jsonl")
        (tmp_path / f"{name}.jsonl").write_text(json.dumps({"image": str(image), "group": name}))
    initial = dict(step=0, cursor=0, adapter={"weight": torch.ones(3)}, adapter_config={"dim": 3},
                   config={"data": {"train_manifest": paths["train"], "val_manifest": paths["val"]}},
                   model_config={}, data_digests={}, revisions={})
    state = {**initial, "step": 4000, "cursor": 8000}
    torch.save(initial, tmp_path / "initial.pt")
    torch.save(state, tmp_path / "best.pt")

    def forbidden(*args, **kwargs):
        raise AssertionError("Must reject leakage before loading a GPU model")

    monkeypatch.setattr("radio_moge.viewer.build_system", forbidden)
    with pytest.raises(ValueError, match="leakage"):
        export_viewer(tmp_path / "best.pt", tmp_path / "initial.pt", paths["val"], tmp_path / "view")
