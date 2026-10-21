import json

import numpy as np
from PIL import Image
import pytest

from radio_moge.data import check_split, evaluation_image, read_manifest
from radio_moge.nyuv2 import prepare, scene_group, split_scenes


def test_scene_split_is_deterministic_disjoint_and_close_to_80_20():
    records = [
        {"image": str(i), "group": f"room{i // 3}", "sha256": str(i)}
        for i in range(60)
    ]
    train, test = split_scenes(records)
    assert len(train) == 48 and len(test) == 12
    check_split(train, test)
    assert (train, test) == split_scenes(records)
    assert (train, test) != split_scenes(records, seed=43)
    assert scene_group("office_0001a") == scene_group("office_0001b")
    assert scene_group("office_0010") != scene_group("office_0001")


def test_matlab_orientation_units_metadata_and_atomic_publication(tmp_path):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "fixture.mat"
    n, h, w = 10, 12, 16
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, (n, h, w, 3), dtype=np.uint8)
    depth = np.broadcast_to(np.arange(h * w, dtype=np.float32).reshape(h, w) / 20, (n, h, w))
    with h5py.File(source, "w") as file:
        file.create_dataset("images", data=rgb.transpose(0, 3, 2, 1))
        file.create_dataset("rawDepths", data=depth.transpose(0, 2, 1))
        for key, values in (
            ("scenes", [f"room_{i // 2:04d}" for i in range(n)]),
            ("sceneTypes", ["room"] * n),
        ):
            refs = file.create_dataset(key, (1, n), dtype=h5py.ref_dtype)
            for i, value in enumerate(values):
                entry = file.create_dataset(f"{key}_{i}", data=[ord(c) for c in value])
                refs[0, i] = entry.ref
    output = tmp_path / "prepared"
    info = prepare(source, output)
    assert info["train"]["images"] == 8 and info["test"]["images"] == 2
    assert not list(tmp_path.glob(".prepared-*"))
    train, test = (read_manifest(output / f"{name}.jsonl") for name in ("train", "test"))
    check_split(train, test)
    for row in train + test:
        i = int(row["id"].split("_")[-1]) - 1
        np.testing.assert_array_equal(np.array(Image.open(row["image"])), rgb[i])
        image, measured = evaluation_image(row)
        assert image.shape == (1, 3, h, w)
        np.testing.assert_array_equal(measured.numpy(), depth[i])
    assert {r["id"] for r in read_manifest(output / "overfit.jsonl")} <= {r["id"] for r in train}
    assert json.loads((output / "dataset_info.json").read_text()) == info
    with pytest.raises(FileExistsError):
        prepare(source, output)


def test_duplicate_pixels_cannot_cross_split():
    records = [
        {"group": "a", "sha256": "same"}, {"group": "b", "sha256": "same"}
    ]
    with pytest.raises(ValueError, match="Duplicate RGB"):
        split_scenes(records, 0.5)
