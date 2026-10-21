import importlib.util
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
from scipy.spatial.transform import Rotation
import torch

from radio_moge.data import TrainingImages, evaluation_image, oriented_rgb


scripts = Path(__file__).parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("prepare_indoor", scripts / "prepare_indoor.py")
prep = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(scripts))
spec.loader.exec_module(prep)
sys.path.pop(0)


@pytest.mark.parametrize("degrees", [0, 90, 180, 270])
def test_rgb_depth_and_training_apply_same_lossless_rotation(tmp_path, degrees):
    array = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    depth = np.arange(24, dtype=np.float32).reshape(4, 6)
    Image.fromarray(array).save(tmp_path / "rgb.png")
    np.save(tmp_path / "depth.npy", depth)
    row = dict(
        image=str(tmp_path / "rgb.png"),
        depth=str(tmp_path / "depth.npy"),
        image_rotation_cw=degrees,
    )
    expected = np.rot90(array, -degrees // 90).copy()
    assert np.array_equal(np.asarray(oriented_rgb(row)), expected)
    rgb, actual_depth = evaluation_image(row)
    assert torch.equal(actual_depth, torch.from_numpy(np.rot90(depth, -degrees // 90).copy()))
    expected_tensor = torch.from_numpy(expected).permute(2, 0, 1).float() / 255
    assert torch.equal(rgb[0], expected_tensor)
    training = TrainingImages([row], augment=False)[(0, 42, *expected.shape[:2])]
    assert torch.equal(training, expected_tensor)


def test_pose_rotation_matches_pinhole_pixel_rotation():
    # Known upright pose maps world +Z to camera -Y.
    upright = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    camera = [640, 480, 500, 510, 310, 230]
    world_point = np.array([0.3, 2.0, 0.2])
    for rotation in [0, 90, 180, 270]:
        native = prep.quarter_turn(rotation).T @ upright
        pose = [0.0, *Rotation.from_matrix(native).as_rotvec().tolist(), 0.0, 0.0, 0.0]
        row = dict(
            pose_axis_angle_translation=pose,
            camera_width_height_fx_fy_cx_cy=camera,
            width=640,
            height=480,
        )
        inferred, projection, dominance = prep.pose_orientation(row)
        assert inferred == rotation and projection > 0.9 and dominance > 0.9
        rotated = prep.orient_record(row, inferred, "test")
        p = native @ world_point
        u, v = camera[2] * p[0] / p[2] + camera[4], camera[3] * p[1] / p[2] + camera[5]
        width, height = camera[:2]
        for _ in range(rotation // 90):
            u, v, width, height = height - 1 - v, u, height, width
        q = Rotation.from_rotvec(rotated["pose_axis_angle_translation"][1:4]).apply(world_point)
        c = rotated["camera_width_height_fx_fy_cx_cy"]
        assert np.allclose([u, v], [c[2] * q[0] / q[2] + c[4], c[3] * q[1] / q[2] + c[5]])


def test_validation_sampling_balances_datasets_and_spreads_scenes():
    rows = [
        dict(dataset=ds, group=f"{ds}-{g}", image=str(i))
        for ds in ["a", "b"]
        for g in range(3)
        for i in range(10)
    ]
    selected = prep.balanced_subset(rows, 5, 42)
    assert len(selected) == 10
    assert sum(r["dataset"] == "a" for r in selected) == 5
    assert len({r["group"] for r in selected}) == 6
    assert selected == prep.balanced_subset(rows, 5, 42)
