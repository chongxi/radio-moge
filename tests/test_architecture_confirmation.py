import importlib.util
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch


scripts = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(scripts))


def load(name):
    spec = importlib.util.spec_from_file_location(name, scripts / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


confirmation = load("confirm_architecture_finalists")
selection = load("select_architecture_finalists")


def test_confirmation_crop_matches_rgb_and_gt_after_rotation(tmp_path):
    # A rectangular synthetic coordinate image makes a wrong crop/rotation visible.
    image = np.zeros((60, 120, 3), dtype=np.uint8)
    image[..., 0] = np.arange(120)[None, :]
    image[..., 1] = np.arange(60)[:, None]
    depth = np.arange(60 * 120, dtype=np.float32).reshape(60, 120) + 1
    Image.fromarray(image).save(tmp_path / "rgb.png")
    np.save(tmp_path / "depth.npy", depth)
    row = dict(
        image=str(tmp_path / "rgb.png"), depth=str(tmp_path / "depth.npy"), image_rotation_cw=90
    )
    rgb, gt = confirmation.geometry_view(row)
    rotated_rgb = Image.fromarray(image).transpose(Image.Transpose.ROTATE_270)
    box = confirmation.center_box(*rotated_rgb.size)
    expected_rgb = np.asarray(rotated_rgb.crop(box).resize((640, 480), Image.Resampling.BILINEAR))
    expected_gt = np.asarray(
        Image.fromarray(np.rot90(depth, -1)).crop(box).resize((640, 480), Image.Resampling.NEAREST)
    )
    assert torch.equal(gt, torch.from_numpy(expected_gt.copy()))
    assert np.array_equal(
        (rgb[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8), expected_rgb
    )


def test_shortlist_deduplicates_flow_nfe_without_using_confirmation():
    def row(name, spatial, summary, seconds, nfe=1):
        return dict(name=name, spatial=spatial, summary=summary, optimizer_seconds=seconds, nfe=nfe)

    rows = [
        row("legacy_2x1280", 0.90, 0.80, 50),
        row("flow", 0.91, 0.86, 55),
        row("flow", 0.92, 0.87, 55, 4),
        row("fast", 0.88, 0.83, 10),
        row("bad", 0.80, 0.70, 60),
    ]
    selected, frontier = selection.select(rows)
    assert len([r for r in selected if r["name"] == "flow"]) == 1
    assert next(r for r in selected if r["name"] == "flow")["nfe"] == 4
    assert "bad" not in {r["name"] for r in frontier}
    assert "legacy_2x1280" in {r["name"] for r in selected}


def test_geometry_subset_is_fixed_by_source_and_includes_all_nyu():
    records = [dict(dataset=source) for source in ["a", "b", "nyuv2"] for _ in range(32)]
    indices = confirmation.geometry_indices(records)
    assert indices == list(range(4)) + list(range(32, 36)) + list(range(64, 96))
