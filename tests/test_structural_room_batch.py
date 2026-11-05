import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import render_structural_room_batch as batch


def test_scene_split_seed_is_stable_and_never_reuses_layout():
    hashes = set()
    for split in ("train", "val"):
        for i in range(100):
            spec = batch.scene_spec(split, i)
            assert spec == batch.scene_spec(split, i)
            assert spec["layout_sha256"] not in hashes
            hashes.add(spec["layout_sha256"])
            assert spec["split"] == split and spec["identity"].split("/")[1] == split
    assert len(hashes) == 200


def test_all_cameras_inside_room_with_multiple_perspective_angles_and_minor_ortho_share():
    for split in ("train", "val"):
        for i in range(128):
            spec = batch.scene_spec(split, i)
            assert not spec["house"]["windows"] and not spec["house"]["doors"]
            assert len(spec["house"]["walls"]) == 4
            keys, angles = set(), []
            ortho_count = 0
            for slot in batch.camera_slots(spec, 20):
                r = slot["request"]
                p = r["position"]
                assert 0 < p["x"] < spec["width"] and 0 < p["y"] < spec["height"]
                assert 0 < p["z"] < spec["depth"]
                if r["orthographic"]:
                    ortho_count += 1
                    scale = r["orthographicSize"]
                    assert r["rotation"] == dict(x=0.0, y=0.0, z=0.0)
                    assert 0 < p["y"] - scale < p["y"] + scale < spec["height"]
                    assert 0 < p["x"] - scale * 4 / 3 < p["x"] + scale * 4 / 3 < spec["width"]
                else:
                    assert "orthographicSize" not in r
                    assert 32 <= r["fieldOfView"] <= 58
                    angles.append((r["rotation"]["y"] + 180) % 360 - 180)
                keys.add(batch.canonical_sha(r))
            assert len(keys) == 20 and ortho_count == 2
            assert min(angles) < -15 and max(angles) > 15
            assert len({x["request"]["position"]["y"] for x in batch.camera_slots(spec, 20)}) > 4


def test_structures_grounded_inside_room_and_members_connected():
    for i in range(32):
        spec = batch.scene_spec("train", i)
        pieces = {p["name"]: p for p in spec["structures"]}
        for p in pieces.values():
            mesh = batch.mesh_for(p)
            xyz = np.array([list(v.values()) for v in mesh["vertices"]]) + p["center"]
            assert (xyz >= (0, -1e-9, 0)).all()
            assert (xyz <= (spec["width"], spec["height"] + 1e-9, spec["depth"])).all()
        left, right, rail, beam = [
            pieces[k]
            for k in (
                "square_column",
                "round_column",
                "horizontal_round_member",
                "rectangular_beam",
            )
        ]
        assert left["center"][1] == right["center"][1] == spec["height"] / 2
        assert rail["center"][0] - rail["length"] / 2 == pytest.approx(left["center"][0])
        assert rail["center"][0] + rail["length"] / 2 == pytest.approx(right["center"][0])
        assert beam["center"][0] - beam["size"][0] / 2 < left["center"][0]
        assert beam["center"][0] + beam["size"][0] / 2 > right["center"][0]


@pytest.mark.parametrize("split,i", [("test", 0), ("train", -1), ("val", 100000)])
def test_invalid_scene_identity_rejected(split, i):
    with pytest.raises(ValueError):
        batch.scene_spec(split, i)


def test_uniform_and_background_fail_closed():
    with pytest.raises(ValueError):
        batch.validate_rgb(np.full((480, 640, 3), 90, np.uint8))
    rgb = np.random.default_rng(1).integers(30, 180, (480, 640, 3), dtype=np.uint8)
    assert batch.validate_rgb(rgb) > 1
    rgb[100, 100] = [255, 0, 255]
    with pytest.raises(ValueError):
        batch.validate_rgb(rgb)


def test_actual_render_loop_preserves_projection_labels_and_closes_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(batch.os, "sched_setaffinity", lambda *args: None)
    actions = []

    class Controller:
        initialization_parameters = dict(makeAgentsVisible=False)
        last_event = SimpleNamespace(metadata=dict(lastActionSuccess=True))

        def step(self, action, **kw):
            actions.append((action, kw))
            if action in ("AddThirdPartyCamera", "UpdateThirdPartyCamera"):
                rgb = np.random.default_rng(len(actions)).integers(
                    30, 180, (480, 640, 3), dtype=np.uint8
                )
                return SimpleNamespace(
                    third_party_camera_frames=[rgb],
                    metadata=dict(
                        thirdPartyCameras=[
                            dict(
                                position=kw["position"],
                                rotation=kw["rotation"],
                                fieldOfView=kw["fieldOfView"],
                            )
                        ]
                    ),
                )
            return SimpleNamespace(metadata=dict(lastActionSuccess=True, actionReturn={}))

        def stop(self):
            pass

    monkeypatch.setattr(batch, "create_controller", lambda *args, **kw: Controller())
    result = batch.run(tmp_path, tmp_path / "run", "train", 0, 1, 4, 60)
    rows = [json.loads(x) for x in (tmp_path / "run/rgb-candidates.jsonl").read_text().splitlines()]
    assert result["all_requested_scenes_closed"] and result["completed"] == 4
    assert [r["projection"] for r in rows] == ["perspective"] * 3 + ["orthographic"]
    assert result["perspective_frames"] == 3 and result["orthographic_frames"] == 1
    assert rows[-1]["camera_model"] == "parallel" and rows[0]["camera_model"] == "pinhole"
    assert all(r["training_admission"] is False and r["geometry_label"] is None for r in rows)
    assert sum(a == "AddThirdPartyCamera" for a, k in actions) == 1
    assert sum(a == "UpdateThirdPartyCamera" for a, k in actions) == 3
    assert json.loads((tmp_path / "run/scene-reports/train-000000.json").read_text())["closed"]


@pytest.mark.parametrize("bad_slots,success", [({12}, True), ({12, 13, 14}, False)])
def test_rejected_entire_frames_are_bound_and_never_replaced(tmp_path, monkeypatch, bad_slots, success):
    monkeypatch.setattr(batch.os, "sched_setaffinity", lambda *args: None)
    cameras = []

    class Controller:
        initialization_parameters = dict(makeAgentsVisible=False)
        last_event = SimpleNamespace(metadata=dict(lastActionSuccess=True))

        def step(self, action, **kw):
            if action in ("AddThirdPartyCamera", "UpdateThirdPartyCamera"):
                slot = len(cameras)
                cameras.append(kw)
                rgb = np.random.default_rng(slot).integers(30, 180, (480, 640, 3), dtype=np.uint8)
                if slot in bad_slots:
                    rgb[305, 525] = [255, 0, 255]
                return SimpleNamespace(third_party_camera_frames=[rgb], metadata=dict(
                    thirdPartyCameras=[dict(position=kw["position"], rotation=kw["rotation"])],
                ))
            return SimpleNamespace(metadata=dict(lastActionSuccess=True, actionReturn={}))

        def stop(self):
            pass

    monkeypatch.setattr(batch, "create_controller", lambda *a, **k: Controller())
    output = tmp_path / "run"
    if not success:
        with pytest.raises(RuntimeError, match="Excess rejected"):
            batch.run(tmp_path, output, "val", 7, 1, 20, 60)
        assert not (output / "report.json").exists()
        assert not (output / "scene-reports/val-000007.json").exists()
        assert len(cameras) == 15
        return
    report = batch.run(tmp_path, output, "val", 7, 1, 20, 60)
    rows = [json.loads(x) for x in (output / "rgb-candidates.jsonl").read_text().splitlines()]
    receipt = json.loads((output / "scene-reports/val-000007.json").read_text())
    assert len(cameras) == 20  # No replacement or repeated camera.
    assert [r["slot"] for r in rows] == [i for i in range(20) if i != 12]
    assert report["completed"] == 19 and report["rejected_frames"] == 1
    assert receipt["attempted_slots"] == list(range(20)) and len(receipt["rejected"]) == 1
    proof = receipt["rejected"][0]
    assert batch.sha(Path(proof["path"])) == proof["sha256"]
    rejection = json.loads(Path(proof["path"]).read_text())
    assert rejection["slot"] == 12 and rejection["training_admission"] is False
    assert batch.sha(Path(rejection["image"]["path"])) == rejection["image"]["sha256"]
    assert all(r["exterior_background_pixels"] == 0 for r in rows)
