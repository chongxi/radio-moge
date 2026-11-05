import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import render_structural_room_batch as batch
import freeze_structural_rgb as freeze


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def ref(path):
    return dict(path=str(path), sha256=batch.sha(path))


@pytest.fixture
def source(tmp_path, monkeypatch, request):
    monkeypatch.setattr(batch.os, "sched_setaffinity", lambda *a: None)

    class Controller:
        initialization_parameters = dict(makeAgentsVisible=False)
        last_event = SimpleNamespace(metadata=dict(lastActionSuccess=True))
        index = -1

        def step(self, action, **kw):
            if action not in ("AddThirdPartyCamera", "UpdateThirdPartyCamera"):
                return SimpleNamespace(metadata=dict(lastActionSuccess=True, actionReturn={}))
            self.index += 1
            rgb = np.broadcast_to(
                (np.arange(640) % 120 + 40).astype(np.uint8)[None, :, None], (480, 640, 3)
            ).copy()
            rgb[2, 3] = [20 + self.index, 70, 90]
            if self.index == 12 and getattr(request, "param", True):
                rgb[305, 525] = [255, 0, 255]
            return SimpleNamespace(
                third_party_camera_frames=[rgb],
                metadata=dict(
                    thirdPartyCameras=[
                        dict(
                            thirdPartyCameraId=0,
                            position=kw["position"],
                            rotation=kw["rotation"],
                            fieldOfView=kw["fieldOfView"],
                        )
                    ]
                ),
            )

        def stop(self):
            pass

    monkeypatch.setattr(batch, "create_controller", lambda *a, **k: Controller())
    root = tmp_path / "batch"
    batch.run(tmp_path, root, "val", 8, 1, 20, 60)
    spec = batch.scene_spec("val", 8)
    plan = tmp_path / "plan.json"
    write(
        plan,
        dict(
            policy=batch.POLICY,
            frame_policy=batch.FRAME_POLICY,
            preserved_validation=dict(batch="old", scene_ordinals=[]),
            scenes=[
                dict(
                    split="val",
                    ordinal=8,
                    scene_sha256=batch.canonical_sha(spec),
                    camera_plan_sha256=batch.canonical_sha(batch.camera_slots(spec, 20)),
                )
            ],
        ),
    )
    selection = tmp_path / "selection.json"
    data = dict(
        policy=batch.POLICY,
        split="val",
        generation_plan=ref(plan),
        expected_scenes=1,
        sources=[
            dict(
                batch=str(root),
                manifest=ref(root / "rgb-candidates.jsonl"),
                termination=ref(root / "report.json"),
                scene_ordinals=[8],
                frame_policy=batch.FRAME_POLICY,
            )
        ],
        isolate_from=[],
    )
    write(selection, data)
    return root, selection, tmp_path / "frozen"


def selected_rows(root):
    return [json.loads(line) for line in (root / "rgb-candidates.jsonl").read_text().splitlines()]


def set_rows(root, selection, rows):
    m = root / "rgb-candidates.jsonl"
    m.write_text("".join(json.dumps(row) + "\n" for row in rows))
    d = json.loads(selection.read_text())
    d["sources"][0]["manifest"] = ref(m)
    write(selection, d)


def test_frozen_custom_source_keeps_projection_and_exact_excluded_slots(source):
    root, selection, out = source
    report = freeze.freeze(selection, batch.sha(selection), out)
    assert report["images"] == 19 and report["scenes"] == 1
    assert report["projections"] == dict(perspective=17, orthographic=2)
    assert report["rejected_frames"] == 1 and report["geometry_admission"] is False
    rows = [json.loads(x) for x in (out / "rgb.jsonl").read_text().splitlines()]
    assert 12 not in [r["slot"] for r in rows]
    assert all(r["training_admission"] is False and r["geometry_admission"] is False for r in rows)
    assert rows[-1]["camera_model"] == "parallel" and rows[0]["camera_model"] == "pinhole"


@pytest.mark.parametrize(
    "change,match",
    [
        ("projection", "Projection"),
        ("pose", "Projection"),
        ("camera_actual", "Actual camera"),
        ("scene", "RGB source"),
        ("geometry", "Feature-only"),
        ("path", "escapes"),
    ],
)
def test_corrupt_contract_rejected_even_when_manifest_hash_is_rebound(source, change, match):
    root, selection, out = source
    rows = selected_rows(root)
    if change == "projection":
        rows[-1]["projection"] = "perspective"
    if change == "pose":
        rows[0]["requested_camera"]["position"]["x"] += 1
    if change == "camera_actual":
        rows[0]["actual_camera"]["rotation"]["y"] += 1
    if change == "scene":
        rows[0]["group"] = "structural-room-rgb-multiview-v1/train/000008"
    if change == "geometry":
        rows[0]["geometry_label"] = "depth.npy"
    if change == "path":
        rows[0]["image"] = str(root.parent / "escaped.png")
    set_rows(root, selection, rows)
    with pytest.raises(ValueError, match=match):
        freeze.freeze(selection, batch.sha(selection), out)
    assert not out.exists()


def test_missing_rejected_slot_is_not_treated_as_a_complete_camera_plan(source):
    root, selection, out = source
    p = root / "scene-reports/val-000008.json"
    r = json.loads(p.read_text())
    r["rejected"] = []
    write(p, r)
    with pytest.raises(ValueError, match="Slots missing"):
        freeze.freeze(selection, batch.sha(selection), out)


def test_unclosed_scene_is_never_promoted(source):
    root, selection, out = source
    p = root / "scene-reports/val-000008.json"
    r = json.loads(p.read_text())
    r["closed"] = False
    write(p, r)
    with pytest.raises(ValueError, match="not a closed"):
        freeze.freeze(selection, batch.sha(selection), out)


def test_pixels_are_decoded_and_background_rule_is_not_just_a_metadata_assertion(source):
    root, selection, out = source
    rows = selected_rows(root)
    row = rows[0]
    p = Path(row["image"])
    a = np.array(Image.open(p))
    a[30, 50] = [255, 0, 255]
    Image.fromarray(a).save(p)
    row["sha256"] = batch.sha(p)
    row["decoded_rgb_sha256"] = hashlib.sha256(a.tobytes()).hexdigest()
    set_rows(root, selection, rows)
    p = root / "scene-reports/val-000008.json"
    r = json.loads(p.read_text())
    r["images"][0]["sha256"] = row["sha256"]
    write(p, r)
    with pytest.raises(ValueError, match="Exterior/background"):
        freeze.freeze(selection, batch.sha(selection), out)


def test_isolation_checks_pixels_independently_of_scene_names(source):
    root, selection, out = source
    r = selected_rows(root)[0]
    p = selection.parent / "other.jsonl"
    p.write_text(
        json.dumps(dict(decoded_rgb_sha256=r["decoded_rgb_sha256"], group="unrelated-group")) + "\n"
    )
    data = json.loads(selection.read_text())
    data["isolate_from"] = [ref(p)]
    write(selection, data)
    with pytest.raises(ValueError, match="exact overlap"):
        freeze.freeze(selection, batch.sha(selection), out)


def test_selection_is_bound_and_failure_does_not_publish(source):
    root, selection, out = source
    with pytest.raises(ValueError, match="Selection hash"):
        freeze.freeze(selection, "0" * 64, out)
    assert not out.exists()


@pytest.mark.parametrize("source", [False], indirect=True)
def test_explicit_preserved_legacy_closed_prefix_remains_valid(source):
    root, selection, out = source
    rows = selected_rows(root)
    for row in rows:
        row.pop("frame_policy")
    set_rows(root, selection, rows)
    receipt_path = root / "scene-reports/val-000008.json"
    receipt = json.loads(receipt_path.read_text())
    for key in ("rejected", "attempted_slots", "frame_policy"):
        receipt.pop(key)
    write(receipt_path, receipt)
    data = json.loads(selection.read_text())
    plan_path = Path(data["generation_plan"]["path"])
    plan = json.loads(plan_path.read_text())
    plan["preserved_validation"] = dict(batch=root.name, scene_ordinals=[8])
    write(plan_path, plan)
    data["generation_plan"] = ref(plan_path)
    data["sources"][0]["frame_policy"] = None
    write(selection, data)
    report = freeze.freeze(selection, batch.sha(selection), out)
    assert report["images"] == 20 and report["rejected_frames"] == 0
    assert report["projections"] == dict(perspective=18, orthographic=2)


def test_asset_mesh_provenance_corruption_is_rejected(source):
    root, selection, out = source
    p = root / "assets/val-000008.json"
    data = json.loads(p.read_text())
    data["objects"][0]["mesh_sha256"] = "0" * 64
    write(p, data)
    with pytest.raises(ValueError, match="Evidence hash differs"):
        freeze.freeze(selection, batch.sha(selection), out)
