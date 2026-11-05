import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import render_structural_room_batch as batch


def fake_runtime(monkeypatch, bad_scenes, bad_camera=False):
    monkeypatch.setattr(batch.os, "sched_setaffinity", lambda *args: None)
    cameras = []

    class Controller:
        initialization_parameters = dict(makeAgentsVisible=False)
        last_event = SimpleNamespace(metadata=dict(lastActionSuccess=True))
        scene = 0
        slot = 0

        def reset(self, **kw):
            self.scene += 1
            self.slot = 0
            return self.last_event

        def step(self, action, **kw):
            if action not in ("AddThirdPartyCamera", "UpdateThirdPartyCamera"):
                return SimpleNamespace(metadata=dict(lastActionSuccess=True, actionReturn={}))
            slot = self.slot
            self.slot += 1
            cameras.append((self.scene, slot))
            rgb = np.random.default_rng(100 * self.scene + slot).integers(
                30, 180, (480, 640, 3), dtype=np.uint8
            )
            if self.scene in bad_scenes and slot in (1, 2, 6):
                rgb[305, 525] = [255, 0, 255]
            position = dict(kw["position"])
            if bad_camera:
                position["x"] += 1
            return SimpleNamespace(
                third_party_camera_frames=[rgb],
                metadata=dict(thirdPartyCameras=[dict(
                    position=position, rotation=kw["rotation"],
                    fieldOfView=kw["fieldOfView"],
                )]),
            )

        def stop(self):
            pass

    monkeypatch.setattr(batch, "create_controller", lambda *args, **kw: Controller())
    return cameras


def test_whole_failed_scene_has_no_manifest_rows_and_next_scene_keeps_original_slots(tmp_path, monkeypatch):
    cameras = fake_runtime(monkeypatch, {1})
    output = tmp_path / "batch"
    report = batch.run(tmp_path, output, "train", 54, 3, 20, 120, 1)
    rows = [json.loads(x) for x in (output / "rgb-candidates.jsonl").read_text().splitlines()]
    assert {r["scene_ordinal"] for r in rows} == {54, 56}
    assert len(rows) == report["completed"] == 40
    assert report["scenes_closed"] == 2 and report["scenes_quarantined"] == 1
    assert report["all_requested_scenes_accounted"] and not report["all_requested_scenes_closed"]
    assert report["next_scene"] == 57
    assert report["rejected_frames"] == 3 and report["held_candidate_frames"] == 4
    assert report["unattempted_held_slots"] == 13
    assert sum(report[k] for k in (
        "completed", "rejected_frames", "held_candidate_frames", "unattempted_held_slots"
    )) == 60
    assert cameras == ([(0, i) for i in range(20)] + [(1, i) for i in range(7)]
                       + [(2, i) for i in range(20)])
    held = json.loads((output / "quarantined-scenes/train-000055.json").read_text())
    assert held["closed"] is False and held["training_admission"] is False
    assert held["unattempted_slots"] == list(range(7, 20))
    assert len(held["candidate_rows_not_admitted"]) == 4 and len(held["rejected"]) == 3
    assert not (output / "scene-reports/train-000055.json").exists()
    for row in held["candidate_rows_not_admitted"]:
        assert batch.sha(Path(row["image"])) == row["sha256"]
    for ref in held["rejected"]:
        assert batch.sha(Path(ref["path"])) == ref["sha256"]
    for scene in (54, 56):
        receipt = json.loads((output / f"scene-reports/train-{scene:06d}.json").read_text())
        assert receipt["closed"] and receipt["attempted_slots"] == list(range(20))
        assert len(receipt["images"]) == 20


def test_quarantine_budget_stops_without_retry_or_admitting_failed_scenes(tmp_path, monkeypatch):
    cameras = fake_runtime(monkeypatch, {0, 1})
    output = tmp_path / "batch"
    with pytest.raises(RuntimeError, match="quarantine budget exceeded"):
        batch.run(tmp_path, output, "train", 54, 3, 20, 120, 1)
    assert not (output / "report.json").exists()
    assert (output / "rgb-candidates.jsonl").read_text() == ""
    assert len(list((output / "quarantined-scenes").glob("*.json"))) == 2
    assert cameras == [(0, i) for i in range(7)] + [(1, i) for i in range(7)]
    status = json.loads((output / "status.json").read_text())
    assert status["completed"] == 0 and status["next_scene"] == 56


def test_camera_contract_error_is_never_reclassified_as_scene_quarantine(tmp_path, monkeypatch):
    fake_runtime(monkeypatch, set(), bad_camera=True)
    output = tmp_path / "batch"
    with pytest.raises(ValueError, match="Camera pose changed"):
        batch.run(tmp_path, output, "train", 54, 3, 20, 120, 1)
    assert not list((output / "quarantined-scenes").glob("*.json"))
    assert (output / "rgb-candidates.jsonl").read_text() == ""


@pytest.mark.parametrize("count,views,budget", [(3, 4, 1), (3, 20, 4), (10, 20, 6), (3, 20, -1)])
def test_scene_quarantine_budget_is_explicit_and_bounded(tmp_path, count, views, budget):
    with pytest.raises(ValueError, match="whole-scene quarantines"):
        batch.run(tmp_path, tmp_path / "bad", "train", 54, count, views, 120, budget)
    assert not (tmp_path / "bad").exists()
