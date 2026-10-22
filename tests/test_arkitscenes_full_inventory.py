import csv
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "arkit_inventory", Path(__file__).parents[1] / "scripts/inventory_arkitscenes_full.py"
)
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


def inputs(tmp_path):
    metadata = tmp_path / "metadata.csv"
    rows = [
        ("1", "10", "Training"),
        ("2", "10", "Training"),
        ("3", "20", "Training"),
        ("4", "20", "Validation"),
        ("5", "30", "Training"),
        ("6", "NA", "Training"),
    ]
    with metadata.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["video_id", "visit_id", "fold"])
        writer.writerows(rows)
    train, heldout = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    train.write_text(
        json.dumps(
            dict(dataset="arkitscenes", video_id="1", group="arkitscenes:visit:10", split="train")
        )
        + "\n"
    )
    heldout.write_text(
        json.dumps(
            dict(dataset="arkitscenes", video_id="5", group="arkitscenes:visit:30", split="val")
        )
        + "\n"
    )
    return metadata, train, heldout


def test_all_captures_in_visit_kept_and_both_heldout_sources_respected(tmp_path):
    metadata, train, heldout = inputs(tmp_path)
    before = [p.read_bytes() for p in (metadata, train, heldout)]
    result = inventory.build_inventory(metadata, train, [heldout])
    assert [r["video_id"] for r in result["captures"]] == ["1", "2"]
    assert result["eligible_training_visits"] == 1
    assert result["unseen_training_captures"] == 1
    assert result["exclusion_counts"] == {"heldout_visit": 2, "unknown_visit": 1}
    assert before == [p.read_bytes() for p in (metadata, train, heldout)]


def test_existing_training_leakage_and_wrong_manifest_role_fail(tmp_path):
    metadata, train, heldout = inputs(tmp_path)
    row = dict(dataset="arkitscenes", video_id="3", group="arkitscenes:visit:20", split="train")
    train.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="excluded"):
        inventory.build_inventory(metadata, train, [heldout])
    with pytest.raises(ValueError, match="role"):
        inventory.build_inventory(metadata, heldout, [train])
