import importlib.util
import json
from pathlib import Path
import sys

import pytest

from radio_moge.config import ROOT, load_config
from test_full_attention_mix import prep, sources


sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("day", ROOT / "scripts/run_attention_day.py")
day = importlib.util.module_from_spec(spec)
spec.loader.exec_module(day)


def prepared_plan(tmp_path):
    base, new = sources(tmp_path)
    output = tmp_path / "full"
    prep.prepare(base, new, output, validation_per_source=1)
    config = load_config(ROOT / "configs/attention-full-day.json")
    config["data"].update(
        train_manifest=str(output / "train.jsonl"), val_manifest=str(output / "val-selection.jsonl")
    )
    for name, folder in new.items():
        if name == "tartanair":
            continue
        completed = (
            (
                [
                    dict(name=f"Structured3D_perspective_{kind}_{i:02d}.zip")
                    for kind in ("full", "empty")
                    for i in range(18)
                    if (kind, i) != ("full", 9)
                ]
                + [dict(name=f"Structured3D_{n}.zip") for n in ("bbox", "annotation_3d")]
            )
            if name == "structured3d"
            else [
                "Store.tar.gz",
                "Home-1.tar.gz",
                "Home-2.tar.gz",
                "Office-1.tar.gz",
                "Office-2.tar.gz",
                "Restaurant.tar.gz",
            ]
        )
        (folder / "status.json").write_text(
            json.dumps(
                dict(
                    stage="archives_complete",
                    completed=completed,
                    missing=["Structured3D_perspective_full_09.zip"]
                    if name == "structured3d"
                    else [],
                )
            )
        )
    initial = tmp_path / "initial.pt"
    initial.write_bytes(b"synthetic initialization for readiness-only tests")
    plan = dict(
        authorized=True,
        selection_images=4,
        prepared_dataset=str(output),
        base_dataset=str(base),
        download_roots={n: str(p) for n, p in new.items()},
        initial_checkpoint=str(initial),
        initial_sha256=prep.digest(initial),
    )
    return plan, config, new


def test_training_gate_requires_all_downloads_and_unchanged_prepared_manifests(tmp_path):
    plan, config, new = prepared_plan(tmp_path)
    assert day.ready(plan, config)["train_images"] == 7
    path = new["irs"] / "status.json"
    saved = path.read_text()
    status = json.loads(saved)
    status["stage"] = "waiting_for_preceding_dataset"
    path.write_text(json.dumps(status))
    with pytest.raises(ValueError, match="download is not complete"):
        day.ready(plan, config)
    path.write_text(saved)
    manifest = Path(config["data"]["val_manifest"])
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest changed"):
        day.ready(plan, config)


def test_training_gate_blocks_still_running_download_and_changed_initial_weights(tmp_path):
    import fcntl

    plan, config, new = prepared_plan(tmp_path)
    with (new["tartanair"] / ".download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            day.ready(plan, config)
    Path(plan["initial_checkpoint"]).write_bytes(b"different weights")
    with pytest.raises(ValueError, match="checkpoint changed"):
        day.ready(plan, config)
