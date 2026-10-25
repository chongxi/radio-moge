import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "backup_verifier", Path(__file__).parents[1] / "scripts/verify_attention_dataset_backup.py"
)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def fixture(root, images=None):
    images = images or ["rgb/scene/a.png", "rgb/scene/b.png"]
    rows = []
    for image in images:
        path = root / image
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(image.encode())
        rows.append(
            dict(
                image=image,
                sha256_kind="encoded_file_bytes",
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    manifest = root / "train.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    pins = {
        "counts": {"train": len(rows)},
        "metadata_files": {
            "train.jsonl": {
                "bytes": manifest.stat().st_size,
                "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            }
        },
    }
    return pins


def test_verifies_every_rgb_and_exact_inventory(tmp_path):
    pins = fixture(tmp_path)
    result = verifier.verify(tmp_path, pins, lambda **kw: None, 0)
    assert result["stage"] == "passed" and result["verified_images"] == 2
    assert result["published"] is False
    (tmp_path / "rgb/extra.png").write_bytes(b"extra")
    with pytest.raises(ValueError, match="Unreferenced"):
        verifier.verify(tmp_path, pins, lambda **kw: None, 0)


def test_detects_rgb_corruption_and_metadata_changes(tmp_path):
    pins = fixture(tmp_path)
    (tmp_path / "rgb/scene/b.png").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="RGB SHA256"):
        verifier.verify(tmp_path, pins, lambda **kw: None, 0)
    (tmp_path / "train.jsonl").write_text("[]\n")
    with pytest.raises(ValueError, match="Metadata SHA256"):
        verifier.verify(tmp_path, pins, lambda **kw: None, 0)


def test_rejects_escape_symlinks_and_duplicate_paths(tmp_path):
    for path in ["../outside", "/outside", "rgb/../outside", "rgb//a", "./rgb/a"]:
        with pytest.raises(ValueError, match="relative path"):
            verifier.relative_path(path)
    pins = fixture(tmp_path, ["rgb/a.png", "rgb/a.png"])
    with pytest.raises(ValueError, match="duplicate"):
        verifier.verify(tmp_path, pins, lambda **kw: None, 0)
    (tmp_path / "rgb/a.png").unlink()
    (tmp_path / "rgb/a.png").symlink_to(tmp_path / "train.jsonl")
    with pytest.raises(ValueError, match="Nonregular"):
        verifier.verify(tmp_path, pins, lambda **kw: None, 0)


def test_rejects_counts_and_files_changed_while_reading(tmp_path):
    pins = fixture(tmp_path)
    pins["counts"]["train"] = 3
    with pytest.raises(ValueError, match="count mismatch"):
        verifier.verify(tmp_path, pins, lambda **kw: None, 0)
    pins["counts"]["train"] = 2

    def mutate(**status):
        if status["stage"] == "checking_rgb_hashes":
            manifest = tmp_path / "train.jsonl"
            manifest.write_text(manifest.read_text() + "\n")

    with pytest.raises((ValueError, json.JSONDecodeError)):
        verifier.verify(tmp_path, pins, mutate, 0)


def add_exclusions(root, pins, rows):
    path = root / "excluded.json"
    path.write_text(json.dumps(rows))
    pins["metadata_files"][path.name] = {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_only_pinned_preparation_exclusions_account_for_extra_rgb(tmp_path):
    pins = fixture(tmp_path)
    excluded = tmp_path / "rgb/excluded.png"
    excluded.write_bytes(b"unused duplicate")
    row = dict(image="rgb/excluded.png", split="train", reason="exact_duplicate")
    add_exclusions(tmp_path, pins, [row])
    result = verifier.verify(tmp_path, pins, lambda **kw: None, 0)
    assert result["verified_images"] == 2 and result["inventory_count"] == 3
    assert result["excluded_images"] == 1 and not result["excluded_bytes_verified"]
    (tmp_path / "rgb/unexpected.png").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="Unreferenced"):
        verifier.verify(tmp_path, pins, lambda **kw: None, 0)


@pytest.mark.parametrize("case", ["missing", "overlap", "duplicate", "escape", "reason"])
def test_exclusion_errors_fail_closed(tmp_path, case):
    pins = fixture(tmp_path)
    row = dict(image="rgb/missing.png", split="train", reason="exact_duplicate")
    if case == "overlap":
        row["image"] = "rgb/scene/a.png"
    elif case == "escape":
        row["image"] = "rgb/../outside.png"
    elif case == "reason":
        row["reason"] = "ignore_corruption"
    add_exclusions(tmp_path, pins, [row, row] if case == "duplicate" else [row])
    with pytest.raises(ValueError):
        verifier.verify(tmp_path, pins, lambda **kw: None, 0)


def test_final_inventory_detects_same_count_replacement(tmp_path):
    pins = fixture(tmp_path)

    def replace(**status):
        if status["stage"] == "checking_final_inventory":
            (tmp_path / "rgb/scene/a.png").rename(tmp_path / "rgb/scene/replaced.png")

    with pytest.raises(ValueError, match="inventory changed"):
        verifier.verify(tmp_path, pins, replace, 0)
