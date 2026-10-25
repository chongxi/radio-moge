import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

scripts = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("raw_verifier", scripts / "verify_raw_archive_backup.py")
raw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(raw)


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest(),
            "sha256_evidence": raw.SMALL_EVIDENCE}


def dataset(root, structured=False):
    name = "a.zip" if structured else "a.tar.gz"
    archive = put(root, "archives/" + name, b"archive-data")
    archive["sha256_evidence"] = raw.ARCHIVE_EVIDENCE
    files = {"archives/" + name: archive}
    for path, value in {
        "receipts/" + name + ".json": {"bytes": archive["bytes"], "sha256": archive["sha256"]},
        "archives/" + name + ".part.json": {"bytes": archive["bytes"], "name": name},
        "status.json": {"stage": "archives_complete_with_missing_assets" if structured
                        else "archives_complete", "missing": ["missing.zip"] if structured else [],
                        "completed": [{"name": name, "sha256": archive["sha256"]}]
                        if structured else [name]},
    }.items():
        files[path] = put(root, path, json.dumps(value).encode())
    files["download.log"] = put(root, "download.log", b"completed\n")
    return {"files": files, "file_count": len(files), "total_bytes": sum(x["bytes"] for x in files.values()),
            "archive_count": 1, "archive_bytes": archive["bytes"],
            "source_stage": "archives_complete_with_missing_assets" if structured else "archives_complete",
            "missing_archives": ["missing.zip"] if structured else []}


def setup_plan(tmp_path):
    base = tmp_path / "raw-backup"
    pins = {"structured": dataset(base / "staging/structured", True),
            "irs": dataset(base / "staging/irs")}
    source = {"datasets": pins, "source_root": "/remote/source"}
    source_pin = put(base, "source-inventory.json", json.dumps(source).encode())
    publication_pin = put(tmp_path, "publication.json", json.dumps(
        {"stage": "published", "passed": True}).encode())
    plan = {"datasets": list(pins), "source_root": source["source_root"],
            "staging": str(base / "staging"), "pinned_source_inventory": pins,
            "source_inventory": "source-inventory.json",
            "source_inventory_sha256": source_pin["sha256"],
            "publication_prerequisites": {"publication.json": publication_pin["sha256"]}}
    plan_pin = put(base, "plan.json", json.dumps(plan).encode())
    status = {"plan_sha256": plan_pin["sha256"], "stage": "all_copies_finished_pending_verification",
              "copied_pending_verification": list(pins), "dataset": None, "child_pid": None,
              "pid": 2147483000}
    put(base, "prepared-dataset-backup-status.json", json.dumps(status).encode())
    return base, plan, plan_pin["sha256"], status


def audit(root, pins, report=lambda **kw: None):
    return raw.verify_dataset(root, pins, raw.ArchiveReader(20, report), report)


@pytest.mark.parametrize("structured", [False, True])
def test_reads_every_destination_file_and_preserves_missing_scope(tmp_path, structured):
    pins = dataset(tmp_path, structured)
    messages = []
    result = audit(tmp_path, pins, lambda **x: messages.append(x))
    assert result["stage"] == "passed" and result["failures"] == 0
    assert result["inventory"] == {"files": 5, "directories": 3, "bytes": pins["total_bytes"]}
    assert set(result["files"]) == set(pins["files"])
    assert result["missing_archives"] == pins["missing_archives"]
    assert not result["published"] and not result["archive_decompression_checked"]
    assert not result["fresh_source_archive_hashes_claimed"]
    assert result["every_destination_file_sha256_verified"]
    assert messages[-1]["stage"] == "checking_final_inventory"


@pytest.mark.parametrize("kind", ["corrupt", "missing", "extra", "empty_dir", "symlink",
                                  "dir_symlink", "fifo", "size"])
def test_rejects_corruption_and_nonexact_tree(tmp_path, kind):
    pins = dataset(tmp_path)
    archive = tmp_path / "archives/a.tar.gz"
    if kind == "corrupt":
        archive.write_bytes(b"corrupt-data")  # Same size, different hash.
    elif kind == "missing":
        archive.unlink()
    elif kind == "extra":
        (tmp_path / "unexpected").write_text("extra")
    elif kind == "empty_dir":
        (tmp_path / "empty").mkdir()
    elif kind == "symlink":
        archive.unlink()
        archive.symlink_to(tmp_path / "download.log")
    elif kind == "dir_symlink":
        (tmp_path / "archives").rename(tmp_path / "elsewhere")
        (tmp_path / "archives").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    elif kind == "fifo":
        archive.unlink()
        os.mkfifo(archive)
    else:
        archive.write_bytes(b"short")
    with pytest.raises(ValueError):
        audit(tmp_path, pins)


@pytest.mark.parametrize("kind", ["receipt_sha", "sidecar_bytes", "status_count", "missing_scope"])
def test_rejects_relational_mismatch_even_with_valid_file_hashes(tmp_path, kind):
    pins = dataset(tmp_path)
    name = {"receipt_sha": "receipts/a.tar.gz.json", "sidecar_bytes": "archives/a.tar.gz.part.json",
            "status_count": "status.json", "missing_scope": "status.json"}[kind]
    value = json.loads((tmp_path / name).read_text())
    if kind == "receipt_sha":
        value["sha256"] = "0" * 64
    elif kind == "sidecar_bytes":
        value["bytes"] += 1
    elif kind == "status_count":
        value["completed"] = ["a.tar.gz", "a.tar.gz"]
    else:
        value["missing"] = ["missing.zip"]
    pins["files"][name] = put(tmp_path, name, json.dumps(value).encode())
    pins["total_bytes"] = sum(x["bytes"] for x in pins["files"].values())
    with pytest.raises(ValueError):
        audit(tmp_path, pins)


def test_catches_change_after_file_was_hashed(tmp_path):
    pins = dataset(tmp_path)
    def report(**change):
        if change.get("stage") == "checking_final_inventory":
            (tmp_path / "archives/a.tar.gz").write_bytes(b"corrupt-data")
    with pytest.raises(ValueError, match="changed during audit"):
        audit(tmp_path, pins, report)


def test_heartbeat_during_one_file_and_detects_replacement(tmp_path):
    path = tmp_path / "archive"
    path.write_bytes(b"a" * 32)
    messages = []
    reader = raw.ArchiveReader(20, lambda **x: messages.append(x), heartbeat_seconds=0, block_bytes=8)
    assert reader.digest(path, raw.identity(path.stat()), "data", "archive") == hashlib.sha256(b"a" * 32).hexdigest()
    assert [x["current_file_bytes_read"] for x in messages] == [8, 16, 24, 32]
    assert reader.bytes_read == 32
    def replace(**change):
        if change["current_file_bytes_read"] == 8:
            replacement = tmp_path / "replacement"
            replacement.write_bytes(b"a" * 32)
            replacement.replace(path)
    reader = raw.ArchiveReader(20, replace, heartbeat_seconds=0, block_bytes=8)
    with pytest.raises(ValueError, match="Changed|Replaced"):
        reader.digest(path, raw.identity(path.stat()), "data", "archive")


@pytest.mark.parametrize("rate", [-1, 0, 21, float("nan"), float("inf")])
def test_cannot_disable_or_exceed_read_limit(rate):
    with pytest.raises(ValueError, match="Read rate"):
        raw.ArchiveReader(rate, lambda **x: None)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "archives/../escape", ".hidden"])
def test_invalid_pin_paths_rejected_before_io(tmp_path, name):
    pins = dataset(tmp_path)
    pins["files"][name] = pins["files"].pop("download.log")
    with pytest.raises(ValueError):
        raw.validate_pins(pins)


def test_full_cli_and_existing_evidence_guard(tmp_path, monkeypatch):
    base, plan, sha, _ = setup_plan(tmp_path)
    monkeypatch.setattr(sys, "argv", ["verify", "--plan", str(base / "plan.json"), "--plan-sha256", sha])
    raw.main()
    report = json.loads((base / "raw-archive-backup-verification.json").read_text())
    assert report["stage"] == "passed" and report["verified_files"] == 10
    assert report["archive_count"] == 2
    assert report["verified_bytes"] == report["bytes_read"] == sum(
        p["total_bytes"] for p in plan["pinned_source_inventory"].values())
    before = (base / "raw-archive-backup-verification-status.json").read_bytes()
    with pytest.raises(ValueError, match="evidence already exists"):
        raw.main()
    assert (base / "raw-archive-backup-verification-status.json").read_bytes() == before


@pytest.mark.parametrize("kind", ["copying", "child", "controller", "checkpoint", "copy_set"])
def test_refuses_active_or_incomplete_copy(tmp_path, kind):
    base, plan, sha, status = setup_plan(tmp_path)
    if kind == "copying":
        status["stage"] = "copying"
    elif kind == "child":
        status["child_pid"] = os.getpid()
    elif kind == "controller":
        status["pid"] = os.getpid()
    elif kind == "checkpoint":
        (base / "prepared-dataset-backup.pause").touch()
    else:
        status["copied_pending_verification"] = ["structured"]
    put(base, "prepared-dataset-backup-status.json", json.dumps(status).encode())
    with pytest.raises(ValueError):
        raw.require_copies_finished(base, plan, sha)


def test_controller_lock_prevents_reading_archive(tmp_path, monkeypatch):
    base, _, sha, _ = setup_plan(tmp_path)
    monkeypatch.setattr(sys, "argv", ["verify", "--plan", str(base / "plan.json"), "--plan-sha256", sha])
    with (base / ".prepared-dataset-backup.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            raw.main()
    assert not (base / "raw-archive-backup-verification-status.json").exists()


def test_pinned_plan_and_source_evidence_cannot_disagree(tmp_path):
    base, plan, sha, _ = setup_plan(tmp_path)
    assert raw.load_plan(base / "plan.json", sha) == plan
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        raw.load_plan(base / "plan.json", "0" * 64)
    modified = copy.deepcopy(plan)
    modified["pinned_source_inventory"]["irs"]["archive_count"] = 2
    pin = put(base, "plan.json", json.dumps(modified).encode())
    with pytest.raises(ValueError, match="disagrees"):
        raw.load_plan(base / "plan.json", pin["sha256"])
