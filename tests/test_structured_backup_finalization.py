import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


scripts = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("finalizer", scripts / "finalize_structured3d_backup.py")
finalizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finalizer)


def fixture(root, pixel_duplicate=False):
    def put(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        return {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
    rgb = put("rgb/a.png", b"rgb-data")
    excluded = put("rgb/b.png", b"different-encoding" if pixel_duplicate else b"rgb-data")
    receipt = put("receipts/a.zip.json", json.dumps({"source_sha256": "abc", "records": [{
        "image": "rgb/b.png", "sha256_kind": "encoded_file_bytes",
        "decoded_rgb_sha256": "pixels", **excluded}]}).encode())
    metadata = {}
    metadata["train.jsonl"] = put("train.jsonl", (json.dumps({
        "image": "rgb/a.png", "sha256_kind": "encoded_file_bytes",
        "decoded_rgb_sha256": "pixels", **rgb}) + "\n").encode())
    metadata["excluded.json"] = put("excluded.json", json.dumps([
        {"image": "rgb/b.png", "reason": "exact_duplicate", "split": "train"}]).encode())
    metadata["dataset_info.json"] = put("dataset_info.json", json.dumps({
        "excluded_exact_duplicates": 1, "provenance": {"archive_sha256": {"a.zip": "abc"}}}
    ).encode())
    pins = {"metadata_files": metadata, "counts": {"train": 1}}
    source = {"stage": "passed", "metadata": metadata,
              "excluded": {"rgb/b.png": excluded}, "receipts": {"receipts/a.zip.json": receipt}}
    return pins, source


def run(root, pins, source, report=lambda **changes: None):
    return finalizer.finalize(root, pins, source, finalizer.Reader(0), report)


def test_supplement_completes_exact_tree_without_rehashing_retained(tmp_path, monkeypatch):
    pins, source = fixture(tmp_path)
    digest = finalizer.Reader.digest
    def tracked(self, path):
        assert path.name != "a.png", "Retained RGB must not be rehashed"
        return digest(self, path)
    monkeypatch.setattr(finalizer.Reader, "digest", tracked)
    result = run(tmp_path, pins, source)
    assert result["stage"] == "passed" and result["excluded_bytes_verified"]
    assert result["retained_images"] == result["excluded_images"] == 1
    assert result["inventory"]["files"] == 6
    assert not result["retained_rehashed"]


def test_supports_byte_different_pixel_duplicates_from_source_verified_receipts(tmp_path):
    pins, source = fixture(tmp_path, pixel_duplicate=True)
    result = run(tmp_path, pins, source)
    assert result["duplicate_receipt_matches"] == {"decoded": 1}


def test_rejects_receipt_without_retained_duplicate(tmp_path):
    pins, source = fixture(tmp_path, pixel_duplicate=True)
    path = tmp_path / "receipts/a.zip.json"
    path.write_bytes(path.read_bytes().replace(b"pixels", b"wrong!"))
    source["receipts"]["receipts/a.zip.json"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="no retained duplicate"):
        run(tmp_path, pins, source)


@pytest.mark.parametrize("kind", ["excluded", "receipt", "metadata", "extra", "missing",
                                  "symlink", "wrong_count", "wrong_source_set"])
def test_rejects_corruption_and_inventory_errors(tmp_path, kind):
    pins, source = fixture(tmp_path)
    if kind == "excluded":
        (tmp_path / "rgb/b.png").write_bytes(b"bad-data")
    elif kind == "receipt":
        (tmp_path / "receipts/a.zip.json").write_bytes(b"damaged")
    elif kind == "metadata":
        (tmp_path / "train.jsonl").write_text("{}\n")
    elif kind == "extra":
        (tmp_path / "extra").write_text("extra")
    elif kind == "missing":
        (tmp_path / "rgb/a.png").unlink()
    elif kind == "symlink":
        (tmp_path / "rgb/a.png").unlink()
        (tmp_path / "rgb/a.png").symlink_to(tmp_path / "rgb/b.png")
    elif kind == "wrong_count":
        pins["counts"]["train"] = 2
    elif kind == "wrong_source_set":
        source["excluded"] = {}
    with pytest.raises(ValueError):
        run(tmp_path, pins, source)


def test_metadata_mutation_during_inventory_is_rejected(tmp_path):
    pins, source = fixture(tmp_path)
    def mutate(**changes):
        if changes["stage"] == "checking_exact_tree":
            path = tmp_path / "excluded.json"
            path.write_bytes(path.read_bytes().replace(b"train", b"tests"))
    with pytest.raises(ValueError, match="Changed"):
        run(tmp_path, pins, source, mutate)


def test_requires_exact_terminal_location_and_pinned_code(tmp_path, monkeypatch):
    code = b"test-pinned-code"
    sha = hashlib.sha256(code).hexdigest()
    monkeypatch.setattr(finalizer, "LEGACY_SHA", sha)
    status = {"stage": "failed", "error": finalizer.EXPECTED_ERROR,
              "code_sha256": sha, "plan_sha256": finalizer.PLAN_SHA,
              "pid": 1, "dataset": finalizer.DATASET, "verified_images": 160891}
    launch = dict(status)
    log = ('line 127, in verify\n'
           'require(not remaining, f"Unreferenced RGB files: {len(remaining)}")\n'
           'ValueError: Unreferenced RGB files: 289\n')
    check = finalizer.validate_terminal_evidence
    check(status, launch, log, code, finalizer.PLAN_SHA)
    with pytest.raises(ValueError, match="post-manifest"):
        check(status, launch, log.replace("line 127", "line 110"), code, finalizer.PLAN_SHA)
    with pytest.raises(ValueError, match="SHA"):
        check(status, launch, log, b"changed code", finalizer.PLAN_SHA)
    status["error"] = "RGB SHA256 mismatch"
    with pytest.raises(ValueError, match="Different legacy"):
        check(status, launch, log, code, finalizer.PLAN_SHA)
