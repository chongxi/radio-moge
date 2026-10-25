import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

scripts = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("irs_finalizer", scripts / "finalize_irs_backup.py")
finalizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finalizer)


def fixture(root):
    def put(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        return {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
    rgb = put("rgb/train.png", b"rgb-data")
    official = put("rgb/official.png", b"official-rgb")
    excluded = put("rgb/duplicate.png", b"rgb-data")
    receipt = put("receipts/a.tar.gz.json", json.dumps({"source_sha256": "abc", "records": [{
        "image": "rgb/duplicate.png", "sha256_kind": "encoded_file_bytes", **excluded}]}).encode())
    metadata = {}
    for split, path, record in [("train", "train", rgb), ("test-official", "official", official)]:
        metadata[split + ".jsonl"] = put(split + ".jsonl", (json.dumps({
            "image": f"rgb/{path}.png", "sha256_kind": "encoded_file_bytes", **record}) + "\n").encode())
    metadata["excluded.json"] = put("excluded.json", json.dumps([
        {"image": "rgb/duplicate.png", "reason": "exact_duplicate", "split": "train"}]).encode())
    info = {"counts": {"train": 1}, "official_test_rgb": 1,
            "extra_holdout_manifests": ["test-official.jsonl"],
            "excluded_exact_duplicates": 1, "provenance": {"archive_sha256": {"a.tar.gz": "abc"}}}
    metadata["dataset_info.json"] = put("dataset_info.json", json.dumps(info).encode())
    pins = {"metadata_files": metadata, "counts": {"train": 1}}
    source = {"stage": "passed", "metadata": metadata, "counts": {"train": 1, "test-official": 1},
              "excluded": {"rgb/duplicate.png": excluded}, "receipts": {"receipts/a.tar.gz.json": receipt}}
    retained = {"counts": source["counts"], "verified_images": 2,
                "verified_rgb_bytes": rgb["bytes"] + official["bytes"]}
    return pins, source, retained, info


def test_completes_official_holdout_and_exclusions_without_retained_rehash(tmp_path, monkeypatch):
    pins, source, retained, _ = fixture(tmp_path)
    digest = finalizer.Reader.digest
    def tracked(self, path):
        assert path.name not in {"train.png", "official.png"}
        return digest(self, path)
    monkeypatch.setattr(finalizer.Reader, "digest", tracked)
    result = finalizer.finalize(tmp_path, pins, source, retained, finalizer.Reader(0), lambda **k: None)
    assert result["counts"] == {"train": 1, "test-official": 1}
    assert result["inventory"]["files"] == 8
    assert not result["retained_rehashed"] and result["excluded_bytes_verified"]
    assert "Completed" in result["retained_hash_evidence"]
    assert "legacy" not in json.dumps(result)


@pytest.mark.parametrize("kind", ["missing_official", "incorrect_official", "retained_bytes",
                                  "corrupt_excluded", "extra_file", "corrupt_receipt"])
def test_rejects_incomplete_or_corrupt_evidence(tmp_path, kind):
    pins, source, retained, _ = fixture(tmp_path)
    if kind == "missing_official":
        del pins["metadata_files"]["test-official.jsonl"]
    elif kind == "incorrect_official":
        source["counts"]["test-official"] = 2
    elif kind == "retained_bytes":
        retained["verified_rgb_bytes"] += 1
    elif kind == "corrupt_excluded":
        (tmp_path / "rgb/duplicate.png").write_bytes(b"bad-data")
    elif kind == "extra_file":
        (tmp_path / "extra").write_text("extra")
    elif kind == "corrupt_receipt":
        (tmp_path / "receipts/a.tar.gz.json").write_text("{}")
    with pytest.raises(ValueError):
        finalizer.finalize(tmp_path, pins, source, retained, finalizer.Reader(0), lambda **k: None)


def test_source_hashes_all_supplemental_files(tmp_path):
    pins, source, _, _ = fixture(tmp_path)
    actual = finalizer.source_audit(tmp_path, pins, finalizer.Reader(0), lambda **k: None)
    for key in ["metadata", "counts", "excluded", "receipts"]:
        assert actual[key] == source[key]


def test_retained_proof_requires_official_count_and_success(tmp_path):
    pins = {"metadata_files": {}}
    proof = {"stage": "passed", "failures": 0, "dataset": finalizer.DATASET, "pid": 42,
             "root": str(tmp_path), "plan_sha256": finalizer.PLAN_SHA,
             "code_sha256": finalizer.VERIFIER_SHA, "metadata": {},
             "counts": dict(finalizer.COUNTS), "verified_images": 206629,
             "inventory_count": 206632, "excluded_images": 3}
    launch = copy.deepcopy(proof)
    finalizer.validate_retained_proof(proof, launch, pins, tmp_path)
    proof["counts"]["test-official"] -= 1
    with pytest.raises(ValueError, match="official holdout"):
        finalizer.validate_retained_proof(proof, launch, pins, tmp_path)
    proof = copy.deepcopy(launch)
    proof["stage"] = "failed"
    with pytest.raises(ValueError, match="did not pass"):
        finalizer.validate_retained_proof(proof, launch, pins, tmp_path)
