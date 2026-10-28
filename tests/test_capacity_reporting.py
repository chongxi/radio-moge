import hashlib
import io
import ast
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from finalize_architecture_backup import extract_bundle
from report_capacity_screen import paired


def test_paired_gains_require_both_seeds_not_only_higher_mean():
    specs = [{"blocks": 4, "width": w} for w in (512, 1024)]
    rows = {
        "combined_4x512_s42": {"spatial/cosine": 0.8, "summary/cosine": 0.7},
        "combined_4x512_s43": {"spatial/cosine": 0.8, "summary/cosine": 0.7},
        "combined_4x1024_s42": {"spatial/cosine": 0.9, "summary/cosine": 0.8},
        "combined_4x1024_s43": {"spatial/cosine": 0.81, "summary/cosine": 0.69},
    }
    result = paired(rows, specs, [42, 43])["4x1024"]
    assert result["mean"]["summary/cosine"] > 0.7
    assert not result["both_cosines_higher_in_each_seed"]
    assert result["paired_difference_vs_4x512"]["43"]["summary/cosine"] < 0


def test_wide_pairing_uses_1024_control_for_each_seed():
    specs = [{"blocks": 4, "width": w} for w in (1024, 2048)]
    rows = {
        f"combined_4x{width}_s{seed}": {
            "spatial/cosine": 0.8 + seed / 1000 + (width == 2048) * 0.01,
            "summary/cosine": 0.7 + seed / 1000 + (width == 2048) * 0.02,
        }
        for width in (1024, 2048)
        for seed in (42, 43)
    }
    result = paired(rows, specs, [42, 43], baseline=(4, 1024))
    assert result["4x2048"]["both_cosines_higher_in_each_seed"]
    assert result["4x2048"]["paired_difference_vs_4x1024"]["43"]["spatial/cosine"] == pytest.approx(
        0.01
    )


def make_archive(path, name="data/metrics.json", content=b'{"value":1}'):
    with tarfile.open(path, "w") as tar:
        entry = tarfile.TarInfo(name)
        entry.size = len(content)
        tar.addfile(entry, io.BytesIO(content))
    return {name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}}


def test_extract_bundle_verifies_contents_and_readback(tmp_path):
    archive = tmp_path / "archive.tar"
    expected = make_archive(archive)
    result = extract_bundle(archive, tmp_path / "out", expected)
    assert result["every_extracted_file_sha256_verified"]
    assert (tmp_path / "out/data/metrics.json").read_bytes() == b'{"value":1}'


@pytest.mark.parametrize("bad_name", ["../outside.json", "/tmp/outside.json"])
def test_extract_bundle_rejects_escape_even_if_hash_matches(tmp_path, bad_name):
    archive = tmp_path / "archive.tar"
    expected = make_archive(archive, bad_name)
    with pytest.raises(AssertionError):
        extract_bundle(archive, tmp_path / "out", expected)


def test_extract_bundle_rejects_changed_bytes(tmp_path):
    archive = tmp_path / "archive.tar"
    expected = make_archive(archive)
    expected["data/metrics.json"]["sha256"] = "0" * 64
    with pytest.raises(AssertionError):
        extract_bundle(archive, tmp_path / "out", expected)
    assert not (tmp_path / "out/data/metrics.json").exists()


def test_monitor_does_not_import_stale_workspace_reports(tmp_path, monkeypatch, capsys):
    script = Path(__file__).parents[1] / "scripts/monitor_architecture_screen.py"
    tree = ast.parse(script.read_text())
    probe = next(
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and "known=json.load(sys.stdin)" in n.value
    )
    (tmp_path / "plan.json").write_text(json.dumps({"candidates": []}))
    (tmp_path / "status.json").write_text(json.dumps({"stage": "complete"}))
    (tmp_path / "curve-verification.json").write_text(json.dumps({"total_proofs": 82}))
    (tmp_path / "continuation.json").write_text(json.dumps({"stage": "outdated"}))
    monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "0 %, 0 MiB")
    exec(compile(probe.replace("ROOT", repr(str(tmp_path))), "remote_probe", "exec"), {})
    files = json.loads(capsys.readouterr().out)["files"]
    assert "status.json" in files
    assert "curve-verification.json" not in files
    assert "continuation.json" not in files
