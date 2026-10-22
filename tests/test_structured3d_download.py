import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


spec = importlib.util.spec_from_file_location(
    "structured_download",
    Path(__file__).parents[1] / "scripts/download_structured3d_perspective.py",
)
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


def test_inventory_explicitly_excludes_panoramas_and_preserves_missing_part(tmp_path):
    names = [
        f"Structured3D_perspective_{kind}_{i:02d}.zip"
        for kind in ("full", "empty")
        for i in range(18)
    ]
    names += ["Structured3D_annotation_3d.zip", "Structured3D_bbox.zip"]
    rows = [
        dict(name=n, url=download.BASE + n, status=200, bytes=10, etag='"version"') for n in names
    ]
    rows[9].update(status=404, bytes=None, etag=None)
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(dict(archives=rows)))
    assert download.inventory(path)[9]["status"] == 404
    rows[0]["name"] = "Structured3D_panorama_00.zip"
    path.write_text(json.dumps(dict(archives=rows)))
    with pytest.raises(ValueError, match="approved"):
        download.inventory(path)


def make_zip(path, member):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(member, b"fixture payload 1234567")
    return path.stat().st_size


def test_all_member_crc_and_safe_paths_are_checked(tmp_path):
    path = tmp_path / "archive.zip"
    size = make_zip(path, "Structured3D/scene_00001/2D_rendering/room/perspective/full/rgb.png")
    receipt = download.verify_archive(path, size)
    assert receipt["scene_ids"] == ["scene_00001"]
    assert receipt["members"] == 1
    raw = path.read_bytes().replace(b"fixture payload", b"corrupt payload")
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="CRC"):
        download.verify_archive(path, size)
    size = make_zip(path, "../escape.png")
    with pytest.raises(ValueError, match="Unsafe"):
        download.verify_archive(path, size)


def test_finished_partial_is_verified_without_download_and_reuse_detects_change(
    tmp_path, monkeypatch
):
    name = "Structured3D_annotation_3d.zip"
    (tmp_path / "archives").mkdir()
    (tmp_path / "receipts").mkdir()
    partial = tmp_path / "archives" / (name + ".part")
    size = make_zip(partial, "Structured3D/scene_00000/annotation_3d.json")
    row = dict(name=name, url=download.BASE + name, etag='"fixed"', bytes=size)
    partial.with_suffix(".part.json").write_text(
        json.dumps({key: row[key] for key in ("url", "bytes", "etag")})
    )
    monkeypatch.setattr(
        download.subprocess, "Popen", lambda *a, **kw: pytest.fail("Unexpected download")
    )

    def record(**kw):
        pass

    receipt = download.download_one(row, tmp_path, record)
    assert not partial.exists()
    assert download.download_one(row, tmp_path, record) == receipt
    (tmp_path / "archives" / name).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed"):
        download.download_one(row, tmp_path, record)
