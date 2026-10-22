import gzip
import importlib.util
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace
import zipfile

from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "archive_rgb", ROOT / "scripts/prepare_archive_rgb.py"
)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def rgb(value, compression=6):
    buffer = io.BytesIO()
    Image.new("RGBA", (32, 24), (value, 11, 97, 255)).save(
        buffer, format="PNG", compress_level=compression
    )
    return buffer.getvalue()


def receipt(path):
    return dict(bytes=path.stat().st_size, sha256=prep.digest(path))


def structured_member(scene, kind="full"):
    return (
        f"Structured3D/scene_{scene:05d}/2D_rendering/27215/perspective/{kind}/0/rgb_rawlight.png"
    )


def test_independent_zip_extraction_missing09_full_empty_same_scene_and_crc(tmp_path):
    out = tmp_path / "rgb"
    rows = []
    for kind in ("full", "empty"):
        archive = tmp_path / f"Structured3D_perspective_{kind}_00.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as file:
            for scene, color in [(95, 10), (3100, 20), (3300, 30)]:
                file.writestr(structured_member(scene, kind), rgb(color + (kind == "empty")))
        rows.extend(prep.extract_zip(archive, receipt(archive), out))
    result = prep.finalize("structured3d", rows, out, dict(missing_archives=["full_09"]))
    assert result["counts"] == dict(train=2, val=2, test=2)
    assert result["primary_group_counts"] == dict(train=1, val=1, test=1)
    assert all(r["native_mode"] == "RGBA" and r["width"] == 32 for r in rows)
    assert all(prep.digest(out / r["image"]) == r["sha256"] for r in rows)
    archive = tmp_path / "Structured3D_perspective_full_00.zip"
    # Same-length payload corruption; updated container SHA cannot hide ZIP CRC failure.
    payload = archive.read_bytes().replace(b"IDAT", b"IDAD", 1)
    archive.write_bytes(payload)
    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        prep.extract_zip(archive, receipt(archive), tmp_path / "bad")


def test_policy_groups_variant_scenes_and_keeps_official_test_separate(tmp_path):
    policy = json.loads((ROOT / "configs/irs-scene-families-v1.json").read_text())
    assert len(policy["groups"]) == 76 and len(policy["family_splits"]) == 17
    assert (
        policy["groups"]["Home/RealisticRenderingDay"]
        == policy["groups"]["Home/RealisticRenderingNight_T1"]
    )
    assert policy["groups"]["Office/OfficeMedley1"] == policy["groups"]["Office/OfficeMedley3_Dark"]
    assert set(policy["family_splits"].values()) == {"train", "val", "test"}
    # All directory variants need an explicit policy. Unexpected names fail closed.
    groups = {
        "Home/A": "Home/A",
        "Home/A_Dark": "Home/A",
        "Office/B": "Office/B",
        "Store/C": "Store/C",
    }
    split = {"Home/A": "train", "Office/B": "val", "Store/C": "test"}
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(dict(groups=groups, family_splits=split)))
    for source, frame in [("train", "00001"), ("test", "00002")]:
        lines = [
            " ".join(
                f"{g}/{prefix}_{frame}.{ext}"
                for prefix, ext in [("l", "png"), ("r", "png"), ("d", "exr"), ("n", "exr")]
            )
            for g in groups
        ]
        (tmp_path / f"irs_{source}.list").write_text("\n".join(lines) + "\n")
    rows, info = prep.irs_inventory(tmp_path, path)
    assert info["counts"] == dict(train=4, val=2, test=2, **{"official-test": 8})
    assert rows["Home/A/l_00001.png"]["group"] == rows["Home/A_Dark/r_00001.png"]["group"]
    assert rows["Home/A/l_00002.png"]["split"] == "official-test"
    with (tmp_path / "irs_train.list").open("a") as file:
        file.write(
            "Home/Unknown/l_1.png Home/Unknown/r_1.png Home/Unknown/d_1.exr Home/Unknown/n_1.exr\n"
        )
    with pytest.raises(ValueError, match="Unreviewed"):
        prep.irs_inventory(tmp_path, path)


def tar_bytes(members):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as file:
        for name, data in members:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            file.addfile(member, io.BytesIO(data))
    return gzip.compress(buffer.getvalue(), mtime=0)


def test_streaming_irs_reads_all_members_and_verifies_even_unselected_bytes(tmp_path):
    path, out = tmp_path / "Store.tar.gz", tmp_path / "prepared"
    official, members = {}, []
    for i, split in enumerate(("train", "val", "test", "official-test")):
        for eye in ("l", "r"):
            canonical = f"Store/Scene{i}/{eye}_00001.png"
            official[canonical] = dict(
                image="rgb/" + canonical,
                dataset="irs",
                split=split,
                group=f"irs:family:Scene{i}",
                official_split="test" if i == 3 else "train",
            )
            members.append(("IRS/" + canonical, rgb(40 + 2 * i + (eye == "r"))))
            members.append((f"IRS/Store/Scene{i}/{eye}_normal.exr", b"label-bytes-ignored"))
    path.write_bytes(tar_bytes(members))
    original = receipt(path)
    rows = prep.extract_tar(path, original, out, official)
    assert len(rows) == 8
    result = prep.finalize("irs", rows, out, {})
    assert result["counts"] == dict(train=2, val=2, test=2) and result["official_test_rgb"] == 2
    assert not list(out.rglob("*.exr"))
    # Valid gzip, but different unselected-label bytes from the verified source.
    path.write_bytes(tar_bytes(members[:-1] + [(members[-1][0], b"changed-label-bytes")]))
    with pytest.raises(ValueError, match="compressed SHA256"):
        prep.extract_tar(path, dict(original, bytes=path.stat().st_size), out, official)


def test_official_test_and_heldout_pixel_duplicates_win_over_training(tmp_path):
    rows = []
    for i, split in enumerate(("train", "val", "test", "official-test")):
        row = dict(image=f"rgb/{i}.png", dataset="irs", split=split, group=f"irs:{i}")
        rows.append(prep.write_rgb(rgb(20 + i), row, tmp_path, "fixture", str(i)))
    # Same pixels as official test, different PNG encoding and filename.
    row = dict(image="rgb/copy.png", dataset="irs", split="train", group="irs:0")
    rows.append(prep.write_rgb(rgb(23, compression=0), row, tmp_path, "fixture", "copy"))
    assert rows[-1]["sha256"] != rows[-2]["sha256"]
    result = prep.finalize("irs", rows, tmp_path, {})
    assert result["counts"]["train"] == 1 and result["excluded_exact_duplicates"] == 1
    assert len((tmp_path / "test-official.jsonl").read_text().splitlines()) == 1


def test_full_structured_ready_gate_and_resume_detects_corrupted_output(tmp_path, monkeypatch):
    import fcntl
    source, out = tmp_path / "source", tmp_path / "prepared"
    (source / "archives").mkdir(parents=True)
    (source / "receipts").mkdir()
    names = [n for n in prep.STRUCTURED_ARCHIVES if n != "Structured3D_perspective_full_09.zip"]
    for name in names:
        p = source / "archives" / name
        with zipfile.ZipFile(p, "w") as file:
            if name == "Structured3D_perspective_full_00.zip":
                for scene, color in [(95, 10), (3100, 20), (3300, 30)]:
                    file.writestr(structured_member(scene), rgb(color))
        (source / "receipts" / (name + ".json")).write_text(json.dumps(receipt(p)))
    status = dict(
        stage="downloading",
        completed=[dict(name=n) for n in names],
        missing=["Structured3D_perspective_full_09.zip"],
    )
    (source / "status.json").write_text(json.dumps(status))
    args = SimpleNamespace(source=source, output=out, dataset="structured3d")
    with pytest.raises(ValueError, match="finish downloading"):
        prep.run(args)
    assert not out.exists()
    status["stage"] = "archives_complete_with_missing_assets"
    (source / "status.json").write_text(json.dumps(status))
    original_extract = prep.extract_zip

    def check_next_dataset_is_not_blocked(*args, **kwargs):
        # IRS waits for Structured3D's download writer lock, not its RGB reader.
        with (source / ".download.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(prep, "extract_zip", check_next_dataset_is_not_blocked)
    result = prep.run(args)
    assert result["stage"] == "ready" and result["counts"] == dict(train=1, val=1, test=1)
    assert prep.run(args)["counts"] == result["counts"]
    image = out / json.loads((out / "train.jsonl").read_text())["image"]
    image.write_bytes(b"bad")
    with pytest.raises(ValueError, match="RGB changed"):
        prep.run(args)
    assert json.loads((out / "dataset_info.json").read_text())["stage"] == "failed"


def test_archive_path_traversal_and_links_are_rejected(tmp_path):
    path = tmp_path / "Structured3D_perspective_full_00.zip"
    with zipfile.ZipFile(path, "w") as file:
        file.writestr("../escape.png", rgb(10))
    with pytest.raises(ValueError, match="Unsafe"):
        prep.extract_zip(path, receipt(path), tmp_path / "out")
    assert not (tmp_path / "escape.png").exists()
    p = tmp_path / "Store.tar.gz"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as file:
        m = tarfile.TarInfo("IRS/Store/link")
        m.type = tarfile.SYMTYPE
        m.linkname = "/tmp"
        file.addfile(m)
    p.write_bytes(gzip.compress(buffer.getvalue()))
    with pytest.raises(ValueError, match="links/special"):
        prep.extract_tar(p, receipt(p), tmp_path / "out", {})
