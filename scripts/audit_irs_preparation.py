"""Independently check the completed IRS inventory and bounded RGB samples.

Read-only for datasets. Writes a completion report only after every check passes.
The full decoder pass belongs to the preparer; this is not a second full rehash.
"""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import time

from PIL import Image


ARCHIVES = (
    "Store.tar.gz",
    "Home-1.tar.gz",
    "Home-2.tar.gz",
    "Office-1.tar.gz",
    "Office-2.tar.gz",
    "Restaurant.tar.gz",
)


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def audit(args):
    started = time.time()
    source, output = args.source.resolve(), args.prepared.resolve()
    info, state = read(output / "dataset_info.json"), read(output / "status.json")
    assert info["stage"] == state["stage"] == "ready"
    assert info["dataset"] == "irs" and info["pilot"] is False
    assert set(state["completed"]) == set(ARCHIVES)
    for path in (source / ".download.lock", output / ".prepare.lock"):
        with path.open("r") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    policy = read(args.policy)
    inputs = {"family_policy": digest(args.policy)}
    expected = {}
    for original in ("train", "test"):
        path = args.metadata / f"irs_{original}.list"
        inputs[path.name] = digest(path)
        for line in path.read_text().splitlines():
            left, right, disparity, normal = line.split()
            directory = str(PurePosixPath(left).parent)
            assert left.replace("/l_", "/r_") == right
            assert all(
                str(PurePosixPath(x).parent) == directory for x in (right, disparity, normal)
            )
            family = policy["groups"][directory]
            for eye, member in (("left", left), ("right", right)):
                image = "rgb/" + member
                assert image not in expected
                expected[image] = dict(
                    dataset="irs",
                    official_split=original,
                    eye=eye,
                    frame=PurePosixPath(member).stem[2:],
                    scene_directory=directory,
                    family=family,
                    group="irs:family:" + family,
                    split=policy["family_splits"][family]
                    if original == "train"
                    else "official-test",
                )
    assert len(expected) == info["raw_extracted_unique_rgb"] == 206632
    assert inputs == info["provenance"]["split_plan"]["inputs"]
    raw, receipts, samples = {}, [], {}
    for archive in ARCHIVES:
        saved = read(output / "receipts" / (archive + ".json"))
        downloaded = read(source / "receipts" / (archive + ".json"))
        stat = (source / "archives" / archive).stat()
        assert saved["source_stat"] == [stat.st_size, stat.st_mtime_ns]
        assert stat.st_size == downloaded["bytes"]
        assert (
            saved["source_sha256"]
            == downloaded["sha256"]
            == info["provenance"]["archive_sha256"][archive]
        )
        for row in saved["records"]:
            image = row["image"]
            assert image in expected and image not in raw
            assert all(row[key] == value for key, value in expected[image].items())
            assert row["source_archive"] == archive
            assert row["sha256_kind"] == "encoded_file_bytes"
            assert all(len(row[k]) == 64 for k in ("sha256", "decoded_rgb_sha256"))
            path = output / image
            assert not path.is_symlink() and path.is_file() and path.stat().st_size == row["bytes"]
            raw[image] = row
            key = (archive, row["scene_directory"], row["split"], row["eye"])
            if key not in samples or image < samples[key]["image"]:
                samples[key] = row
        receipts.append(
            dict(archive=archive, rgb=len(saved["records"]), source_receipt_and_stat_match=True)
        )
    assert set(raw) == set(expected)
    assert read(output / "repeated-archive-members.json") == []
    manifests, retained, hashes, families = {}, {}, {}, {}
    for split in ("train", "val", "test", "test-official"):
        path = output / (split + ".jsonl")
        manifests[path.name] = digest(path)
        assert manifests[path.name] == info["manifest_sha256"][path.name]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            assert row == raw[row["image"]]
            assert row["split"] == ("official-test" if split == "test-official" else split)
            assert row["image"] not in retained
            retained[row["image"]] = row
        hashes[split] = [{r[k] for r in rows} for k in ("sha256", "decoded_rgb_sha256")]
        families[split] = {r["group"] for r in rows}
        expected_count = (
            info["official_test_rgb"] if split == "test-official" else info["counts"][split]
        )
        assert len(rows) == expected_count
        if split != "test-official":
            assert all(len(h) == len(rows) for h in hashes[split])
    assert set(manifests) == set(info["manifest_sha256"])
    keys = list(hashes)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            assert all(not x.intersection(y) for x, y in zip(hashes[a], hashes[b]))
            if "test-official" not in (a, b):
                assert not families[a].intersection(families[b])
    excluded = read(output / "excluded.json")
    assert len(excluded) == info["excluded_exact_duplicates"] == 3
    assert set(raw) - set(retained) == {r["image"] for r in excluded}
    retained_encoded = {r["sha256"] for r in retained.values()}
    retained_decoded = {r["decoded_rgb_sha256"] for r in retained.values()}
    for entry in excluded:
        row = raw[entry["image"]]
        assert entry["split"] == row["split"] and entry["reason"] == "exact_duplicate"
        assert row["sha256"] in retained_encoded or row["decoded_rgb_sha256"] in retained_decoded
    for row in samples.values():
        path = output / row["image"]
        assert digest(path) == row["sha256"]
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            assert rgb.size == (row["width"], row["height"])
            pixels = f"{rgb.width}x{rgb.height}:RGB:".encode() + rgb.tobytes()
            assert hashlib.sha256(pixels).hexdigest() == row["decoded_rgb_sha256"]
    assert read(output / "dataset_info.json") == info
    return dict(
        stage="passed",
        checked_at=time.time(),
        elapsed_seconds=time.time() - started,
        dataset=str(output),
        counts=info["counts"],
        official_test_rgb=info["official_test_rgb"],
        raw_rgb=len(raw),
        retained_rgb=len(retained),
        excluded_exact_duplicates=len(excluded),
        manifest_sha256=manifests,
        archive_receipt_checks=receipts,
        raw_split_counts=dict(Counter(r["split"] for r in raw.values())),
        primary_group_counts={s: len(families[s]) for s in ("train", "val", "test")},
        all_manifest_paths_present_and_sizes_match=True,
        full_official_inventory_match=True,
        primary_families_disjoint=True,
        official_test_excluded_from_primary=True,
        hashes_disjoint=True,
        writer_locks_released=True,
        sample_count=len(samples),
        samples=[r["image"] for r in samples.values()],
        limits="All stored hashes, source receipts, inventory, split membership and file sizes checked; "
        "bounded samples independently decoded and rehashed. Not a second full-file rehash. "
        "Official-test families overlap primary families; official-test is audit-only.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "prepared", "metadata", "policy", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = audit(args)
    fd, temporary = tempfile.mkstemp(prefix=".irs-audit-", dir=args.report.parent)
    with os.fdopen(fd, "w") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    os.replace(temporary, args.report)
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}), flush=True)
