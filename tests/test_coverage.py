import hashlib
import json

import pytest
import torch

from radio_moge import data
from radio_moge.coverage import checkpoint_coverage, file_sha256, replay, sampling_records


@pytest.fixture
def inputs(tmp_path):
    # Unequal scene lengths and repeated draws distinguish exact coverage from
    # presentation count or uniform image sampling. Image files need not exist.
    rows = [dict(dataset="a", group=g) for g in ["long", "long", "long", "short"]]
    rows.append(dict(dataset="b", group="only"))
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(dict(row, image=f"missing-{i}.png")) for i, row in enumerate(rows))
        + "\n"
    )
    config = {
        "train": {"seed": 42, "batch_size": 3, "accumulation": 2},
        "data": {
            "train_manifest": str(manifest),
            "mixture_weights": {"a": 0.7, "b": 0.3},
            "buckets": [[480, 640], [640, 480]],
        },
    }
    return rows, manifest, config


def test_consumed_prefix_excludes_future_batches_and_repeated_images(inputs):
    rows, _, config = inputs
    # Golden trace from pinned 71549cc: [2,3,0],[4,4,3],[3,4,3],[0,3,4].
    # Row 1 is first visited in the next microbatch, which is not consumed yet.
    result = replay(rows, config, step=2, cursor=4)
    assert result["total_draws"] == 12
    assert result["unique_images"] == 4
    assert result["per_source"]["a"] == {
        "draws": 8,
        "available_images": 4,
        "unique_images": 3,
        "image_coverage_fraction": 0.75,
        "available_groups": 2,
        "unique_groups": 2,
        "group_coverage_fraction": 1.0,
    }
    assert result["per_source"]["b"]["draws"] == 4
    assert result["per_source"]["b"]["unique_images"] == 1
    assert replay(rows, config, step=3, cursor=6)["unique_images"] == 5
    assert replay(rows, config, step=0, cursor=0)["unique_images"] == 0
    with pytest.raises(ValueError, match="cursor"):
        replay(rows, config, step=2, cursor=5)


def test_manifest_identity_includes_order_and_unused_fields(inputs):
    rows, manifest, _ = inputs
    expected = file_sha256(manifest)
    assert sampling_records(manifest, expected) == rows
    lines = manifest.read_text().splitlines()
    manifest.write_text("\n".join(reversed(lines)) + "\n")
    with pytest.raises(ValueError, match="SHA256"):
        sampling_records(manifest, expected)


def test_report_binds_checkpoint_manifest_and_sampler(inputs, tmp_path, monkeypatch):
    _, manifest, config = inputs
    checkpoint = tmp_path / "frozen.pt"
    torch.save(
        {"config": config, "step": 2, "cursor": 4, "data_digests": {"train": "audit"}}, checkpoint
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CPU replay queried CUDA"))
    expected = file_sha256(manifest)
    report = checkpoint_coverage(checkpoint, expected, file_sha256(data.__file__))
    assert report["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert report["manifest_sha256"] == expected
    assert report["unique_images"] == 4
    assert report["checkpoint_metadata"]["data_digests"] == {"train": "audit"}
    with pytest.raises(ValueError, match="Sampler"):
        checkpoint_coverage(checkpoint, expected, "0" * 64)
