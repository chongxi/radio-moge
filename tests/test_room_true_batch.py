from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch

from radio_moge.corpus_admission import file_ref
from radio_moge.data import TrainingImages
from radio_moge.rgb_input_audit import RGBInputAudit, load_reference_records
from radio_moge.rgb_replacement import ReplacementRecords
from radio_moge.train import train
from run_room_feature_epoch import SingleEpochBatches
from run_room_feature_mix import initial_state_check
from run_room_true_batch_epoch import (
    TrueBatchSampler,
    BatchCoverage,
    check_contract,
    check_performance,
    updates_for,
    BUCKET_POLICY,
    SECONDS,
    IMAGES,
)
from test_procthor_rgb_training import contract, dummy_records, spec
from test_rgb_reference import feature_config
from test_training import tiny_system
from test_performance import profile
from verify_corpus_stage_startup import equal_finite


@pytest.mark.parametrize("batch", [64, 128])
@pytest.mark.parametrize("n", [1, 3, 7, 8, 15, 16, 17, 50787])
def test_sampler_keeps_identity_seeds_complete_coverage_and_true_shapes(batch, n):
    records = dummy_records(53, n)
    buckets = [[480, 640], [640, 480], [400, 640]]
    settings = spec(213264)
    count = ((n + batch // 8 - 1) // (batch // 8)) * batch // 4
    old = list(SingleEpochBatches(records, settings, 4, buckets, 43, 0, count))

    def sampler(start, count):
        return TrueBatchSampler(records, settings, 4, buckets, 43, start, count, actual_batch=batch)

    new = list(sampler(0, count))
    assert list(sampler(3, max(0, count - 3))) == new[3:]
    indices = Counter()
    for a, b in zip(old, new, strict=True):
        for x, y in zip(a, b, strict=True):
            assert x[:2] == y[:2]
            if y[0] >= 53:
                indices[y[0] - 53] += 1
    assert indices == {i: 1 for i in range(n)}
    for i in range(0, count, batch // 4):
        window = [k for b in new[i : i + batch // 4] for k in b]
        assert len(window) == batch and len({k[2:] for k in window}) == 1


def fixture_contract(batch=128):
    config, parent, _ = contract()
    config['train'].update(batch_size=4, precision='bf16')
    config["evaluation"]["dataset_variants"]["structural-rooms"] = dict(
        manifest="/struct", max_side=640, num_tokens=1200
    )
    parent.update(config=deepcopy(config), step=258946, cursor=101576)
    config["data"]["rgb_replacement"]["real_start_cursor"] += parent["cursor"]
    config["train"].update(
        steps=parent["step"] + updates_for(batch),
        max_seconds=SECONDS,
        eval_every=32000 // batch,
        save_every=128,
        precision="bf16",
    )
    parent["config"]["train"]["precision"] = "bf16"
    plan = dict(
        authorized=True,
        actual_batch=batch,
        coverage_goal="every_frozen_synthetic_rgb_once",
        bucket_policy=BUCKET_POLICY,
        synthetic_images=IMAGES,
        additional_updates=updates_for(batch),
        source_step=parent["step"],
        source_parent_cursor=parent["cursor"],
        source_cursor=config["data"]["rgb_replacement"]["real_start_cursor"],
        maximum_training_seconds=SECONDS,
        synthetic_manifest=config["data"]["rgb_replacement"]["manifest"],
        validation_manifest=dict(path="/val"),
        structural_validation_manifest=dict(path="/struct"),
    )
    return config, parent, plan


@pytest.mark.parametrize(
    "change",
    [None, "weight", "lr", "steps", "cursor", "bucket", "manifest", "eval", "batch", "policy"],
)
def test_scoped_contract_rejects_unintended_training_changes(change):
    cfg, parent, plan = fixture_contract()
    if change == "weight":
        cfg["loss"]["cosine"] = 0.1
    if change == "lr":
        cfg["train"]["lr"] = 4e-5
    if change == "steps":
        cfg["train"]["steps"] += 1
    if change == "cursor":
        cfg["data"]["rgb_replacement"]["real_start_cursor"] = 0
    if change == "bucket":
        cfg["data"]["buckets"] = [[480, 640]]
    if change == "manifest":
        cfg["data"]["rgb_replacement"]["manifest"]["path"] = "/other"
    if change == "eval":
        cfg["train"]["eval_at_start"] = True
    if change == "batch":
        plan["actual_batch"] = 32
    if change == "policy":
        plan["bucket_policy"] = "mixed_shapes"
    if change is None:
        check_contract(cfg, parent, plan)
    else:
        with pytest.raises(ValueError):
            check_contract(cfg, parent, plan)


@pytest.mark.parametrize("bad", ["split", "no_prefetch", "auto", "old_batch"])
def test_performance_must_be_actual_large_batch(bad):
    cfg, parent, plan = fixture_contract()
    perf = dict(
        authorized=True,
        effective_batch=128,
        compute_batch=128,
        workers=4,
        data_pipeline="prepared",
        compact_transport=True,
        gpu_prefetch=True,
        anchor_cursor=0,
        anchor_step=parent["step"],
        anchor_schedule_position=float(parent["step"]),
    )
    check_performance(cfg, parent, plan, perf)
    if bad == "split":
        perf["compute_batch"] = 32
    if bad == "no_prefetch":
        perf["gpu_prefetch"] = False
    if bad == "auto":
        perf["autotune"] = True
    if bad == "old_batch":
        perf["effective_batch"] = 32
    with pytest.raises(ValueError):
        check_performance(cfg, parent, plan, perf)


@pytest.mark.parametrize("batch", [64, 128])
def test_real_trainer_one_forward_and_backward_per_update_with_preserved_adam(
    tmp_path, monkeypatch, batch
):
    cfg = feature_config(tmp_path)
    cfg["train"].update(eval_at_start=False, log_every=1, eval_every=100, accumulation=8)
    cfg["output_dir"] = str(tmp_path / "parent")
    source = train(cfg, stop_after=2, system_adapter=tiny_system())
    parent = torch.load(source, weights_only=True)
    manifest = Path(cfg["data"]["train_manifest"])
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    for row in rows:
        row.update(sha256=file_ref(row["image"])["sha256"], dataset="real")
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    real = load_reference_records(manifest)
    image = tmp_path / "synthetic.png"
    Image.fromarray(np.full((40, 60, 3), [30, 80, 150], dtype=np.uint8)).save(image)
    generated = tmp_path / "synthetic.jsonl"
    generated.write_text(
        json.dumps(
            dict(
                image=str(image),
                sha256=file_ref(image)["sha256"],
                dataset="procthor_empty",
                group="generated",
            )
        )
        + "\n"
    )
    records = ReplacementRecords(real, load_reference_records(generated))
    cfg["data"]["rgb_replacement"] = dict(spec(parent["cursor"]), manifest=file_ref(generated))
    cfg["train"].update(steps=4, max_seconds=600, lr_schedule="constant")
    cfg["output_dir"] = str(tmp_path / "large")
    perf = dict(
        profile(2, 0, batch, batch),
        data_pipeline="prepared",
        compact_transport=True,
        gpu_prefetch=False,
        base_effective_batch=32,
        schedule_ratio=batch / 32,
    )
    keys = [
        list(k)
        for b in TrueBatchSampler(
            records,
            cfg["data"]["rgb_replacement"],
            4,
            cfg["data"]["buckets"],
            cfg["train"]["seed"],
            0,
            2 * batch // 4,
            actual_batch=batch,
        )
        for k in b
    ]
    calls = []

    class Audit(RGBInputAudit):
        def load_records(self, path):
            return records

        def check_fork(self, *args):
            pass

        def check_batch_transition(self, *args):
            pass

        def before_updates(self, system, adapter, statistics, records, config, output):
            initial_state_check(
                torch.load(output / "initial.pt", weights_only=True), parent, cfg, perf
            )
            original = system.encode

            def encode(image, *args, **kwargs):
                calls.append(len(image))
                return original(image, *args, **kwargs)

            monkeypatch.setattr(system, "encode", encode)
            monkeypatch.setattr(
                system.decoder, "forward", lambda *a: pytest.fail("Geometry decoder called")
            )

    audit = Audit(cfg["output_dir"], 2, 4, batch // 4, batch, keys)
    monkeypatch.setattr(
        "radio_moge.train.evaluate", lambda *a, **kw: {f"feature/{k}": 0.0 for k in cfg["loss"]}
    )
    with patch(
        "radio_moge.train.ReplacementBatches",
        lambda *a, **kw: TrueBatchSampler(*a, actual_batch=batch, **kw),
    ):
        final = torch.load(
            train(
                cfg,
                fork_from=source,
                data_fork=True,
                batch_fork=True,
                performance=perf,
                reference=audit,
                system_adapter=tiny_system(),
            ),
            weights_only=True,
        )
    assert calls == [batch, batch] and final["cursor"] == 2 * batch // 4 and final["step"] == 4
    assert equal_finite(parent["statistics"], final["statistics"])
    assert {int(v["step"]) for v in final["optimizer"]["state"].values()} == {4}
    logs = [
        json.loads(line)
        for line in (Path(cfg["output_dir"]) / "train.jsonl").read_text().splitlines()
    ]
    assert all(
        r["actual_forward_batch"] == batch and r["model_forwards_per_update"] == 1 for r in logs
    )
    assert all(r["effective_batch"] == batch and r["input_wait_seconds"] >= 0 for r in logs)
    data = TrainingImages(records, True, compact_transport=True)
    assert data[keys[0]].shape == torch.Size([3, *keys[0][2:]])


def test_coverage_rejects_shape_split_and_duplicate():
    records = dummy_records(2, 17)
    for i, row in enumerate(records):
        row.update(image=f'/image-{i}', sha256=str(i), dataset='real' if i < 2 else 'procthor_empty')
    cov = BatchCoverage(2, records[2:], 128)
    rows = []
    for i in range(128):
        n = 2 + i // 8 if i % 8 == 7 else i % 2
        r = records[n]
        rows.append(
            dict(key=[n, i, 480, 640], image=r["image"], sha256=r["sha256"], dataset=r["dataset"])
        )
    indices = cov.validate(rows)
    assert len(indices) == 16
    cov.commit(indices)
    with pytest.raises(ValueError):
        cov.validate(rows)
    cov = BatchCoverage(2, records[2:], 128)
    rows[0]["key"][2:] = [640, 480]
    with pytest.raises(ValueError):
        cov.validate(rows)
