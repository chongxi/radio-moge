from copy import deepcopy
from collections import Counter
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
from run_room_true_batch_epoch import TrueBatchSampler
from run_room_lr_arm import (
    check_contract,
    check_performance,
    compare_arm_configs,
    initial_state_check,
    COVERAGE_GOAL,
    UPDATES,
    SECONDS,
    SYNTHETIC_PRESENTATIONS,
)
from review_room_lr_arm import epoch_shape, endpoints
from verify_room_lr_finish import check_closure
from test_room_true_batch import fixture_contract
from test_procthor_rgb_training import dummy_records, spec
from test_rgb_reference import feature_config
from test_training import tiny_system
from test_performance import profile
from verify_corpus_stage_startup import equal_finite


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def contract(lr=1e-5):
    cfg, parent, plan = fixture_contract()
    parent["performance"] = dict(effective_batch=128, compute_batch=128)
    parent["config"]["train"].update(seed=43, update_norm_every=20, weight_decay=0.01)
    parent["config"]["data"]["augment"] = True
    cfg["data"]["augment"] = True
    cfg["train"]["weight_decay"] = 0.01
    cfg["train"].update(
        lr=lr,
        steps=parent["step"] + UPDATES,
        max_seconds=SECONDS,
        save_every=250,
        seed=43,
        update_norm_every=20,
    )
    plan.update(
        arm="lr1e-5" if lr == 1e-5 else "lr2e-5",
        learning_rate=lr,
        coverage_goal=COVERAGE_GOAL,
        additional_updates=UPDATES,
        maximum_training_seconds=SECONDS,
        synthetic_presentations=SYNTHETIC_PRESENTATIONS,
    )
    return cfg, parent, plan


@pytest.mark.parametrize("lr", [1e-5, 2e-5])
@pytest.mark.parametrize(
    "bad", [None, "lr", "beta", "cursor", "steps", "weight", "eval", "budget", "batch", "seed"]
)
def test_narrow_lr_contract(lr, bad):
    cfg, parent, plan = contract(lr)
    if bad == "lr":
        cfg["train"]["lr"] = 4e-5
    if bad == "beta":
        cfg["train"]["weight_decay"] += 0.01
    if bad == "cursor":
        cfg["data"]["rgb_replacement"]["real_start_cursor"] += 1
    if bad == "steps":
        cfg["train"]["steps"] += 1
    if bad == "weight":
        cfg["loss"]["cosine"] = 0.1
    if bad == "eval":
        cfg["train"]["eval_at_start"] = True
    if bad == "budget":
        plan["additional_updates"] = 2000
    if bad == "batch":
        plan["actual_batch"] = 64
    if bad == "seed":
        cfg["train"]["seed"] += 1
    if bad is None:
        check_contract(cfg, parent, plan)
    else:
        with pytest.raises(ValueError):
            check_contract(cfg, parent, plan)


def test_arm_only_lr_and_labels_may_differ():
    a, _, _ = contract(1e-5)
    b, _, _ = contract(2e-5)
    compare_arm_configs(a, b)
    b["data"]["augment"] = not b["data"]["augment"]
    with pytest.raises(ValueError):
        compare_arm_configs(a, b)


@pytest.mark.parametrize("bad", ["compute", "auto", "prefetch"])
def test_same_actual128_runtime(bad):
    cfg, parent, plan = contract()
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
    if bad == "compute":
        perf["compute_batch"] = 64
    if bad == "auto":
        perf["autotune"] = True
    if bad == "prefetch":
        perf["gpu_prefetch"] = False
    with pytest.raises(ValueError):
        check_performance(cfg, parent, plan, perf)


def test_exact_prefix_same_keys_and_buckets_with_no_repeat():
    records = dummy_records(53, 50787)

    def make():
        return [
            k
            for group in TrueBatchSampler(
                records,
                spec(314864),
                4,
                [[480, 640], [640, 480], [400, 640]],
                43,
                0,
                UPDATES * 32,
                actual_batch=128,
            )
            for k in group
        ]

    a, b = make(), make()
    assert a == b and len(a) == 128000
    synthetic = Counter(k[0] for k in a if k[0] >= 53)
    assert len(synthetic) == 16000 and set(synthetic.values()) == {1}
    assert all(len({tuple(k[2:]) for k in a[n : n + 128]}) == 1 for n in range(0, len(a), 128))
    _, _, plan = contract()
    assert epoch_shape(plan) == (128, 1000, 250)
    assert endpoints(262121) == [262371, 262621, 262871, 263121]


def state_fixture():
    cfg = {"train": {"lr": 2e-5}}
    parent = dict(
        step=12,
        cursor=4,
        optimizer={
            "state": {
                0: {
                    "step": torch.tensor(12.0),
                    "exp_avg": torch.tensor([0.2]),
                    "exp_avg_sq": torch.tensor([0.3]),
                }
            },
            "param_groups": [
                dict(
                    lr=1e-5,
                    initial_lr=1e-5,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=0.01,
                    params=[0],
                )
            ],
        },
        adapter={"x": torch.tensor([1.0])},
        statistics={"x": torch.tensor([2.0])},
        rng={"x": torch.tensor([3])},
        adapter_config={"a": 1},
        model_config={"m": 1},
        revisions={"teacher": "fixed"},
    )
    value = deepcopy(parent)
    value.update(
        cursor=0,
        config=cfg,
        performance={},
        scheduler=dict(base_lrs=[2e-5], _last_lr=[2e-5], last_epoch=0, _step_count=1),
    )
    value["optimizer"]["param_groups"][0].update(lr=2e-5, initial_lr=2e-5)
    return value, parent, cfg


@pytest.mark.parametrize(
    "bad", [None, "moment", "counter", "beta", "epsilon", "rng", "adapter", "scheduler"]
)
def test_only_actual_lr_policy_fields_change_initial_state(bad):
    value, parent, cfg = state_fixture()
    if bad == "moment":
        value["optimizer"]["state"][0]["exp_avg"] += 0.01
    if bad == "counter":
        value["optimizer"]["state"][0]["step"] += 1
    if bad == "beta":
        value["optimizer"]["param_groups"][0]["betas"] = (0.9, 0.99)
    if bad == "epsilon":
        value["optimizer"]["param_groups"][0]["eps"] = 1e-6
    if bad == "rng":
        value["rng"]["x"] += 1
    if bad == "adapter":
        value["adapter"]["x"] += 1
    if bad == "scheduler":
        value["scheduler"]["last_epoch"] = 1
    if bad is None:
        initial_state_check(value, parent, cfg, {})
    else:
        with pytest.raises(ValueError):
            initial_state_check(value, parent, cfg, {})


def closure():
    from test_room_epoch_review import closed

    plan, cfg, completion, review, audit, coverage, _ = closed()
    plan.update(
        actual_batch=128,
        additional_updates=1000,
        coverage_goal=COVERAGE_GOAL,
        synthetic_presentations=16000,
    )
    endpoint = plan["source_step"] + 1000
    cfg["train"]["steps"] = endpoint
    completion.update(new_updates=1000, budget_complete=True, full_epoch_complete=False)
    completion["stop_report"]["step"] = endpoint
    review.update(new_updates=1000, step=endpoint, budget_complete=True, full_epoch_complete=False)
    expected = dict(
        expected=50787,
        committed=16000,
        unique=16000,
        missing=34787,
        repeated=0,
        each_image_exactly_once=False,
    )
    sources = dict(procthor_empty=12500, structural_rooms=3500, original=112000)
    audit.update(
        optimizer_updates=1000,
        cursor=32000,
        presentations=128000,
        source_counts=sources,
        resolution_counts={"three": 128000},
    )
    review["actual_inputs"].update(
        presentations=128000, source_counts=sources, synthetic_coverage=expected
    )
    coverage.update(
        step=endpoint, updates=1000, budget_completed=True, full_epoch_completed=False, **expected
    )
    return plan, cfg, completion, review, audit, coverage, [1] * 16000 + [0] * 34787


@pytest.mark.parametrize("bad", [None, "partial", "count", "hash", "geometry", "batch"])
def test_budget_closure_is_not_a_full_epoch(bad):
    args = closure()
    if bad == "partial":
        args[2]["budget_complete"] = False
    if bad == "count":
        args[-1][0] = 0
    if bad == "hash":
        args[3]["actual_inputs"]["actual_input_prefix"]["sha256"] = "different"
    if bad == "geometry":
        args[1]["ordinary_geometry"] = {}
    if bad == "batch":
        args[0]["actual_batch"] = 32
    if bad is None:
        assert check_closure(*args) == args[0]["source_step"] + 1000
    else:
        with pytest.raises(ValueError):
            check_closure(*args)


@pytest.mark.parametrize("lr", [1e-5, 2e-5])
def test_real_trainer_one_forward_and_backward_per_update_with_preserved_adam(
    tmp_path, monkeypatch, lr
):
    batch = 128
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
    cfg["train"].update(steps=4, max_seconds=600, lr_schedule="constant", lr=lr)
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

    assert all(r["lr_used"] == lr for r in logs)


@pytest.mark.parametrize("bad", [None, "input", "budget", "baseline", "metric"])
def test_final_pair_requires_same_actual_population_and_budget(bad):
    from review_room_lr_pair import compare_final, VIEWS

    a = dict(
        new_updates=1000,
        budget_complete=True,
        step=263121,
        parent={"sha256": "same"},
        actual_inputs={"actual_input_prefix": {"sha256": "pixels"}},
        views={name: {"metrics": {"test": {"before": 1.0, "final": 0.9}}} for name in VIEWS},
        against_parent={
            "metrics": {
                k: {"final": v}
                for k, v in dict(
                    spatial_cosine=0.96, summary_cosine=0.93, lidar_abs_rel=0.09
                ).items()
            }
        },
    )
    b = deepcopy(a)
    if bad == "input":
        b["actual_inputs"]["actual_input_prefix"]["sha256"] = "different"
    if bad == "budget":
        b["new_updates"] = 999
    if bad == "baseline":
        b["parent"]["sha256"] = "different"
    if bad == "metric":
        b["views"]["original"]["metrics"]["test"]["before"] = 2.0
    if bad is None:
        contrasts, joint = compare_final(a, b)
        assert len(contrasts) == 6 and contrasts["original"]["test"]["difference"] == 0
        assert not joint["all_three_improve"]
    else:
        with pytest.raises(ValueError):
            compare_final(a, b)
