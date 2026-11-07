from copy import deepcopy
import json

import pytest
import torch

from radio_moge.data import DeterministicBatches
from radio_moge.rgb_input_audit import ReferenceRecords
from radio_moge.rgb_replacement import ReplacementRecords
from radio_moge.optimizer_trial import CoreOptimizerTrial, optimizer_counts, policy
from run_room_muon_continuation import (
    MultiEpochBatches,
    MultiEpochCoverage,
    check_continued_counts,
    initial_state_check,
    check_contract,
    EPOCHS,
    BATCH,
    IMAGES,
    UPDATES,
    UPDATES_PER_EPOCH,
    SECONDS,
    CADENCE,
    COVERAGE_GOAL,
    BUCKET_POLICY,
)
from verify_corpus_stage_startup import equal_finite
from test_optimizer_trial import model


def records(n=19):
    def rows(size, dataset):
        value = ReferenceRecords(
            [
                dict(image=f"{dataset}/{i}.jpg", sha256=f"{i:064x}", dataset=dataset)
                for i in range(size)
            ]
        )
        value.metadata_sha256 = dataset
        return value

    return ReplacementRecords(rows(31, "real"), rows(n, "structural_rooms"))


def sampler(rows, start=0, count=192):
    return MultiEpochBatches(
        rows,
        dict(every_images=8, real_start_cursor=37, seed=43),
        4,
        [[4, 6], [6, 4], [4, 8]],
        43,
        start,
        count,
    )


@pytest.mark.parametrize("n", [1, 16, 19, 32, 33])
def test_all_three_passes_tails_real_keys_crops_and_arbitrary_resume(n):
    rows = records(n)
    groups = ((n + 15) // 16) * 32
    actual = list(sampler(rows, count=groups * 3 + 32))
    original = list(DeterministicBatches(31, 4, [[4, 6], [6, 4], [4, 8]], 43, 37, len(actual)))
    tracker = MultiEpochCoverage(31, rows[31:])
    orders = []
    for epoch in range(3):
        section = actual[epoch * groups : (epoch + 1) * groups]
        indices = [k[0] - 31 for group in section for k in group if k[0] >= 31]
        assert sorted(indices) == list(range(n))
        orders.append(indices)
    if n > 1:
        assert len({tuple(o) for o in orders}) > 1
    for index, (group, old) in enumerate(zip(actual, original, strict=True)):
        assert [k[1] for k in group] == [k[1] for k in old]
        for k, o in zip(group, old, strict=True):
            if k[0] < 31:
                assert k[:2] == o[:2]
        if index >= groups * 3:
            assert all(k[0] < 31 for k in group)
    for start in [1, 31, groups - 1, groups, groups + 1, groups * 2 + 1]:
        assert list(sampler(rows, start=start, count=13)) == actual[start : start + 13]
    for i in range(0, groups * 3, 32):
        keys = [k for g in actual[i : i + 32] for k in g]
        assert len({k[2:] for k in keys}) == 1
        payload = [dict(rows[k[0]], key=list(k)) for k in keys]
        tracker.commit(tracker.validate(payload))
    assert tracker.report()["every_image_once_per_epoch"]
    assert tracker.report()["committed"] == n * 3
    assert all(t.counts == [1] * n for t in tracker.trackers)


def test_duplicate_inside_pass_rejected_not_hidden_as_next_epoch():
    rows = records()
    keys = [k for g in sampler(rows, count=64) for k in g]
    tracker = MultiEpochCoverage(31, rows[31:])
    first = [dict(rows[k[0]], key=list(k)) for k in keys[:128]]
    tracker.commit(tracker.validate(first))
    second = [dict(rows[k[0]], key=list(k)) for k in keys[128:]]
    slot = next(i for i, r in enumerate(second) if r["key"][0] >= 31)
    second[slot] = next(r for r in first if r["key"][0] >= 31)
    with pytest.raises(ValueError, match="repeated"):
        tracker.validate(second)


@pytest.mark.parametrize("bad", [None, "key", "group", "duplicate", "partial"])
def test_independent_saved_prefix_checks_all_pass_boundaries(tmp_path, bad):
    from review_room_muon_prefix import committed_prefix

    rows = records()
    planned = [list(k) for g in sampler(rows, count=192) for k in g]
    payload = []
    for number in range(6):
        images = [dict(rows[k[0]], key=k) for k in planned[number * 128 : (number + 1) * 128]]
        payload.append(
            dict(
                update=101 + number,
                logical_images=images,
                compute_groups=[
                    dict(
                        shape=[128, 3, *images[0]["key"][2:]], logical_image_slots=list(range(128))
                    )
                ],
            )
        )
    if bad == "key":
        payload[2]["logical_images"][1]["key"] = [0, 0, 4, 6]
    if bad == "group":
        payload[2]["compute_groups"][0]["shape"][0] = 64
    if bad == "duplicate":
        payload[1]["logical_images"][7] = deepcopy(payload[0]["logical_images"][7])
        planned[128 + 7] = payload[1]["logical_images"][7]["key"]
    if bad == "partial":
        payload.pop()
    path = tmp_path / "inputs.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in payload))
    if bad:
        with pytest.raises(ValueError):
            committed_prefix(path, planned, rows[31:], 31, 100, 6, 128)
    else:
        proof = committed_prefix(path, planned, rows[31:], 31, 100, 6, 128)
        assert proof["synthetic_coverage"]["every_image_once_per_epoch"]
        assert proof["source_counts"]["structural_rooms"] == 57


@pytest.mark.parametrize("bad", [None, "partial", "missing", "duplicates", "hash", "geometry"])
def test_final_closure_requires_all_three_passes(bad):
    from test_room_lr_pair import closure as old_closure
    from verify_room_muon_finish import check_closure

    plan, cfg, completion, review, audit, coverage, _ = old_closure()
    new_cfg, _, new_plan = contract()
    plan.update(new_plan)
    cfg.update(new_cfg)
    endpoint = plan["source_step"] + UPDATES
    completion.update(new_updates=UPDATES, budget_complete=True)
    completion["stop_report"]["step"] = endpoint
    review.update(new_updates=UPDATES, step=endpoint, budget_complete=True)
    sources = dict(
        procthor_empty=40000 * 3, structural_rooms=10787 * 3, original=UPDATES * 128 - IMAGES * 3
    )
    audit.update(
        optimizer_updates=UPDATES,
        cursor=UPDATES * 32,
        presentations=UPDATES * 128,
        source_counts=sources,
        resolution_counts={"three": UPDATES * 128},
    )
    summary = dict(
        completed_epochs=3,
        committed=IMAGES * 3,
        expected=IMAGES * 3,
        repeated_within_epochs=0,
        every_image_once_per_epoch=True,
    )
    coverage.update(step=endpoint, updates=UPDATES, budget_completed=True, **summary)
    review["actual_inputs"].update(
        presentations=UPDATES * 128,
        source_counts=sources,
        synthetic_coverage=summary,
        synthetic_identity_and_no_within_epoch_repeats_verified=True,
    )
    counts = [[1] * IMAGES for _ in range(3)]
    if bad == "partial":
        completion["budget_complete"] = False
    if bad == "missing":
        counts[2][0] = 0
    if bad == "duplicates":
        counts[1][0] = 2
    if bad == "hash":
        review["actual_inputs"]["actual_input_prefix"]["sha256"] = "bad"
    if bad == "geometry":
        cfg["ordinary_geometry"] = {}
    args = (plan, cfg, completion, review, audit, coverage, counts)
    if bad:
        with pytest.raises(ValueError):
            check_closure(*args)
    else:
        assert check_closure(*args) == endpoint


def source_counts():
    return dict(
        parameters=100,
        core_parameters=16,
        auxiliary_parameters=84,
        adam_states=84,
        core_adam_steps=[],
        auxiliary_adam_steps=[263121],
        muon_states=16,
        muon_group_steps=[1000],
        new_optimizer_steps=1000,
    )


@pytest.mark.parametrize(
    "bad", ["muon_group_steps", "new_optimizer_steps", "auxiliary_adam_steps", None]
)
def test_continued_counts_keep_both_optimizer_ages(bad):
    source = source_counts()
    final = deepcopy(source)
    final.update(
        auxiliary_adam_steps=[263121 + UPDATES],
        muon_group_steps=[1000 + UPDATES],
        new_optimizer_steps=1000 + UPDATES,
    )
    if bad:
        final[bad] = [UPDATES] if isinstance(final[bad], list) else UPDATES
        with pytest.raises(ValueError, match="reset"):
            check_continued_counts(final, source, UPDATES)
    else:
        check_continued_counts(final, source, UPDATES)


def contract():
    from test_room_lr_pair import contract as previous

    config, parent, _ = previous()
    parent["config"]["optimizer_trial"] = policy("core-muon")
    parent["config"]["tracking"] = {"enabled": False}
    parent["config"]["evaluation"]["checkpoint_metrics"] = {}
    parent["step"] = 263121
    parent["cursor"] = 32000
    config = deepcopy(parent["config"])
    config["train"].update(
        steps=263121 + UPDATES,
        max_seconds=SECONDS,
        eval_every=CADENCE,
        save_every=CADENCE,
        eval_at_start=False,
    )
    cursor = parent["config"]["data"]["rgb_replacement"]["real_start_cursor"] + 32000
    config["data"]["rgb_replacement"]["real_start_cursor"] = cursor
    plan = dict(
        authorized=True,
        actual_batch=BATCH,
        epochs=EPOCHS,
        updates_per_epoch=UPDATES_PER_EPOCH,
        additional_updates=UPDATES,
        synthetic_images=IMAGES,
        synthetic_presentations=IMAGES * EPOCHS,
        coverage_goal=COVERAGE_GOAL,
        source_step=263121,
        source_parent_cursor=32000,
        source_cursor=cursor,
        maximum_training_seconds=SECONDS,
        bucket_policy=BUCKET_POLICY,
        source_optimizer_counts=source_counts(),
        synthetic_manifest=config["data"]["rgb_replacement"]["manifest"],
    )
    return config, parent, plan


@pytest.mark.parametrize(
    "bad", [None, "lr", "loss", "momentum", "augmentation", "budget", "cursor"]
)
def test_continuation_contract_rejects_unrelated_changes(bad):
    config, parent, plan = contract()
    if bad == "lr":
        config["train"]["lr"] *= 2
    if bad == "loss":
        config["loss"]["cosine"] = 0.1
    if bad == "momentum":
        config["optimizer_trial"]["muon"]["mu"] = 0.9
    if bad == "augmentation":
        config["data"]["augment"] = {"changed": True}
    if bad == "budget":
        plan["additional_updates"] += 1
    if bad == "cursor":
        config["data"]["rgb_replacement"]["real_start_cursor"] += 1
    if bad:
        with pytest.raises(ValueError):
            check_contract(config, parent, plan)
    else:
        check_contract(config, parent, plan)


def test_production_continuation_restores_hybrid_optimizer_exactly(tmp_path):
    from radio_moge.train import train
    from radio_moge.rgb_input_audit import RGBInputAudit, load_reference_records
    from radio_moge.corpus_admission import file_ref
    from test_performance import feature_config, profile
    from test_training import tiny_system

    cfg = feature_config(tmp_path)
    cfg["train"].update(
        lr=1e-5,
        weight_decay=0.01,
        eval_at_start=False,
        log_every=1,
        eval_every=100,
        lr_schedule="constant",
    )
    cfg["output_dir"] = str(tmp_path / "parent")
    system, _ = tiny_system()
    adapter = model()
    parent_path = train(cfg, stop_after=2, system_adapter=(system, adapter))
    state = torch.load(parent_path, weights_only=True)
    opt = CoreOptimizerTrial(adapter.named_parameters(), cfg["train"], policy("core-muon"))
    opt.load_parent_adam(state["optimizer"])
    for _ in range(3):
        for p in adapter.parameters():
            p.grad = torch.randn_like(p)
        opt.step()
    state["adapter"] = deepcopy(adapter.state_dict())
    state["optimizer"] = deepcopy(opt.state_dict())
    state["step"] = 5
    state["config"]["optimizer_trial"] = policy("core-muon")
    torch.save(state, parent_path)
    manifest = __import__("pathlib").Path(cfg["data"]["train_manifest"])
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    for row in rows:
        row.update(sha256=file_ref(row["image"])["sha256"], dataset="real")
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    rows = load_reference_records(manifest)
    cfg["optimizer_trial"] = policy("core-muon")
    cfg["train"].update(steps=7, max_seconds=600)
    cfg["output_dir"] = str(tmp_path / "continued")
    perf = dict(
        profile(5, 0, 16, 8), data_pipeline="prepared", compact_transport=True, gpu_prefetch=False
    )

    class Audit(RGBInputAudit):
        preserves_core_optimizer = True

        def load_records(self, _):
            return rows

        def check_fork(self, *args):
            pass

        def make_optimizer(self, adapter, settings):
            return CoreOptimizerTrial(adapter.named_parameters(), settings, policy("core-muon"))

        def restore_optimizer(self, optimizer, parent):
            optimizer.load_state_dict(parent["optimizer"])

        def before_updates(self, system, adapter, statistics, records, config, output):
            actual = torch.load(output / "initial.pt", weights_only=True)
            initial_state_check(actual, state, cfg, perf)
            corrupt = deepcopy(actual)
            next(iter(corrupt["optimizer"]["muon"]["state"].values()))["momentum"].zero_()
            with pytest.raises(ValueError, match="optimizer"):
                initial_state_check(corrupt, state, cfg, perf)

    audit = Audit(cfg["output_dir"], 5, 4, 4, 8)
    system, _ = tiny_system()
    final = train(
        cfg,
        fork_from=parent_path,
        data_fork=True,
        performance=perf,
        reference=audit,
        system_adapter=(system, model()),
    )
    actual = torch.load(final, weights_only=True)
    check_continued_counts(
        optimizer_counts(actual["optimizer"]), optimizer_counts(state["optimizer"]), 2
    )
    assert actual["step"] == 7 and actual["cursor"] == 8
    assert "core_matrix_optimizer_state" not in actual["fork_provenance"]["reset"]
    assert "core_Muon_momentum_and_group_counters" in actual["fork_provenance"]["restored"]
    assert equal_finite(state["optimizer"], torch.load(parent_path, weights_only=True)["optimizer"])
