from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from radio_moge.architecture_screen import JointScreenAdapter
from radio_moge.optimizer_trial import (
    CoreOptimizerTrial,
    policy,
    partition,
    optimizer_counts,
    check_counts,
)
from radio_moge.checkpoint import rng_state
from verify_corpus_stage_startup import equal_finite
from test_architecture_screen import small_spec
from test_room_lr_pair import contract
from run_room_optimizer_arm import check_contract, compare_arm_configs


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def model():
    value = JointScreenAdapter(
        small_spec(
            blocks=4,
            registers=4,
            gate=True,
            qk_norm=True,
            layer_scale=0.01,
            activation="swiglu",
            position="rope",
        ),
        8,
        8,
        6,
        6,
    )
    value.config = dict(
        kind="joint_attention",
        target_dim=6,
        target_summary_dim=6,
        spatial_dim=8,
        summary_dim=8,
        indices=[0, 1, 2, 3],
    )
    value.indices = (0, 1, 2, 3)
    return value


def warm(module, steps=3):
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-5, weight_decay=0.01)
    for _ in range(steps):
        for p in module.parameters():
            p.grad = torch.randn_like(p)
        optimizer.step()
    return deepcopy(optimizer.state_dict())


def test_partition_actual_parameter_roles_not_tensor_dimension():
    m = model()
    named, core, aux = partition(m.named_parameters())
    assert (len(named), len(core), len(aux)) == (100, 16, 84)
    assert "registers" in dict(aux) and dict(aux)["registers"].ndim == 2
    assert "summary_input.weight" in dict(aux) and "spatial_output.1.weight" in dict(aux)
    assert all(p.ndim == 2 for _, p in core)
    with pytest.raises(ValueError, match="aliased"):
        partition(named + [named[0]])


@pytest.mark.parametrize("mode", ["core-reset-adamw", "core-muon"])
def test_source_transition_and_serialized_resume_are_exact(mode):
    torch.manual_seed(43)
    m = model()
    parent = warm(m)
    original = deepcopy(parent)
    before_rng = rng_state()
    opt = CoreOptimizerTrial(m.named_parameters(), dict(lr=1e-5, weight_decay=0.01), policy(mode))
    opt.load_parent_adam(parent)
    assert equal_finite(rng_state(), before_rng)
    check_counts(optimizer_counts(opt.state_dict()), mode, 3, 0)
    core_names = set(opt.identity()["core_names"])
    for i, (n, p) in enumerate(m.named_parameters()):
        if n not in core_names:
            assert equal_finite(opt.state[p], parent["state"][i])
    for p in m.parameters():
        p.grad = torch.randn_like(p)
    opt.step()
    check_counts(optimizer_counts(opt.state_dict()), mode, 3, 1)
    assert equal_finite(parent, original), "source state must not alias live CPU state"
    saved_model, saved_opt = deepcopy(m.state_dict()), deepcopy(opt.state_dict())
    restored = model()
    restored.load_state_dict(saved_model)
    other = CoreOptimizerTrial(
        restored.named_parameters(), dict(lr=1e-5, weight_decay=0.01), policy(mode)
    )
    other.load_state_dict(saved_opt)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 0.5)
    scheduler_other = torch.optim.lr_scheduler.LambdaLR(other, lambda _: 0.5)
    for a, b in zip(m.parameters(), restored.parameters(), strict=True):
        a.grad = torch.randn_like(a)
        b.grad = a.grad.clone()
    opt.step()
    other.step()
    scheduler.step()
    scheduler_other.step()
    assert equal_finite(m.state_dict(), restored.state_dict())
    assert equal_finite(opt.state_dict(), other.state_dict())
    check_counts(optimizer_counts(opt.state_dict()), mode, 3, 2)
    opt.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in m.parameters())
    with pytest.raises(ValueError, match="explicit parent"):
        opt.load_state_dict(parent)
    corrupt = deepcopy(saved_opt)
    corrupt["optimizer_trial"]["core_names"] = []
    with pytest.raises(ValueError, match="identity differs"):
        opt.load_state_dict(corrupt)


@pytest.mark.parametrize("mode", ["core-reset-adamw", "core-muon"])
def test_auxiliary_updates_match_native_adamw_and_core_updates_once(mode):
    m = model()
    parent = warm(m)
    original = deepcopy(m)
    baseline = torch.optim.AdamW(original.parameters(), lr=1e-5, weight_decay=0.01)
    baseline.load_state_dict(deepcopy(parent))
    trial = CoreOptimizerTrial(m.named_parameters(), dict(lr=1e-5, weight_decay=0.01), policy(mode))
    trial.load_parent_adam(parent)
    core = set(trial.identity()["core_names"])
    if mode == "core-reset-adamw":
        for n, p in original.named_parameters():
            if n in core:
                del baseline.state[p]
    else:
        from dion import Muon

        ref_muon = Muon(
            [p for n, p in original.named_parameters() if n in core],
            lr=1e-5,
            weight_decay=0.01,
            **policy(mode)["muon"],
        )
    for (name, a), (_, b) in zip(m.named_parameters(), original.named_parameters(), strict=True):
        a.grad = torch.randn_like(a)
        b.grad = a.grad.clone()
    trial.step()
    if mode == "core-muon":
        ref_muon.step()
        for n, p in original.named_parameters():
            if n in core:
                p.grad = None
    baseline.step()
    assert equal_finite(m.state_dict(), original.state_dict())


def test_muon_missing_gradient_does_not_decay_or_create_adam_state():
    m = model()
    parent = warm(m)
    opt = CoreOptimizerTrial(
        m.named_parameters(), dict(lr=1e-5, weight_decay=0.01), policy("core-muon")
    )
    opt.load_parent_adam(parent)
    opt.zero_grad(set_to_none=True)
    before = deepcopy(m.state_dict())
    opt.step()
    assert equal_finite(before, m.state_dict())
    assert len(opt.state) == 84 and len(opt.muon.state) == 16


@pytest.mark.parametrize("mode", ["core-reset-adamw", "core-muon"])
@pytest.mark.parametrize("bad", [None, "loss", "lr", "cursor", "policy", "budget", "baseline"])
def test_scoped_configuration_rejects_unrelated_changes(mode, bad):
    cfg, parent, plan = contract()
    cfg["optimizer_trial"] = policy(mode)
    plan.update(arm=mode, reference_arm=None)
    if bad == "loss":
        cfg["loss"]["cosine"] = 0.1
    if bad == "lr":
        cfg["train"]["lr"] *= 2
    if bad == "cursor":
        cfg["data"]["rgb_replacement"]["real_start_cursor"] += 1
    if bad == "policy":
        cfg["optimizer_trial"]["muon"]["nesterov"] = False
    if bad == "budget":
        plan["additional_updates"] += 1
    if bad == "baseline":
        plan["reference_arm"] = "legacy-unchecked"
    if bad:
        with pytest.raises(ValueError):
            check_contract(cfg, parent, plan)
    else:
        check_contract(cfg, parent, plan)
        other = deepcopy(cfg)
        other["optimizer_trial"] = policy("core-muon")
        compare_arm_configs(cfg, other)


def test_production_loop_optimizer_hook_and_initial_state(tmp_path):
    from radio_moge.train import train
    from radio_moge.rgb_input_audit import RGBInputAudit
    from radio_moge.rgb_input_audit import load_reference_records
    from radio_moge.corpus_admission import file_ref
    from test_performance import feature_config, profile
    from test_training import tiny_system

    cfg = feature_config(tmp_path)
    cfg["train"].update(
        lr=1e-5, weight_decay=0.01, eval_at_start=False, log_every=1, eval_every=100
    )
    cfg["output_dir"] = str(tmp_path / "parent")
    system, _ = tiny_system()
    parent_path = train(cfg, stop_after=2, system_adapter=(system, model()))
    parent = torch.load(parent_path, weights_only=True)
    manifest = Path(cfg["data"]["train_manifest"])
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    for row in records:
        row.update(sha256=file_ref(row["image"])["sha256"], dataset="real")
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records))
    rows = load_reference_records(manifest)
    cfg["train"].update(steps=4, max_seconds=600, lr_schedule="constant")
    cfg["optimizer_trial"] = policy("core-muon")
    cfg["output_dir"] = str(tmp_path / "trial")
    perf = dict(
        profile(2, 0, 16, 8), data_pipeline="prepared", compact_transport=True, gpu_prefetch=False
    )

    class Audit(RGBInputAudit):
        def load_records(self, _):
            return rows

        def check_fork(self, *args):
            pass

        def make_optimizer(self, adapter, settings):
            self.optimizer = CoreOptimizerTrial(
                adapter.named_parameters(), settings, cfg["optimizer_trial"]
            )
            return self.optimizer

        def restore_optimizer(self, optimizer, state):
            optimizer.load_parent_adam(state["optimizer"])

        def before_updates(self, system, adapter, statistics, records, config, output):
            initial = torch.load(output / "initial.pt", weights_only=True)
            for section in ("adapter", "statistics", "rng"):
                assert equal_finite(initial[section], parent[section])
            check_counts(optimizer_counts(initial["optimizer"]), "core-muon", 2, 0)

    audit = Audit(cfg["output_dir"], 2, 4, 4, 8)
    system, _ = tiny_system()
    final = train(
        cfg,
        fork_from=parent_path,
        data_fork=True,
        performance=perf,
        reference=audit,
        system_adapter=(system, model()),
    )
    state = torch.load(final, weights_only=True)
    check_counts(optimizer_counts(state["optimizer"]), "core-muon", 2, 2)
    assert state["step"] == 4 and state["cursor"] == 8
    assert "optimizer_moments" not in state["fork_provenance"]["restored"]
    assert state["fork_provenance"]["optimizer_trial"] == cfg["optimizer_trial"]


@pytest.mark.parametrize("mode", ["core-reset-adamw", "core-muon"])
def test_final_counter_verifier_rejects_global_age_for_reset_core(mode):
    m = model()
    parent = warm(m)
    opt = CoreOptimizerTrial(m.named_parameters(), dict(lr=1e-5, weight_decay=0.01), policy(mode))
    opt.load_parent_adam(parent)
    for p in m.parameters():
        p.grad = torch.randn_like(p)
    opt.step()
    counts = optimizer_counts(opt.state_dict())
    check_counts(counts, mode, 3, 1)
    counts["new_optimizer_steps"] = 4
    with pytest.raises(ValueError, match="state/counters"):
        check_counts(counts, mode, 3, 1)


def test_fixed_final_comparison_rejects_unmatched_inputs_and_partial_budget():
    from review_room_optimizer_pair import contrast, VIEWS

    a = dict(
        new_updates=1000,
        budget_complete=True,
        step=263121,
        parent={"sha256": "same"},
        actual_inputs=dict(actual_input_prefix=dict(sha256="same")),
        views={v: dict(metrics=dict(cos=dict(before=0.8, final=0.9))) for v in VIEWS},
        against_parent=dict(
            metrics={
                k: dict(final=x)
                for k, x in dict(spatial_cosine=0.9, summary_cosine=0.9, lidar_abs_rel=0.1).items()
            }
        ),
    )
    b = deepcopy(a)
    result = contrast(a, b)
    assert not result["primary_joint"]["all_three_improve"]
    assert set(result["all_six_view_metrics"]) == VIEWS
    b["actual_inputs"]["actual_input_prefix"]["sha256"] = "different"
    with pytest.raises(ValueError, match="actual input"):
        contrast(a, b)
    b = deepcopy(a)
    b["budget_complete"] = False
    with pytest.raises(ValueError):
        contrast(a, b)
