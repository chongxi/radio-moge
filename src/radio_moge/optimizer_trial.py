"""Explicit core-matrix optimizer experiments; the ordinary AdamW path is unchanged.

Dion is imported only for the Muon arm from a separately pinned source tree.
Auxiliary AdamW uses PyTorch, including its original moments and per-tensor age.
"""

from copy import deepcopy
import re

import torch


DION_REVISION = "7692479288f928a3d61d1cc58137b4dbe4f71000"
CORE_PATTERN = re.compile(r"blocks\.[0-9]+\.(qkv|projection|fc1|fc2)\.weight")
MODES = ("core-reset-adamw", "core-muon")


def policy(mode):
    if mode not in MODES:
        raise ValueError("Unknown scoped optimizer experiment")
    return dict(
        schema=1,
        mode=mode,
        core_selection="blocks.*.{qkv,projection,fc1,fc2}.weight",
        auxiliary="torch.optim.AdamW; preserve source moments and counters",
        core_initial_state="fresh; no transfer of Adam moments into Muon",
        core_lr=1e-5,
        dion_revision=DION_REVISION,
        muon=dict(
            mu=0.95,
            nesterov=True,
            adjust_lr="rms_norm",
            epsilon=1e-8,
            flatten=False,
            cautious_wd=False,
            use_polar_express=False,
            use_gram_newton_schulz=False,
            use_triton=False,
        ),
        matrix_partition="stored unsplit matrices; no QKV/head/SwiGLU splitting",
        lr_scaling="Dion rms_norm: 0.2 * sqrt(max(rows, columns)); decay uses base LR",
        numerical_orthogonalizer="pinned Dion five-stage Newton-Schulz coefficients",
    )


def partition(named_parameters):
    named = list(named_parameters)
    if len({n for n, _ in named}) != len(named) or len({id(p) for _, p in named}) != len(named):
        raise ValueError("Parameter names or objects are aliased")
    core = [(n, p) for n, p in named if CORE_PATTERN.fullmatch(n)]
    if not core or any(p.ndim != 2 or not p.requires_grad for _, p in core):
        raise ValueError("Muon requires explicitly named trainable core matrices")
    auxiliary = [(n, p) for n, p in named if not CORE_PATTERN.fullmatch(n)]
    if any(not p.requires_grad for _, p in auxiliary):
        raise ValueError("Frozen backbones must not enter the adapter optimizer")
    return named, core, auxiliary


class CoreOptimizerTrial(torch.optim.AdamW):
    """One Torch optimizer handle, native AdamW auxiliaries, optional Dion core.

    The parent AdamW parameter order is retained. The Muon arm hides core grads
    only during AdamW.step, so every tensor is updated exactly once. Nested Muon
    state and a named partition are serialized and checked on every load.
    """

    def __init__(self, named_parameters, settings, trial):
        if trial != policy(trial.get("mode")):
            raise ValueError("Optimizer policy differs from the frozen implementation")
        self.named, self.core, self.auxiliary = partition(named_parameters)
        self.trial = deepcopy(trial)
        self.committed_steps = 0
        self.muon = None
        super().__init__(
            [p for _, p in self.named], lr=settings["lr"], weight_decay=settings["weight_decay"]
        )
        if trial["mode"] == "core-muon":
            from dion import Muon

            self.muon = Muon(
                [p for _, p in self.core],
                lr=trial["core_lr"],
                weight_decay=settings["weight_decay"],
                **trial["muon"],
            )

    def identity(self):
        return dict(
            policy=self.trial,
            names=[n for n, _ in self.named],
            shapes=[list(p.shape) for _, p in self.named],
            core_names=[n for n, _ in self.core],
            auxiliary_names=[n for n, _ in self.auxiliary],
        )

    def load_parent_adam(self, state):
        if "optimizer_trial" in state or len(state["param_groups"]) != 1:
            raise ValueError("Switching requires the declared plain AdamW parent")
        group = state["param_groups"][0]
        if len(group["params"]) != len(self.named) or set(state["state"]) != set(group["params"]):
            raise ValueError("Parent AdamW state must cover the complete named adapter")
        # Torch may retain the CPU step tensor even for CUDA parameters. Copy
        # all source buffers so neither moments nor counters can alias a receipt.
        source = deepcopy(state)
        super().load_state_dict(source)
        for _, p in self.core:
            del self.state[p]
        self.committed_steps = 0
        if self.muon is not None:
            if any(torch.count_nonzero(s["momentum"]) for s in self.muon.state.values()):
                raise ValueError("Muon momentum must be fresh before parent transition")

    def state_dict(self):
        value = super().state_dict()
        value["optimizer_trial"] = dict(self.identity(), committed_steps=self.committed_steps)
        if self.muon is not None:
            value["muon"] = self.muon.state_dict()
        return value

    def load_state_dict(self, state_dict):
        proof = state_dict.get("optimizer_trial")
        if (
            not proof
            or {k: v for k, v in proof.items() if k != "committed_steps"} != self.identity()
        ):
            raise ValueError(
                "Use explicit parent transition; checkpoint optimizer identity differs"
            )
        if ("muon" in state_dict) != (self.muon is not None):
            raise ValueError("Muon state missing or present in the reset control")
        # A gate may restore this same saved snapshot more than once. Native
        # load_state_dict can alias same-device buffers, including host Adam
        # counters on CUDA. Never let a subsequent step mutate that snapshot.
        state_dict = deepcopy(state_dict)
        super().load_state_dict({k: state_dict[k] for k in ("state", "param_groups")})
        self.committed_steps = int(proof["committed_steps"])
        if self.muon is not None:
            self.muon.load_state_dict(state_dict["muon"])
            if any(p in self.state for _, p in self.core):
                raise ValueError("Muon core also has Adam state")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.muon is None:
            super().step()
        else:
            gradients = [p.grad for _, p in self.core]
            try:
                for _, p in self.core:
                    p.grad = None
                super().step()
            finally:
                for (_, p), grad in zip(self.core, gradients, strict=True):
                    p.grad = grad
            # The bounded experiment uses a constant scheduler. Preserve the
            # configured core/auxiliary LR ratio when an explicit scheduler is used.
            for group in self.muon.param_groups:
                group["lr"] = self.trial["core_lr"] * self.param_groups[0]["lr"] / 1e-5
                group["weight_decay"] = self.param_groups[0]["weight_decay"]
            self.muon.step()
        self.committed_steps += 1
        return loss


def optimizer_counts(state):
    proof = state["optimizer_trial"]
    names = proof["names"]
    ids = state["param_groups"][0]["params"]
    by_name = dict(zip(names, ids, strict=True))
    core_steps = [
        int(state["state"][by_name[n]]["step"])
        for n in proof["core_names"]
        if by_name[n] in state["state"]
    ]
    auxiliary_steps = [int(state["state"][by_name[n]]["step"]) for n in proof["auxiliary_names"]]
    return dict(
        parameters=len(names),
        core_parameters=len(proof["core_names"]),
        auxiliary_parameters=len(proof["auxiliary_names"]),
        adam_states=len(state["state"]),
        core_adam_steps=sorted(set(core_steps)),
        auxiliary_adam_steps=sorted(set(auxiliary_steps)),
        muon_states=len(state.get("muon", {}).get("state", {})),
        muon_group_steps=[int(g["step"]) for g in state.get("muon", {}).get("param_groups", [])],
        new_optimizer_steps=proof["committed_steps"],
    )


def check_counts(counts, mode, parent_step, updates, total=100, core=16):
    expected = dict(
        parameters=total,
        core_parameters=core,
        auxiliary_parameters=total - core,
        adam_states=total - core if mode == "core-muon" or updates == 0 else total,
        core_adam_steps=[updates] if mode == "core-reset-adamw" and updates else [],
        auxiliary_adam_steps=[parent_step + updates],
        muon_states=core if mode == "core-muon" else 0,
        muon_group_steps=[updates] if mode == "core-muon" else [],
        new_optimizer_steps=updates,
    )
    if counts != expected:
        raise ValueError(f"Wrong scoped optimizer state/counters: {counts}; expected {expected}")
