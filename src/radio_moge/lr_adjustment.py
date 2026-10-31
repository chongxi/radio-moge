"""Explicit constant-LR changes on exact-state resume, without resetting coverage."""

from copy import deepcopy
import math
import re
import time


def apply_lr_adjustment(config, state, optimizer, scheduler, request):
    history = deepcopy(state.get("lr_adjustments", []))
    if request.get("authorized") is not True or not re.fullmatch(
        r"[a-z0-9-]{1,64}", request.get("id", "")
    ):
        raise ValueError("An identified, authorized LR adjustment is required")
    if any(item["id"] == request["id"] for item in history):
        raise ValueError("Learning-rate adjustment already applied")
    if (
        config["train"].get("lr_schedule") != "constant"
        or not state.get("fork_provenance", {}).get("data_change")
        or config["data"].get("coverage_epochs") != 1
    ):
        raise ValueError("LR adjustment requires an existing constant-LR coverage run")
    if (
        state["step"] != request["expected_step"]
        or state["cursor"] != request["expected_cursor"]
        or state["runtime_limits"] != request["runtime_limits"]
        or state["fork_provenance"]["budget_started_at"] != request["budget_started_at"]
    ):
        raise ValueError("Resume step, data cursor or original budget changed")
    previous, target = request["previous_lr"], request["learning_rate"]
    if (
        not all(
            type(x) in (int, float) and math.isfinite(x) and 0 < x <= 1e-3
            for x in (previous, target)
        )
        or previous == target
    ):
        raise ValueError("Expected distinct finite positive learning rates")
    if any(group["lr"] != previous for group in optimizer.param_groups):
        raise ValueError("Checkpoint learning rate differs from reviewed source")
    # Preserve moments, Adam counters, scheduler position, RNG, bests and data cursor.
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] = target
    scheduler.base_lrs = [target] * len(optimizer.param_groups)
    scheduler._last_lr = list(scheduler.base_lrs)
    history.append(
        dict(
            request,
            applied_at=time.time(),
            effective_after_step=state["step"],
            scheduler_last_epoch=scheduler.last_epoch,
            optimizer_state_reset=False,
            data_cursor_reset=False,
            budget_extended=False,
        )
    )
    return history


def expected_constant_lr(initial_lr, step, history):
    value = initial_lr
    previous_step = -1
    for change in history:
        boundary = change["effective_after_step"]
        if boundary <= previous_step or change["previous_lr"] != value:
            raise ValueError("Inconsistent learning-rate adjustment history")
        previous_step = boundary
        if step <= boundary:
            return value
        value = change["learning_rate"]
    return value
