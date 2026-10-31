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


def reviewed_source_lr(state):
    """Resolve an adjusted constant-LR parent's rate for a new data fork.

    The immutable initial config can predate authorized resume adjustments. A
    ledger is accepted only when the saved optimizer and scheduler agree with
    its final value. This does not authorize an additional LR increase.
    """
    initial = state['config']['train']['lr']
    history = state.get('lr_adjustments', [])
    if not history:
        return initial
    if state['config']['train'].get('lr_schedule') != 'constant':
        raise ValueError('Adjusted data-fork source must use a constant LR')
    value, previous_step, ids = initial, -1, set()
    for change in history:
        identity = change.get('id', '')
        boundary = change.get('effective_after_step')
        target = change.get('learning_rate')
        if (change.get('authorized') is not True
                or not re.fullmatch(r'[a-z0-9-]{1,64}', identity) or identity in ids
                or type(boundary) is not int or not previous_step < boundary <= state['step']
                or change.get('expected_step') != boundary
                or change.get('previous_lr') != value
                or type(target) not in (int, float) or not math.isfinite(target)
                or not 0 < target <= 1e-3
                or any(change.get(key) is not False for key in
                       ['optimizer_state_reset', 'data_cursor_reset', 'budget_extended'])):
            raise ValueError('Unverified or inconsistent source LR adjustment ledger')
        ids.add(identity)
        value, previous_step = target, boundary
    groups = state.get('optimizer', {}).get('param_groups', [])
    scheduler = state.get('scheduler', {})
    if (not groups or any(g.get('lr') != value or g.get('initial_lr') != value for g in groups)
            or scheduler.get('base_lrs') != [value] * len(groups)
            or scheduler.get('_last_lr') != [value] * len(groups)):
        raise ValueError('Source optimizer/scheduler differ from the reviewed LR ledger')
    return value
