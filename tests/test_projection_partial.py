from copy import deepcopy

import pytest

from review_projection_partial import partial_scope


def fixture():
    stop = dict(step=232421, reason='wall_clock_budget', final_evaluation='complete',
        max_seconds=2700, optimizer_stopped_elapsed_seconds=2706)
    log = [dict(step=i, total=.2, effective_batch=32, logical_data_cursor=(i-232288)*8)
           for i in range(232289, 232422)]
    audit = dict(optimizer_updates=133, cursor=1064, presentations=4256,
        planned_keys_matched=True, unused_prefetch_excluded=True, actual_compute_groups_verified=True)
    return stop, log, audit


def test_partial_is_accounted_without_relaxing_fixed_endpoint():
    stop, log, audit = fixture()
    assert partial_scope(stop, log, audit) == 133
    for change in (dict(step=232538), dict(reason='step_limit'),
                   dict(final_evaluation='pending'), dict(optimizer_stopped_elapsed_seconds=2600)):
        with pytest.raises(ValueError):
            partial_scope(dict(stop, **change), log, audit)


def test_missing_updates_nonfinite_loss_and_uncommitted_inputs_fail():
    stop, log, audit = fixture()
    invalid = deepcopy(log)
    invalid[10]['total'] = float('nan')
    for rows, evidence in ((log[:-1], audit), (invalid, audit),
            (log, dict(audit, presentations=4288)), (log, dict(audit, cursor=1072)),
            (log, dict(audit, unused_prefetch_excluded=False))):
        with pytest.raises(ValueError):
            partial_scope(stop, rows, evidence)
