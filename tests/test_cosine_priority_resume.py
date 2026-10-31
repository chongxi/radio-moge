from copy import deepcopy

import pytest

from test_data_continuation import module


def inputs():
    plan = dict(authorized=True, stop_on_metric_regression=False, preserve_exact_training_state=True,
                source_identity=[1, 2, 3, 4, 5], source_checkpoint='/source.pt', resume_step=95909,
                resume_cursor=12336, learning_rate=5e-6, original_budget_started_at=1000,
                original_deadline=44200, outer_deadline=44800)
    proof = dict(stage='passed', all_state_tensors_finite=True, config_and_digests_match=True,
                 identity=plan['source_identity'], source_checkpoint='/source.pt', step=95909,
                 cursor=12336, learning_rate=5e-6, original_budget_started_at=1000,
                 runtime_limits=dict(steps=134025, max_seconds=43200))
    return plan, proof, dict(train=dict(steps=134025, max_seconds=43200))


def test_resume_retains_remaining_absolute_budget():
    validate = module('run_cosine_priority_resume').validate
    assert validate(*inputs(), 5000) == 39800
    assert validate(*inputs(), 8000) == 36800


@pytest.mark.parametrize('change', ['cursor', 'lr', 'identity', 'source', 'deadline', 'expired', 'policy'])
def test_rejects_state_changes_or_silent_budget_extension(change):
    plan, proof, config = deepcopy(inputs())
    now = 5000
    if change == 'cursor':
        plan['resume_cursor'] = 0
    elif change == 'lr':
        plan['learning_rate'] = 1e-5
    elif change == 'identity':
        plan['source_identity'] = [9] * 5
    elif change == 'source':
        plan['source_checkpoint'] = '/old.pt'
    elif change == 'deadline':
        plan['original_deadline'] += 3600
    elif change == 'expired':
        now = 44200
    else:
        plan['stop_on_metric_regression'] = True
    with pytest.raises(ValueError):
        module('run_cosine_priority_resume').validate(plan, proof, config, now)


def test_authorized_lr_request_retains_source_state_and_deadline():
    plan, proof, config = inputs()
    plan['learning_rate'] = 1e-5
    request = dict(authorized=True, id='user-lr1e5', previous_lr=5e-6, learning_rate=1e-5,
                   expected_step=95909, expected_cursor=12336,
                   runtime_limits=proof['runtime_limits'], budget_started_at=1000)
    validate = module('run_cosine_priority_resume').validate
    assert validate(plan, proof, config, 5000, request) == 39800
    for key, value in [('expected_cursor', 0), ('previous_lr', 2.5e-6),
                       ('budget_started_at', 5000), ('learning_rate', 2e-5)]:
        bad = dict(request, **{key: value})
        with pytest.raises(ValueError):
            validate(plan, proof, config, 5000, bad)


def test_initialization_interruption_requires_no_new_state():
    validate = module('run_cosine_priority_resume').validate_previous_stop
    plan, _, _ = inputs()
    old = dict(pid=47404, last_exit_code=-2)
    interrupted = dict(stage='passed', controller_pid=47404, no_new_optimizer_updates=True,
                       interrupted_before_restore=True, processes_absent=True,
                       source_content_identity=plan['source_identity'][:4],
                       resume_step=95909, resume_cursor=12336)
    validate(plan, old, interrupted)
    for key in ['no_new_optimizer_updates', 'interrupted_before_restore', 'processes_absent']:
        with pytest.raises(ValueError):
            validate(plan, old, dict(interrupted, **{key: False}))
    with pytest.raises(ValueError):
        validate(plan, dict(old, last_exit_code=1), interrupted)
