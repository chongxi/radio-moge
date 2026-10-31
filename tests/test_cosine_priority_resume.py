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
