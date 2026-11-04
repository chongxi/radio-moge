from copy import deepcopy

import pytest
import torch

from run_ordinary_geometry_training import verify_endpoint


def endpoint():
    return dict(stop=dict(step=232538, reason='step_limit', final_evaluation='complete'),
        log=[dict(step=i, total=.1) for i in range(232289, 232539)],
        audit=dict(optimizer_updates=250, cursor=2000, presentations=8000, planned_keys_matched=True,
                   unused_prefetch_excluded=True, actual_compute_groups_verified=True),
        initial=dict(data_digests={'train':'fixed'}, revisions={'model':'fixed'}, statistics={'rms':torch.ones(2)}),
        final=dict(step=232538, cursor=2000, config={'test':True}, data_digests={'train':'fixed'},
                   revisions={'model':'fixed'}, statistics={'rms':torch.ones(2)}, adapter={'w':torch.ones(2)},
                   optimizer={'state':{0:{'step':torch.tensor(232538.),'exp_avg':torch.ones(2)}}}),
        config={'test':True})


def test_completed_endpoint_and_partial_wall_clock_endpoint_are_distinct():
    value = endpoint()
    verify_endpoint(**value)
    value['stop'].update(step=232537, reason='wall_clock_budget')
    with pytest.raises(ValueError, match='fixed250 endpoint'):
        verify_endpoint(**value)


@pytest.mark.parametrize('change', ['missing_update', 'nonfinite_loss', 'wrong_adam', 'changed_rms',
                                  'nonfinite_adam', 'short_input_audit'])
def test_final_proof_rejects_incomplete_or_changed_actual_run(change):
    value = deepcopy(endpoint())
    if change == 'missing_update':
        value['log'].pop()
    elif change == 'nonfinite_loss':
        value['log'][-1]['total'] = float('nan')
    elif change == 'wrong_adam':
        value['final']['optimizer']['state'][0]['step'] -= 1
    elif change == 'changed_rms':
        value['final']['statistics']['rms'][0] += 1
    elif change == 'nonfinite_adam':
        value['final']['optimizer']['state'][0]['exp_avg'][0] = float('inf')
    else:
        value['audit']['presentations'] -= 4
    with pytest.raises(ValueError):
        verify_endpoint(**value)
