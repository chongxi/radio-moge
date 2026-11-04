from copy import deepcopy

import pytest
import torch

from run_ordinary_geometry_training import model_backend_scope, require_runtime_backend, verify_endpoint


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


def backend_report():
    return dict(default_backend_gradient_bitwise_equality_claimed=False,
        diagnostic_backend_used_for_training=False, feature_cases=[dict(
            runtime_attention_modes=dict(flash=True, mem_efficient=True, math=True, cudnn=True),
            runtime_attention_backend_restored=True,
            runtime_forward=dict(runtime_feature_pixels_losses_exact=True, backward_performed=False),
            diagnostic_math_backward=dict(feature_pixels_losses_gradients_exact=True))])


def test_runtime_backend_is_bound_separately_from_diagnostic_gradient_scope():
    report = backend_report()
    modes = model_backend_scope(report)
    require_runtime_backend(modes)
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with sdpa_kernel(SDPBackend.MATH):
        with pytest.raises(ValueError, match='differs from the actual model gate'):
            require_runtime_backend(modes)


@pytest.mark.parametrize('change', ['claim_default_exact', 'diagnostic_for_training', 'changed_case', 'no_cases'])
def test_publication_rejects_backend_scope_mislabeling(change):
    report = backend_report()
    if change == 'claim_default_exact':
        report['default_backend_gradient_bitwise_equality_claimed'] = True
    elif change == 'diagnostic_for_training':
        report['diagnostic_backend_used_for_training'] = True
    elif change == 'changed_case':
        report['feature_cases'].append(deepcopy(report['feature_cases'][0]))
        report['feature_cases'][-1]['runtime_attention_modes']['flash'] = False
    else:
        report['feature_cases'] = []
    with pytest.raises(ValueError):
        model_backend_scope(report)
