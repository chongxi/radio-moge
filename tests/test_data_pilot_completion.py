from copy import deepcopy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from verify_data_pilot_completion import review, REMOTE


def fixture():
    plan = dict(phase='pilot', additional_updates=3000, parent_step=100,
                final_step=3100, continuation_guardrails={'old/error': {'max_ratio': 1.03}})
    rows = [dict(step=s, **{'old/error': v}) for s, v in
            [(100, .07), (1100, .069), (2100, .08), (3100, .075)]]
    audits, wandb = {}, {}
    for row in rows:
        s = row['step']
        views = {}
        for name in ['original', 'native-640x400', 'dataset-scannetpp-dslr']:
            views[name] = dict(images=128 if name.startswith('dataset-') else 1268,
                               nyu_sensor_rows=0 if name.startswith('dataset-') else 116,
                               postprocess_failures=0, all_per_image_values_finite=True,
                               exact_identity_order_source_match=True)
        audits[s] = dict(stage='passed', step=s, views=views)
        wandb[s] = dict(stage='passed', step=s, verified_metrics=403, max_metric_difference=0,
                        config_verified=True, id='run', state='finished' if s == 3100 else 'running')
    return plan, rows, audits, wandb


def test_qualified_earlier_source_preserves_final_regression_record():
    result = review(*fixture(), 1100)
    assert result['selected_checkpoint_qualified']
    assert not result['final_checkpoint_qualified']
    assert result['failed_checkpoint_steps'] == [2100, 3100]
    with pytest.raises(ValueError, match='guardrails'):
        review(*fixture(), 3100)


@pytest.mark.parametrize('failure', ['missing', 'unfinished', 'view', 'nonfinite', 'metrics'])
def test_missing_or_failed_evidence_cannot_release_source(failure):
    plan, rows, audits, wandb = deepcopy(fixture())
    if failure == 'missing':
        rows.pop()
    elif failure == 'unfinished':
        wandb[3100]['state'] = 'running'
    elif failure == 'view':
        audits[2100]['views']['original']['exact_identity_order_source_match'] = False
    elif failure == 'nonfinite':
        rows[1]['old/error'] = float('nan')
    else:
        wandb[2100]['max_metric_difference'] = .01
    with pytest.raises(ValueError):
        review(plan, rows, audits, wandb, 1100)


def test_remote_payload_is_valid_python():
    compile(REMOTE, '<remote verifier>', 'exec')
