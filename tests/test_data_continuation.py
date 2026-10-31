import importlib.util
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'scripts' / (name+'.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_continuation_gate_rejects_old_domain_regression_despite_new_domain_gain():
    gate = module('audit_data_evaluations').continuation_gate
    base = {'old/abs_rel': .07, 'new/cosine': .95}
    policy = {'old/abs_rel': {'max_ratio': 1.03}, 'new/cosine': {'max_drop': .002}}
    assert gate(base, {'old/abs_rel': .071, 'new/cosine': .96}, policy)['stage'] == 'passed'
    assert gate(base, {'old/abs_rel': .08, 'new/cosine': .98}, policy)['stage'] == 'needs_inspection'
    with pytest.raises(ValueError):
        gate(base, {'old/abs_rel': float('nan'), 'new/cosine': .96}, policy)
    with pytest.raises(KeyError):
        gate(base, {'old/abs_rel': .07}, policy)


def test_controller_budget_and_output_cannot_silently_change():
    validate = module('run_data_continuation').validate_plan
    plan = dict(authorized=True, additional_updates=3000, parent_step=93367,
                learning_rate=1e-5, max_seconds=10800, effective_batch=32, remote_root='/new')
    config = dict(data={'buckets': [[480, 640], [640, 480], [400, 640]]}, output_dir='/new/train',
                  train=dict(steps=96367, lr=1e-5, lr_schedule='constant', max_seconds=10800))
    validate(plan, config, dict(effective_batch=32))
    config['output_dir'] = '/old/train'
    with pytest.raises(ValueError):
        validate(plan, config, dict(effective_batch=32))
