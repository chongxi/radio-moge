import importlib.util
import math
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'wandb_verification', Path(__file__).parents[1]/'scripts/wandb_verification.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_wandb_json_rounding_is_reported_and_meaningful_changes_rejected():
    expected = {'baseline': .9593845226106881, 'started_at': 1793288222.1457853,
                'lr': 1.5e-5, 'steps': 91367, 'enabled': True, 'tags': ['continuation']}
    actual = dict(expected, baseline=.959384522610688, started_at=1793288222.145785)
    differences = module.compare_metadata(actual, expected)
    assert {row['path'] for row in differences} == {'/baseline', '/started_at'}
    assert module.compare_metadata({'value': 1}, {'value': 1.0}) == [
        {'path': '/value', 'absolute_difference': 0,
         'representation_change': 'equal integer/float'}]
    with pytest.raises(AssertionError):
        module.compare_metadata({'value': 1.1}, {'value': 1})
    for change in ({'steps': 91368}, {'lr': 1.6e-5}, {'enabled': 1},
                   {'tags': []}, {'baseline': float('nan')},
                   {'baseline': math.nextafter(math.nextafter(expected['baseline'], 0), 0)}):
        with pytest.raises(AssertionError):
            module.compare_metadata(dict(expected, **change), expected)
    with pytest.raises(AssertionError):
        module.compare_metadata({**expected, 'extra': 1}, expected)
    remote = {}
    exec(compile(module.REMOTE_COMPARISON, '<remote comparison>', 'exec'), remote)
    assert remote['compare_metadata'](actual, expected) == differences
