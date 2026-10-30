"""Mixed-resolution backup preserves baseline timing and independent view choices."""

import importlib.util
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
spec = importlib.util.spec_from_file_location(
    'resolution_backup', Path(__file__).resolve().parents[1] / 'scripts/backup_resolution_training.py')
backup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backup)


def fixture():
    native = 'resolution/native-640x400/'
    specs = {'original': dict(metric='cos', mode='max', baseline=.9),
             'native': dict(metric=native+'err', mode='min')}
    rows = [dict(step=step, cos=cos, postprocess_failed=0,
                 **{native+'err': err, native+'postprocess_failed': failed})
            for step, cos, err, failed in [(91367, .9, .1, 0), (92367, .95, .08, 0),
                                           (93367, .94, .01, 1), (94367, .95, .08, 0)]]
    return specs, rows


def test_initial_before_evaluation_does_not_invent_native_selections():
    specs, rows = fixture()
    before = backup.selections_at(specs, rows, 91367, before_initial=True)
    assert set(before) == {'original'}
    baseline = backup.selections_at(specs, rows, 91367)
    assert baseline['native']['step'] == 91367
    assert baseline['native']['value'] == .1


def test_independent_failures_and_ties_keep_correct_checkpoints():
    specs, rows = fixture()
    best = backup.selections_at(specs, rows, 94367)
    assert best['original']['step'] == best['native']['step'] == 92367
    rows[1]['postprocess_failed'] = 1
    assert backup.selections_at(specs, rows, 92367)['native']['step'] == 91367


def test_reject_nonfinite_history():
    specs, rows = fixture()
    rows[1]['cos'] = float('nan')
    with pytest.raises(AssertionError):
        backup.selections_at(specs, rows, 94367)


def test_remote_verifier_compiles():
    compile(backup.remote_verifier(), 'resolution-backup-remote', 'exec')
