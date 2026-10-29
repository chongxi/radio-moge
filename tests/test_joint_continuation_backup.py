"""Independent checkpoint selection and hardlink-preserving backup safeguards."""

import importlib.util
import inspect
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
spec = importlib.util.spec_from_file_location(
    'continuation_backup', Path(__file__).resolve().parents[1] / 'scripts/backup_joint_continuation.py')
backup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backup)


def test_selections_keep_independent_history_and_ignore_failures():
    specs = {'spatial': dict(metric='cos', mode='max', baseline=.9),
             'depth': dict(metric='err', mode='min', baseline=.1)}
    rows = [dict(step=62367, cos=.95, err=.2, postprocess_failed=0),
            dict(step=63367, cos=.94, err=.08, postprocess_failed=0),
            dict(step=64367, cos=.99, err=.01, postprocess_failed=1),
            dict(step=65367, cos=.95, err=.08, postprocess_failed=0)]
    result = backup.selections_at(specs, rows, 65367)
    assert result['spatial']['step'] == 62367
    assert result['depth']['step'] == 63367
    assert backup.selections_at(specs, rows, 62367)['depth']['step'] == 61367


def fixture_source():
    primary = dict(relative='train/best.pt', source='/remote/train/best.pt', bytes=4,
                   sha256='recorded-hash', identity=[1, 2, 4, 5, 6], checkpoint={'step': 86367})
    alias = dict(primary, relative='train/best-summary.pt', source='/remote/train/best-summary.pt')
    return dict(files=[primary, alias], aliases={alias['relative']: primary['relative']},
                bundled_metadata=[])


def test_one_transfer_preserves_alias_without_overwriting(tmp_path):
    source = fixture_source()
    receipts = backup.transfer_items(source, tmp_path)
    assert len(receipts) == 1
    primary = Path(receipts[0]['destination'])
    primary.parent.mkdir()
    primary.write_bytes(b'data')
    alias = tmp_path / 'train/best-summary.pt'
    alias.write_bytes(b'nope')
    with pytest.raises(AssertionError, match='never overwrite'):
        backup.restore_aliases(source, tmp_path, receipts)
    assert alias.read_bytes() == b'nope'
    alias.unlink()
    with pytest.raises(AssertionError):
        backup.restore_aliases(source, tmp_path, [])
    backup.restore_aliases(source, tmp_path, receipts)
    assert alias.samefile(primary)
    backup.restore_aliases(source, tmp_path, receipts)
    assert alias.samefile(primary)


def test_alias_requires_matching_source_identity_and_hash(tmp_path):
    source = fixture_source()
    source['files'][1]['sha256'] = 'different'
    with pytest.raises(AssertionError):
        backup.transfer_items(source, tmp_path)


def test_remote_verifier_compiles():
    compile(inspect.getsource(backup.selections_at) + '\n' + backup.REMOTE,
            'continuation-backup-remote', 'exec')
