import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


scripts = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('scanner_supplement',
                                             scripts / 'fetch_scannetpp_scanner_poses.py')
poses = importlib.util.module_from_spec(spec)
spec.loader.exec_module(poses)
sys.path.remove(str(scripts))


def test_scan_indices_and_rotations_are_checked():
    value = np.repeat(np.eye(4)[None], 4, axis=0)
    assert poses.check_poses(value.tolist(), [1, 3])['poses'] == 4
    with pytest.raises(ValueError, match='no corresponding'):
        poses.check_poses(value, [4])
    value[1, 0, 0] = -1
    with pytest.raises(ValueError, match='proper'):
        poses.check_poses(value, [1])


def test_transport_never_exposes_private_url():
    class BrokenSession:
        def get(self, url, **kwargs):
            raise OSError('secret query: ' + url)
    with pytest.raises(poses.SafeDownloadError) as caught:
        poses.fetch(BrokenSession(), 'https://example.invalid/?token=secret-example')
    assert 'secret-example' not in str(caught.value)
    assert caught.value.__suppress_context__


def test_supplement_refuses_overwrite(tmp_path):
    with pytest.raises(FileExistsError):
        poses.run(tmp_path, tmp_path, tmp_path / 'missing-private-file')
