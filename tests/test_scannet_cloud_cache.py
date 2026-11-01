"""Reject swapped or corrupted reference branches before final-model comparisons."""

import base64
import gzip
import importlib.util
import io
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

spec = importlib.util.spec_from_file_location(
    'scannet_cloud_cache', Path(__file__).resolve().parents[1] / 'scripts/scannet_cloud_cache.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path):
    rgb = np.zeros((400, 640, 3), dtype=np.uint8)
    mask = np.ones((400, 640), dtype=bool)
    cameras = np.stack([np.eye(3)] * 3)
    saved = dict(rgb=rgb, sensor_depth=np.ones((400, 640), dtype=np.float32), cameras=cameras)
    names = ('student', 'parent', 'teacher')
    for index, name in enumerate(names):
        saved[name + '_depth'] = np.full((400, 640), index + 1, dtype=np.float32)
        saved[name + '_mask'] = mask.copy()
    np.savez_compressed(tmp_path / 'test.npz', **saved)
    png = io.BytesIO()
    Image.fromarray(rgb).save(png, format='PNG')
    blob = b''.join(saved[n + '_depth'][::2, ::2].astype('<f4').tobytes() for n in names)
    blob += mask[::2, ::2].astype(np.uint8).tobytes() * 3
    sample = dict(id='test', rgb='data:image/png;base64,' + base64.b64encode(png.getvalue()).decode(),
                  feature_grid_hw=[25, 40], preprocessing=dict(crop_xyxy=[0, 40, 640, 440]),
                  cameras=cameras.tolist(), stride=2,
                  payload=base64.b64encode(gzip.compress(blob)).decode())
    return saved, sample


def test_reference_roles_preserved(tmp_path):
    saved, sample = fixture(tmp_path)
    actual = module.cached_reference(tmp_path, sample)
    assert all(np.array_equal(actual[k], v) for k, v in saved.items())
    assert actual['student_depth'][0, 0] == 1
    assert actual['teacher_depth'][0, 0] == 3


def test_swapped_old_parent_and_teacher_rejected(tmp_path):
    saved, sample = fixture(tmp_path)
    saved['parent_depth'], saved['teacher_depth'] = saved['teacher_depth'], saved['parent_depth']
    np.savez_compressed(tmp_path / 'test.npz', **saved)
    with pytest.raises(AssertionError):
        module.cached_reference(tmp_path, sample)


@pytest.mark.parametrize('field', ['rgb', 'student_depth', 'teacher_mask', 'cameras', 'sensor_depth'])
def test_corrupted_reference_rejected(tmp_path, field):
    saved, sample = fixture(tmp_path)
    saved[field].flat[0] = np.nan if field == 'sensor_depth' else 0 if field == 'teacher_mask' else 7
    np.savez_compressed(tmp_path / 'test.npz', **saved)
    with pytest.raises(AssertionError):
        module.cached_reference(tmp_path, sample)
