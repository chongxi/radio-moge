"""Cached comparisons must keep identical RGB, camera and sampled geometry."""

import base64
import gzip
import importlib.util
import io
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


spec = importlib.util.spec_from_file_location(
    'resolution_clouds', Path(__file__).resolve().parents[1]/'scripts/export_resolution_clouds.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path):
    rgb = np.zeros((400, 640, 3), dtype=np.uint8)
    depth = np.ones((400, 640), dtype=np.float32)
    mask = np.ones((400, 640), dtype=bool)
    cameras = np.stack([np.eye(3), np.eye(3)])
    saved = dict(rgb=rgb, sensor_depth=depth, student_depth=depth, teacher_depth=depth,
                 student_mask=mask, teacher_mask=mask, cameras=cameras)
    np.savez_compressed(tmp_path/'test.npz', **saved)
    png = io.BytesIO()
    Image.fromarray(rgb).save(png, format='PNG')
    blob = depth[::2, ::2].astype('<f4').tobytes()*2 + mask[::2, ::2].astype(np.uint8).tobytes()*2
    sample = dict(id='test', rgb='data:image/png;base64,'+base64.b64encode(png.getvalue()).decode(),
                  feature_grid_hw=[25, 40], preprocessing=dict(crop_xyxy=[0, 40, 640, 440]),
                  cameras=cameras.tolist(), stride=2, payload=base64.b64encode(gzip.compress(blob)).decode())
    return saved, sample


def test_unchanged_cached_arrays_pass(tmp_path):
    saved, sample = fixture(tmp_path)
    actual = module.cached_sample(tmp_path, sample)
    assert all(np.array_equal(actual[k], v) for k, v in saved.items())


@pytest.mark.parametrize('field', ['rgb', 'student_depth', 'teacher_mask', 'cameras', 'sensor_depth'])
def test_changed_cache_is_rejected(tmp_path, field):
    saved, sample = fixture(tmp_path)
    array = saved[field].copy()
    array.flat[0] = np.nan if field == 'sensor_depth' else 0 if field == 'teacher_mask' else 7
    saved[field] = array
    np.savez_compressed(tmp_path/'test.npz', **saved)
    with pytest.raises(AssertionError):
        module.cached_sample(tmp_path, sample)
