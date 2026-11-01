"""Verify the immutable three-way RGB/geometry cache before reusing two branches."""

import base64
import gzip
import io
from pathlib import Path

import numpy as np
from PIL import Image


def cached_reference(root, sample):
    sid = sample['id']
    assert Path(sid).name == sid
    names = ('student', 'parent', 'teacher')
    with np.load(root / f'{sid}.npz', allow_pickle=False) as data:
        saved = {k: data[k].copy() for k in data.files}
    rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(sample['rgb'].split(',', 1)[1]))))
    assert rgb.shape == (400, 640, 3) and np.array_equal(rgb, saved['rgb'])
    assert saved['sensor_depth'].shape == (400, 640)
    assert np.isfinite(saved['sensor_depth']).all() and (saved['sensor_depth'] >= 0).all()
    assert sample['feature_grid_hw'] == [25, 40]
    assert sample['preprocessing']['crop_xyxy'] == [0, 40, 640, 440]
    assert np.array_equal(saved['cameras'], np.asarray(sample['cameras']))
    assert saved['cameras'].shape == (3, 3, 3)
    for name in names:
        depth, mask = saved[name + '_depth'], saved[name + '_mask']
        assert depth.shape == mask.shape == (400, 640) and mask.dtype == bool
        assert mask.any() and np.isfinite(depth[mask]).all() and (depth[mask] > 0).all()
    stride = sample['stride']
    assert stride == 2
    blob = b''.join(np.asarray(saved[n + '_depth'][::stride, ::stride], dtype='<f4').tobytes()
                    for n in names)
    blob += b''.join(saved[n + '_mask'][::stride, ::stride].astype(np.uint8).tobytes()
                     for n in names)
    assert blob == gzip.decompress(base64.b64decode(sample['payload']))
    return saved
