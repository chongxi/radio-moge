from pathlib import Path
import importlib.util
import json

import cv2
import numpy as np
import pytest

from radio_moge.scannetpp_validation import (
    aligned_view, internal_holdout, read_video_frame, registered_frames, select_spaced_frames,
)


def camera():
    return dict(width=1920, height=1440, parameters=[1400, 1400, 960, 720, 0, 0, 0, 0])


def test_registered_frames_use_filename_index_and_own_camera(tmp_path):
    c, i = tmp_path/'cameras.txt', tmp_path/'images.txt'
    c.write_text('2 OPENCV 1920 1440 1400 1400 960 720 0 0 0 0\n')
    i.write_text('# comment\n900 1 0 0 0 0 0 0 2 video/frame_000010.jpg\n\n'
                 '400 1 0 0 0 0 0 0 2 video/frame_000030.jpg\n\n')
    rows = registered_frames(c, i)
    assert sorted(rows) == [10, 30] and rows[30]['camera_id'] == 2
    i.write_text(i.read_text() + '1 1 0 0 0 0 0 0 2 video/frame_000010.jpg\n\n')
    with pytest.raises(ValueError, match='identity'):
        registered_frames(c, i)


def test_selection_is_sorted_temporal_strata_without_replacement():
    frames = dict.fromkeys(reversed(range(0, 200, 10)))
    assert select_spaced_frames(frames) == [20, 70, 120, 170]
    assert internal_holdout('020312de8d')
    with pytest.raises(ValueError):
        internal_holdout('../outside')
    with pytest.raises(ValueError):
        select_spaced_frames({1: None}, 4)


def test_zero_distortion_preserves_scaled_camera_rays_and_sensor_values():
    rgb = np.zeros((1440, 1920, 3), np.uint8)
    mask = np.full_like(rgb, 255)
    depth = np.broadcast_to(np.arange(256, dtype=np.float32)[None, :]+1, (192, 256)).copy()
    _, got, proof = aligned_view(rgb, mask, depth, camera())
    # Output768 is3x native depth256. A common point projects to column3*x.
    for x in [20, 40, 100, 200]:
        assert got[300, 3*x] == depth[100, x]
    np.testing.assert_allclose(proof['output_intrinsic'], [[560, 0, 384], [0, 560, 288], [0, 0, 1]])
    assert proof['depth_units'] == 'metres' and not proof['metric_scale_fitting']


def test_nonzero_distortion_keeps_rgb_and_depth_correspondence():
    c = camera()
    c['parameters'][4:] = [.05, -.04, .001, -.002]
    # Identical planar labeling at the officially scaled native coordinates.
    rgb = np.zeros((1440, 1920, 3), np.uint8)
    rgb[:, 960:] = 200
    depth = np.ones((192, 256), np.float32)
    depth[:, 128:] = 2
    mask = np.full_like(rgb, 255)
    out, got, _ = aligned_view(rgb, mask, depth, c)
    assert np.all(got[100:400, 100:300] == 1)
    assert np.all(got[100:400, 460:660] == 2)
    assert np.all(out[100:400, 100:300] == 0)
    assert np.all(out[100:400, 460:660] == 200)


def test_anonymized_and_zero_depth_regions_stay_invalid():
    rgb = np.full((1440, 1920, 3), 100, np.uint8)
    mask = np.full_like(rgb, 255)
    mask[400:800, 400:800] = 0
    depth = np.full((192, 256), 2.0, np.float32)
    depth[120:160, 120:160] = 0
    out, got, _ = aligned_view(rgb, mask, depth, camera())
    assert not got[180:300, 180:300].any() and not out[180:300, 180:300].any()
    assert not got[380:450, 380:450].any()
    depth[0, 0] = np.nan
    with pytest.raises(ValueError, match='native LiDAR'):
        aligned_view(rgb, mask, depth, camera())


def test_existing_backend_seeks_exact_generated_frame_indices(tmp_path):
    path = Path(tmp_path)/'known.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
    assert writer.isOpened()
    for index in range(12):
        writer.write(np.full((48, 64, 3), index*20, np.uint8))
    writer.release()
    capture = cv2.VideoCapture(str(path))
    try:
        for index in [9, 2, 11, 0]:
            frame = read_video_frame(capture, index)
            assert abs(float(frame.mean()) - index*20) < 2
    finally:
        capture.release()


def test_preparer_requires_completed_receipts_and_crc_without_bulk_reads(tmp_path):
    path = Path(__file__).parents[1]/'scripts/prepare_scannetpp_depth_validation.py'
    spec = importlib.util.spec_from_file_location('depth_preparer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data, control, scene = tmp_path/'data', tmp_path/'control', '020312de8d'
    (control/'receipts').mkdir(parents=True)
    (control/'extractions').mkdir()
    (data/'data'/scene/'iphone').mkdir(parents=True)
    (data/'data'/scene/'iphone/rgb.mkv').write_bytes(b'x')
    extraction_paths = []
    for name in ['rgb.mkv', 'rgb_mask.zip', 'depth.zip', 'colmap.zip', 'pose_intrinsic_imu.zip']:
        relative = f'data/{scene}/iphone/{name}'
        (control/'receipts'/(module.sha(relative.encode())+'.json')).write_text(json.dumps(
            dict(path=relative, completed_at=1, bytes=1, sha256='0'*64)))
        if name.endswith('.zip'):
            ep = control/'extractions'/(module.sha(str(data/relative).encode())+'.json')
            ep.write_text(json.dumps(dict(stage='passed', crc_checked=True, completed_at=2)))
            extraction_paths.append(ep)
    # Archives do not exist: only their prior verified receipts are inspected.
    assert len(module.completed_assets(data, control, scene, 3)) == 5
    with pytest.raises(ValueError, match='extraction/CRC'):
        module.completed_assets(data, control, scene, 1.5)
    extraction_paths[0].write_text(json.dumps(dict(stage='passed', crc_checked=False, completed_at=2)))
    with pytest.raises(ValueError, match='extraction/CRC'):
        module.completed_assets(data, control, scene, 3)
