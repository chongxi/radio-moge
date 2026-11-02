import numpy as np
from PIL import Image
import pytest

from cache_scannetpp_geometry import camera_for_row, colmap_poses, conservative_anonymization
from radio_moge.geometry_sources import scannetpp_nerfstudio_to_colmap_c2w


def test_native_colmap_pose_parse_and_camera_binding(tmp_path):
    path = tmp_path / 'images.txt'
    path.write_text('# header\n1 1 0 0 0 1 2 3 4 image.JPG\n\n')
    poses = colmap_poses(path)
    np.testing.assert_allclose(poses['image.JPG'][:3, 3], [-1, -2, -3])
    transform = scannetpp_nerfstudio_to_colmap_c2w(poses['image.JPG'])
    camera = dict(camera_model='PINHOLE', w=640, h=480, fl_x=300, fl_y=301, cx=320, cy=240)
    row = dict(view='image.JPG', width=640, height=480,
               camera_width_height_fx_fy_cx_cy=[640, 480, 300, 301, 320, 240],
               transform_matrix=transform)
    frames = {'image.JPG': dict(transform_matrix=transform)}
    _, pose = camera_for_row(row, camera, frames, poses)
    np.testing.assert_allclose(pose, poses['image.JPG'])
    with pytest.raises(ValueError):
        camera_for_row(row, dict(camera, cx=321), frames, poses)
    with pytest.raises(ValueError):
        camera_for_row(row, dict(camera, k1=.01), frames, poses)


def test_mask_downsampling_does_not_erase_anonymized_holes(tmp_path):
    mask = np.full((64, 96), 255, np.uint8)
    mask[30, 30] = 0
    mask[40, 60] = 254  # Nonbinary uncertain source pixels are also not valid.
    path = tmp_path / 'mask.png'
    Image.fromarray(mask).save(path)
    support = conservative_anonymization(path, (64, 96), (16, 24))
    assert support.sum() > 100
    assert not support[7, 7] and not support[10, 15]
    assert not support[:2].any() and not support[:, :2].any()
