import pytest
import torch

from scripts.export_spatial_cosine_clouds import native_center_crop
from radio_moge.upstream import radio_input


def test_640x400_preserves_rgb_and_sensor_pixels_and_native_grid():
    depth = torch.arange(480 * 640).reshape(480, 640).float()
    image = depth[None, None].expand(1, 3, -1, -1).clone()
    cropped, sensor, metadata = native_center_crop(image, depth, (640, 400))
    assert metadata["crop_xyxy"] == [0, 40, 640, 440]
    assert torch.equal(sensor, depth[40:440])
    assert torch.equal(cropped[0, 0], sensor)
    assert cropped.shape == (1, 3, 400, 640)
    radio = type("Radio", (), dict(patch_size=16, min_resolution_step=16))()
    native = radio_input(cropped, radio, num_tokens=None)
    assert torch.equal(native, cropped)
    assert (native.shape[-2] // 16, native.shape[-1] // 16) == (25, 40)
    # A fixed 1200-token resize would invalidate this native-resolution experiment.
    assert radio_input(cropped, radio, num_tokens=1200).shape[-2:] != (400, 640)


@pytest.mark.parametrize("size", [(1280, 720), (640, 401), (0, 400)])
def test_native_crop_refuses_upsampling_or_invalid_patch_sizes(size):
    with pytest.raises(ValueError):
        native_center_crop(torch.zeros(1, 3, 480, 640), torch.zeros(480, 640), size)


def test_native_crop_requires_registered_depth():
    with pytest.raises(ValueError):
        native_center_crop(torch.zeros(1, 3, 480, 640), torch.zeros(400, 640), (640, 400))
