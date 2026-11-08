"""Public geometry conventions, filtering, input ownership and portable output."""

import numpy as np
from PIL import Image
import pytest
import torch

from radio_moge import PointCloudPipeline
from radio_moge.pipeline import _image_tensor


class FixedGeometry(torch.nn.Module):
    device = torch.device("cpu")
    metadata = {"training_step": 272646}

    def infer(self, image, *, features=None, validate=True):
        h, w = image.shape[-2:]
        depth = torch.ones(h, w)
        depth[:, w // 2:] = 2
        k = torch.tensor([[1., 0, .5], [0, 1., .5], [0, 0, 1.]])
        v, u = torch.meshgrid((torch.arange(h) + .5) / h, (torch.arange(w) + .5) / w,
                              indexing="ij")
        points = torch.stack(((u - .5) * depth, (v - .5) * depth, depth), -1)
        return dict(depth=depth, intrinsics=k, points=points, mask=torch.ones(h, w).bool())


def test_geometry_edges_projection_colors_and_ply(tmp_path):
    pipeline = PointCloudPipeline(FixedGeometry())
    rgb = np.arange(6 * 10 * 3, dtype=np.uint8).reshape(6, 10, 3)
    before = rgb.copy()
    result = pipeline(rgb)
    expected = np.ones((6, 10), dtype=bool)
    expected[:, 4:6] = False  # only the two columns straddling the discontinuity
    np.testing.assert_array_equal(result.mask, expected)
    np.testing.assert_array_equal(result.rgb, rgb)
    np.testing.assert_array_equal(rgb, before)
    assert result.base_mask.all()
    assert not torch.equal(result.mask, result.base_mask)
    k = result.intrinsics.numpy()
    assert (k[0, 0], k[1, 1], k[0, 2], k[1, 2]) == (10, 6, 4.5, 2.5)
    v, u = np.mgrid[:6, :10]
    xyz = result.points.numpy()
    np.testing.assert_allclose(xyz[..., 0] / xyz[..., 2] * k[0, 0] + k[0, 2], u, atol=1e-6)
    np.testing.assert_allclose(xyz[..., 1] / xyz[..., 2] * k[1, 1] + k[1, 2], v, atol=1e-6)
    cv, colors = result.pointcloud(numpy=True)
    gl, _ = result.pointcloud(numpy=True, coordinate_system="opengl")
    np.testing.assert_array_equal(gl, cv * [1, -1, -1])
    np.testing.assert_array_equal(colors, rgb[expected])
    assert cv.shape == (48, 3)
    path = result.save_ply(tmp_path / "test.ply")
    header, payload = path.read_bytes().split(b"end_header\n", 1)
    assert b"element vertex 48" in header
    packed = np.frombuffer(payload, dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
    np.testing.assert_array_equal(packed["xyz"], cv)
    np.testing.assert_array_equal(packed["rgb"], colors)
    arrays = result.numpy()
    assert arrays["normal"] is None and arrays["depth"].dtype == np.float32


def test_no_filter_is_explicit_and_input_formats_preserve_pixels():
    rgb = np.arange(6 * 10 * 3, dtype=np.uint8).reshape(6, 10, 3)
    pipeline = PointCloudPipeline(FixedGeometry(), edge_threshold=None)
    for image in [rgb, Image.fromarray(rgb), torch.from_numpy(rgb).permute(2, 0, 1)]:
        result = pipeline(image)
        assert result.mask.all()
        np.testing.assert_array_equal(result.rgb, rgb)
    readonly = rgb.copy()
    readonly.flags.writeable = False
    np.testing.assert_array_equal(pipeline(readonly).rgb, rgb)
    np.testing.assert_array_equal(pipeline(rgb[..., ::-1]).rgb, rgb[..., ::-1])
    from dataclasses import replace
    empty = pipeline(rgb)
    empty = replace(empty, mask=torch.zeros_like(empty.mask))
    xyz, colors = empty.pointcloud(numpy=True)
    assert xyz.shape == colors.shape == (0, 3)


@pytest.mark.parametrize("bad", [np.zeros((4, 5, 3), np.float32), np.zeros((4, 5), np.uint8),
    torch.zeros(1, 3, 4, 5), torch.zeros(3, 0, 5), torch.ones(3, 4, 5) * 255,
    torch.ones(3, 4, 5) * float("nan"), torch.zeros(3, 4, 5, dtype=torch.int32)])
def test_bad_image_contract(bad):
    with pytest.raises(ValueError):
        _image_tensor(bad, True)


@pytest.mark.parametrize("threshold", [0, -1, float("nan"), float("inf")])
def test_bad_filter_threshold(threshold):
    with pytest.raises(ValueError):
        PointCloudPipeline(FixedGeometry(), edge_threshold=threshold)


def test_bundle_checks_missing_corrupt_and_escaping_files(tmp_path):
    import json
    from radio_moge.bundle import read_bundle, sha256
    from radio_moge.deployment import PINS

    files = {}
    for name in ["adapter.pt", "decoder.pt", "radio.pth.tar", "config.json"]:
        path = tmp_path / name
        path.write_bytes(b"test")
        files[name] = dict(bytes=4, sha256=sha256(path))
    manifest = dict(format="radio-moge-pointcloud", format_version=1, revisions=PINS, files=files)
    (tmp_path / "bundle.json").write_text(json.dumps(manifest))
    read_bundle(tmp_path)
    (tmp_path / "adapter.pt").write_bytes(b"oops")
    with pytest.raises(ValueError, match="checksum"):
        read_bundle(tmp_path)
    read_bundle(tmp_path, verify=False)
    (tmp_path / "decoder.pt").unlink()
    with pytest.raises(ValueError, match="Missing"):
        read_bundle(tmp_path, verify=False)
    files["../outside"] = dict(bytes=4, sha256="bad")
    (tmp_path / "decoder.pt").write_bytes(b"test")
    (tmp_path / "bundle.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        read_bundle(tmp_path, verify=False)
