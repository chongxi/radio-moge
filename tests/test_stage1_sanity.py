import numpy as np

from export_stage1_sanity import bilinear, encode_cloud


def test_prediction_sampling_uses_normalized_pixel_centers():
    h, w = 8, 12
    yy, xx = np.meshgrid((np.arange(h) + .5) / h, (np.arange(w) + .5) / w, indexing='ij')
    points = np.stack([2 * xx - yy, 3 * yy, xx + yy + 1], -1)
    uv = np.array([[.25, .35], [.5, .5], [.75, .65]])
    expected = np.column_stack([2 * uv[:, 0] - uv[:, 1], 3 * uv[:, 1], uv.sum(1) + 1])
    np.testing.assert_allclose(bilinear(points, uv), expected, atol=1e-12)
    np.testing.assert_allclose(bilinear(points[..., 2], uv), expected[:, 2], atol=1e-12)


def test_cloud_binary_preserves_raw_coordinates_and_exact_color_correspondence():
    points = np.array([[1, 2, 3], [4, 5, -1], [np.nan, 0, 4], [7, 8, 9]], dtype=float)
    colors = np.arange(12, dtype=np.uint8).reshape(4, 3)
    before = points.copy()
    raw, report = encode_cloud(points, colors)
    assert report['count'] == report['full_count'] == 2
    np.testing.assert_array_equal(np.frombuffer(raw, '<f4', count=6).reshape(2, 3), points[[0, 3]])
    np.testing.assert_array_equal(np.frombuffer(raw, np.uint8, offset=24).reshape(2, 3), colors[[0, 3]])
    np.testing.assert_array_equal(points, before)


def test_display_decimation_never_creates_or_repeats_observations():
    points = np.column_stack([np.arange(100), np.zeros(100), np.arange(100) + 1.])
    colors = np.repeat(np.arange(100, dtype=np.uint8)[:, None], 3, axis=1)
    raw, report = encode_cloud(points, colors, max_points=12)
    exported = np.frombuffer(raw, '<f4', count=36).reshape(12, 3)
    assert report['full_count'] == 100 and report['count'] == 12
    assert len(np.unique(exported[:, 0])) == 12
    np.testing.assert_array_equal(exported, points[exported[:, 0].astype(int)])
