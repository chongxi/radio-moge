import gzip

import numpy as np

from radio_moge.viewer import summarize, write_depths


def test_browser_depth_format_preserves_meters_invalid_values_and_plane_order(tmp_path):
    arrays = [np.array([[1.25, 80.125], [0, np.nan]], dtype=np.float32) + i for i in range(4)]
    path = tmp_path / 'sample.depth.gz'
    write_depths(path, arrays)
    # The browser's Float32Array must recover unscaled metric depths and sensor holes.
    decoded = np.frombuffer(gzip.decompress(path.read_bytes()), dtype='<f4').reshape(4, 2, 2)
    for i in range(4):
        np.testing.assert_array_equal(decoded[i], arrays[i])


def test_summary_retains_failed_predictions_in_metric_counts():
    samples = []
    for value, failed in [(0.1, False), (1.0, True)]:
        samples.append(dict(metrics={k: dict(abs_rel=value, rmse=2, delta1=0, invalid_fraction=int(failed))
                                     for k in ('before', 'after', 'teacher')},
                            failures={'after': 'focal recovery'} if failed else {}))
    result = summarize(samples)
    assert result['after']['abs_rel_count'] == 2
    assert result['after']['failures'] == 1
    assert result['after']['abs_rel'] == 0.55
    assert result['after']['invalid_fraction'] == 0.5
