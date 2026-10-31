import struct
import zlib

import numpy as np
import pytest

from radio_moge.scannetpp_depth import decode_frame, read_selected_depth


def raw_deflate(data):
    encoder = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    return encoder.compress(data) + encoder.flush()


def literal_lz4(data):
    # A valid independent fixture with no matches, including extended lengths.
    length = len(data)
    if length < 15:
        return bytes([length << 4]) + data
    rest = length - 15
    return b'\xf0' + b'\xff' * (rest // 255) + bytes([rest % 255]) + data


def test_native_uint16_millimetres_and_float32_metres_have_same_scale():
    mm = np.array([[0, 1000, 2000], [3000, 4000, 65535]], dtype='<u2')
    metres = mm.astype('<f4') / 1000
    a, ac = decode_frame(literal_lz4(mm.tobytes()), mm.shape)
    b, bc = decode_frame(raw_deflate(metres.tobytes()), mm.shape)
    np.testing.assert_array_equal(a, metres)
    np.testing.assert_array_equal(b, metres)
    assert ac == 'lz4_uint16_mm' and bc == 'deflate_float32_metres'


def test_selective_framed_mixed_codecs_preserve_frame_ids(tmp_path):
    expected = [np.full((192, 256), i * 1000, dtype='<u2') for i in range(6)]
    payloads = [literal_lz4(a.tobytes()) if i % 2 == 0
                else raw_deflate((a.astype('<f4') / 1000).tobytes())
                for i, a in enumerate(expected)]
    path = tmp_path / 'depth.bin'
    path.write_bytes(b''.join(struct.pack('<I', len(p)) + p for p in payloads))
    selected, proof = read_selected_depth(path, [4, 1])
    assert set(selected) == {1, 4} and proof['format'] == 'per_frame'
    assert proof['decoded_frames'] == 2 and proof['units'] == 'metres'
    for i in selected:
        np.testing.assert_array_equal(selected[i], expected[i].astype(np.float32) / 1000)


def test_streaming_global_deflate_selects_noncontiguous_frames(tmp_path):
    expected = np.random.default_rng(7).uniform(0, 9, (8, 192, 256)).astype('<f4')
    path = tmp_path / 'depth.bin'
    path.write_bytes(raw_deflate(expected.tobytes()))
    selected, proof = read_selected_depth(path, [0, 3, 7])
    assert proof['format'] == 'global' and proof['decoded_frames'] == 3
    for i in selected:
        np.testing.assert_array_equal(selected[i], expected[i])


@pytest.mark.parametrize('payload', [b'', b'\x10a\x00\x00', b'\xf0\xff',
                                    raw_deflate(b'\0' * 10000)])
def test_frame_decoder_rejects_corruption_or_wrong_output_size(payload):
    with pytest.raises(ValueError):
        decode_frame(payload, (2, 3))


def test_missing_frames_and_truncated_blocks_are_rejected(tmp_path):
    payload = literal_lz4(np.zeros((2, 3), dtype='<u2').tobytes())
    path = tmp_path / 'depth.bin'
    path.write_bytes(struct.pack('<I', len(payload)) + payload)
    with pytest.raises(ValueError, match='missing'):
        read_selected_depth(path, [1], (2, 3))
    path.write_bytes(raw_deflate(np.zeros((2, 2, 3), dtype='<f4').tobytes()))
    with pytest.raises(ValueError, match='ends before'):
        read_selected_depth(path, [2], (2, 3))


@pytest.mark.parametrize('indices', [[], [1, 1], [-1], [True], list(range(129))])
def test_invalid_or_unbounded_selection_rejected_before_read(tmp_path, indices):
    with pytest.raises(ValueError):
        read_selected_depth(tmp_path / 'absent', indices)
