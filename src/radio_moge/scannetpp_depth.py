"""Bounded selective decoding of official ScanNet++ iPhone depth.bin formats.

Format reference: scannetpp/scannetpp, iphone/prepare_iphone_data.py.
Uses the existing system liblz4 through its bounded decompression API; no install.
Returned arrays are native-resolution float32 metres. Alignment/masks are the
preparation caller's responsibility; this module never creates training records.
"""
import ctypes
import ctypes.util
from functools import lru_cache
from pathlib import Path
import zlib

import numpy as np


@lru_cache(maxsize=1)
def _lz4():
    name = ctypes.util.find_library('lz4')
    if name is None:
        raise RuntimeError('An existing system liblz4 is required for LZ4 frames')
    library = ctypes.CDLL(name)
    library.LZ4_decompress_safe.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.c_int, ctypes.c_int]
    library.LZ4_decompress_safe.restype = ctypes.c_int
    return library


def _inflate_exact(payload, expected):
    decoder = zlib.decompressobj(-zlib.MAX_WBITS)
    output = decoder.decompress(payload, expected + 1)
    if (len(output) != expected or not decoder.eof
            or decoder.unused_data or decoder.unconsumed_tail):
        raise ValueError('Depth frame has an invalid decompressed size or boundary')
    return output


def decode_frame(payload, shape=(192, 256)):
    count = int(np.prod(shape))
    destination = ctypes.create_string_buffer(count * 2)
    source = ctypes.create_string_buffer(payload)
    written = _lz4().LZ4_decompress_safe(source, destination, len(payload), count * 2)
    if written == count * 2:
        values = np.frombuffer(destination.raw, dtype='<u2').reshape(shape)
        return values.astype(np.float32) / 1000, 'lz4_uint16_mm'
    try:
        raw = _inflate_exact(payload, count * 4)
    except (zlib.error, ValueError) as error:
        raise ValueError('Depth frame is neither bounded LZ4 uint16 nor raw DEFLATE float32') from error
    return np.frombuffer(raw, dtype='<f4').reshape(shape).copy(), 'deflate_float32_metres'


def _framed(file, indices, shape):
    wanted, result, codecs = set(indices), {}, set()
    maximum = max(indices)
    frame_bytes = int(np.prod(shape)) * 4
    size = file.seek(0, 2)
    file.seek(0)
    for index in range(maximum + 1):
        header = file.read(4)
        if len(header) != 4:
            raise ValueError('Requested depth frame is missing')
        length = int.from_bytes(header, 'little')
        if not 0 < length <= frame_bytes * 2 + 65536 or file.tell() + length > size:
            raise ValueError('Invalid compressed depth frame boundary')
        if index in wanted:
            result[index], codec = decode_frame(file.read(length), shape)
            codecs.add(codec)
        else:
            file.seek(length, 1)
    return result, sorted(codecs)


def _global_deflate(file, indices, shape):
    wanted, result = set(indices), {}
    frame_bytes = int(np.prod(shape)) * 4
    decoder, buffer, index = zlib.decompressobj(-zlib.MAX_WBITS), bytearray(), 0
    while index <= max(indices):
        compressed = file.read(65536)
        if not compressed:
            raise ValueError('Global depth stream ends before a requested frame')
        while compressed:
            before = len(compressed)
            try:
                output = decoder.decompress(compressed, frame_bytes - len(buffer))
            except zlib.error as error:
                raise ValueError('Invalid global raw DEFLATE depth stream') from error
            compressed = decoder.unconsumed_tail
            buffer.extend(output)
            if len(buffer) == frame_bytes:
                if index in wanted:
                    result[index] = np.frombuffer(buffer, dtype='<f4').reshape(shape).copy()
                index += 1
                buffer.clear()
                if index > max(indices):
                    return result, ['global_deflate_float32_metres']
            if decoder.eof:
                raise ValueError('Global depth stream ends before a requested frame')
            if not output and len(compressed) == before:
                raise ValueError('Global depth decoder made no progress')
    raise AssertionError('Unreachable depth decode state')


def read_selected_depth(path, indices, shape=(192, 256)):
    """Read at most 128 selected frames; never inflate a whole scene in memory.

    Framed streams skip unselected payloads. Global raw DEFLATE must be streamed
    through the last selected frame, with at most one decompressed frame buffered.
    File integrity must already be established by the download receipts.
    """
    indices = list(indices)
    if (not 1 <= len(indices) <= 128 or len(set(indices)) != len(indices)
            or any(type(i) is not int or i < 0 for i in indices)
            or len(shape) != 2 or any(type(d) is not int or not 1 <= d <= 2048 for d in shape)):
        raise ValueError('Expected distinct nonnegative frame indices and bounded image shape')
    with Path(path).open('rb') as file:
        header = file.read(4)
        length = int.from_bytes(header, 'little')
        limit = int(np.prod(shape)) * 8 + 65536
        framed = False
        if len(header) == 4 and 0 < length <= limit:
            payload = file.read(length)
            if len(payload) == length:
                try:
                    decode_frame(payload, shape)
                    framed = True
                except ValueError:
                    pass
        file.seek(0)
        result, codecs = _framed(file, indices, shape) if framed else _global_deflate(file, indices, shape)
    return result, dict(format='per_frame' if framed else 'global', codecs=codecs,
                        requested_frames=sorted(indices), decoded_frames=len(result),
                        units='metres', shape=list(shape), unselected_frames_not_retained=True)
