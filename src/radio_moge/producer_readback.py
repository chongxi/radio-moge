"""Reuse a completed producer readback for an explicitly reviewed source subset.

This does not claim a new array census or reconstruct pixel histograms from
quantiles. The loader still hashes and decodes every actual training payload.
"""
import hashlib
import math
from pathlib import Path

from .geometry_cache import canonical_json, validate_member_reference
from .geometry_sampling import source_supervision


MODE = 'completed_producer_readback_with_per_read_sha256'


def check_pack(generation, index_path, receipt_path, receipt, tar):
    completed = generation.get('completed_at')
    if (generation.get('stage') != 'complete' or generation.get('mode') != 'generate'
            or generation.get('training_admission') is not False
            or generation.get('training_started') is not False
            or generation.get('optimizer_updates') != 0
            or generation.get('verified_cache_members') != generation.get('completed')
            or not isinstance(completed, (int, float)) or not math.isfinite(completed)):
        raise ValueError('Completed producer array readback is required')
    for path in (index_path, receipt_path, tar):
        info = Path(path).stat()
        if info.st_mtime > completed:
            raise ValueError('Producer input changed after completed readback')
    if tar.stat().st_size != receipt['bytes']:
        raise ValueError('Producer pack size changed')


def member_summary(entry):
    validate_member_reference(entry)
    meta = entry['geometry_metadata']
    shape, n = meta['shape'], meta['valid_pixels']
    q = entry['depth_quantiles_m']
    if (len(shape) != 2 or any(type(x) is not int or x <= 0 for x in shape)
            or type(n) is not int or not 0 < n <= math.prod(shape)
            or len(q) != 5 or any(not math.isfinite(x) or x <= 0 for x in q)
            or list(q) != sorted(q)
            or not math.isclose(entry['valid_fraction'], n / math.prod(shape), rel_tol=0, abs_tol=1e-12)
            or hashlib.sha256(canonical_json(meta)).hexdigest() != entry['key']):
        raise ValueError('Invalid completed-producer receipt statistics')
    mode = source_supervision(meta)
    pseudo = mode == 'relative_pseudo'
    return dict(depth_quantiles=q, external_flag=False, coverage=dict(
        statistics_kind='producer_receipt_quantiles', supervision=mode,
        grid_pixels=math.prod(shape), valid_pixels=n, valid_fraction=entry['valid_fraction'],
        depth_units='relative_to_image_valid_median' if pseudo else 'meters_camera_z',
        quantile_probabilities=[0, .01, .5, .99, 1],
        depth_quantiles=[x / q[2] for x in q] if pseudo else q,
        histogram_counts=None, per_pixel_histogram_recomputed=False))
