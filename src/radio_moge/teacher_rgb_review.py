"""Exact, evidenced input exclusions; teacher masks cannot validate blank RGB."""
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

from .corpus_admission import checked_json, checked_path


REASON = 'reviewed_spatially_constant_rgb'


def verify_constant_rgb(row, color):
    data = Path(row['image']).read_bytes()
    if hashlib.sha256(data).hexdigest() != row['sha256']:
        raise ValueError('Reviewed RGB bytes changed')
    with Image.open(io.BytesIO(data)) as image:
        if (image.size != (row['width'], row['height'])
                or image.getexif().get(274, 1) != 1 or row.get('image_rotation_cw', 0)):
            raise ValueError('Reviewed RGB dimensions or orientation changed')
        extrema = image.convert('RGB').getextrema()
    if list(extrema) != [(int(value), int(value)) for value in color] or len(color) != 3:
        raise ValueError('Only exactly spatially constant reviewed RGB can be excluded')


def reviewed_rgb_exclusions(review_ref, rows, native_outcome_ref):
    review = checked_json(review_ref)
    if (review.get('stage') != 'rgb_input_review_complete' or review.get('passed') is not True
            or review.get('training_admission') is not False
            or review.get('decision') != 'exclude_exactly_spatially_constant_rgb'
            or review.get('scope_images') != len(rows)):
        raise ValueError('Explicit complete RGB input review required')
    audit = checked_json(review['diagnostic'])
    if (audit['stage'] != 'rgb_diagnostic_complete_not_accepted'
            or audit['native_outcomes'] != native_outcome_ref
            or audit['processed'] != len(rows) or audit['rgb_sha256_verified'] is not True):
        raise ValueError('RGB diagnostic differs from the exact native review scope')
    pending = {row['corpus_ordinal']: row for row in rows}
    if len(pending) != len(rows):
        raise ValueError('Repeated input RGB identity')
    excluded = {}
    with checked_path(audit['records']).open() as file:
        for line in file:
            item = json.loads(line)
            ordinal = item['corpus_ordinal']
            row = pending.pop(ordinal, None)
            if (row is None or row['split'] != 'train'
                    or any(item[key] != row[key] for key in ('image', 'sha256', 'group'))
                    or item['shape'] != [row['height'], row['width'], 3]):
                raise ValueError('RGB diagnostic changed or repeated a frozen input')
            uniform = item['channel_min'] == item['channel_max']
            if uniform != item['spatially_constant']:
                raise ValueError('RGB diagnostic constant flag disagrees with extrema')
            if uniform:
                verify_constant_rgb(row, item['channel_min'])
                excluded[ordinal] = item
    if (pending or sorted(excluded) != sorted(review['excluded_corpus_ordinals'])
            or sorted(excluded) != sorted(r['corpus_ordinal'] for r in audit['spatially_constant_images'])):
        raise ValueError('RGB review does not account for every input and exact exclusion')
    return excluded


def accounted_preflight(completed, rejections, rows, excluded):
    expected = {r['corpus_ordinal'] for r in rows} & set(excluded)
    rejected = [r['image']['corpus_ordinal'] for r in rejections]
    return bool(completed > 0 and completed + len(rejections) == len(rows)
        and len(set(rejected)) == len(rejected) and set(rejected) == expected
        and all(r['reason'] == REASON for r in rejections))
