"""Read-only RGB identity and sampling diagnostics; no automatic exclusions."""
from collections import Counter, defaultdict
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image


ARCHIVE_HASH_SOURCES = {'irs', 'structured3d', 'tartanair'}


def decoded_hash(image):
    """Canonical, size-bound RGB hash, independent of the file container."""
    return hashlib.sha256(f'{image.width}x{image.height}:RGB:'.encode() + image.tobytes()).hexdigest()


def dhash128(image):
    """Horizontal + vertical grayscale comparisons; a retrieval descriptor only."""
    gray = image.convert('L')
    horizontal = np.asarray(gray.resize((9, 8), Image.Resampling.BILINEAR))
    vertical = np.asarray(gray.resize((8, 9), Image.Resampling.BILINEAR))
    bits = np.concatenate(((horizontal[:, 1:] > horizontal[:, :-1]).ravel(),
                           (vertical[1:, :] > vertical[:-1, :]).ravel()))
    return np.packbits(bits).tobytes().hex()


def inspect_rgb(row):
    """Verify frozen bytes/metadata, then describe the exact oriented training RGB."""
    data = Path(row['image']).read_bytes()
    if row.get('sha256_kind') != 'encoded_file_bytes' or hashlib.sha256(data).hexdigest() != row['sha256']:
        raise ValueError('Frozen encoded RGB hash differs')
    if row.get('split') != 'train':
        raise ValueError('Only frozen training RGB belongs in this census')
    with Image.open(io.BytesIO(data)) as file:
        if file.getexif().get(274, 1) != 1:
            raise ValueError('Undocumented EXIF orientation')
        image = file.convert('RGB')
    original_size = image.size
    raw_hash = decoded_hash(image)
    supplied = row.get('decoded_rgb_sha256')
    if supplied:
        if row['dataset'] == 'nyuv2':
            expected = hashlib.sha256(image.tobytes()).hexdigest()
        elif row['dataset'] in ARCHIVE_HASH_SOURCES:
            expected = raw_hash
        else:
            raise ValueError('Unreviewed stored decoded-hash convention')
        if supplied != expected:
            raise ValueError('Stored decoded RGB hash differs under its documented convention')
    rotation = row.get('image_rotation_cw', 0)
    if type(rotation) is not int or rotation not in (0, 90, 180, 270):
        raise ValueError('Undocumented RGB rotation')
    if rotation:
        image = image.transpose({90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180,
                                 270: Image.Transpose.ROTATE_90}[rotation])
    if ('width' in row or 'height' in row) and image.size != (row.get('width'), row.get('height')):
        raise ValueError('Frozen oriented RGB dimensions differ')
    return image, dict(raw_canonical_rgb_sha256=raw_hash, canonical_rgb_sha256=decoded_hash(image),
        stored_decoded_hash_verified=bool(supplied), original_width=original_size[0],
        original_height=original_size[1], width=image.width, height=image.height,
        image_rotation_cw=rotation, dhash128=dhash128(image))


def choose_groups(groups, limit=16, coco_limit=128):
    """Largest/smallest and stable-hash groups; all small environment families."""
    bins = defaultdict(list)
    for group in groups:
        bins[group['dataset']].append(group)
    result = []
    for dataset in sorted(bins):
        rows = sorted(bins[dataset], key=lambda g: (g['images'], g['group']))
        count = coco_limit if dataset == 'coco2017' else limit
        selected = {g['group']: g for g in (rows[0], rows[-1])}
        for row in sorted(rows, key=lambda g: hashlib.sha256(('rgb-audit-43:' + g['group']).encode()).digest()):
            if len(selected) >= count:
                break
            selected[row['group']] = row
        result.extend(selected[k] for k in sorted(selected))
    return result


def choose_rows(rows):
    """Consecutive first/middle/last quartets plus four stable-hash rows."""
    ordered = sorted(rows, key=lambda r: r['corpus_ordinal'])
    middle = max(0, len(ordered) // 2 - 2)
    picks = ordered[:4] + ordered[middle:middle + 4] + ordered[-4:]
    picks += sorted(ordered, key=lambda r: (r['sha256'], r['corpus_ordinal']))[:4]
    return sorted({r['corpus_ordinal']: r for r in picks}.values(), key=lambda r: r['corpus_ordinal'])


def pair_diagnostics(records, threshold=8, limit=8):
    """Exhaust all pairs in this small diagnostic sample, never infer prevalence."""
    counts, examples, exact = Counter(), defaultdict(list), defaultdict(list)
    integers = [int(row['dhash128'], 16) for row in records]
    compared = Counter()
    for i, a in enumerate(records):
        exact[a['canonical_rgb_sha256']].append(a['corpus_ordinal'])
        for j in range(i):
            b = records[j]
            category = ('cross_dataset' if a['dataset'] != b['dataset'] else
                        a['dataset'] + (':same_group' if a['group'] == b['group'] else ':cross_group'))
            compared[category] += 1
            distance = (integers[i] ^ integers[j]).bit_count()
            if distance <= threshold:
                counts[category] += 1
                pair = dict(a=b['corpus_ordinal'], b=a['corpus_ordinal'], distance=distance)
                best = examples[category]
                best.append(pair)
                best.sort(key=lambda v: (v['distance'], v['a'], v['b']))
                del best[limit:]
    return dict(sample_images=len(records), pairs_compared=dict(compared),
        descriptor_bits=128, retrieval_threshold=threshold, candidate_pairs=dict(counts),
        examples=dict(examples), canonical_exact_clusters=[v for v in exact.values() if len(v) > 1],
        requires_visual_review=True, representative_prevalence_estimate=False,
        full_corpus_near_duplicate_scan=False, automatic_exclusions=0)
