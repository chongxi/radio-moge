"""Descriptive corpus coverage; statistics never constitute quality acceptance."""
from collections import Counter, defaultdict
import json
import math

import numpy as np


METRIC_EDGES = (.5, 1., 2., 5., 10., 20., 50., 200.)
RELATIVE_EDGES = (.25, .5, 1., 2., 4., 8.)
SUPPORT_EDGES = (.01, .05, .1, .25, .5, .75, .95)
GROUP_MEANING = {
    'arkitscenes': 'visit; multiple videos of a visit are not independent groups',
    'hypersim': 'scene; cameras and trajectory frames share the scene',
    'structured3d': 'scene; rooms and furnishing variants share the scene',
    'scannetpp_dslr': 'scan scene; DSLR views share the scene',
    'tartanair': 'environment family; day/night factory variants are merged',
    'irs': 'environment family; eyes and adjacent frames share the family',
    'nyuv2': 'published scene identity',
    'coco2017': 'image identity only; independent physical scenes are unknown',
}


def geometry_coverage(depth, valid, supervision):
    """Count only verified observations; estimated pseudo scale is discarded."""
    if (valid.dtype != np.bool_ or depth.shape != valid.shape or not valid.any()
            or supervision not in ('metric_native', 'relative_pseudo')):
        raise ValueError('Coverage requires valid supported geometry and explicit supervision')
    values = np.asarray(depth[valid], dtype=np.float64)
    if not np.isfinite(values).all() or not (values > 0).all():
        raise ValueError('Invalid supported depth in coverage')
    median = float(np.median(values))
    pseudo = supervision == 'relative_pseudo'
    if pseudo:
        values /= median
    edges = RELATIVE_EDGES if pseudo else METRIC_EDGES
    counts = np.bincount(np.searchsorted(edges, values, side='right'), minlength=len(edges) + 1)
    return dict(schema_version=1, supervision=supervision, grid_pixels=int(valid.size),
        valid_pixels=int(valid.sum()), valid_fraction=float(valid.mean()),
        depth_units='relative_to_image_valid_median' if pseudo else 'meters_camera_z',
        bin_upper_edges=[*edges, None], histogram_counts=counts.tolist(),
        median_depth_m=None if pseudo else median)


def metadata_coverage(db, *, accounting=False, write_group=None):
    """Full metadata counts, group concentration and exact decoded-RGB repeats.

    Exact repeats are not perceptual near-duplicate detection. Group labels are
    source-specific and never promoted to a claim of independent scenes.
    """
    disposition = 'disposition' if accounting else "'inventory_only'"
    source_counts = [dict(dataset=d, disposition=s, images=n) for d, s, n in db.execute(
        f'SELECT dataset,{disposition},COUNT(*) FROM images GROUP BY 1,2 ORDER BY 1,2')]
    groups, tops, totals, squares = Counter(), defaultdict(list), Counter(), Counter()
    full_groups, fully_excluded = Counter(), Counter()
    rows = db.execute(f'SELECT dataset,scene_group,{disposition},COUNT(*) FROM images '
                      'GROUP BY 1,2,3 ORDER BY 1,2,3')
    current, counts = None, {}

    def flush(key, values):
        if key is None:
            return
        dataset, group = key
        full_groups[dataset] += 1
        if accounting and values.get('selected', 0) == 0:
            fully_excluded[dataset] += 1
        if write_group is not None:
            write_group(dict(dataset=dataset, group=group, counts=values))
        n = values.get('selected' if accounting else 'inventory_only', 0)
        if n:
            groups[dataset] += 1
            totals[dataset] += n
            squares[dataset] += n * n
            top = tops[dataset]
            top.append(dict(group=group, images=n))
            top.sort(key=lambda v: (-v['images'], v['group']))
            del top[10:]

    for dataset, group, outcome, n in rows:
        key = (dataset, group)
        if key != current:
            flush(current, counts)
            current, counts = key, {}
        counts[outcome] = n
    flush(current, counts)
    concentration = [dict(dataset=d, group_meaning=GROUP_MEANING.get(d, 'unreviewed grouping'),
        scope_groups=full_groups[d], retained_groups=groups[d], fully_excluded_groups=fully_excluded[d],
        largest_group_fraction=tops[d][0]['images'] / totals[d] if totals[d] else None,
        effective_group_count=totals[d] ** 2 / squares[d] if squares[d] else 0.,
        effective_group_definition='inverse group-concentration sum; not independent scene count',
        largest_groups=tops[d]) for d in sorted(full_groups)]
    decoded = "json_extract(row_json,'$.decoded_rgb_sha256')"
    valid_hash = f"(length({decoded})=64 AND {decoded} NOT GLOB '*[^0-9a-f]*')"
    image_hashes = [dict(dataset=d, images=n, rows_with_decoded_hash=h or 0,
                         missing_or_invalid_decoded_hash=n - (h or 0)) for d, n, h in db.execute(
        f'SELECT dataset,COUNT(*),SUM({valid_hash}) FROM images GROUP BY dataset ORDER BY dataset')]
    duplicates = dict(clusters=0, excess_rows=0, cross_dataset_clusters=0, examples=[])
    for checksum, count, sources in db.execute(
            f'SELECT {decoded},COUNT(*),COUNT(DISTINCT dataset) FROM images '
            f'WHERE {valid_hash} GROUP BY 1 HAVING COUNT(*)>1 ORDER BY 2 DESC,1'):
        duplicates['clusters'] += 1
        duplicates['excess_rows'] += count - 1
        duplicates['cross_dataset_clusters'] += int(sources > 1)
        if len(duplicates['examples']) < 20:
            duplicates['examples'].append(dict(decoded_rgb_sha256=checksum, rows=count, datasets=sources))
    duplicates['scope'] = 'all frozen rows, including exclusions; metadata hashes only, no RGB reread'
    return dict(source_counts=source_counts, groups=concentration, decoded_hash_coverage=image_hashes,
        exact_decoded_duplicates=duplicates, perceptual_near_duplicates_checked=False,
        indoor_outdoor_image_content_checked=False)


def selected_geometry_coverage(db):
    """Aggregate statistics recorded during the full verified payload reread."""
    result = {}
    for dataset, kind, encoded in db.execute("SELECT dataset,label_kind,coverage_json FROM images "
                                             "WHERE disposition='selected' ORDER BY ordinal"):
        if not encoded:
            raise ValueError('Selected image lacks verified per-pixel coverage statistics')
        value = json.loads(encoded)
        pseudo = kind == 'teacher_pseudo'
        expected_mode = 'relative_pseudo' if pseudo else 'metric_native'
        expected_units = 'relative_to_image_valid_median' if pseudo else 'meters_camera_z'
        edges = RELATIVE_EDGES if pseudo else METRIC_EDGES
        counts = value['histogram_counts']
        if (value['supervision'] != expected_mode or value['depth_units'] != expected_units
                or value['bin_upper_edges'] != [*edges, None]
                or len(counts) != len(edges) + 1
                or any(type(n) is not int or n < 0 for n in counts)
                or sum(counts) != value['valid_pixels']
                or not 0 < value['valid_pixels'] <= value['grid_pixels']
                or not math.isclose(value['valid_fraction'], value['valid_pixels'] / value['grid_pixels'],
                                    abs_tol=1e-12, rel_tol=0)
                or (pseudo and value['median_depth_m'] is not None)):
            raise ValueError('Coverage statistics violate depth/support provenance')
        key = (dataset, kind)
        if key not in result:
            result[key] = dict(dataset=dataset, label_kind=kind, supervision=expected_mode,
                depth_units=expected_units, bin_upper_edges=[*edges, None], histogram_counts=[0] * len(counts),
                mean_per_image_bin_fraction=[0.] * len(counts), images=0, grid_pixels=0, valid_pixels=0,
                image_mean_valid_fraction=0., support_bin_upper_edges=[*SUPPORT_EDGES, None],
                support_histogram=[0] * (len(SUPPORT_EDGES) + 1))
        total = result[key]
        total['images'] += 1
        for field in ('grid_pixels', 'valid_pixels'):
            total[field] += value[field]
        total['image_mean_valid_fraction'] += value['valid_fraction']
        total['support_histogram'][int(np.searchsorted(SUPPORT_EDGES, value['valid_fraction'], side='right'))] += 1
        for i, count in enumerate(counts):
            total['histogram_counts'][i] += count
            total['mean_per_image_bin_fraction'][i] += count / value['valid_pixels']
    for row in result.values():
        row['image_mean_valid_fraction'] /= row['images']
        row['pixel_weighted_valid_fraction'] = row['valid_pixels'] / row['grid_pixels']
        row['mean_per_image_bin_fraction'] = [v / row['images'] for v in row['mean_per_image_bin_fraction']]
    if not result:
        raise ValueError('No selected geometry coverage')
    return [result[key] for key in sorted(result)]
