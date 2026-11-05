"""Audit closed full-grid evidence and render masks; no rays/models or admission."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from probe_scannetpp_v2_source_pilot import bound_path, read
from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import publish_json, sha_file
from radio_moge.saved_angle_match import match_saved_angles
from radio_moge.scanner_support import ScannerSupportContract, dense_image_support, station_support
from run_scannetpp_dense_source import ORDINAL, implementation
from scannetpp_dense_source import merge_masks

MASKS = ['conditional_support', 'numerical_support', 'both_stations_evaluated',
         'scan0_support', 'scan1_support', 'scan0_evaluated', 'scan1_evaluated']
REASONS = ['structural_footprint_unresolved', 'exact_target_visibility_unresolved',
           'raw_patch_or_direction_or_hit_incomplete', 'nominal_range_budget_exceeded']


def validate_closed(report, plan, arrays):
    h, w = plan['grid']
    n = h*w
    if (report['stage'] != 'full_grid_conditional_source_complete' or report['mode'] != 'full'
            or report['corpus_ordinal'] != ORDINAL or plan['corpus_ordinal'] != ORDINAL
            or plan['mode'] != 'full' or plan['scanner_order'] != ['0', '1']
            or report['full_grid_evaluated'] is not True
            or report['evaluated_station_slots'] != 2*n or report['requested_station_slots'] != 2*n
            or report['full_grid_pixels'] != n or report['parity_passed'] is not True
            or report['parity_slots'] != 96 or report['training_admission'] is not False
            or report['source_quality_accepted'] is not False or report['optimizer_updates'] != 0):
        raise ValueError('Closed full-grid scope, original parity and no admission required')
    if set(arrays) != set(MASKS) or any(a.shape != (h, w) or a.dtype != bool for a in arrays.values()):
        raise ValueError('All seven original boolean grids required')
    if not all(arrays[k].all() for k in ('both_stations_evaluated', 'scan0_evaluated', 'scan1_evaluated')):
        raise ValueError('Every original pixel must be evaluated by both scanners')


def add_observation(row, seen, reconstructed, reasons, reason_bits, arrays, ordinal, scene):
    h, w = seen.shape[1:]
    if row['scan'] not in ('0', '1') or row['corpus_ordinal'] != ordinal or row['scene'] != scene:
        raise ValueError('Original station/image identity required')
    i = int(row['scan'])
    y, x = row['pixel_yx']
    if not (type(y) is int and type(x) is int and 0 <= y < h and 0 <= x < w and row['slot'] == y*w+x):
        raise ValueError('Exact full-grid slot identity required')
    if seen[i, y, x]:
        raise ValueError('Duplicate station slot')
    computed = station_support(row)
    if computed != row['decision']:
        raise ValueError('Saved source decision differs from unchanged frozen rule')
    numerical = arrays['numerical_support'][y, x]
    if numerical:
        measurement = row['measurement']
        if measurement['slot'] != row['slot'] or match_saved_angles(measurement) != row['angle_matches']:
            raise ValueError('Original raw pixels/nominal direction residual evidence differs')
    elif ('measurement' in row or computed['conditional_station_support']
          or row['exact_target_visibility']['status'] != 'DSLR_numerically_unsupported'):
        raise ValueError('Numerically missing slot cannot claim source evidence')
    seen[i, y, x] = True
    reconstructed[i, y, x] = computed['conditional_station_support']
    reasons[i].update(computed['reasons'])
    for reason in computed['reasons']:
        if reason not in REASONS:
            raise ValueError('Unexpected source failure reason')
        reason_bits[i, y, x] |= 1 << REASONS.index(reason)


def rows(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def render(output, rgb, depth, arrays):
    numeric, mask = arrays['numerical_support'], arrays['conditional_support']
    h, w = depth.shape
    rgb = np.asarray(rgb.resize((w, h), Image.Resampling.LANCZOS))
    value = np.clip(np.log(np.maximum(depth, .05)/.05)/np.log(20/.05), 0, 1)
    colors = (np.stack([value, 1-abs(2*value-1), 1-value], -1)*255).astype(np.uint8)
    numeric_depth, accepted_depth = colors.copy(), colors.copy()
    numeric_depth[~numeric] = 0
    accepted_depth[~mask] = 0
    category = np.full((*depth.shape, 3), [84, 93, 107], np.uint8)
    category[numeric & ~mask] = [239, 163, 66]
    category[mask] = [39, 207, 139]
    overlay = np.rint(.5*rgb+.5*category).astype(np.uint8)
    panels = dict(rgb=rgb, numerical_depth=numeric_depth, retained_depth=accepted_depth,
                  support_categories=category, rgb_overlay=overlay,
                  scanner0_mask=np.repeat((arrays['scan0_support']*255).astype(np.uint8)[..., None], 3, axis=-1),
                  scanner1_mask=np.repeat((arrays['scan1_support']*255).astype(np.uint8)[..., None], 3, axis=-1))
    refs = {}
    for key, a in panels.items():
        p = output/(key+'.png')
        Image.fromarray(a).save(p)
        refs[key] = file_ref(p)
    page = Image.new('RGB', (3*w, 2*(h+38)), 'white')
    draw = ImageDraw.Draw(page)
    for index, (key, title) in enumerate([
        ('rgb', 'Perspective DSLR RGB'), ('numerical_depth', 'Numerical candidate depth: fixed log0.05-20m'),
        ('retained_depth', 'Conditional source-supported depth: SAME colors'),
        ('rgb_overlay', 'Green retained / orange evidence insufficient / gray no numerical support'),
        ('scanner0_mask', 'Fixed scanner0 conditional support'), ('scanner1_mask', 'Fixed scanner1 conditional support')]):
        x, y = (index % 3)*w, (index//3)*(h+38)
        draw.text((x+5, y+9), title, fill='black')
        page.paste(Image.fromarray(panels[key]), (x, y+38))
    p = output/'page.png'
    page.save(p)
    refs['page'] = file_ref(p)
    return refs


def run(root, source, output):
    report_ref = file_ref(source/'report.json')
    report = read(report_ref, root)
    plan = read(report['plan'], root)
    from dataclasses import asdict
    if (report['implementation'] != implementation() or plan['implementation'] != implementation()
            or plan['policy'] != asdict(ScannerSupportContract()) or plan['evidence'] != report['plan_evidence']):
        raise ValueError('Exact completed producer and frozen policy required')
    benchmark = read(plan['benchmark'], root)
    if benchmark['parity_passed'] is not True or benchmark['implementation'] != implementation():
        raise ValueError('Bound completed benchmark required')
    with np.load(bound_path(report['mask'], root)) as loaded:
        arrays = {k: loaded[k] for k in loaded.files}
    validate_closed(report, plan, arrays)
    h, w = plan['grid']
    record = read(plan['row']['saved_v2_record'], root)
    with np.load(bound_path(record['arrays'], root)) as a, np.load(bound_path(plan['row']['saved_numerical_mask'], root)) as m:
        depth = a['final_depth']
        if not np.array_equal(m['consensus_support'], arrays['numerical_support']):
            raise ValueError('Full-grid input mask differs from saved numerical result')
    targets_seen = np.zeros((h, w), bool)
    for t in rows(bound_path(report['targets'], root)):
        y, x = t['pixel_yx']
        if (not (0 <= y < h and 0 <= x < w) or t['slot'] != y*w+x or targets_seen[y, x]
                or t['numerical_support'] != bool(arrays['numerical_support'][y, x])
                or t['saved_z_m'] != float(depth[y, x])):
            raise ValueError('Complete original target grid/values required')
        targets_seen[y, x] = True
    if not targets_seen.all():
        raise ValueError('Missing target slots')
    seen = np.zeros((2, h, w), bool)
    reconstructed = np.zeros_like(seen)
    reasons, reason_bits = [Counter(), Counter()], np.zeros((2, h, w), np.uint8)
    chunk_hashes = set()
    for index, ref in enumerate(report['chunks']):
        if ref['path'] in chunk_hashes:
            raise ValueError('Duplicate chunk reference')
        chunk_hashes.add(ref['path'])
        for row in rows(bound_path(ref, root)):
            add_observation(row, seen, reconstructed, reasons, reason_bits, arrays, ORDINAL, plan['binding']['scene'])
        if index % 256 == 0:
            print(json.dumps(dict(checked_chunks=index+1, total_chunks=len(report['chunks']), observations=int(seen.sum()))), flush=True)
    if not seen.all():
        raise ValueError('Missing original station observations')
    excluded = {json.loads(s)['corpus_ordinal'] for s in bound_path(plan['evidence']['exclusions'], root).read_text().splitlines()}
    merged, evaluated = merge_masks(arrays['numerical_support'], reconstructed, seen, ORDINAL in excluded)
    for key, actual in dict(conditional_support=merged, both_stations_evaluated=evaluated,
            scan0_support=reconstructed[0], scan1_support=reconstructed[1],
            scan0_evaluated=seen[0], scan1_evaluated=seen[1]).items():
        if not np.array_equal(actual, arrays[key]):
            raise ValueError('Reconstructed full-grid masks differ from saved output')
    coverage = dense_image_support(arrays['numerical_support'], merged, True)
    if coverage != report['dense_coverage']:
        raise ValueError('Saved coverage differs from all-pixel denominator')
    image = plan['row']['original_selection']['image']
    rgb_path = Path(image['image'])
    if sha_file(rgb_path) != image['sha256']:
        raise ValueError('Original perspective RGB changed')
    with Image.open(rgb_path) as source_rgb:
        if source_rgb.size != (image['width'], image['height']) or source_rgb.getexif().get(274, 1) != 1:
            raise ValueError('Original RGB grid/orientation changed')
        rgb = source_rgb.convert('RGB')
    output.mkdir(exist_ok=False)
    panels = render(output, rgb, depth, arrays)
    data = output/'pixel-inspection.npz'
    np.savez_compressed(data, depth=depth, **arrays, scan_reason_bits=reason_bits)
    numeric = arrays['numerical_support']
    result = dict(stage='complete_saved_dense_mask_verified_visual_evidence_rendered_not_reviewed',
        source_report=report_ref, plan=report['plan'], corpus_ordinal=ORDINAL, scene=plan['binding']['scene'],
        RGB=dict(path=str(rgb_path), sha256=image['sha256']), grid=[h, w],
        target_slots_verified=h*w, station_slots_verified=int(seen.sum()), chunks_verified=len(chunk_hashes),
        saved_masks_exactly_reconstructed=True, source_decisions_recomputed=True,
        coverage=coverage, numerical_pixels=int(numeric.sum()),
        retained_fraction_of_numerical_pixels=float(merged.sum()/numeric.sum()) if numeric.any() else None,
        category_counts=dict(no_numerical_support=int((~numeric).sum()),
            numerical_but_source_unresolved=int((numeric & ~merged).sum()), conditional_retained=int(merged.sum())),
        station_support_pixels=reconstructed.sum(axis=(1, 2)).tolist(),
        overlapping_station_reason_counts=[dict(c) for c in reasons], reason_bit_order=REASONS,
        panels=panels, pixel_inspection=file_ref(data),
        depths_changed=False, missing_pixels_filled=False, mesh_reads=0, new_ray_queries=0,
        RGB_reads=1, new_numerical_renders=0, physical_accuracy_accepted=False,
        source_quality_accepted=False, training_admission=False, optimizer_updates=0,
        code=file_ref(Path(__file__)),
        caveats=['Single development view, not a source-wide quality estimate.',
            'Missing/occluded/ambiguous source evidence does not prove wrong depth.',
            'Two fixed scanner stations and1cm engineering filter are not independent physical calibration.'])
    publish_json(output/'report.json', result)
    print(json.dumps({k: result[k] for k in ('stage', 'coverage', 'category_counts')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.source.resolve(), args.output.resolve())
