"""Compare every retained raw pixel to an exactly matching saved nominal ray."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from probe_scannetpp_v2_source_pilot import read
from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import publish_json
from radio_moge.saved_angle_match import match_saved_angles


PRIOR_SHA = '241b6ffe307e5088a534b39ce5860b2784eb46037c448b80c0c798b7371de1bb'


def summary(rows):
    if len(rows) != 768 or len({(r['corpus_ordinal'], r['scan'], r['slot']) for r in rows}) != 768:
        raise ValueError('All 768 unique original station slots required')
    slots = defaultdict(list)
    for row in rows:
        slots[(row['corpus_ordinal'], row['slot'])].append(row)
    if len(slots) != 384 or any({r['scan'] for r in v} != {'0', '1'} for v in slots.values()):
        raise ValueError('All 384 original DSLR slots with both stations required')
    def complete(row):
        return row['angle_matches'].get('all_nine_residuals_available', False)
    def structural(row):
        return complete(row) and row['projected_footprint']['sampled_C0_correspondence']
    available = [r['angle_matches'] for r in rows if 'pixels' in r['angle_matches']]
    residuals = [p['signed_range_residual_m'] for m in available for p in m['pixels']
                 if 'signed_range_residual_m' in p]
    structure_residuals = [p['signed_range_residual_m'] for r in rows if structural(r)
                          for p in r['angle_matches']['pixels']]
    distinct_directions, distinct_pixels = set(), set()
    for row in rows:
        for pixel in row['angle_matches'].get('pixels', []):
            if 'signed_range_residual_m' in pixel:
                distinct_directions.add((row['scene'], row['scan'],
                    pixel['raw']['azim'], pixel['raw']['elev']))
            if 'raw' in pixel:
                distinct_pixels.add((row['scene'], row['scan'], *pixel['pixel_yx']))
    def quantiles(values):
        return (dict(zip(('min', 'p05', 'median', 'p95', 'max'),
            np.quantile(values, [0, .05, .5, .95, 1]).tolist())) if values else None)
    result = dict(DSLR_slots=384, station_slots=768,
        numerically_unsupported_station_slots=768-len(available),
        expected_raw_pixel_slots=len(available)*9,
        raw_pixel_statuses=dict(Counter(p['status'] for m in available for p in m['pixels'])),
        conditional_residual_pixels=len(residuals),
        sum_of_per_patch_distinct_residual_directions=sum(m['distinct_residual_directions'] for m in available),
        distinct_station_direction_keys=len(distinct_directions),
        unique_retained_source_pixels=len(distinct_pixels),
        complete_rawpatch_direction_multiplicity=dict(Counter(
            m['distinct_residual_directions'] for m in available if m['all_nine_residuals_available'])),
        signed_residual_m=quantiles(residuals), absolute_residual_m=quantiles(np.abs(residuals).tolist()),
        structural_signed_residual_m=quantiles(structure_residuals),
        structural_absolute_residual_m=quantiles(np.abs(structure_residuals).tolist()),
        pixel_statistics_not_independent_trials=True, physical_accepted_pixels=0)
    for label, predicate in [('complete_nominal_comparison', complete),
                             ('structural_complete_nominal_comparison', structural)]:
        result[label+'_station_slots'] = sum(predicate(r) for r in rows)
        result[label+'_DSLR_any_station'] = sum(any(predicate(r) for r in v) for v in slots.values())
        result[label+'_DSLR_both_stations'] = sum(all(predicate(r) for r in v) for v in slots.values())
    return result


def run(root, output):
    prior_ref = file_ref(root/'scannetpp-native-v2-projected-footprint-01/report.json')
    if prior_ref['sha256'] != PRIOR_SHA:
        raise ValueError('Frozen structural evidence required')
    prior = read(prior_ref, root)
    output.mkdir(exist_ok=False)
    contract = dict(prior=prior_ref,
        direction_identity='exact stored integer azim/elev pair; no snapping, wrapping, interpolation or new rays',
        range_convention='raw millimeters /1000 minus saved whole-mesh scanner-local radial parameter',
        support='original mask255 and50<rawmillimeters<65535; unsupported values retained without residual',
        ordering='nine spatial pixel slots around retained center; independently match all nine original probes',
        duplicate_angles='preserve each pixel and count distinct nominal directions separately',
        missing='all station/DSLR slots retained; missing patch/angle/hit explicit',
        nominal_quantization='same stored angles do not establish equal physical rays or pose calibration',
        structural='report C0 selected-patch condition separately; never suppress conflicting source residuals',
        scope='conditional internal source comparison only, not a label acceptance contract',
        physical_error_tolerance_m=None, continuous_uncertainty_bound=False,
        independent_physical_calibration=False, training_admission=False)
    publish_json(output/'contract.json', contract)
    cache, rows, refs = {}, [], []
    def saved(ref):
        key = (ref['path'], ref['sha256'])
        if key not in cache:
            cache[key] = read(ref, root)
        return cache[key]
    for ref in prior['observations']:
        batch = []
        for row in saved(ref):
            station = saved(row['station'])
            targets = saved(station['targets'])
            target, measurement = targets['slots'][row['slot']], station['measurements'][row['slot']]
            if (targets['corpus_ordinal'] != row['corpus_ordinal'] or station['scan'] != row['scan']
                    or target['slot'] != row['slot'] or measurement['slot'] != row['slot']
                    or target['pixel_yx'] != row['pixel_yx'] or targets['scene'] != row['scene']):
                raise ValueError('Original source identity changed')
            if target['numerical_support']:
                if measurement['scanner_identity'] != [row['scene'], row['scan']]:
                    raise ValueError('Original scanner identity changed')
                result = match_saved_angles(measurement)
            else:
                result = dict(status='DSLR_numerically_unsupported', physical_accuracy_accepted=False)
            batch.append(dict(row, projected_observations=ref, angle_matches=result))
        path = output/Path(ref['path']).name
        publish_json(path, batch)
        refs.append(file_ref(path))
        rows.extend(batch)
    examples = []
    for before in prior['counterexamples_preserved']:
        key = (before['corpus_ordinal'], before['scan'], before['slot'])
        after = next(r for r in rows if (r['corpus_ordinal'], r['scan'], r['slot']) == key)
        if before['range_evidence'] != after['range_evidence']:
            raise ValueError('Earlier source gap evidence changed')
        examples.append(after)
    result = dict(stage='saved_nominal_angle_comparison_complete_not_accepted', prior=prior_ref,
        contract=file_ref(output/'contract.json'), observations=refs, coverage=summary(rows),
        counterexamples_preserved=examples, prior_counterexample=prior['prior_counterexample'],
        new_source_reads=0, new_mesh_reads=0, new_RGB_reads=0, new_rays=0,
        optimizer_updates=0, gpu_used=False, confirmation_images_read=0,
        source_quality_accepted=False, labels_changed=False, training_admission=False,
        whole_source_acceptance_contract_complete=False,
        code=[file_ref(Path(__file__)), file_ref(Path(__file__).parents[1]/'src/radio_moge/saved_angle_match.py')])
    publish_json(output/'report.json', result)
    print(result['coverage'], flush=True)
    print(file_ref(output/'report.json'), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
