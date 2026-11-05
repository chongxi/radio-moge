"""Match original scanner pixels to saved nominal-angle whole-mesh probes.

Exact integer angles are exact *stored* identities, not a physical angular
calibration. The range residual remains conditional on the released pose,
quantized angle, ray tracer and shared acquisition. No interpolation or fitting.
"""
import math


OFFSETS = [(a, p) for a in (-1, 0, 1) for p in (-1, 0, 1)]


def integer(value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError('Original integer storage required')
    return value


def match_saved_angles(measurement):
    cy, cx = map(integer, measurement['scanner_pixel_yx'])
    az, pol = (integer(measurement[k]) for k in
               ('stored_azimuth_millirad', 'stored_polar_millirad'))
    samples = measurement['discrete_angle_sensitivity']['samples']
    if [(integer(s['azimuth_offset_millirad']), integer(s['polar_offset_millirad']))
            for s in samples] != OFFSETS:
        raise ValueError('Original ordered nine probes required')
    probes = {(az+a, pol+p): (i, s['nearest_whole_mesh_hit'])
              for i, ((a, p), s) in enumerate(zip(OFFSETS, samples))}
    for _, hit in probes.values():
        if hit is not None and (integer(hit['face_id']) < 0
                or not math.isfinite(hit['radial_range_m']) or hit['radial_range_m'] <= 0):
            raise ValueError('Invalid saved first hit')
    patch = measurement['raw_patch']
    y0, x0, y1, x1 = map(integer, patch['bounds_yxyx'])
    if not (0 <= y0 <= cy < y1 and 0 <= x0 <= cx < x1
            and cy-1 <= y0 and y1 <= cy+2 and cx-1 <= x0 and x1 <= cx+2):
        raise ValueError('Raw patch must be the retained center neighborhood')
    values = patch['values']
    for key in ('azim', 'elev', 'mask', 'range'):
        grid = values[key]
        if len(grid) != y1-y0 or any(len(row) != x1-x0 for row in grid):
            raise ValueError('Raw patch shape disagrees with its bounds')
        for row in grid:
            for value in row:
                integer(value)
    if (values['azim'][cy-y0][cx-x0], values['elev'][cy-y0][cx-x0]) != (az, pol):
        raise ValueError('Center angle identity differs from saved probe origin')
    pixels = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            y, x = cy+dy, cx+dx
            row = dict(offset_yx=[dy, dx], pixel_yx=[y, x],
                nominal_angle_matched=False, raw_supported=False,
                physical_accuracy_accepted=False)
            if not (y0 <= y < y1 and x0 <= x < x1):
                row['status'] = 'raw_pixel_not_retained'
                pixels.append(row)
                continue
            raw = {k: values[k][y-y0][x-x0] for k in ('azim', 'elev', 'mask', 'range')}
            if not (0 <= raw['mask'] <= 255 and 0 <= raw['range'] <= 65535):
                raise ValueError('Invalid raw storage range')
            row.update(raw=raw, measured_range_m=raw['range']/1000,
                raw_supported=raw['mask'] == 255 and 50 < raw['range'] < 65535)
            key = (raw['azim'], raw['elev'])
            if key not in probes:
                row['status'] = 'nominal_direction_not_probed'
            else:
                index, hit = probes[key]
                row.update(nominal_angle_matched=True, saved_probe_index=index,
                    saved_whole_mesh_first_hit=hit)
                if not row['raw_supported']:
                    row['status'] = 'raw_range_or_mask_unsupported'
                elif hit is None:
                    row['status'] = 'saved_whole_mesh_no_hit'
                else:
                    row.update(status='conditional_same_stored_angle_residual',
                        signed_range_residual_m=row['measured_range_m']-hit['radial_range_m'])
            pixels.append(row)
    matched = [p for p in pixels if 'signed_range_residual_m' in p]
    nominal = [p for p in pixels if p['nominal_angle_matched']]
    return dict(pixels=pixels, expected_raw_pixels=9, retained_raw_pixels=(y1-y0)*(x1-x0),
        supported_raw_pixels=sum(p['raw_supported'] for p in pixels),
        nominal_matched_pixels=len(nominal), residual_pixels=len(matched),
        distinct_nominal_directions=len({p['saved_probe_index'] for p in nominal}),
        distinct_residual_directions=len({p['saved_probe_index'] for p in matched}),
        all_nine_residuals_available=len(matched) == 9,
        signed_residual_interval_m=([min(p['signed_range_residual_m'] for p in matched),
                                    max(p['signed_range_residual_m'] for p in matched)] if matched else None),
        quantized_nominal_angles_not_physical_rays=True, independent_physical_calibration=False,
        continuous_uncertainty_bound=False, physical_accuracy_accepted=False,
        whole_source_acceptance=False, new_rays=0)
