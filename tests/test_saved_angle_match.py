from copy import deepcopy

import pytest

from radio_moge.saved_angle_match import OFFSETS, match_saved_angles


def fixture():
    # Deliberately reverse physical pixel order relative to probe order.
    return dict(scanner_pixel_yx=[4, 9], stored_azimuth_millirad=120,
        stored_polar_millirad=700,
        discrete_angle_sensitivity=dict(samples=[dict(azimuth_offset_millirad=a,
            polar_offset_millirad=p, nearest_whole_mesh_hit=dict(face_id=i,
                radial_range_m=2+i*.001)) for i, (a, p) in enumerate(OFFSETS)]),
        raw_patch=dict(bounds_yxyx=[3, 8, 6, 11], values=dict(
            azim=[[121]*3, [120]*3, [119]*3], elev=[[701, 700, 699]]*3,
            mask=[[255]*3 for _ in range(3)], range=[[2010]*3 for _ in range(3)])))


def test_exact_angle_identity_not_raw_array_order_drives_matching():
    result = match_saved_angles(fixture())
    assert [p['saved_probe_index'] for p in result['pixels']] == list(range(8, -1, -1))
    assert result['pixels'][0]['signed_range_residual_m'] == pytest.approx(.002)
    assert result['all_nine_residuals_available']
    assert result['distinct_residual_directions'] == 9
    assert not result['physical_accuracy_accepted']
    assert result['new_rays'] == 0


def test_quantized_duplicate_angles_keep_pixels_but_not_independent_ray_counts():
    m = fixture()
    m['raw_patch']['values']['azim'] = [[120]*3]*3
    m['raw_patch']['values']['elev'] = [[700]*3]*3
    result = match_saved_angles(m)
    assert result['residual_pixels'] == 9
    assert result['distinct_residual_directions'] == 1


def test_unmatched_direction_is_not_snapped_and_invalid_pixel_is_not_filled():
    m = fixture()
    m['raw_patch']['values']['azim'][0][0] = 122
    m['raw_patch']['values']['mask'][0][1] = 0
    m['discrete_angle_sensitivity']['samples'][6]['nearest_whole_mesh_hit'] = None
    result = match_saved_angles(m)
    assert [p['status'] for p in result['pixels'][:3]] == [
        'nominal_direction_not_probed', 'raw_range_or_mask_unsupported', 'saved_whole_mesh_no_hit']
    assert result['residual_pixels'] == 6
    assert not result['all_nine_residuals_available']


def test_border_preserves_nine_expected_pixel_slots():
    m = fixture()
    m['raw_patch']['bounds_yxyx'][0] = 4
    for k in m['raw_patch']['values']:
        m['raw_patch']['values'][k] = m['raw_patch']['values'][k][1:]
    result = match_saved_angles(m)
    assert len(result['pixels']) == 9 and result['retained_raw_pixels'] == 6
    assert [p['status'] for p in result['pixels'][:3]] == ['raw_pixel_not_retained']*3
    assert not result['all_nine_residuals_available']


@pytest.mark.parametrize('kind', ['center', 'order', 'shape', 'integer', 'bounds', 'nonfinite'])
def test_malformed_or_changed_evidence_rejected(kind):
    m = fixture()
    if kind == 'center':
        m['stored_azimuth_millirad'] += 1
    elif kind == 'order':
        m['discrete_angle_sensitivity']['samples'].reverse()
    elif kind == 'shape':
        m['raw_patch']['values']['mask'].pop()
    elif kind == 'integer':
        m['raw_patch']['values']['azim'][0][0] = 120.1
    elif kind == 'bounds':
        m['raw_patch']['bounds_yxyx'][0] -= 1
    else:
        m['discrete_angle_sensitivity']['samples'][0]['nearest_whole_mesh_hit']['radial_range_m'] = float('nan')
    with pytest.raises(ValueError):
        match_saved_angles(m)


@pytest.mark.parametrize('mm,depth,sign', [(2692, 2.62, 1), (3040, 3.114, -1), (1500, .08369, 1)])
def test_source_counterexamples_remain_signed_not_threshold_filtered(mm, depth, sign):
    m = fixture()
    m['raw_patch']['values']['range'] = [[mm]*3]*3
    for sample in m['discrete_angle_sensitivity']['samples']:
        sample['nearest_whole_mesh_hit']['radial_range_m'] = depth
    before = deepcopy(m)
    result = match_saved_angles(m)
    assert m == before
    assert all(p['signed_range_residual_m']*sign > .05 for p in result['pixels'])
    assert not result['whole_source_acceptance']


def test_complete_denominators_no_best_scanner_selection():
    from audit_scannetpp_saved_angle_matches import summary
    rows = [dict(corpus_ordinal=o, scene=str(o), slot=i, scan=s, angle_matches={},
                 projected_footprint=dict(sampled_C0_correspondence=False))
            for o in range(4) for i in range(96) for s in ('0', '1')]
    rows[0]['angle_matches'] = match_saved_angles(fixture())
    rows[0]['projected_footprint']['sampled_C0_correspondence'] = True
    result = summary(rows)
    assert result['station_slots'] == 768 and result['DSLR_slots'] == 384
    assert result['structural_complete_nominal_comparison_DSLR_any_station'] == 1
    assert result['structural_complete_nominal_comparison_DSLR_both_stations'] == 0
    assert result['physical_accepted_pixels'] == 0
    rows[2]['angle_matches'] = deepcopy(rows[0]['angle_matches'])
    result = summary(rows)
    assert result['sum_of_per_patch_distinct_residual_directions'] == 18
    assert result['distinct_station_direction_keys'] == 9
    assert result['unique_retained_source_pixels'] == 9
    with pytest.raises(ValueError, match='768 unique'):
        summary(rows[:-1]+rows[:1])
