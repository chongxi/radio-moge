from copy import deepcopy

import numpy as np
import pytest

from radio_moge.scanner_support import (
    ScannerSupportContract, dense_image_support, exact_target_visibility,
    paired_support, station_support,
)


def target():
    return dict(face_id=2, point_world=[0., 0., 2.],
                vertices_world=[[-1., -1., 2.], [1., -1., 2.], [0., 1., 2.]])


def hit(depth=2., face=2):
    return dict(face_id=face, point_world=[0., 0., depth], radial_range_m=depth)


def row(scan='0'):
    return dict(scene='s', corpus_ordinal=1, slot=0, pixel_yx=[12, 15], scan=scan,
        projected_footprint=dict(sampled_C0_correspondence=True),
        exact_target_visibility=exact_target_visibility(target(), np.eye(4), hit()),
        angle_matches=dict(all_nine_residuals_available=True, pixels=[dict(
            status='conditional_same_stored_angle_residual', nominal_angle_matched=True,
            raw_supported=True, signed_range_residual_m=.001) for _ in range(9)]))


def test_exact_target_visibility_preserves_rounded_full_published_parameter():
    pose = np.eye(4)
    pose[2, 2] = 1.0000006
    h = hit()
    h['radial_range_m'] = 2/pose[2, 2]
    result = exact_target_visibility(target(), pose, h)
    assert result['exact_target_first_surface']
    assert result['first_minus_target_radial_m'] == 0
    assert not result['unique_first_face_proven']
    assert not result['measured_range_at_exact_target_direction']
    assert not result['physical_accuracy_accepted']


@pytest.mark.parametrize('h,status', [(None, 'exact_target_ray_no_hit'),
    (hit(.008, 3), 'exact_target_occluded'), (hit(2.1), 'exact_target_first_hit_disagrees'),
    (hit(2., 3), 'exact_target_coincident_other_face')])
def test_first_hit_missing_occluded_or_coincident_never_skipped(h, status):
    result = exact_target_visibility(target(), np.eye(4), h)
    assert result['status'] == status and not result['exact_target_first_surface']


def test_boundary_and_wrong_ray_rejected_without_fitting():
    t = target()
    t['point_world'] = [0., 1., 2.]
    h = dict(face_id=2, point_world=t['point_world'], radial_range_m=float(np.sqrt(5)))
    assert exact_target_visibility(t, np.eye(4), h)['status'] == 'exact_target_triangle_boundary'
    h['point_world'] = [.01, 1., 2.]
    with pytest.raises(ValueError, match='unnormalized'):
        exact_target_visibility(t, np.eye(4), h)


def test_centimeter_policy_is_fixed_budget_not_physical_claim_or_tunable():
    r = row()
    r['angle_matches']['pixels'][0]['signed_range_residual_m'] = .01
    assert station_support(r)['conditional_station_support']
    r['angle_matches']['pixels'][0]['signed_range_residual_m'] = np.nextafter(.01, np.inf)
    result = station_support(r)
    assert not result['conditional_station_support']
    assert result['reasons'] == ['nominal_range_budget_exceeded']
    assert not result['physical_accuracy_accepted']
    with pytest.raises(ValueError, match='no runtime tuning'):
        ScannerSupportContract(maximum_nominal_range_residual_m=.02)


def test_missing_or_structurally_unresolved_evidence_never_passes():
    r = row()
    r['angle_matches']['all_nine_residuals_available'] = False
    r['angle_matches']['pixels'][1].pop('signed_range_residual_m')
    r['projected_footprint']['sampled_C0_correspondence'] = False
    r.pop('exact_target_visibility')
    result = station_support(r)
    assert len(result['reasons']) == 3 and not result['conditional_station_support']


def test_favorable_station_cannot_override_peer_conflict_or_prior_exclusion():
    rows = [row('0'), row('1')]
    assert paired_support(rows, ['0', '1'], True, False)['conditional_sample_candidate']
    rows[1]['angle_matches']['pixels'][0]['signed_range_residual_m'] = -.068
    before = deepcopy(rows)
    result = paired_support(rows, ['0', '1'], True, False)
    assert not result['conditional_sample_candidate'] and rows == before
    assert -.068 in result['stations'][1]['residuals_preserved']
    assert not paired_support([row('0'), row('1')], ['0', '1'], True, True)['conditional_sample_candidate']
    assert not paired_support([row('0'), row('1')], ['0', '1'], False, False)['conditional_sample_candidate']


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'replacement', 'different_target', 'bad_pixel'])
def test_station_identity_and_evidence_cannot_be_substituted(kind):
    rows = [row('0'), row('1')]
    if kind == 'missing':
        rows.pop()
    elif kind == 'duplicate':
        rows[1]['scan'] = '0'
    elif kind == 'replacement':
        rows[1]['scan'] = '2'
    elif kind == 'different_target':
        rows[1]['slot'] = 1
    else:
        rows[1]['angle_matches']['pixels'][0]['raw_supported'] = False
    with pytest.raises(ValueError):
        paired_support(rows, ['0', '1'], True, False)


def test_sparse_probe_coverage_is_never_dense_image_coverage():
    a = np.ones((20, 100), bool)
    b = np.zeros_like(a)
    b.ravel()[:100] = True
    assert not dense_image_support(a, b, False)['conditional_dense_coverage_sufficient']
    assert dense_image_support(a, b, True)['conditional_dense_coverage_sufficient']
    b.ravel()[99] = False
    assert not dense_image_support(a, b, True)['conditional_dense_coverage_sufficient']
    with pytest.raises(ValueError, match='cannot create'):
        dense_image_support(~a, b, True)
    small = np.ones((4, 4), bool)
    assert not dense_image_support(small, small, True)['conditional_dense_coverage_sufficient']


def test_new_query_uses_exact_target_direction_unshifted_and_preserves_unsupported():
    from probe_scannetpp_exact_target_visibility import query_targets
    t = dict(target(), slot=0, pixel_yx=[2, 4], numerical_support=True)
    missing = dict(slot=1, pixel_yx=[3, 4], numerical_support=False)
    pose = np.eye(4)
    pose[2, 2] = 1.0000006
    class Engine:
        def intersects_location(self, origins, rays, multiple_hits):
            assert not multiple_hits and origins.shape == (1, 3)
            assert np.array_equal(origins, [[0., 0., 0.]])
            assert np.array_equal(rays, [[0., 0., 1.0000006]])
            return np.array([[0., 0., 2.]]), np.array([0]), np.array([2])
    result = query_targets([t, missing], pose, Engine())
    assert result[0]['visibility']['exact_target_first_surface']
    assert result[1]['visibility']['status'] == 'DSLR_numerically_unsupported'


def test_no_new_ray_for_fully_unsupported_station():
    from probe_scannetpp_exact_target_visibility import query_targets
    class Engine:
        def intersects_location(self, *args, **kwargs):
            raise AssertionError('Unsupported slots must not query')
    result = query_targets([dict(slot=0, pixel_yx=[0, 0], numerical_support=False)], np.eye(4), Engine())
    assert not result[0]['visibility']['exact_target_first_surface']


def test_policy_report_keeps_all_sample_slots_and_never_admits_dense_images():
    from audit_scannetpp_conditional_source_support import coverage
    points = [dict(corpus_ordinal=o, slot=i, decision=dict(conditional_sample_candidate=False,
        reasons=['both_selected_stations_not_supported'], stations=[dict(reasons=['missing'])]*2))
        for o in range(4) for i in range(96)]
    points[0]['decision'] = dict(conditional_sample_candidate=True, reasons=[], stations=[dict(reasons=[])]*2)
    report = coverage(points)
    assert report['conditional_sample_candidates'] == 1 and report['DSLR_slots'] == 384
    assert report['per_image'][0]['sampled_fraction'] == 1/96
    assert not report['per_image'][0]['image_eligible_for_training']
    assert report['source_accepted_images'] == 0
    with pytest.raises(ValueError, match='384unique'):
        coverage(points[:-1]+points[:1])
