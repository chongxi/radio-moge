from copy import deepcopy

import numpy as np
import pytest

from radio_moge.source_footprint import (
    FootprintContract, extract_topology, planar_footprint, sampled_range_evidence,
    validate_saved_hits,
)


def fixture():
    v = np.array([[-1., -1., 2], [1., -1., 2], [-1., 1., 2], [1., 1., 2]])
    f = np.array([[0, 1, 2], [1, 3, 2]])
    target = dict(face_id=0, point_world=[-.3, -.3, 2.], vertices_world=v[f[0]].tolist())
    hits = [dict(face_id=i % 2, point_world=[.3, .3, 2.] if i % 2 else [-.3, -.3, 2.],
                 radial_range_m=2.) for i in range(9)]
    return v, f, target, hits


def classify(v, f, target, hits, origin=None):
    return planar_footprint(target, hits, np.zeros(3) if origin is None else origin,
                            extract_topology(v, f, {target['face_id'], *(h['face_id'] for h in hits if h)}))


def test_coplanar_triangle_seam_is_not_a_different_surface_and_winding_is_irrelevant():
    v, f, t, h = fixture()
    for reversed_winding in (False, True):
        if reversed_winding:
            f[1] = f[1, ::-1]
        result = classify(v, f, t, h)
        assert result['diagnostic_planar_footprint']
        assert not result['physical_accuracy_accepted']


def test_detached_coordinate_identical_face_is_not_welded():
    v, f, t, h = fixture()
    v = np.concatenate([v, v[f[1]]])
    f[1] = [4, 5, 6]
    assert classify(v, f, t, h)['status'] == 'connectivity_unproven_with_retained_faces'


@pytest.mark.parametrize('shift', [.0005, -1.995])
def test_depth_jump_or_subcentimeter_foreground_is_not_skipped(shift):
    v, f, t, h = fixture()
    v = np.concatenate([v, v[f[1]]+[0, 0, shift]])
    f[1] = [4, 5, 6]
    for hit in h:
        if hit['face_id'] == 1:
            hit['point_world'][2] += shift
    assert classify(v, f, t, h)['status'] == 'nonplanar_or_depth_discontinuous_unresolved'


def test_crease_is_unresolved_not_flattened():
    v, f, t, h = fixture()
    v[3, 2] = 2.1
    for hit in h:
        if hit['face_id'] == 1:
            hit['point_world'][2] = 2.03
    assert classify(v, f, t, h)['status'] == 'nonplanar_or_depth_discontinuous_unresolved'


def test_global_third_face_cannot_be_hidden_by_selected_subset():
    v, f, t, h = fixture()
    v = np.concatenate([v, [[0., 0., 3.]]])
    f = np.concatenate([f, [[1, 2, 4]]])
    # Face2 not selected; its edge incidence must still be counted.
    assert classify(v, f, t, h)['status'] == 'nonmanifold_shared_edge_unresolved'


def test_same_side_overlap_does_not_prove_surface_continuity():
    v, f, t, h = fixture()
    v[3] = [0., 0., 2.]
    f[1] = [0, 1, 3]
    for hit in h:
        if hit['face_id'] == 1:
            hit['point_world'] = [0., -.5, 2.]
    assert classify(v, f, t, h)['status'] == 'overlapping_same_side_faces_unresolved'


def test_grazing_same_plane_remains_unresolved():
    v, f, t, h = fixture()
    assert classify(v, f, t, h, np.array([100., 0., 0.]))['status'] == 'grazing_footprint_unresolved'


def test_internal_edge_allowed_exposed_edge_unresolved():
    v, f, t, h = fixture()
    h[0]['point_world'] = [0., 0., 2.]
    assert classify(v, f, t, h)['diagnostic_planar_footprint']
    h[0]['point_world'] = [0., -1., 2.]
    assert classify(v, f, t, h)['status'] == 'exposed_or_unretained_edge_unresolved'


def test_missing_hit_keeps_denominator_and_partial_interval():
    v, f, t, h = fixture()
    h[4] = None
    result = classify(v, f, t, h)
    assert result['missing_hits'] == 1 and result['status'] == 'first_hit_missing'
    evidence = sampled_range_evidence(h, patch(2000))
    assert evidence['saved_first_hits'] == 8
    assert evidence['first_hit_interval_m'] == [2., 2.]
    assert evidence['status'] == 'source_patch_or_first_hits_incomplete'


def patch(mm):
    return dict(values=dict(range=[[mm]*3]*3, mask=[[255]*3]*3))


@pytest.mark.parametrize('depth,measurement,expected', [(2.62, 2692, .072), (3.114, 3040, -.074),
                                                       (.08369004726409912, 1500, 1.416309952735901)])
def test_source_counterexamples_remain_visible_even_without_topology(depth, measurement, expected):
    hits = [dict(radial_range_m=depth, face_id=i) for i in range(9)]
    result = sampled_range_evidence(hits, patch(measurement))
    assert result['signed_interval_separation_m'] == pytest.approx(expected)
    assert not result['physical_accuracy_accepted']
    assert not result['continuous_uncertainty_bound']


def test_angle_sensitive_envelope_not_reported_as_center_error_bound():
    hits = [dict(radial_range_m=2.68157) for _ in range(8)]+[dict(radial_range_m=2.633562)]
    p = patch(2630)
    p['values']['range'][0] = [2633]*3
    result = sampled_range_evidence(hits, p)
    assert result['signed_interval_separation_m'] == pytest.approx(-.000562)
    p['values']['mask'][0] = [0]*3
    assert sampled_range_evidence(hits, p)['status'] == 'source_patch_or_first_hits_incomplete'


def test_rounded_pose_preserves_radial_parameter_without_normalizing():
    rays = np.tile([0., 0., 1.0000006], (9, 1))
    hits = [dict(radial_range_m=2., point_world=[0., 0., 2.0000012]) for _ in range(9)]
    validate_saved_hits(hits, np.zeros(3), rays)
    with pytest.raises(ValueError, match='radial parameter'):
        validate_saved_hits(hits, np.zeros(3), rays/np.linalg.norm(rays, axis=1, keepdims=True))


def test_changed_hit_and_contract_rejected():
    v, f, t, h = fixture()
    changed = deepcopy(h)
    changed[0]['point_world'][2] += .01
    with pytest.raises(ValueError, match='not on its original'):
        classify(v, f, t, changed)
    with pytest.raises(ValueError, match='no runtime tuning'):
        FootprintContract(numerical_plane_distance_m=.002)


def test_all_slots_counted_even_when_no_station_can_qualify():
    from audit_scannetpp_saved_source_footprints import coverage
    rows = [dict(corpus_ordinal=o, slot=i, scan=s, footprint=dict(status='unresolved'))
            for o in range(4) for i in range(96) for s in ('0', '1')]
    rows[0]['footprint']['diagnostic_planar_footprint'] = True
    result = coverage(rows)
    assert result['DSLR_slots'] == 384 and result['station_slots'] == 768
    assert result['planar_at_least_one_station'] == 1 and result['planar_both_stations'] == 0
    assert result['physical_accepted_pixels'] == 0
    with pytest.raises(ValueError, match='768 distinct'):
        coverage(rows[:-1]+[rows[0]])


def test_global_incidence_chunking_does_not_change_topology():
    v, f, _, _ = fixture()
    assert extract_topology(v, f, {0, 1}, chunk=1) == extract_topology(v, f, {0, 1}, chunk=64)
