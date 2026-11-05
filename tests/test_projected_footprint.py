from copy import deepcopy

import numpy as np
import pytest

from radio_moge.projected_footprint import (
    ProjectedFootprintContract, area, clip_polygon, projected_footprint, separated_triangles,
    vertex_cones_separated, vertex_links_are_manifold,
)
from radio_moge.source_footprint import extract_topology, sampled_range_evidence


def intersections(v, f, rays, pose):
    """Small independent analytic fixture: solve barycentrics for every face."""
    hits = []
    for local in rays:
        ray = pose[:3, :3] @ local
        candidates = []
        for face, triangle in enumerate(v[f]):
            a, b, c = triangle
            system = np.column_stack([b-a, c-a, -ray])
            if abs(np.linalg.det(system)) < 1e-14:
                continue
            u, w, t = np.linalg.solve(system, pose[:3, 3]-a)
            if min(u, w, 1-u-w) >= -1e-12 and t > 0:
                candidates.append(dict(face_id=face, radial_range_m=float(t),
                                       point_world=(pose[:3, 3]+t*ray).tolist()))
        hits.append(min(candidates, key=lambda x: x['radial_range_m']) if candidates else None)
    return hits


def fixture(bend=0.):
    v = np.array([[x, y, 2+bend*abs(x)] for x in (-1., 0., 1.) for y in (-1., 1.)])
    f = np.array([[0, 2, 1], [2, 3, 1], [2, 4, 3], [4, 5, 3]])
    rays = np.array([[x, y, 1.] for x in (-.2, 0., .2) for y in (-.19, 0., .19)])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    pose = np.eye(4)
    hits = intersections(v, f, rays, pose)
    t = hits[4]
    target = dict(face_id=t['face_id'], point_world=t['point_world'],
                  vertices_world=v[f[t['face_id']]].tolist())
    return v, f, rays, pose, hits, target


def check(v, f, rays, pose, hits, target):
    topology = extract_topology(v, f, {target['face_id'], *(h['face_id'] for h in hits if h)})
    return projected_footprint(target, hits, pose, rays, topology)


@pytest.mark.parametrize('bend', [0., .08, .8, 2.])
def test_visible_planar_bent_and_sharp_crease_can_be_C0_but_not_certified_smooth(bend):
    result = check(*fixture(bend))
    assert result['sampled_C0_correspondence'], result
    assert result['selected_faces'] == 4
    assert not result['smoothness_certified'] and not result['physical_accuracy_accepted']
    assert not result['whole_mesh_visibility_between_samples_proven']
    if bend:
        assert result['maximum_view_oriented_dihedral_degrees'] > 0
    if bend == 2.:
        assert result['maximum_view_oriented_dihedral_degrees'] > 120


def test_piecewise_surface_does_not_clear_source_gap():
    args = fixture(.8)
    assert check(*args)['sampled_C0_correspondence']
    evidence = sampled_range_evidence(args[4], dict(values=dict(range=[[3500]*3]*3, mask=[[255]*3]*3)))
    assert evidence['signed_interval_separation_m'] > 1
    assert not evidence['physical_accuracy_accepted']


def test_rounded_published_pose_preserves_local_radial_ranges():
    v, f, rays, pose, _, target = fixture(.08)
    pose[2, 2] = 1.0000006
    hits = intersections(v, f, rays, pose)
    target.update(face_id=hits[4]['face_id'], point_world=hits[4]['point_world'],
                  vertices_world=v[f[hits[4]['face_id']]].tolist())
    assert check(v, f, rays, pose, hits, target)['sampled_C0_correspondence']
    changed = deepcopy(hits)
    for hit in changed:
        hit['radial_range_m'] *= 1.0000006
    with pytest.raises(ValueError, match='radial parameter'):
        check(v, f, rays, pose, changed, target)


def test_detached_coordinates_are_not_welded():
    v, f, rays, pose, hits, target = fixture(.08)
    v = np.concatenate([v, v[[2, 3]]])
    f[2:] = np.where(f[2:] == 2, 6, np.where(f[2:] == 3, 7, f[2:]))
    assert check(v, f, rays, pose, hits, target)['status'] == 'connectivity_unproven_with_retained_faces'


def test_nonmanifold_global_edge_not_hidden_by_selected_faces():
    v, f, rays, pose, hits, target = fixture()
    v = np.concatenate([v, [[0., 0., 3.]]])
    f = np.concatenate([f, [[2, 3, 6]]])
    assert check(v, f, rays, pose, hits, target)['status'] == 'nonmanifold_edge_unresolved'


def test_vertex_links_reject_bow_tie_and_duplicate_faces():
    assert vertex_links_are_manifold([[0, 1, 2], [0, 2, 3]])
    assert not vertex_links_are_manifold([[0, 1, 2], [0, 3, 4]])
    assert not vertex_links_are_manifold([[0, 1, 2], [2, 1, 0]])


def test_connected_fold_overlaps_in_scanner_projection():
    v = np.array([[-1., -1., 2.], [1., -1., 2.], [-1., 1., 2.], [-.5, -.5, 3.]])
    f = np.array([[0, 1, 2], [1, 3, 2]])
    rays = np.array([[x, y, 1.] for x in (-.25, -.2, -.15) for y in (-.25, -.2, -.15)])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    pose = np.eye(4)
    hits = intersections(v, f, rays, pose)
    target = dict(face_id=1, point_world=v[f[1]].mean(axis=0).tolist(), vertices_world=v[f[1]].tolist())
    result = check(v, f, rays, pose, hits, target)
    assert result['status'] == 'projected_fold_or_edge_uncertainty'


def test_missing_hit_and_missing_topology_explicit():
    args = list(fixture(.08))
    args[4][0] = None
    assert check(*args)['status'] == 'first_hit_missing'
    v, f, rays, pose, hits, target = fixture(.08)
    topology = extract_topology(v, f, [target['face_id']])
    assert projected_footprint(target, hits, pose, rays, topology)['status'] == 'topology_unavailable'


def test_target_outside_tested_angles_is_not_certified():
    v, f, rays, pose, hits, target = fixture()
    target['point_world'] = [-.8, .5, 2.]
    target['face_id'] = 0
    target['vertices_world'] = v[f[0]].tolist()
    assert check(v, f, rays, pose, hits, target)['status'] == 'target_outside_sampled_ray_hull'


def test_polygon_intersection_contact_and_overlap_areas():
    a = np.array([[0., 0.], [1., 0.], [0., 1.]])
    b = np.array([[1., 0.], [1., 1.], [0., 1.]])
    assert area(clip_polygon(a, b, 1e-14)) == pytest.approx(0, abs=1e-14)
    assert area(clip_polygon(a, a+.1, 1e-14)) == pytest.approx(.32, abs=1e-14)
    assert len(clip_polygon(a, b+3., 1e-14)) == 0


def test_runtime_threshold_tuning_rejected():
    with pytest.raises(ValueError, match='Frozen'):
        ProjectedFootprintContract(minimum_incidence_cos=.001)


def test_unshifted_subcentimeter_foreground_is_not_discarded():
    v, f, rays, pose, _, target = fixture()
    v = np.concatenate([v, [[-1., -1., .005], [1., -1., .005], [0., 1., .005]]])
    f = np.concatenate([f, [[6, 7, 8]]])
    hits = intersections(v, f, rays, pose)
    assert all(h['face_id'] == 4 for h in hits)
    assert check(v, f, rays, pose, hits, target)['status'] == 'connectivity_unproven_with_retained_faces'


def test_horizon_crossing_triangle_unresolved_without_trimming_it():
    v = np.array([[-1., -1., 2.], [1., -1., 2.], [0., 1., -.5]])
    f = np.array([[0, 1, 2]])
    rays = np.array([[x, y, 1.] for x in (-.05, 0., .05) for y in (-.45, -.4, -.35)])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    pose = np.eye(4)
    hits = intersections(v, f, rays, pose)
    target = dict(face_id=0, point_world=hits[4]['point_world'], vertices_world=v.tolist())
    assert check(v, f, rays, pose, hits, target)['status'] == 'chart_horizon_unresolved'


def test_grazing_patch_is_conditionally_excluded():
    v = np.array([[1., -1., 20.], [1., 1., 20.], [1., -1., 40.], [1., 1., 40.]])
    f = np.array([[0, 1, 2], [1, 3, 2]])
    rays = np.array([[1., y, z] for z in (29., 30., 31.) for y in (-.2, 0., .2)])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    pose = np.eye(4)
    hits = intersections(v, f, rays, pose)
    target = dict(face_id=hits[4]['face_id'], point_world=hits[4]['point_world'],
                  vertices_world=v[f[hits[4]['face_id']]].tolist())
    assert check(v, f, rays, pose, hits, target)['status'] == 'grazing_patch_unresolved'


@pytest.mark.parametrize('hole_halfwidth', [.2, 1e-10])
def test_sample_hits_do_not_hide_an_uncovered_hole_in_angular_hull(hole_halfwidth):
    h = hole_halfwidth
    xy = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.],
                   [-h, -h], [h, -h], [h, h], [-h, h]])
    v = np.column_stack([xy, np.full(8, 2.)])
    f = np.array([t for i in range(4) for t in
                  [(i, (i+1) % 4, 4+i), ((i+1) % 4, 4+(i+1) % 4, 4+i)]])
    samples = np.concatenate([v[f].mean(axis=1), v[f[:1]].mean(axis=1)])
    rays = samples/np.linalg.norm(samples, axis=1, keepdims=True)
    pose = np.eye(4)
    hits = intersections(v, f, rays, pose)
    target = dict(face_id=hits[4]['face_id'], point_world=hits[4]['point_world'],
                  vertices_world=v[f[hits[4]['face_id']]].tolist())
    result = check(v, f, rays, pose, hits, target)
    assert not result['sampled_C0_correspondence']
    assert result['status'] in ('exposed_patch_boundary_touches_ray_hull',
                                'shared_vertex_contact_unresolved', 'projected_triangle_degenerate')


def test_all_missing_station_slots_are_kept_in_coverage():
    from audit_scannetpp_projected_footprints import summarize
    rows = [dict(corpus_ordinal=i, scan=s, slot=k,
        projected_footprint=dict(sampled_C0_correspondence=False, status='missing'))
        for i in range(4) for s in ('0', '1') for k in range(96)]
    rows[0]['projected_footprint'].update(sampled_C0_correspondence=True, selected_faces=2)
    result = summarize(rows)
    assert result['DSLR_slots'] == 384 and result['station_slots'] == 768
    assert result['structural_observations'] == 1 and result['physical_accepted_pixels'] == 0
    assert result['structural_and_sampled_overlap_observations'] == 0
    with pytest.raises(ValueError, match='768 unique'):
        summarize(rows[:-1]+[rows[0]])


def test_narrow_overlap_and_uncertain_unindexed_contact_never_certified_by_small_area():
    a = np.array([[0., 0.], [1., 0.], [0., 1.]])
    b = np.array([[1., 0.], [1., 1.], [0., 1.]])
    eps = 1e-12
    assert not separated_triangles(a, b, eps)
    assert not separated_triangles(a, b-1e-14, eps)
    assert not separated_triangles(a, b+1e-14, eps)
    assert separated_triangles(a, b+.01, eps)
    c = np.array([[0., 0.], [-1., 0.], [0., -1.]])
    assert vertex_cones_separated(a, c, np.array([0., 0.]), eps)
    assert not vertex_cones_separated(a, a, np.array([0., 0.]), eps)
