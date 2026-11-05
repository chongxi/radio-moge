from types import SimpleNamespace

import numpy as np
import pytest

from probe_scannetpp_v2_source_pilot import (
    ORDINALS, nearest_world_hits, summarize, target_slots, validate_slots, visibility,
)
from radio_moge.mesh_clip import MeshClipContract


class Plane:
    def __init__(self, depths):
        self.depths = depths

    def intersects_location(self, origins, rays, multiple_hits):
        assert not multiple_hits
        t = (np.asarray(self.depths)[:, None]-origins[:, 2])/rays[:, 2]
        t[t < 0] = np.inf
        face = t.argmin(axis=0)
        best = t[face, np.arange(len(rays))]
        ids = np.flatnonzero(np.isfinite(best))
        return origins[ids]+best[ids, None]*rays[ids], ids, face[ids]


def comparison(z=2, hit=True, boundary=False):
    return dict(intersects_exact_triangle=hit, triangle_range_m=z, boundary_hit=boundary)


def test_unshifted_visibility_keeps_subcentimeter_occluder():
    first = nearest_world_hits(Plane([.005, 2.]), np.eye(4), [[0., 0., 1.]])[0]
    assert first['radial_range_m'] == .005
    assert visibility(comparison(), first, 1) == 'occluded_by_earlier_mesh_surface'


def test_full_published_matrix_preserves_scanner_radial_parameter():
    pose = np.eye(4)
    pose[2, 2] = 1.0000006
    local = np.array([[0., 0., 1.]])
    first = nearest_world_hits(Plane([2.]), pose, local)[0]
    assert first['radial_range_m'] == pytest.approx(2/pose[2, 2], abs=1e-14)
    assert first['radial_range_m'] != 2.


@pytest.mark.parametrize('value,first,face,expected', [
    (comparison(hit=False), dict(radial_range_m=2, face_id=0), 0, 'target_triangle_missed'),
    (comparison(), None, 0, 'whole_mesh_hit_missing'),
    (comparison(), dict(radial_range_m=3, face_id=0), 0, 'target_first_hit_disagrees'),
    (comparison(), dict(radial_range_m=2, face_id=1), 0, 'coincident_different_face_ambiguous'),
    (comparison(boundary=True), dict(radial_range_m=2, face_id=0), 0, 'target_triangle_boundary'),
    (comparison(), dict(radial_range_m=2, face_id=0), 0, 'same_target_is_first_mesh_surface'),
])
def test_visibility_states_do_not_grant_unmatched_targets(value, first, face, expected):
    assert visibility(value, first, face) == expected


def test_clipped_target_capture_does_not_skip_near_guard():
    case = dict(grid=[1, 1], intrinsics=[[1., 0, .5], [0, 1., .5], [0, 0, 1.]], c2w=np.eye(4))
    mesh = SimpleNamespace(vertices=np.array([[-1., -1., 2], [1., -1., 2], [0., 1., 2]]),
                           faces=np.array([[0, 1, 2]]))
    with pytest.raises(ValueError, match='exactly reproduced'):
        target_slots(mesh, Plane([.05001, 2]), case, [[0, 0]],
                     np.array([[2]], np.float32), np.array([[True]]), MeshClipContract())
    result = target_slots(mesh, Plane([2]), case, [[0, 0]],
                          np.array([[2]], np.float32), np.array([[True]]), MeshClipContract())
    assert result[0]['face_id'] == 0 and result[0]['point_world'] == [0., 0., 2.]


def test_all384_slots_preserved_and_replacement_rejected():
    rows = [dict(corpus_ordinal=o, group=f'scene:{o}', role='ordinary_control', grid=[80, 120],
                 pixel_yx=[[5+y*10, 5+x*10] for y in range(8) for x in range(12)]) for o in ORDINALS]
    validate_slots(rows)
    rows[0]['pixel_yx'][-1] = rows[0]['pixel_yx'][0]
    with pytest.raises(ValueError, match='ninety-six'):
        validate_slots(rows)


def test_missing_slots_and_dependent_scanner_pixels_remain_counted():
    records = [dict(status='DSLR_numerically_unsupported') for _ in range(90)]
    records += [dict(status='same_target_is_first_mesh_surface', scanner_pixel_yx=[1, 2],
                     center_diagnostic_comparable=True, center_signed_residual_m=.01,
                     all_nine_diagnostic_comparable=False) for _ in range(6)]
    result = summarize(records)
    assert result['slots'] == 96 and result['center_diagnostic_comparable'] == 6
    assert result['unique_matched_scanner_pixels'] == 1
    assert not result['observations_are_independent']
    records[-1]['scanner_identity'] = ['other-scene', '0']
    assert summarize(records)['unique_matched_scanner_pixels'] == 2
