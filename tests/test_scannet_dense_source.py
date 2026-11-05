from copy import deepcopy

import numpy as np
from PIL import Image
import pytest

from probe_scannetpp_triangle_scan_support import directions, nearest_angle_pixel
from probe_scannetpp_v2_source_pilot import target_slots
from radio_moge.mesh_clip import MeshClipContract
from radio_moge.scanner_support import dense_image_support, paired_support
from radio_moge.source_footprint import extract_topology
from run_scannetpp_source_confirmation import source_case
from scannetpp_dense_source import AngleLookup, StationSession, TopologyIndex, merge_masks
from test_scannet_source_confirmation import Plane, source_fixture


@pytest.mark.parametrize('kind', ['random', 'all_ties', 'seam', 'thin', 'duplicates'])
def test_cached_lookup_matches_original_exactly_including_ties_and_eviction(kind):
    rng = np.random.default_rng(487)
    shape = (1, 181) if kind == 'thin' else (107, 131)
    az = rng.integers(0, 6284, shape, np.uint16)
    pol = rng.integers(0, 3142, shape, np.uint16)
    if kind == 'all_ties':
        az[:] = 1000
        pol[:] = 785
    if kind == 'seam':
        az[:, :65], az[:, 65:] = 0, 6283
    if kind == 'duplicates':
        az[::2], pol[::2] = 1000, 785
    lookup = AngleLookup(az, pol, window_limit=2)
    queries = list(rng.normal(size=(16, 3)))+[directions(az[0, 0], pol[0, 0])]
    for q in queries + queries[::-1]:
        original, cached = nearest_angle_pixel(az, pol, q), lookup(q)
        assert original[:2] == cached[:2]
        assert np.array_equal(original[2], cached[2])
        assert original[3] == cached[3]
        assert len(lookup.windows) <= 2


def test_global_topology_exact_nonmanifold_and_unselected_faces():
    vertices = np.array([[0., 0, 1], [1, 0, 1], [0, 1, 1], [0, -1, 1], [0, 0, 2]])
    faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])
    index = TopologyIndex(vertices, faces, chunk=1)
    for ids in ([0], [0, 1], [2, 0, 1]):
        assert index.selected(ids) == extract_topology(vertices, faces, ids, chunk=1)
    assert index.selected([0, 1])['edge_incidence']['1'] == 3
    with pytest.raises(ValueError, match='memory bound'):
        TopologyIndex(vertices, faces, maximum_faces=2)


@pytest.mark.parametrize('excluded,conflict', [(False, False), (True, False), (False, True)])
def test_optimized_station_equals_full_legacy_decisions_and_source_evidence(tmp_path, excluded, conflict):
    binding, record, arrays, mesh = source_fixture(tmp_path, conflict)
    output = tmp_path/'legacy'
    output.mkdir()
    baseline, _ = source_case(binding, record, arrays, mesh, Plane(), {1} if excluded else set(), tmp_path, output)
    targets = target_slots(mesh, Plane(), record, binding['pixel_yx'], arrays['final_depth'],
                           arrays['final_support'], MeshClipContract())
    topology = TopologyIndex(mesh.vertices, mesh.faces)
    optimized = []
    for i, scanner in enumerate(binding['scanners']):
        raw = {k: np.asarray(Image.open(ref['path'])).copy() for k, ref in scanner['inputs'].items()}
        session = StationSession(raw, scanner['pose'], Plane(), topology, hit_cache_limit=1)
        row = session.evaluate(targets, 1, 's', str(i))[0]
        old = baseline[0]['station_evidence'][i]
        for key in ('projected_footprint', 'angle_matches', 'exact_target_visibility'):
            assert row[key] == old[key]
        # Repeated direction reuse must not cache a per-target final decision.
        again = session.evaluate(targets, 1, 's', str(i))[0]
        assert row == again and session.new_nine_ray_queries == 1
        moved = deepcopy(targets)
        moved[0]['point_world'][2] += .1
        with pytest.raises(ValueError, match='not on its retained triangle'):
            session.evaluate(moved, 1, 's', str(i))
        optimized.append(row)
    assert paired_support(optimized, ['0', '1'], True, excluded) == baseline[0]['decision']


def test_masks_cannot_extrapolate_missing_station_or_sparse_coverage():
    numeric = np.ones((10, 10), bool)
    masks = [numeric.copy(), numeric.copy()]
    evaluated = [numeric.copy(), numeric.copy()]
    masks[1][0] = False
    evaluated[1][0] = False
    result, complete = merge_masks(numeric, masks, evaluated, False)
    assert result.sum() == 90 and not complete.all()
    assert not dense_image_support(numeric, result, bool(complete.all()))['conditional_dense_coverage_sufficient']
    assert not merge_masks(numeric, masks, evaluated, True)[0].any()
    masks[1][0, 0] = True
    with pytest.raises(ValueError, match='must be evaluated'):
        merge_masks(numeric, masks, evaluated, False)


def test_all_unsupported_slots_keep_both_denominators_and_issue_no_rays(tmp_path):
    binding, record, arrays, mesh = source_fixture(tmp_path)
    class NoQueries:
        def intersects_location(self, *args, **kw):
            raise AssertionError('No mesh queries for numerical missing slots')
    targets = [dict(slot=i, pixel_yx=[0, i], numerical_support=False, saved_z_m=0.) for i in range(64)]
    scan = binding['scanners'][0]
    raw = {k: np.asarray(Image.open(ref['path'])).copy() for k, ref in scan['inputs'].items()}
    session = StationSession(raw, scan['pose'], NoQueries(), TopologyIndex(mesh.vertices, mesh.faces))
    rows = session.evaluate(targets, 1, 's', '0')
    assert len(rows) == 64
    assert all(not r['decision']['conditional_station_support'] for r in rows)
    assert session.new_nine_ray_queries == 0
