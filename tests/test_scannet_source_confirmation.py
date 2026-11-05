from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

import run_scannetpp_source_confirmation as runner
from probe_scannetpp_triangle_scan_support import directions
from radio_moge.corpus_admission import file_ref
from radio_moge.mesh_clip import MeshClipContract
from radio_moge.scanner_support import ScannerSupportContract


def scope():
    rows, originals = [], []
    for i in range(16):
        pose = np.eye(4)
        peer = pose.copy()
        peer[0, 3] = .1
        chosen = dict(corpus_ordinal=i, group=f'scannetpp:s{i}', role='confirmation',
            image=dict(height=427, width=640, scene=f's{i}', dataset='scannetpp_dslr', split='train'))
        originals.append(deepcopy(chosen))
        rows.append(dict(corpus_ordinal=i, scene=f's{i}', role='confirmation',
            original_selection=chosen, grid=[427, 640],
            pixel_yx=[[int((y+.5)*427/8), int((x+.5)*640/12)] for y in range(8) for x in range(12)],
            scanners=[dict(scan=str(n), pose=p.tolist(), inputs=dict.fromkeys(('range', 'mask', 'azim', 'elev'), {}))
                      for n, p in enumerate((pose, peer))]))
    return rows, originals


def test_exact_confirmation_identity_scope():
    rows, original = scope()
    runner.validate_scope(rows, original)


@pytest.mark.parametrize('fault', ['drop', 'duplicate', 'retag', 'shiftpixel', 'swapscan', 'samepose', 'samegroup', 'panorama'])
def test_no_substitution_or_missing_slots_in_confirmation(fault):
    rows, original = scope()
    if fault == 'drop':
        rows.pop()
    elif fault == 'duplicate':
        rows[1] = deepcopy(rows[0])
    elif fault == 'retag':
        rows[0]['role'] = 'ordinary_control'
    elif fault == 'shiftpixel':
        rows[0]['pixel_yx'][0][1] += 1
    elif fault == 'swapscan':
        rows[0]['scanners'][1]['scan'] = '2'
    elif fault == 'samepose':
        rows[0]['scanners'][1]['pose'] = rows[0]['scanners'][0]['pose']
    elif fault == 'samegroup':
        original.append(dict(group=original[0]['group'], role='ordinary_control'))
    else:
        rows[0]['original_selection']['image']['dataset'] = 'panocam'
    with pytest.raises(ValueError):
        runner.validate_scope(rows, original)


def test_stale_actual_fixture_cannot_open_confirmation(monkeypatch):
    monkeypatch.setattr(runner, 'implementation', lambda: {'real_current_code': 'abc'})
    fixture = dict(stage='current_confirmation_fixtures_passed', passed=True,
        implementation=runner.implementation(), policy=asdict(ScannerSupportContract()),
        clip_contract=MeshClipContract().as_dict(), confirmation_images_read=0)
    runner.validate_fixture(fixture)
    fixture['implementation'] = {'old_code': 'abc'}
    with pytest.raises(ValueError, match='Fresh actual'):
        runner.validate_fixture(fixture)


def point_rows(rows):
    return [dict(corpus_ordinal=r['corpus_ordinal'], slot=s, pixel_yx=p,
        decision=dict(conditional_sample_candidate=False, reasons=['missing']),
        station_evidence=[dict(corpus_ordinal=r['corpus_ordinal'], slot=s, pixel_yx=p, scan=scan)
                          for scan in ('0', '1')]) for r in rows for s, p in enumerate(r['pixel_yx'])]


def test_missing_evidence_does_not_shrink_coverage_denominators():
    rows, _ = scope()
    points = point_rows(rows)
    summary = runner.coverage(points, rows)
    assert summary['DSLR_slots'] == 1536 and summary['station_slots'] == 3072
    assert summary['conditional_sample_candidates'] == 0
    assert summary['point_reason_counts'] == {'missing': 1536}
    assert not summary['dense_source_mask_constructed']
    points.pop()
    with pytest.raises(ValueError, match='1536'):
        runner.coverage(points, rows)


@pytest.mark.parametrize('fault', ['duplicate', 'replace_station', 'drop_station', 'wrong_pixel'])
def test_aggregate_scope_rejects_dropped_or_replaced_data(fault):
    rows, _ = scope()
    points = point_rows(rows)
    if fault == 'duplicate':
        points[1] = deepcopy(points[0])
    elif fault == 'replace_station':
        points[0]['station_evidence'][1]['scan'] = '2'
    elif fault == 'drop_station':
        points[0]['station_evidence'].pop()
    else:
        points[0]['station_evidence'][1]['pixel_yx'] = [0, 0]
    with pytest.raises(ValueError):
        runner.coverage(points, rows)


class Plane:
    def intersects_location(self, origins, rays, multiple_hits):
        assert not multiple_hits
        t = (2-origins[:, 2])/rays[:, 2]
        ids = np.flatnonzero(t > 0)
        return origins[ids]+t[ids, None]*rays[ids], ids, np.zeros(len(ids), np.int64)


def source_fixture(tmp_path, conflict=False):
    ray = directions(1000, 785)
    target = ray*2/ray[2]
    camera = np.eye(4)
    camera[:2, 3] = target[:2]
    mesh = SimpleNamespace(vertices=target+np.array([[-.1, -.1, 0], [.1, -.1, 0], [0., .1, 0]]),
                           faces=np.array([[0, 1, 2]]))
    poses = [np.eye(4), np.eye(4)]
    poses[1][0, 3] = .0001
    poses[1][2, 2] = 1.0000006
    pose_path = tmp_path/'poses.json'
    pose_path.write_text(json.dumps([p.tolist() for p in poses]))
    scans = []
    az, pol = np.meshgrid(np.arange(998, 1003, dtype=np.uint16), np.arange(783, 788, dtype=np.uint16))
    for i, pose in enumerate(poses):
        d = directions(az, pol)
        ranges = np.rint(2000/(d[..., 2]*pose[2, 2])).astype(np.uint16)
        if conflict and i:
            ranges += 100
        refs = {}
        for key, a in dict(range=ranges, mask=np.full((5, 5), 255, np.uint8), azim=az, elev=pol).items():
            p = tmp_path/f'{i}-{key}.png'
            Image.fromarray(a).save(p)
            refs[key] = file_ref(p)
        scans.append(dict(scan=str(i), pose=pose.tolist(), inputs=refs))
    binding = dict(corpus_ordinal=1, scene='s', pixel_yx=[[0, 0]],
                   scanner_poses=file_ref(pose_path), scanners=scans)
    record = dict(grid=[1, 1], intrinsics=[[1., 0, .5], [0, 1., .5], [0, 0, 1.]],
                  c2w=camera.tolist(), source_refs=[])
    arrays = dict(final_depth=np.array([[2.]], np.float32), final_support=np.ones((1, 1), bool))
    return binding, record, arrays, mesh


@pytest.mark.parametrize('excluded,conflict', [(False, False), (True, False), (False, True)])
def test_actual_source_helper_composition_preserves_peer_gap_and_exclusion(tmp_path, excluded, conflict):
    binding, record, arrays, mesh = source_fixture(tmp_path, conflict)
    output = tmp_path/'output'
    output.mkdir()
    points, refs = runner.source_case(binding, record, arrays, mesh, Plane(), {1} if excluded else set(), tmp_path, output)
    assert len(points) == 1 and len(points[0]['station_evidence']) == 2
    assert points[0]['decision']['conditional_sample_candidate'] == (not excluded and not conflict), points
    assert len(refs['stations']) == 2
    assert not points[0]['decision']['training_admission']
    if conflict:
        peer = points[0]['station_evidence'][1]
        assert min(p['signed_range_residual_m'] for p in peer['angle_matches']['pixels']) > .09


def test_all_unsupported_source_case_retains_slots_without_mesh_queries(tmp_path):
    binding, record, arrays, mesh = source_fixture(tmp_path)
    binding['pixel_yx'] = [[y, x] for y in range(8) for x in range(12)]
    record['grid'] = [8, 12]
    arrays = dict(final_depth=np.zeros((8, 12), np.float32), final_support=np.zeros((8, 12), bool))
    class NoQueries:
        def intersects_location(self, *a, **kw):
            raise AssertionError('Unsupported source slots must not query mesh')
    output = tmp_path/'out'
    output.mkdir()
    points, _ = runner.source_case(binding, record, arrays, mesh, NoQueries(), set(), tmp_path, output)
    assert len(points) == 96
    assert all(not p['decision']['conditional_sample_candidate'] for p in points)
    assert all(len(p['station_evidence']) == 2 for p in points)
