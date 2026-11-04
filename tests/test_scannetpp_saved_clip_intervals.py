from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from audit_scannetpp_saved_clip_intervals import audit, check_ray, interval_hit
from probe_scannetpp_native_near_surfaces import analytic_hits
from radio_moge.corpus_admission import file_ref


def intersections(depths, camera_ray=(0., 0., 1.)):
    vertices = np.array([[[-50, -50, z], [50, -50, z], [0, 50, z]] for z in depths]).reshape(-1, 3)
    faces = np.arange(len(vertices)).reshape(-1, 3)
    result = analytic_hits(vertices, faces, [0, 0, 0], camera_ray)
    for hit in result['first_hits']:
        hit.update(camera_z_m=hit['t'], point_world=(hit['t']*np.asarray(camera_ray)).tolist())
    return result


def test_near_clip_exposes_rear_surface_instead_of_discarding_original_pixel():
    result = interval_hit(intersections([.02, 2.]))
    assert result['outcome'] == 'different_first_hit'
    assert result['hit']['camera_z_m'] == 2.
    # Independent shifted-origin ray agrees for this off-axis plane geometry.
    vertices = np.array([[-5, -5, 2.], [5, -5, 2.], [0, 5, 2.]])
    d = np.array([.5, .25, 1.])
    shifted = analytic_hits(vertices, np.array([[0, 1, 2]]), .05*d, d)
    assert shifted['first_hits'][0]['t']+.05 == pytest.approx(2.)


@pytest.mark.parametrize('depths,outcome', [
    ([.02], 'no_intersection_inside_clip_interval'),
    ([21., 25.], 'no_intersection_inside_clip_interval'),
    ([.08, 2.], 'unchanged_first_hit'),
    ([], 'no_positive_intersection'),
    ([.05, 2.], 'clip_boundary_ambiguous'),
    ([20.], 'clip_boundary_ambiguous'),
])
def test_missing_far_and_boundary_outcomes(depths, outcome):
    assert interval_hit(intersections(depths))['outcome'] == outcome


def test_more_than_eight_saved_hits_cannot_be_treated_as_complete():
    saved = intersections([.001*i for i in range(1, 10)]+[2.])
    assert len(saved['first_hits']) == 8 and saved['positive_triangle_intersections'] == 10
    with pytest.raises(ValueError, match='prefix is insufficient'):
        interval_hit(saved)


def test_oblique_translated_rotated_ray_uses_camera_z_not_radial_range():
    camera = np.array([2., 0., 1.])
    angle = .7
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = [50, -10, 3]
    world = camera @ rotation.T
    raw = intersections([.03, 2.], camera)
    for hit in raw['first_hits']:
        hit['point_world'] = (pose[:3, 3]+hit['t']*world).tolist()
    ray = dict(camera_ray=camera.tolist(), ray_direction_world=world.tolist(),
               ray_origin_world=pose[:3, 3].tolist(), analytical=raw)
    check_ray(dict(c2w=pose.tolist()), ray)
    assert np.linalg.norm(np.array(raw['first_hits'][0]['point_world'])-pose[:3, 3]) > .05
    assert interval_hit(raw)['hit']['camera_z_m'] == 2.
    bad = deepcopy(ray)
    bad['ray_direction_world'] = (world/np.linalg.norm(world)).tolist()
    with pytest.raises(ValueError, match='camera rays differ'):
        check_ray(dict(c2w=pose.tolist()), bad)


def fixture(tmp_path):
    case = dict(corpus_ordinal=7, group='scannetpp:test', c2w=np.eye(4).tolist(),
        image={'path': '/unread/image.jpg', 'sha256': 'a'*64}, source_refs=[],
        rays=[dict(pixel_yx=[1, 2], selection=['existing_selection'], cached_valid=True,
                   cached_z_m=.02, ray_origin_world=[0, 0, 0], ray_direction_world=[0, 0, 1],
                   camera_ray=[0, 0, 1], analytical=intersections([.02, 2.]))])
    target = tmp_path/'7.json'
    target.write_text(json.dumps(case))
    case_ref = file_ref(target)
    case_ref['path'] = '/original/7.json'
    source = tmp_path/'source.json'
    producer = Path(__file__).resolve().parents[1]/'scripts/probe_scannetpp_native_near_surfaces.py'
    source.write_text(json.dumps(dict(stage='bounded_raw_mesh_ray_diagnosis_complete',
        training_admission=False, optimizer_updates=0, code=file_ref(producer), cases=[case_ref])))
    return source


def test_bound_mirror_reads_metadata_only_and_does_not_publish_labels(tmp_path):
    source = fixture(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    report = audit(source, tmp_path/'out', tmp_path)
    assert report['rays'] == report['below_near_exposes_farther_hit'] == 1
    assert report['supported_outcomes'] == {'different_first_hit': 1}
    assert not report['official_renderer_executed'] and not report['geometry_edited']
    assert not report['source_quality_accepted'] and not report['meshes_reread']
    for p, value in before.items():
        assert p.read_bytes() == value


def test_changed_mirror_fails_before_publication(tmp_path):
    source = fixture(tmp_path)
    target = tmp_path/'7.json'
    target.write_text(target.read_text()+' ')
    with pytest.raises(ValueError, match='content changed'):
        audit(source, tmp_path/'out', tmp_path)
    assert not (tmp_path/'out').exists()
