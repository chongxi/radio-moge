import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from render_structural_room_samples import (box_mesh, camera_requests, cylinder_mesh,
                                            mesh_for, room, structures)


@pytest.mark.parametrize('mesh', [box_mesh((.6, 4.2, .6)),
                                 cylinder_mesh(.3, 4.2, 'y'), cylinder_mesh(.2, 3.6, 'x')])
def test_closed_primitives_have_outward_triangles_and_normals(mesh):
    vertices = np.array([list(v.values()) for v in mesh['vertices']])
    normals = np.array([list(v.values()) for v in mesh['normals']])
    faces = vertices[np.array(mesh['triangles']).reshape(-1, 3)]
    cross = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    assert np.all(np.linalg.norm(cross, axis=1) > 0)
    assert np.all(np.einsum('ij,ij->i', cross, faces.mean(axis=1)) > 0)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-14)
    # Geometric, not duplicated-index, edges must meet exactly twice, with opposite direction.
    edges = {}
    for face in faces:
        for a, b in zip(face, np.roll(face, -1, axis=0)):
            a, b = tuple(np.round(a, 9)), tuple(np.round(b, 9))
            key = tuple(sorted((a, b)))
            edges.setdefault(key, []).append((a, b))
    assert all(len(v) == 2 and v[0] == v[1][::-1] for v in edges.values())


@pytest.mark.parametrize('variant', range(3))
def test_closed_room_and_attached_structures_are_well_inside_scene(variant):
    house = room(variant)
    assert not house['windows'] and not house['doors'] and not house['objects']
    assert len(house['walls']) == 4
    assert house['proceduralParameters']['ceilingMaterial']['name']
    assert house['proceduralParameters']['skyboxId'] == 'SkyGarden'
    pieces = structures(variant)
    assert {'square_column', 'round_column', 'rectangular_beam',
            'horizontal_round_member'} <= {x['name'] for x in pieces}
    for piece in pieces:
        vertices = np.array([list(v.values()) for v in mesh_for(piece)['vertices']])
        vertices += piece['center']
        assert np.all(vertices >= (0, -1e-9, 0))
        assert np.all(vertices <= (10, 4.2 + 1e-9, 14))
        assert vertices[:, 2].min() - 2 > 8
    beam = next(x for x in pieces if x['name'] == 'rectangular_beam')
    rail = next(x for x in pieces if x['name'] == 'horizontal_round_member')
    assert beam['center'][0] - beam['size'][0] / 2 < 3.2
    assert beam['center'][0] + beam['size'][0] / 2 > 6.8
    assert rail['center'][0] - rail['length'] / 2 == 3.2
    assert rail['center'][0] + rail['length'] / 2 == 6.8


def test_projection_options_are_explicit_and_camera_faces_far_wall():
    options = camera_requests()
    assert [v['orthographic'] for _, v in options] == [False, True]
    assert 'comparison_only' in options[1][0]
    for _, request in options:
        assert request['position'] == dict(x=5., y=2.1, z=2.)
        assert request['rotation'] == dict(x=0., y=0., z=0.)
        assert request['skyboxColor'] == '#ff00ff'
        assert 14 - request['position']['z'] == 12
    assert options[0][1]['fieldOfView'] == 26


def test_invalid_primitive_dimensions_are_rejected():
    with pytest.raises(ValueError):
        box_mesh((1, 0, 1))
    with pytest.raises(ValueError):
        cylinder_mesh(.2, 1, 'z')
