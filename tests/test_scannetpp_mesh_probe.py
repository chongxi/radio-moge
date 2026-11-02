import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

trimesh = pytest.importorskip('trimesh')
pytest.importorskip('embreex')
RayMeshIntersector = pytest.importorskip('trimesh.ray.ray_pyembree').RayMeshIntersector


scripts = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('mesh_probe', scripts / 'probe_scannetpp_mesh.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
sys.path.remove(str(scripts))


def test_renderer_returns_first_surface_in_metric_coordinates():
    vertices = np.array([[-10, -10, 3], [10, -10, 3], [10, 10, 3], [-10, 10, 3]], float)
    mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [0, 2, 3]], process=False)
    ray = RayMeshIntersector(mesh)
    origins = np.zeros((3, 3))
    directions = np.array([[0, 0, 1], [.2, 0, 1], [0, 1, 0]], float)
    points, indices, _ = ray.intersects_location(origins, directions, multiple_hits=False)
    assert set(indices) == {0, 1}
    np.testing.assert_allclose(points[:, 2], 3)
    np.testing.assert_allclose(points[:, 0], directions[indices, 0] * 3, atol=1e-6)


def test_colmap_parser_skips_observation_rows(tmp_path):
    path = tmp_path / 'images.txt'
    path.write_text('# header\n1 1 0 0 0 1 2 3 1 other.JPG\n0 0 -1\n'
                    '2 1 0 0 0 4 5 6 1 chosen.JPG\n\n')
    c2w = probe.colmap_pose(path, 'chosen.JPG')
    np.testing.assert_array_equal(c2w[:3, 3], [-4, -5, -6])
