import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image
import pytest


scripts = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('native_probe', scripts / 'probe_native_depth.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
sys.path.remove(str(scripts))


def test_plane_geometry_detects_wrong_range_convention(tmp_path):
    archive = tmp_path / 'structured3d-perspective-v1-20261022/archives/sample.zip'
    archive.parent.mkdir(parents=True)
    prefix = 'Structured3D/scene_00000/view'
    buffer = io.BytesIO()
    Image.fromarray(np.full((64, 96), 4000, np.uint16)).save(buffer, format='PNG')
    layout = dict(junctions=[dict(coordinate=[x, y]) for x, y in
                             [(0, 0), (95, 0), (95, 63), (0, 63)]],
                  planes=[dict(normal=[0, 0, -1], offset=4000, visible_mask=[[0, 1, 2, 3]])])
    with zipfile.ZipFile(archive, 'w') as file:
        for name in ('depth.png', 'normal.png'):
            file.writestr(prefix + '/' + name, buffer.getvalue())
        file.writestr(prefix + '/camera_pose.txt', '0 0 0 0 0 1 0 1 0 .7 .5 1')
        file.writestr(prefix + '/layout.json', json.dumps(layout))
    row = dict(source_member=prefix + '/rgb_rawlight.png', source_archive=archive.name,
               scene='00000', room='0', configuration='empty', view='0')
    evidence = probe.Evidence(tmp_path / 'evidence')
    result = probe.structured3d_probe(evidence, tmp_path, [row])[0]
    assert result['plane_residual_m']['z_pixelhalf']['max'] == 0
    assert result['plane_residual_m']['range_pixelhalf']['p50'] > .1
    assert result['training_admission'] is False
    with pytest.raises(FileExistsError):
        probe.Evidence(tmp_path / 'evidence')


def test_quantiles_do_not_fabricate_empty_support():
    assert probe.quantiles(np.array([np.nan, np.inf])) is None
    assert probe.quantiles(np.array([1., np.nan, 3.]))['p50'] == 2
