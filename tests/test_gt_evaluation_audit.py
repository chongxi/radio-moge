import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_gt_evaluations import LASER_REQUIRED, audit_laser


def evidence():
    expected = [dict(id='curved', scene='a', split='val', planes=[]),
                dict(id='wall', scene='b', split='val', planes=[[0, 0, 1, -2]])]
    rows = [dict(id=r['id'], scene=r['scene'], **dict.fromkeys(LASER_REQUIRED, .2))
            for r in expected]
    rows[0].update(gt_plane_fraction=0, gt_plane=0, rmse=.8)
    rows[1].update(plane_rmse_m=.03, rmse=.2)
    metrics = {k: math.fsum(r[k] for r in rows if k in r) / sum(k in r for r in rows)
               for k in LASER_REQUIRED | {'plane_rmse_m'}}
    metrics.update(images=2, plane_images=1)
    log = {'gt_laser_known_camera/' + k: v for k, v in metrics.items()}
    return rows, metrics, log, expected


def test_all_geometry_images_contribute_while_plane_score_uses_only_planes():
    args = evidence()
    proof = audit_laser(*args)
    assert proof['images'] == 2 and proof['scenes'] == 2
    assert args[1]['rmse'] == .5 and args[1]['plane_rmse_m'] == .03


@pytest.mark.parametrize('corruption', ['aggregate', 'order', 'missing_geometry', 'nan',
                                      'wrong_split', 'missing_plane', 'logged_metric'])
def test_rejects_incomplete_or_inconsistent_gt_evaluation(corruption):
    rows, metrics, log, expected = evidence()
    if corruption == 'aggregate':
        metrics['rmse'] = .2
    elif corruption == 'order':
        rows.reverse()
    elif corruption == 'missing_geometry':
        for row in rows:
            del row['gt_points']
        del metrics['gt_points']
    elif corruption == 'nan':
        rows[0]['gt_normal'] = float('nan')
    elif corruption == 'wrong_split':
        expected[0]['split'] = 'train'
    elif corruption == 'missing_plane':
        del rows[1]['plane_rmse_m']
    else:
        log['gt_laser_known_camera/rmse'] = 0
    with pytest.raises(AssertionError):
        audit_laser(rows, metrics, log, expected)
