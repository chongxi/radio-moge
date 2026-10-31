import importlib.util
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
spec = importlib.util.spec_from_file_location(
    'resolution_audit', Path(__file__).resolve().parents[1] / 'scripts/audit_resolution_evaluations.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    rows = [dict(image='a', dataset='nyuv2', postprocess_failed=0, **{'gt_teacher/abs_rel': .1}),
            dict(image='b', dataset='other', postprocess_failed=0)]
    metrics = {'images': 2, 'gt_teacher/abs_rel': .1, 'gt_teacher/abs_rel/count': 1,
               'dataset/nyuv2/images': 1, 'macro/gt_teacher/abs_rel': .1,
               'postprocess_failed': 0}
    expected = [('a', 'nyuv2'), ('b', 'other')]
    return rows, metrics, metrics.copy(), expected, '', [r.copy() for r in rows]


def test_saved_view_recomputation_and_identity():
    args = fixture()
    assert module.check_view(*args)['nyu_sensor_rows'] == 1
    args[0].reverse()
    with pytest.raises(AssertionError):
        module.check_view(*args)


@pytest.mark.parametrize('corruption', ['aggregate', 'frozen_teacher', 'failure', 'nonfinite'])
def test_audit_rejects_corrupted_evidence(corruption):
    args = fixture()
    if corruption == 'aggregate':
        args[1]['gt_teacher/abs_rel'] = .2
    elif corruption == 'frozen_teacher':
        args[5][0]['gt_teacher/abs_rel'] = .2
    elif corruption == 'failure':
        args[0][0]['postprocess_failed'] = 1
    else:
        args[0][0]['extra'] = float('nan')
    with pytest.raises(AssertionError):
        module.check_view(*args)


def test_explicit_sensor_identity_supports_lidar_without_relabeling_nyu():
    rows = [dict(image='lidar-frame', dataset='scannetpp_iphone_lidar', postprocess_failed=0,
                 **{'gt_teacher/abs_rel': .1, 'gt_student/abs_rel': .2})]
    expected = [('lidar-frame', 'scannetpp_iphone_lidar')]
    reference = [r.copy() for r in rows]
    metrics = {'images': 1, 'gt_teacher/abs_rel': .1, 'gt_student/abs_rel': .2}
    proof = module.check_view(rows, metrics, metrics, expected, '', reference,
                              expected_sensor_ids=expected)
    assert proof['sensor_rows'] == 1 and proof['nyu_sensor_rows'] == 0
    assert proof['sensor_source_counts'] == {'scannetpp_iphone_lidar': 1}
    with pytest.raises(AssertionError):
        module.check_view(rows, metrics, metrics, expected, '', reference)
    with pytest.raises(AssertionError):
        module.check_view(rows, metrics, metrics, expected, '', reference, expected_sensor_ids=[])
    del reference[0]['gt_student/abs_rel']
    with pytest.raises(AssertionError):
        module.check_view(rows, metrics, metrics, expected, '', reference,
                          expected_sensor_ids=expected)
