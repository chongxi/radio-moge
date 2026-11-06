import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('lidar_audit',Path(__file__).parents[1]/'scripts/audit_lidar_metric.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_metric_units_and_support_are_not_prediction_dependent():
    p=np.array([1.,2.,np.nan,-1.,0.,9.,np.inf,1.25])
    t=np.array([1.,1.,2.,2.,0.,np.nan,4.,1.])
    r=audit.reference_metrics(p,t)
    assert r['pixels']==6
    assert r['abs_rel']==pytest.approx(4.25/6)
    assert r['invalid_fraction']==.5
    assert r['delta1']==pytest.approx(1/6)
    assert audit.reference_metrics(p*1000,t*1000)['abs_rel']==r['abs_rel']


def test_absolute_scale_error_remains_in_primary_score():
    t=np.array([1.,2.,3.,4.])
    r=audit.reference_metrics(2*t,t)
    assert r['abs_rel']==1
    assert r['diagnostic_median_scale']==.5
    assert r['diagnostic_scaled_abs_rel']==0


def test_shape_and_empty_sensor_fail():
    with pytest.raises(ValueError,match='grids'):
        audit.reference_metrics(np.ones((2,3)),np.ones((3,2)))
    with pytest.raises(ValueError,match='Empty'):
        audit.reference_metrics(np.ones(3),np.array([0,np.nan,-1]))


def test_nearest_pixel_center_convention():
    source=np.arange(12).reshape(3,4)
    expected=np.array([[0,0,1,1,2,2,3,3],[0,0,1,1,2,2,3,3],
                       [4,4,5,5,6,6,7,7],[4,4,5,5,6,6,7,7],
                       [8,8,9,9,10,10,11,11],[8,8,9,9,10,10,11,11]])
    np.testing.assert_array_equal(audit.independent_nearest(source,(6,8)),expected)


def test_metric_aggregate_mean_can_differ_from_pooled_pixels():
    a=audit.reference_metrics(np.array([2.]),np.array([1.]))
    b=audit.reference_metrics(np.ones(3),np.ones(3))
    assert (a['abs_rel']+b['abs_rel'])/2 == .5
    assert audit.reference_metrics(np.array([2.,1.,1.,1.]),np.ones(4))['abs_rel']==.25
