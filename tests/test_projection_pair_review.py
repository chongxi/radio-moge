from copy import deepcopy

import pytest

from review_projection_pair_arm import compare_view, metrics_for


def fixture():
    old = [dict(image=f'{i}.png', dataset='nyuv2' if i < 1012 else 'coco2017',
                **{'spatial/cosine': .9, 'summary/cosine': .8, 'gt_teacher/abs_rel': .1})
           for i in range(1268)]
    new = deepcopy(old[:1012])
    return dict(old_rows=old, new_rows=new, old_protocol=dict(images=1268, max_side=640),
        new_protocol=dict(images=1012, max_side=640), new_metrics=metrics_for(new), view='original',
        kept=[(r['image'], r['dataset']) for r in new])


def test_only_predeclared_source_count_changes():
    args = fixture()
    assert compare_view(**args)['images'] == 1012


@pytest.mark.parametrize('kind', ['order', 'teacher', 'protocol', 'count', 'metric', 'kept'])
def test_saved_evidence_drift_rejected(kind):
    args = fixture()
    if kind == 'order':
        args['new_rows'].reverse()
    elif kind == 'teacher':
        args['new_rows'][0]['gt_teacher/abs_rel'] = .2
    elif kind == 'protocol':
        args['new_protocol']['max_side'] = 480
    elif kind == 'count':
        args['old_protocol']['images'] = 1012
    elif kind == 'metric':
        args['new_metrics']['macro/spatial/cosine'] += .01
    else:
        args['kept'].reverse()
    with pytest.raises(ValueError):
        compare_view(**args)


def test_lidar_cannot_change_protocol_count():
    args = fixture()
    args.update(view='dataset-scannetpp-lidar', old_rows=args['new_rows'],
                old_protocol=dict(images=1012, max_side=640))
    assert compare_view(**args)['protocol_count_only_transition'] is False
    args['new_protocol']['images'] = 120
    with pytest.raises(ValueError, match='protocol'):
        compare_view(**args)
