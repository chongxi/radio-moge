from copy import deepcopy

import pytest

from review_projection_filtered_metrics import COSINES, cosine_summary, identities, reaggregate


def fixture():
    rows = [dict(image=str(i), dataset=s, **{k: v for k in COSINES}) for i, (s, v) in
            enumerate([('a', .5), ('a', .7), ('b', .9), ('coco2017', .1)])]
    metrics = {'macro/' + k: v for k, v in cosine_summary(rows)['macro'].items()}
    kept = identities(rows[:-1])
    return rows, metrics, identities(rows), kept


def test_equal_source_macro_not_image_weighted_and_preserves_historical():
    args = fixture()
    before = deepcopy(args)
    result = reaggregate(*args)
    assert result['filtered']['macro']['spatial/cosine'] == (.6 + .9) / 2
    assert result['historical']['images'] == 4 and result['filtered']['images'] == 3
    assert args == before


@pytest.mark.parametrize('change', ['duplicate', 'reordered', 'missing', 'nan', 'bad_macro', 'selective_hold'])
def test_mismatched_scope_or_metric_fails(change):
    rows, metrics, expected, kept = fixture()
    if change == 'duplicate':
        rows[1]['image'] = rows[0]['image']
    elif change == 'reordered':
        rows.reverse()
    elif change == 'missing':
        del rows[0]['spatial/cosine']
    elif change == 'nan':
        rows[0]['summary/cosine'] = float('nan')
    elif change == 'bad_macro':
        metrics['macro/spatial/cosine'] += .01
    else:
        kept.pop()
    with pytest.raises(ValueError):
        reaggregate(rows, metrics, expected, kept)
