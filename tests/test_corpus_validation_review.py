from copy import deepcopy

import pytest

from review_corpus_validation import check_aggregates, extend_metric, paired_rows


def test_pairing_rejects_reordered_images_duplicates_and_changed_teacher():
    original = [dict(image=str(i), dataset='source', **{'gt_teacher/abs_rel': .1}) for i in range(3)]
    paired_rows(original, deepcopy(original))
    for current in (original[::-1], original[:2], [original[0]] * 3):
        with pytest.raises(ValueError, match='identity/order'):
            paired_rows(original, current)
    changed = deepcopy(original)
    changed[1]['gt_teacher/abs_rel'] = .2
    with pytest.raises(ValueError, match='Frozen teacher'):
        paired_rows(original, changed)


def test_known_camera_support_cannot_change_with_predictions():
    a = [dict(id='one', scene='s', gt_valid_fraction=.8, gt_plane_fraction=.5, abs_rel=.3)]
    b = [dict(a[0], abs_rel=.2)]
    paired_rows(a, b, ('id', 'scene'), ('gt_valid_fraction', 'gt_plane_fraction'))
    b[0]['gt_valid_fraction'] = .7
    with pytest.raises(ValueError, match='fixed GT support'):
        paired_rows(a, b, ('id', 'scene'), ('gt_valid_fraction', 'gt_plane_fraction'))


def test_macro_weights_datasets_equally_even_when_image_counts_differ():
    names = ('spatial/cosine', 'summary/cosine', 'teacher_depth/abs_rel')
    rows = [dict(dataset='a', **dict.fromkeys(names, .2)) for _ in range(3)]
    rows.append(dict(dataset='b', **dict.fromkeys(names, .8)))
    metrics = dict(images=4, **{'macro/' + k: .5 for k in names})
    check_aggregates(rows, metrics)
    metrics['macro/spatial/cosine'] = .35
    with pytest.raises(ValueError, match='macro metric'):
        check_aggregates(rows, metrics)


def test_metric_history_preserves_prior_steps_and_replaces_stale_deltas():
    old = dict(before=.1, step1000=.09, step2000=.08, percent_from_1000=-99)
    new = extend_metric(old, .07, 2250, 2000)
    assert new['step1000'] == .09 and new['step2000'] == .08 and new['step2250'] == .07
    assert new['percent_from_2000'] == pytest.approx(-12.5)
    assert 'percent_from_1000' not in new and old['percent_from_1000'] == -99
    with pytest.raises(ValueError):
        extend_metric(new, .07, 2250, 2000)
    with pytest.raises(ValueError):
        extend_metric(old, float('nan'), 2250, 2000)
