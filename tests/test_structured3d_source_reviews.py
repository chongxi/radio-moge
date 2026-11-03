from copy import deepcopy

import pytest

from publish_structured3d_source_reviews import source_disposition


def fixture():
    row = dict(image='/original.png', sha256='a' * 64, group='structured3d:scene:00103',
               dataset='structured3d', split='train')
    return dict(ordinal=532493, row=row, outcome='native_candidate', reason='quality_review_pending',
                key='b' * 64, metrics=dict(valid_fraction=.8, layout_verified_pixels=1000,
                median_depth_m=3), exclusions={}, flags={})


def excluded(args):
    args['exclusions'][args['ordinal']] = dict(args['row'], corpus_ordinal=args['ordinal'],
        decision='exclude', scope='geometry_only', rgb_replay_excluded=False)
    args['flags'][args['key']] = dict(args['row'], corpus_ordinal=args['ordinal'])
    return args


def test_native_candidate_and_exact_quarantine_remain_distinct():
    args = fixture()
    assert source_disposition(**args) == 'native_source_eligible'
    args = excluded(args)
    original = deepcopy(args)
    assert source_disposition(**args) == 'excluded_geometry'
    assert args == original


@pytest.mark.parametrize('change', ['missing_exclusion', 'missing_flag', 'changed_rgb',
                                   'wrong_ordinal', 'rgb_excluded', 'unsupported'])
def test_flag_exclusion_and_identity_cannot_be_silently_bypassed(change):
    args = excluded(fixture())
    if change == 'missing_exclusion':
        args['exclusions'].clear()
    elif change == 'missing_flag':
        args['flags'].clear()
    elif change == 'changed_rgb':
        args['exclusions'][args['ordinal']]['sha256'] = 'c' * 64
    elif change == 'wrong_ordinal':
        args['flags'][args['key']]['corpus_ordinal'] += 1
    elif change == 'rgb_excluded':
        args['exclusions'][args['ordinal']]['rgb_replay_excluded'] = True
    else:
        args = fixture()
        args['metrics']['valid_fraction'] = .049
    with pytest.raises(ValueError):
        source_disposition(**args)


def test_native_rejection_requires_exact_reviewed_reason():
    args = excluded(fixture())
    args.update(outcome='rejected_needs_disposition', reason='insufficient_layout_reference', key=None)
    args['flags'].clear()
    with pytest.raises(ValueError, match='exact reviewed disposition'):
        source_disposition(**args)
    args['exclusions'][args['ordinal']]['native_reason'] = args['reason']
    assert source_disposition(**args) == 'excluded_geometry'
    args['exclusions'][args['ordinal']]['native_reason'] = 'upstream_errata'
    with pytest.raises(ValueError):
        source_disposition(**args)
