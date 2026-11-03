from copy import deepcopy

import pytest

from publish_arkit_source_review import (FARO_EXCLUSIONS, FARO_RETAINED,
                                       checked_faro_records, source_disposition)


def fixture():
    return dict(ordinal=27177, row=dict(image='/original.png', sha256='a' * 64,
        group='arkitscenes:visit:one', dataset='arkitscenes', split='train'),
        outcome='native_candidate', reason='candidate_pending_source_quality_review', key='b' * 64,
        metrics=dict(support=.9, median_depth=.765, rgb_mae=.002, rgb_p90=.004),
        exclusions={}, flags={})


def excluded(args):
    args['exclusions'][args['ordinal']] = dict(args['row'], corpus_ordinal=args['ordinal'],
        decision='exclude', scope='geometry_only', rgb_replay_excluded=False)
    args['flags'][args['key']] = dict(args['row'], corpus_ordinal=args['ordinal'])
    return args


def test_native_and_quarantined_routes_preserve_original_inputs():
    args = fixture()
    assert source_disposition(**args) == 'native_source_eligible'
    args = excluded(args)
    original = deepcopy(args)
    assert source_disposition(**args) == 'excluded_geometry'
    assert args == original


@pytest.mark.parametrize('change', ['missing_decision', 'missing_flag', 'wrong_rgb',
                                   'wrong_ordinal', 'rgb_exclusion', 'invalid_support',
                                   'bad_registration', 'nonfinite', 'native_flag'])
def test_known_flags_identity_and_native_support_cannot_be_bypassed(change):
    args = excluded(fixture())
    if change == 'missing_decision':
        args['exclusions'].clear()
    elif change == 'missing_flag':
        args['flags'].clear()
    elif change == 'wrong_rgb':
        args['exclusions'][args['ordinal']]['sha256'] = 'c' * 64
    elif change == 'wrong_ordinal':
        args['flags'][args['key']]['corpus_ordinal'] += 1
    elif change == 'rgb_exclusion':
        args['exclusions'][args['ordinal']]['rgb_replay_excluded'] = True
    elif change == 'native_flag':
        args['outcome'] = 'flagged_candidate'
    else:
        args = fixture()
        if change == 'invalid_support':
            args['metrics']['support'] = .049
        elif change == 'bad_registration':
            args['metrics']['rgb_p90'] = .081
        else:
            args['metrics']['median_depth'] = float('nan')
    with pytest.raises(ValueError):
        source_disposition(**args)


def test_rejection_requires_exact_reviewed_reason_but_allows_legacy_geometry_scope():
    args = excluded(fixture())
    args.update(outcome='rejected', reason='low_registered_support', key=None)
    args['flags'].clear()
    with pytest.raises(ValueError, match='exact reviewed disposition'):
        source_disposition(**args)
    decision = args['exclusions'][args['ordinal']]
    decision.update(native_reason=args['reason'], scope='geometry_supervision')
    assert source_disposition(**args) == 'excluded_geometry'
    decision['native_reason'] = 'image_registration_failed'
    with pytest.raises(ValueError):
        source_disposition(**args)


def faro_fixture():
    rows = [dict(corpus_ordinal=i, image=dict(path=f'/{i}.png', sha256='a' * 64),
                 page=dict(path=f'/page-{i}.png', sha256='b' * 64))
            for i in sorted(set(FARO_EXCLUSIONS) | FARO_RETAINED)]
    diagnostic = dict(exact_highres_intrinsics_now_obtained=True,
        original_native_arrays_and_masks_unchanged=True, calibration_fit_performed=False, records=rows)
    review = dict(stage='exact_publisher_camera_visual_review_complete', images_reviewed=9,
        pages_reviewed=9, exact_highres_intrinsics_ambiguity_resolved_for_these_nine=True,
        records=[dict(x, page_actually_viewed=True, observation='reviewed case') for x in deepcopy(rows)])
    return review, diagnostic


def test_exact_visual_scope_is_not_replaced_by_a_numeric_error_threshold():
    review, diagnostic = faro_fixture()
    assert len(checked_faro_records(review, diagnostic)) == 9
    for change in ('duplicate', 'missing', 'changed_image', 'changed_page', 'unviewed', 'fitted'):
        a, b = deepcopy(review), deepcopy(diagnostic)
        if change == 'duplicate':
            b['records'].append(b['records'][0])
        elif change == 'missing':
            b['records'].pop()
        elif change == 'changed_image':
            a['records'][0]['image']['sha256'] = 'c' * 64
        elif change == 'changed_page':
            a['records'][0]['page']['sha256'] = 'c' * 64
        elif change == 'unviewed':
            a['records'][0]['page_actually_viewed'] = False
        else:
            b['calibration_fit_performed'] = True
        with pytest.raises(ValueError):
            checked_faro_records(a, b)
