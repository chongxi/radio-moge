from copy import deepcopy

import pytest

from publish_tartanair_source_review import check_candidate_scope, source_disposition


def candidate():
    return dict(ordinal=1, row=dict(dataset='tartanair', split='train', image='/rgb.png',
        sha256='a' * 64, group='tartanair:environment:gascola'), outcome='native_candidate',
        reason='candidate_pending_source_quality_review', key='b' * 64,
        metrics=dict(support=.5, task_support=.9, median_depth=10., p99_depth=30.,
                     rgb_mae=.15, max_stereo_error=.9),
        proof=dict(path='/receipt.json', sha256='c' * 64), exclusions={}, closed=set())


def rejection(scope='geometry_supervision'):
    args = candidate()
    args.update(outcome='rejected', reason='insufficient_stereo_verified_support', key=None, closed={1})
    args['metrics']['support'] = .04
    args['exclusions'][1] = dict(args['row'], corpus_ordinal=1, decision='exclude',
        scope=scope, rgb_replay_excluded=False, evidence=[args['proof']])
    return args


def test_original_support_boundary_and_legacy_exact_rejection_evidence_are_preserved():
    args = candidate()
    args['metrics']['support'] = .05
    original = deepcopy(args)
    assert source_disposition(**args) == 'native_source_eligible'
    assert args == original
    for scope in ('geometry_only', 'geometry_supervision'):
        args = rejection(scope)
        assert 'native_reason' not in args['exclusions'][1]
        assert 'valid_fraction' not in args['exclusions'][1]
        assert source_disposition(**args) == 'excluded_geometry'


@pytest.mark.parametrize('name,value', [('support', .049), ('support', float('nan')),
    ('support', True), ('task_support', .49), ('median_depth', float('inf')),
    ('median_depth', .01), ('p99_depth', 201), ('p99_depth', 9),
    ('rgb_mae', 1.01), ('max_stereo_error', 2)])
def test_native_statistics_cannot_admit_outside_policy_candidates(name, value):
    args = candidate()
    args['metrics'][name] = value
    with pytest.raises(ValueError):
        source_disposition(**args)


@pytest.mark.parametrize('change', ['unknown', 'excluded', 'closed', 'wrong_split', 'missing_key'])
def test_unknown_or_disposed_candidates_cannot_be_selected(change):
    args = candidate()
    if change == 'unknown':
        args['outcome'] = 'flagged_candidate'
    elif change == 'excluded':
        args['exclusions'][1] = rejection()['exclusions'][1]
    elif change == 'closed':
        args['closed'] = {1}
    elif change == 'wrong_split':
        args['row']['split'] = 'test'
    else:
        args['key'] = None
    with pytest.raises(ValueError):
        source_disposition(**args)


@pytest.mark.parametrize('change', ['missing_decision', 'missing_closure', 'wrong_rgb',
    'wrong_reason', 'wrong_ref', 'rgb_exclusion', 'support_drift', 'minimum_drift', 'not_below_minimum'])
def test_rejection_requires_exact_reviewed_rgb_and_evidence(change):
    args = rejection()
    decision = args['exclusions'][1]
    if change == 'missing_decision':
        args['exclusions'].clear()
    elif change == 'missing_closure':
        args['closed'].clear()
    elif change == 'wrong_rgb':
        decision['sha256'] = 'f' * 64
    elif change == 'wrong_reason':
        args['reason'] = 'unexpected'
    elif change == 'wrong_ref':
        decision['evidence'] = [dict(path='/other.json', sha256='c' * 64)]
    elif change == 'rgb_exclusion':
        decision['rgb_replay_excluded'] = True
    elif change == 'support_drift':
        decision['valid_fraction'] = .01
    elif change == 'minimum_drift':
        decision['minimum_image_valid_fraction'] = .01
    else:
        args['metrics']['support'] = .05
    with pytest.raises(ValueError):
        source_disposition(**args)


def reviews():
    entry = dict(corpus_ordinal=1, entry_key='b' * 64,
                 receipt=dict(path='/receipt.json', sha256='c' * 64))
    image = candidate()['row']
    page = dict(path='/page.jpg', sha256='d' * 64)
    rendered = dict(records=[dict(entry, image=image,
        arrays_camera_mask_and_statistics_reproduced_exactly=True)], pages=[page],
        arrays_camera_mask_and_statistics_reproduced_exactly=True,
        source_member_sha_crc_verified=True, all_selected_views_included=True)
    views = [dict(entry, image=image['image'], sha256=image['sha256'], augmented=augmented,
        seed=seed, flipped=flipped, selected_observations=n, sparse_geometry_skipped=n < 32,
        native_to_target_max_relative_error=0.) for augmented, seed, flipped, n in
        [(False, 0, False, 5912), (True, 43, False, 0), (True, 0, True, 0)]]
    return dict(selection=[entry], rendered=rendered,
        visual=dict(reviewed_images=1, all_rendered_images_viewed=True, pages_viewed=[page]),
        crops=dict(actual_production_loader_used=True, original_mask_unchanged=True,
            all_observations_unique_original_samples=True, images=1, views=3, sparse_views=2),
        views=views, sparse=dict(all_rendered_pages_inspected=True,
            minimum_observations_unchanged=32, new_geometry_exclusions=0, rgb_excluded=False,
            corpus_ordinal=1, selected_counts=[5912, 0, 0]))


def test_all_selected_views_bind_source_arrays_and_keep_the_sparse_geometry_skip():
    args = reviews()
    original = deepcopy(args)
    check_candidate_scope(**args)
    assert args == original


@pytest.mark.parametrize('change', ['unviewed', 'different_page', 'duplicate', 'wrong_rgb',
    'wrong_member', 'missing_view', 'missing_flip', 'wrong_skip', 'mapping_error', 'filled_support'])
def test_visual_scope_and_actual_loader_evidence_cannot_be_substituted(change):
    args = reviews()
    if change == 'unviewed':
        args['visual']['all_rendered_images_viewed'] = False
    elif change == 'different_page':
        args['visual']['pages_viewed'] = []
    elif change == 'duplicate':
        args['rendered']['records'] *= 2
    elif change == 'wrong_rgb':
        args['views'][0]['sha256'] = 'e' * 64
    elif change == 'wrong_member':
        args['views'][0]['entry_key'] = 'e' * 64
    elif change == 'missing_view':
        args['views'].pop()
    elif change == 'missing_flip':
        args['views'][2]['flipped'] = False
    elif change == 'wrong_skip':
        args['views'][1]['sparse_geometry_skipped'] = False
    elif change == 'mapping_error':
        args['views'][0]['native_to_target_max_relative_error'] = .1
    else:
        args['crops']['original_mask_unchanged'] = False
    with pytest.raises(ValueError):
        check_candidate_scope(**args)
