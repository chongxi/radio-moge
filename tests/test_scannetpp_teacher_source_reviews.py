from copy import deepcopy

import pytest

from publish_scannetpp_teacher_source_reviews import COUNTS, validate_review_inputs


def fixture():
    audit = dict(stage='teacher_outcomes_accounted_not_accepted', dataset='scannetpp_dslr',
        expected_scope=35783, candidate_policy_counts=COUNTS,
        outcome_counts={'teacher_candidate':35783}, candidate_images=35783, rejected_images=0,
        scenes=25, producer_array_readback_complete=True, native_source_flags_cleared=False)
    visual = dict(stage='visual_review_complete',
        decision='eligible_for_relative_pseudo_source_review_with_documented_limits',
        reviewed_images=100, reviewed_pages=list(range(34)), reviewed_scenes=list(range(25)),
        all_rendered_pages_actually_inspected=True, additional_candidate_quarantine_ordinals=[],
        additional_geometry_exclusions=[], native_source_flags_cleared=False)
    rendered = dict(rendered_images=100, scenes=25, all_selected_views_included=True,
        selected_payloads_and_rgb_rechecked=True, original_anonymity_masks_rechecked=True,
        pages=visual['reviewed_pages'], records=[dict(scene=i//4, corpus_ordinal=i,
            native_source_review_unresolved=True, support_within_eroded_anonymity_mask=True) for i in range(100)])
    return audit, visual, rendered


def test_only_the_complete_reviewed_scope_can_use_empty_quarantine():
    validate_review_inputs(*fixture())
    for scope, key, value in ((0, 'producer_array_readback_complete', False),
            (0, 'native_source_flags_cleared', True), (0, 'rejected_images', 1),
            (0, 'candidate_policy_counts', {}), (1, 'all_rendered_pages_actually_inspected', False),
            (1, 'additional_candidate_quarantine_ordinals', [10]),
            (1, 'additional_geometry_exclusions', [10]),
            (2, 'original_anonymity_masks_rechecked', False), (2, 'pages', [])):
        args = deepcopy(fixture())
        args[scope][key] = value
        with pytest.raises(ValueError, match='actual all-scene'):
            validate_review_inputs(*args)


def test_duplicate_views_missing_scene_or_cleared_native_flag_cannot_pass():
    for key, value in (('corpus_ordinal', 1), ('scene', 99),
                       ('native_source_review_unresolved', False),
                       ('support_within_eroded_anonymity_mask', False)):
        args = fixture()
        args[2]['records'][0][key] = value
        with pytest.raises(ValueError):
            validate_review_inputs(*args)
