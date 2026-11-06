from copy import deepcopy

import pytest

from prepare_room_epoch_pool import check_review


@pytest.mark.parametrize('diagnostic', [False, True])
@pytest.mark.parametrize('mutation', [None, 'admit_reject', 'replace', 'count', 'unseen', 'binding'])
def test_expansion_preserves_new_batch_review_and_diagnostic_only_rejection(diagnostic, mutation):
    frozen = dict(stage='frozen_verified_structural_feature_rgb', split='train',
        feature_training_eligible=True, every_selected_rgb_decoded_and_hashed=True,
        images=1962 if diagnostic else 1981, manifest={'sha256':'manifest'})
    closure = dict(manifest=frozen['manifest'], freeze={'sha256':'freeze'})
    closure['qualified_images' if diagnostic else 'images'] = frozen['images']
    visual = dict(source_report=closure['freeze'], all64_images_actually_viewed=True,
        all16_pages_actually_viewed=True, new_exclusions=[], feature_rgb_data_eligible=True)
    if diagnostic:
        visual.update(rejected_diagnostic_images=1, admitted_sample_images=63,
                      rejected_image_admitted=False, replacement_images=0)
    if mutation == 'admit_reject': visual['rejected_image_admitted'] = True
    elif mutation == 'replace': visual['replacement_images'] = 1
    elif mutation == 'count': frozen['images'] += 1
    elif mutation == 'unseen': visual['all64_images_actually_viewed'] = False
    elif mutation == 'binding': visual['source_report'] = {'sha256':'other'}
    before = deepcopy((closure, frozen, visual))
    if mutation is None:
        check_review(closure, frozen, visual)
        assert (closure, frozen, visual) == before
    else:
        with pytest.raises(ValueError): check_review(closure, frozen, visual)
