import json

import pytest
from PIL import Image

from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import publish_json, sha_file
from radio_moge.teacher_rgb_review import (REASON, accounted_preflight,
                                          reviewed_rgb_exclusions, verify_constant_rgb)


def fixture(tmp_path):
    image = tmp_path / 'blank.png'
    Image.new('RGB', (8, 6), 'white').save(image)
    row = dict(image=str(image), sha256=sha_file(image), corpus_ordinal=4,
               width=8, height=6, split='train', group='hypersim:fixture')
    item = dict(row, shape=[6, 8, 3], spatially_constant=True,
                channel_min=[255] * 3, channel_max=[255] * 3)
    records = tmp_path / 'records.jsonl'
    records.write_text(json.dumps(item) + '\n')
    native = dict(path='fixture-native-report', sha256='a' * 64)
    audit = dict(stage='rgb_diagnostic_complete_not_accepted', native_outcomes=native,
                 processed=1, rgb_sha256_verified=True, records=file_ref(records),
                 spatially_constant_images=[item])
    publish_json(tmp_path / 'audit.json', audit)
    review = dict(stage='rgb_input_review_complete', passed=True, training_admission=False,
                  decision='exclude_exactly_spatially_constant_rgb', scope_images=1,
                  diagnostic=file_ref(tmp_path / 'audit.json'), excluded_corpus_ordinals=[4])
    publish_json(tmp_path / 'review.json', review)
    return row, native, review


def test_review_exclusion_is_bound_to_every_frozen_rgb_and_actual_constant_pixels(tmp_path):
    row, native, review = fixture(tmp_path)
    assert set(reviewed_rgb_exclusions(file_ref(tmp_path / 'review.json'), [row], native)) == {4}
    with pytest.raises(ValueError, match='native review scope'):
        reviewed_rgb_exclusions(file_ref(tmp_path / 'review.json'), [row], dict(native, sha256='b' * 64))
    with pytest.raises(ValueError, match='frozen input'):
        reviewed_rgb_exclusions(file_ref(tmp_path / 'review.json'), [dict(row, split='val')], native)
    review['excluded_corpus_ordinals'] = []
    (tmp_path / 'review.json').write_text(json.dumps(review))
    with pytest.raises(ValueError, match='exact exclusion'):
        reviewed_rgb_exclusions(file_ref(tmp_path / 'review.json'), [row], native)


def test_one_pixel_variation_cannot_be_claimed_as_exactly_blank(tmp_path):
    row, _, _ = fixture(tmp_path)
    image = Image.open(row['image']).copy()
    image.putpixel((2, 3), (254, 255, 255))
    image.save(row['image'])
    with pytest.raises(ValueError, match='bytes changed'):
        verify_constant_rgb(row, [255] * 3)
    row['sha256'] = sha_file(row['image'])
    with pytest.raises(ValueError, match='exactly spatially constant'):
        verify_constant_rgb(row, [255] * 3)


def test_preflight_needs_positive_cache_and_exact_reviewed_rejections():
    rows = [dict(corpus_ordinal=4), dict(corpus_ordinal=5)]
    rejection = dict(image=rows[0], reason=REASON)
    assert accounted_preflight(1, [rejection], rows, {4: {}})
    assert not accounted_preflight(1, [dict(rejection, reason='unexpected')], rows, {4: {}})
    assert not accounted_preflight(0, [rejection], rows[:1], {4: {}})
    assert not accounted_preflight(2, [], rows, {4: {}})
    assert not accounted_preflight(1, [rejection, rejection], rows, {4: {}})
