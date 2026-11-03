from copy import deepcopy

import pytest

from publish_hypersim_native_source_review import validate_exclusion_identity


def fixture():
    original = dict(image='/frozen/white.jpg', sha256='a' * 64,
                    dataset='hypersim', group='hypersim:original')
    decision = dict(image=original['image'], sha256=original['sha256'], dataset='hypersim',
                    corpus_ordinal=210812, decision='exclude')
    return original, decision


def test_legacy_exclusion_binds_original_identity_without_rewriting_record():
    original, decision = fixture()
    saved = deepcopy(decision)
    validate_exclusion_identity(decision, original, 210812)
    assert decision == saved and 'group' not in decision
    validate_exclusion_identity(dict(decision, group=original['group']), original, 210812)


@pytest.mark.parametrize('key,value', [('image', '/other.jpg'), ('sha256', 'b' * 64),
    ('dataset', 'coco2017'), ('group', 'hypersim:other'), ('corpus_ordinal', 210813),
    ('decision', 'select')])
def test_wrong_legacy_or_explicit_identity_still_fails(key, value):
    original, decision = fixture()
    with pytest.raises(ValueError, match='changed frozen RGB'):
        validate_exclusion_identity(dict(decision, **{key: value}), original, 210812)


@pytest.mark.parametrize('key', ['sha256', 'corpus_ordinal'])
def test_missing_identity_cannot_be_inferred_from_group(key):
    original, decision = fixture()
    decision['group'] = original['group']
    decision.pop(key)
    with pytest.raises(ValueError, match='changed frozen RGB'):
        validate_exclusion_identity(decision, original, 210812)
