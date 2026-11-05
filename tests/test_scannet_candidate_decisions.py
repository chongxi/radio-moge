from copy import deepcopy

import pytest

from publish_scannetpp_candidate_decisions import ORDINALS, reviewed_candidates


def fixture():
    flags = [dict(corpus_ordinal=o, image=f'image/{o}.jpg', sha256='a'*64,
                  group='scannetpp:scene', entry_key=f'{o:064x}', member_sha256='b'*64,
                  receipt=dict(path=f'{o}.json', sha256='c'*64), policy_sha256='d'*64)
             for o in sorted(ORDINALS)]
    records = [dict(f, image=dict(path=f['image'], sha256=f['sha256']),
                    actual_page_viewed=True, decision='quarantine_candidate') for f in flags]
    return dict(stage='visual_review_complete_source_quality_unresolved',
                source_quality_accepted=False, quarantined_candidate_ordinals=sorted(ORDINALS),
                records=records), flags


def test_exact_scope_and_no_record_mutation():
    review, flags = fixture()
    original = deepcopy((review, flags))
    assert set(reviewed_candidates(review, flags)) == ORDINALS
    assert (review, flags) == original


@pytest.mark.parametrize('field,value', [
    ('actual_page_viewed', False), ('entry_key', 'wrong'), ('member_sha256', 'e'*64),
    ('group', 'scannetpp:other'), ('policy_sha256', 'f'*64),
    ('image', dict(path='changed.jpg', sha256='a'*64)),
    ('receipt', dict(path='other.json', sha256='c'*64)),
])
def test_changed_identity_or_unviewed_record_rejected(field, value):
    review, flags = fixture()
    review['records'][0][field] = value
    with pytest.raises(ValueError, match='identity|review'):
        reviewed_candidates(review, flags)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'neighbor'])
def test_no_scope_expansion_or_omission(change):
    review, flags = fixture()
    if change == 'missing':
        flags.pop()
    elif change == 'duplicate':
        flags[-1] = flags[0]
    else:
        flags[-1]['corpus_ordinal'] += 1
    with pytest.raises(ValueError, match='nineteen'):
        reviewed_candidates(review, flags)
