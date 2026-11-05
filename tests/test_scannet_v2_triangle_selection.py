from copy import deepcopy

import pytest

from probe_scannetpp_v2_disagreement_triangles import remaining_disagreements


def evidence():
    ref = dict(path='/source.json', sha256='a'*64)
    record = dict(path='/100.json', sha256='b'*64)
    rows = [dict(ordinal=100, pixel_yx=[i, 2], depth=1., gpu=1.0001, software=1.01,
                 source=dict(path='/arrays', sha256='c'*64)) for i in range(32)]
    previous = dict(source=ref, stage='bounded_single_triangle_diagnosis_complete',
        training_admission=False, cases=[dict(corpus_ordinal=100, pixel_yx=[i, 2],
            record=record, arrays=dict(path='/arrays', sha256='c'*64),
            saved_cpu_z_m=1., saved_cuda_z_m=1.0001, saved_public_z_m=1.01) for i in range(5)])
    return ref, dict(records=[record]), dict(source=ref, disagreements=rows,
                                            training_admission=False), previous


def test_exact_remaining27_preserves_inputs_and_never_repeats_completed_pixels():
    args = evidence()
    original = deepcopy(args)
    selected, expected = remaining_disagreements(*args)
    assert selected == {100: [(i, 2) for i in range(5, 32)]}
    assert len(expected) == 32 and args == original


@pytest.mark.parametrize('change', ['source', 'duplicate', 'missing', 'unknown_previous',
                                   'duplicate_previous', 'array', 'depth', 'record'])
def test_stale_changed_missing_or_duplicate_evidence_cannot_change_scope(change):
    args = list(evidence())
    if change == 'source':
        args[2]['source'] = dict(path='/other', sha256='d'*64)
    elif change == 'duplicate':
        args[2]['disagreements'][-1] = args[2]['disagreements'][0]
    elif change == 'missing':
        args[2]['disagreements'].pop()
    elif change == 'unknown_previous':
        args[3]['cases'][0]['pixel_yx'] = [99, 2]
    elif change == 'duplicate_previous':
        args[3]['cases'][1] = args[3]['cases'][0]
    elif change == 'array':
        args[3]['cases'][0]['arrays']['sha256'] = 'd'*64
    elif change == 'depth':
        args[3]['cases'][0]['saved_public_z_m'] = 2.
    elif change == 'record':
        args[3]['cases'][0]['record'] = dict(path='/100.json', sha256='e'*64)
    with pytest.raises(ValueError):
        remaining_disagreements(*args)
