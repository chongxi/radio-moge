from collections import Counter
from copy import deepcopy

import numpy as np
import pytest

from radio_moge.scanner_support import station_support
from review_scannetpp_dense_source import MASKS, add_observation, validate_closed


def closed():
    plan = dict(grid=[2, 3], mode='full', corpus_ordinal=851268, scanner_order=['0', '1'])
    report = dict(stage='full_grid_conditional_source_complete', mode='full', corpus_ordinal=851268,
        full_grid_evaluated=True, evaluated_station_slots=12, requested_station_slots=12,
        full_grid_pixels=6, parity_passed=True, parity_slots=96, training_admission=False,
        source_quality_accepted=False, optimizer_updates=0)
    arrays = {k: np.ones((2, 3), bool) for k in MASKS}
    return report, plan, arrays


@pytest.mark.parametrize('fault', ['none', 'partial', 'lost_grid', 'wrong_dtype', 'missing_station', 'wrong_scan', 'no_parity'])
def test_full_denominator_and_evaluation_grid_required(fault):
    report, plan, arrays = closed()
    if fault == 'partial':
        report['full_grid_evaluated'] = False
    elif fault == 'lost_grid':
        arrays.pop('scan1_evaluated')
    elif fault == 'wrong_dtype':
        arrays['conditional_support'] = arrays['conditional_support'].astype(np.uint8)
    elif fault == 'missing_station':
        arrays['scan1_evaluated'][0, 0] = False
    elif fault == 'wrong_scan':
        plan['scanner_order'] = ['0', '2']
    elif fault == 'no_parity':
        report['parity_passed'] = False
    if fault == 'none':
        validate_closed(report, plan, arrays)
    else:
        with pytest.raises(ValueError):
            validate_closed(report, plan, arrays)


def unsupported_row():
    row = dict(scan='0', corpus_ordinal=851268, scene='scene', pixel_yx=[0, 1], slot=1,
        projected_footprint=dict(sampled_C0_correspondence=False), angle_matches={},
        exact_target_visibility=dict(status='DSLR_numerically_unsupported', exact_target_first_surface=False))
    row['decision'] = station_support(row)
    return row


@pytest.mark.parametrize('fault', ['none', 'duplicate', 'forged_support', 'wrong_slot', 'wrong_identity'])
def test_saved_slot_recomputation_detects_duplicate_forged_and_replaced_observations(fault):
    seen, mask = np.zeros((2, 2, 3), bool), np.zeros((2, 2, 3), bool)
    bits, counts = np.zeros((2, 2, 3), np.uint8), [Counter(), Counter()]
    arrays = dict(numerical_support=np.zeros((2, 3), bool))
    row = unsupported_row()
    original = deepcopy(row)
    if fault == 'forged_support':
        row['decision']['conditional_station_support'] = True
    elif fault == 'wrong_slot':
        row['slot'] = 4
    elif fault == 'wrong_identity':
        row['corpus_ordinal'] = 851269
    elif fault == 'duplicate':
        add_observation(row, seen, mask, counts, bits, arrays, 851268, 'scene')
    if fault == 'none':
        add_observation(row, seen, mask, counts, bits, arrays, 851268, 'scene')
        assert seen.sum() == 1 and not mask.any()
        assert bits[0, 0, 1] == 7 and row == original
    else:
        with pytest.raises(ValueError):
            add_observation(row, seen, mask, counts, bits, arrays, 851268, 'scene')
