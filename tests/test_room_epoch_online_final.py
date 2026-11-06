"""Final validation must survive rectangular history without hiding bad records."""
import pytest

from verify_room_epoch_online import exact_values, final_validation_record


KEYS = ('validation/cosine', 'validation/error')


def result(internal_step=11, cosine=.96, error=.1):
    return {'_step': internal_step, 'optimizer_step': 100,
            KEYS[0]: cosine, KEYS[1]: error}


def test_null_padded_final_training_row_is_not_a_validation():
    train = result(10, None, None) | {'train/total': .15}
    final = result()
    selected, ignored = final_validation_record([train, final], 100, KEYS)
    assert selected == final
    assert ignored == [{'internal_step': 10, 'optimizer_step': 100,
                        'all_expected_validation_values_null_or_absent': True}]
    assert exact_values({k: selected[k] for k in KEYS},
                        {KEYS[0]: .96, KEYS[1]: .1}, 'final') == []


def test_null_only_rows_cannot_supply_missing_validation():
    with pytest.raises(ValueError, match='Missing or repeated'):
        final_validation_record([result(10, None, None)], 100, KEYS)


@pytest.mark.parametrize('rows', [[], [result(), result(12)]])
def test_missing_or_duplicate_complete_records_fail(rows):
    with pytest.raises(ValueError, match='Missing or repeated'):
        final_validation_record(rows, 100, KEYS)


@pytest.mark.parametrize('partial', [result(10, None, .1),
                                    {'optimizer_step': 100, KEYS[0]: .96}])
def test_partial_record_is_not_hidden_by_a_complete_record(partial):
    with pytest.raises(ValueError, match='Partially missing'):
        final_validation_record([partial, result()], 100, KEYS)


def test_zeros_are_real_values_and_other_steps_are_excluded():
    final = result(error=0)
    selected, ignored = final_validation_record(
        [result() | {'optimizer_step': 99}, final], 100, KEYS)
    assert selected == final
    assert ignored == []


def test_numeric_mismatch_still_fails():
    selected, _ = final_validation_record([result()], 100, KEYS)
    with pytest.raises(AssertionError, match='Metadata value mismatch'):
        exact_values({k: selected[k] for k in KEYS},
                     {KEYS[0]: .96, KEYS[1]: .2}, 'final')


@pytest.mark.parametrize('invalid', [float('nan'), float('inf')])
def test_nonfinite_is_not_treated_as_null_padding(invalid):
    selected, ignored = final_validation_record([result(error=invalid)], 100, KEYS)
    assert ignored == []
    with pytest.raises(AssertionError, match='Nonfinite metadata'):
        exact_values({k: selected[k] for k in KEYS},
                     {KEYS[0]: .96, KEYS[1]: .1}, 'final')


def test_empty_metric_contract_fails():
    with pytest.raises(ValueError, match='metrics are required'):
        final_validation_record([result()], 100, [])
