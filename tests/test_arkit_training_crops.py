import pytest

from probe_arkit_training_crops import probe_entry, select_rotations
from test_geometry_sampling import packed_fixture


def test_rotation_sampling_keeps_exclusions_out_and_preserves_each_stratum():
    rows = [(i, f'visit:{i // 3}', .05 + i / 10000, 1. + i, (i // 40) * 90)
            for i in range(160)]
    excluded = {0, 40, 80, 120}
    chosen = select_rotations(rows, excluded, 'policy')
    assert chosen == select_rotations(list(reversed(rows)), excluded, 'policy')
    assert not set(chosen) & excluded
    for turn, first in zip((0, 90, 180, 270), (0, 40, 80, 120)):
        assert first + 1 in chosen and first + 39 in chosen
        assert all(f'rotation_{turn}' in chosen[i] for i in chosen if first <= i < first + 40)
    for bad_rows, bad_exclusions in ((rows + rows[:1], excluded),
                                     (rows[:120], {0, 40, 80}), (rows, {999})):
        with pytest.raises(ValueError, match='four-orientation'):
            select_rotations(bad_rows, bad_exclusions, 'policy')


@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
def test_actual_multigrid_packed_loader_checks_centered_augmented_and_reflected_points(tmp_path, rotation):
    pack, entry = packed_fixture(tmp_path, rotation)
    records = probe_entry(pack, entry, (20, 32))
    assert len(records) == 3 and {r['flipped'] for r in records} == {False, True}
    assert all(r['unique_native_observations'] for r in records)
    assert all(r['native_to_target_max_relative_error'] < 1e-6 for r in records)
    assert all(r['selected_observations'] <= r['native_valid_pixels'] for r in records)
    assert all(r['supervision'] == 'metric_native' for r in records)
