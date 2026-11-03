import torch

from probe_tartanair_candidate_crops import probe_entry
from test_tartanair_outcome_visuals import fixture


def test_actual_native_crop_retains_identity_unique_points_and_true_flip(tmp_path):
    torch.set_num_threads(1)
    entry, pack, _, _, _ = fixture(tmp_path)
    rows = probe_entry(pack, entry)
    assert len(rows) == 3 and {r['flipped'] for r in rows} == {False, True}
    assert all(r['unique_native_observations'] and r['selected_observations'] > 32 for r in rows)
    assert max(r['native_to_target_max_relative_error'] for r in rows) < 1e-7
