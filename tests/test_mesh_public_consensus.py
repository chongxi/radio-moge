import numpy as np
import pytest

from radio_moge.mesh_clip import MeshClipContract, select_public_consensus


def test_disagreements_in_both_directions_missing_and_guard_are_invalid_without_replacement():
    cpu = np.array([[2., 2., 2., 2., .05010007321834564, 0.]], np.float32)
    valid = cpu > 0
    reference = np.array([[2.001, 1.99, 2.01, 0., .05009995400905609, 5.]], np.float32)
    original = cpu.copy(), valid.copy(), reference.copy()
    depth, mask, reason, stats = select_public_consensus(cpu, valid, reference, MeshClipContract())
    assert reason.tolist() == [[1, 4, 4, 2, 3, 0]]
    assert mask.tolist() == [[True, False, False, False, False, False]]
    assert depth.tolist() == [[2., 0., 0., 0., 0., 0.]]
    assert stats['input_verified_pixels'] == 5 and stats['rejected_pixels'] == 4
    for a, b in zip(original, (cpu, valid, reference)):
        np.testing.assert_array_equal(a, b)


def test_empty_and_unchanged_selection_keep_exact_float32_values():
    cpu = np.array([[.1, 2., 19.9]], np.float32)
    for support in (np.ones_like(cpu, bool), np.zeros_like(cpu, bool)):
        depth, mask, _, stats = select_public_consensus(cpu, support, cpu, MeshClipContract())
        np.testing.assert_array_equal(mask, support)
        np.testing.assert_array_equal(depth[support], cpu[support])
        assert depth.dtype == cpu.dtype and stats['rejected_pixels'] == 0


@pytest.mark.parametrize('kind', ['shape', 'nan', 'negative_reference', 'invalid_cpu_guard', 'nonbool'])
def test_malformed_arrays_cannot_claim_consensus(kind):
    cpu = np.ones((2, 3), np.float32)
    valid, reference = np.ones_like(cpu, bool), cpu.copy()
    if kind == 'shape':
        reference = reference[:, :2]
    elif kind == 'nan':
        reference[0, 0] = np.nan
    elif kind == 'negative_reference':
        reference[0, 0] = -1
    elif kind == 'invalid_cpu_guard':
        cpu[0, 0] = .05
    elif kind == 'nonbool':
        valid = valid.astype(np.uint8)
    with pytest.raises(ValueError):
        select_public_consensus(cpu, valid, reference, MeshClipContract())
