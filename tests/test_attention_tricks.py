import pytest
import torch

from radio_moge.adapters import RadioFeatures
from radio_moge.architecture_screen import make_candidate
from radio_moge.attention_tricks import attention_candidates


def small(variant):
    spec = next(x for x in attention_candidates() if x['variant'] == variant)
    return dict(spec, width=32, blocks=2, heads={'heads16': 2, 'heads64': 8}.get(variant, 4))


def features():
    return RadioFeatures(torch.randn(2, 8, 2, 3), torch.randn(2, 12),
                         tuple(torch.randn(2, 8, 2, 3) for _ in range(4)))


def model(variant):
    torch.manual_seed(42)
    return make_candidate(small(variant), dict(spatial_dim=8, summary_dim=12,
                                             target_dim=6, target_summary_dim=10))


def test_prespecified_single_changes_and_real_parameter_counts():
    rows = attention_candidates()
    assert len(rows) == len({r['name'] for r in rows}) == 10
    for seed in (42, 43):
        group = {r['variant']: r for r in rows if r['seed'] == seed}
        for variant, field in [('heads16', 'heads'), ('heads64', 'heads'),
                               ('qknorm', 'qk_norm'), ('summary_readout', 'summary_readout')]:
            assert {k for k in group[variant] if group[variant].get(k) != group['baseline'].get(k)} == {
                'name', 'variant', field}
    with torch.device('meta'):
        counts = {r['variant']: sum(p.numel() for p in make_candidate(r).parameters())
                  for r in rows[:5]}
    assert counts['baseline'] == 221739844
    assert counts['heads16'] < counts['baseline'] < counts['heads64']
    assert counts['qknorm'] > counts['baseline']
    assert counts['baseline'] < counts['summary_readout'] < counts['baseline'] * 1.1


@pytest.mark.parametrize('variant', ['baseline', 'heads16', 'heads64', 'qknorm', 'summary_readout'])
def test_all_variants_learn_both_outputs_with_finite_gradients(variant):
    m = model(variant)
    output = m(features())
    assert output.spatial.shape == (2, 6, 2, 3) and output.summary.shape == (2, 10)
    (output.spatial.square().mean() + output.summary.square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in m.parameters() if p.numel())
    assert any(torch.count_nonzero(p.grad) for p in m.parameters() if p.numel())


def test_summary_readout_preserves_shared_initialization_and_spatial_output():
    base, readout = model('baseline'), model('summary_readout')
    for name, value in base.state_dict().items():
        assert torch.equal(value, readout.state_dict()[name])
    f = features()
    a, b = base(f), readout(f)
    torch.testing.assert_close(a.spatial, b.spatial, rtol=0, atol=0)
    assert not torch.equal(a.summary, b.summary)
    with torch.no_grad():
        readout.summary_readout.scale.zero_()
    torch.testing.assert_close(a.summary, readout(f).summary, rtol=0, atol=0)


def test_qk_normalization_acts_on_projected_heads_and_can_be_disabled_exactly():
    base, normalized = model('baseline'), model('qknorm')
    for name, value in base.state_dict().items():
        assert torch.equal(value, normalized.state_dict()[name])
    f = features()
    assert not torch.equal(base(f).spatial, normalized(f).spatial)
    for block in normalized.blocks:
        assert block.q_norm.normalized_shape == (8,)
        x = torch.randn(2, 4, 11, 8) * 9 + 4
        torch.testing.assert_close(block.q_norm(x).mean(-1), torch.zeros(2, 4, 11), atol=1e-6, rtol=0)
        block.q_norm, block.k_norm = torch.nn.Identity(), torch.nn.Identity()
    a, b = base(f), normalized(f)
    torch.testing.assert_close(a.spatial, b.spatial, rtol=0, atol=0)
    torch.testing.assert_close(a.summary, b.summary, rtol=0, atol=0)
