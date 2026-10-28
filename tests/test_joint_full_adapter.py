import torch

from radio_moge.adapters import RadioFeatures, build_adapter
from radio_moge.architecture_screen import make_candidate
from radio_moge.attention_tricks import attention_candidates
from radio_moge.metrics import dataset_metrics


def test_selected_seed_survives_backbone_rng_and_checkpoint_roundtrip():
    spec = next(x for x in attention_candidates() if x['variant'] == 'qknorm' and x['seed'] == 43)
    spec = dict(spec, width=32, blocks=2, heads=4)
    dims = dict(spatial_dim=8, summary_dim=12, target_dim=6, target_summary_dim=10)
    torch.manual_seed(43)
    screened = make_candidate(spec, dims)
    torch.manual_seed(9876)
    before = torch.random.get_rng_state().clone()
    full = build_adapter(kind='joint_attention', spec=spec, initialization_seed=43, **dims)
    assert torch.equal(before, torch.random.get_rng_state())
    for key, value in screened.state_dict().items():
        torch.testing.assert_close(value, full.state_dict()[key], atol=0, rtol=0)
    f = RadioFeatures(torch.randn(2, 8, 2, 3), torch.randn(2, 12),
                      tuple(torch.randn(2, 8, 2, 3) for _ in range(4)))
    expected, actual = screened(f), full(f)
    torch.testing.assert_close(expected.spatial, actual.spatial, atol=0, rtol=0)
    torch.testing.assert_close(expected.summary, actual.summary, atol=0, rtol=0)
    (actual.spatial.square().mean() + actual.summary.square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in full.parameters())
    restored = build_adapter(**full.config)
    restored.load_state_dict(full.state_dict(), strict=True)
    torch.testing.assert_close(restored(f).summary, actual.summary, atol=0, rtol=0)


def test_feature_macro_is_source_balanced_and_never_hides_missing_rows():
    rows = [dict(dataset='a', **{'spatial/cosine': .8})] * 3
    rows += [dict(dataset='b', **{'spatial/cosine': .4})]
    result = dataset_metrics(rows)
    assert abs(result['macro/spatial/cosine'] - .6) < 1e-12
    assert abs(result['dataset/a/spatial/cosine'] - .8) < 1e-12
    assert 'macro/spatial/cosine' not in dataset_metrics(rows + [dict(dataset='b')])


def test_joint_cleans_every_consumed_radio_input_before_projection():
    from radio_moge.adapters import zero_dominant_channels, zero_dominant_summary_channels

    spec = next(x for x in attention_candidates() if x['variant'] == 'qknorm' and x['seed'] == 43)
    spec = dict(spec, width=32, blocks=2, heads=4)
    dims = dict(spatial_dim=12, summary_dim=24, target_dim=6, target_summary_dim=8)
    raw = build_adapter(kind='joint_attention', spec=spec, initialization_seed=43, **dims)
    clean = build_adapter(kind='joint_attention', initialization_seed=43, **dims,
                          spec=dict(spec, summary_cleaning='per-token-zero-channel'))
    features = RadioFeatures(torch.randn(2, 12, 2, 3), torch.ones(2, 24),
                             tuple(torch.ones(2, 12, 2, 3) for _ in range(4)))
    features.summary[0, 2] = 100
    for i, f in enumerate(features.intermediates):
        f[:, i] = 100
    snapshots = [x.clone() for x in (features.spatial, features.summary, *features.intermediates)]
    manual = RadioFeatures(features.spatial, zero_dominant_summary_channels(features.summary, 12),
                           tuple(zero_dominant_channels(f) for f in features.intermediates))
    actual, expected = clean(features), raw(manual)
    torch.testing.assert_close(actual.spatial, expected.spatial, atol=0, rtol=0)
    torch.testing.assert_close(actual.summary, expected.summary, atol=0, rtol=0)
    for x, before in zip((features.spatial, features.summary, *features.intermediates), snapshots):
        assert torch.equal(x, before)
    restored = build_adapter(**clean.config)
    restored.load_state_dict(clean.state_dict(), strict=True)
    torch.testing.assert_close(restored(features).summary, actual.summary, atol=0, rtol=0)


def test_joint_adapter_trains_and_resumes_with_full_trainer(tmp_path):
    from test_training import config_for, tiny_system
    from radio_moge.train import train

    cfg = config_for(tmp_path)
    cfg['output_dir'] = str(tmp_path / 'joint')
    cfg['train'].update(steps=4, seed=43)
    cfg['data']['val_manifest'] = None
    cfg['loss'] = dict(spatial=1., summary=1., cosine=1., summary_cosine=1.)
    spec = next(x for x in attention_candidates() if x['variant'] == 'qknorm' and x['seed'] == 43)
    cfg['adapter'] = dict(kind='joint_attention', spec=dict(spec, width=32, heads=4, blocks=2),
                          initialization_seed=43)
    system, _ = tiny_system()
    adapter = build_adapter(**cfg['adapter'], spatial_dim=8, summary_dim=8,
                            target_dim=6, target_summary_dim=6)
    first = train(cfg, stop_after=2, system_adapter=(system, adapter))
    state = torch.load(first, weights_only=True)
    assert state['step'] == 2 and state['cursor'] == 4
    restored = build_adapter(**state['adapter_config'])
    final = train(cfg, resume=first, system_adapter=(system, restored))
    state = torch.load(final, weights_only=True)
    assert state['step'] == 4 and state['cursor'] == 8
    assert all(torch.isfinite(v).all() for v in state['adapter'].values())
    assert all(p.grad is None and not p.requires_grad for m in
               (system.radio, system.encoder, system.decoder) for p in m.parameters())
