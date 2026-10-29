from copy import deepcopy

import pytest
import torch

from radio_moge.data import read_manifest
from radio_moge.prefetch import PreparedWindows
from radio_moge.performance import grouped_window
from radio_moge.train import make_loader, train
from test_performance import feature_config, profile
from test_training import tiny_system


@pytest.mark.parametrize('workers', [0, 2])
@pytest.mark.parametrize('compact', [False, True])
def test_prepared_real_augmented_pixels_exact_with_resume_cursor(tmp_path, workers, compact):
    cfg = feature_config(tmp_path)
    cfg['data'].update(workers=workers, buckets=[[32, 48], [48, 32]])
    records = read_manifest(cfg['data']['train_manifest'])
    loader = make_loader(records, cfg, 19, 24)
    iterator = iter(loader)
    expected = [list(grouped_window(iterator, 8, 16)) for _ in range(3)]
    cfg['data']['compact_transport'] = compact
    prepared = PreparedWindows(make_loader(records, cfg, 19, 24), 8, 16, 3)
    try:
        for reference, actual in zip(expected, prepared, strict=True):
            assert len(reference) == len(actual)
            for a, b in zip(reference, actual, strict=True):
                assert torch.equal(a, b)
    finally:
        prepared.close()
    assert not prepared.thread.is_alive()


def test_prefetch_shutdown_discards_unused_windows_and_forwards_errors():
    tensors = [torch.ones(4, 3, 8, 8) for _ in range(80)]
    prepared = PreparedWindows(tensors, 8, 16, 10)
    next(prepared)
    prepared.close()
    assert not prepared.thread.is_alive()

    def broken():
        yield torch.ones(4, 3, 8, 8)
        raise ValueError('decode failed')

    prepared = PreparedWindows(broken(), 8, 16, 1)
    with pytest.raises(ValueError, match='decode failed'):
        next(prepared)
    assert not prepared.thread.is_alive()


def test_pipeline_only_resume_preserves_exact_adam_weights_lr_and_cursor(tmp_path):
    cfg = feature_config(tmp_path)
    cfg['data']['buckets'] = [[32, 48], [48, 32]]
    cfg['output_dir'] = str(tmp_path / 'parent')
    checkpoint = train(cfg, stop_after=2, system_adapter=tiny_system())
    states = []
    perf = profile(2, 8, effective=32, compute=16)
    for mode in ['legacy', 'prepared']:
        local = deepcopy(cfg)
        local['output_dir'] = str(tmp_path / mode)
        result = train(local, resume=checkpoint, stop_after=4, system_adapter=tiny_system(),
                       performance=dict(perf, data_pipeline=mode, compact_transport=mode == 'prepared'))
        states.append(torch.load(result, weights_only=True))
    a, b = states
    assert a['step'] == b['step'] == 4 and a['cursor'] == b['cursor'] == 24
    assert a['scheduler'] == b['scheduler']
    for key in a['adapter']:
        assert torch.equal(a['adapter'][key], b['adapter'][key])
    for i, row in a['optimizer']['state'].items():
        for key, value in row.items():
            assert torch.equal(value, b['optimizer']['state'][i][key])


def test_disposable_pipeline_probe_restores_full_state(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from radio_moge.data_pipeline_probe import probe
    from radio_moge.losses import FeatureStatistics
    from radio_moge.checkpoint import rng_state
    import radio_moge.train as training
    cfg = feature_config(tmp_path)
    cfg['output_dir'] = str(tmp_path / 'parent')
    state = torch.load(train(cfg, stop_after=2, system_adapter=tiny_system()), weights_only=True)
    state['performance'] = profile(2, 8, 32, 16, 4)
    system, adapter = tiny_system()
    stats = FeatureStatistics(6, 6)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['train']['lr'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    scheduler.load_state_dict(deepcopy(state['scheduler']))
    device = SimpleNamespace(device=torch.device('cpu'), backward=lambda x: x.backward(),
                             clip_grad_norm_=torch.nn.utils.clip_grad_norm_)
    for name in ('synchronize', 'empty_cache', 'reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 100)
    # Multiprocess pixel identity is tested separately; keep this 144-update test small.
    original_loader = make_loader

    def small_loader(records, config, *args):
        config = deepcopy(config)
        config['data']['workers'] = 0
        return original_loader(records, config, *args)

    monkeypatch.setattr(training, 'make_loader', small_loader)
    output = tmp_path / 'probe'
    output.mkdir()
    result = probe(system, adapter, adapter, stats, optimizer, scheduler, device, nullcontext,
                   read_manifest(cfg['data']['train_manifest']), cfg, state,
                   dict(state['performance'], compact_transport=True), output)
    assert all(result[k] == v for k, v in state['performance'].items())
    assert result['compact_transport'] is True
    for k, v in adapter.state_dict().items():
        assert torch.equal(v, state['adapter'][k])
    for i, row in optimizer.state_dict()['state'].items():
        for key, value in row.items():
            assert torch.equal(value, state['optimizer']['state'][i][key])
    assert scheduler.state_dict() == state['scheduler']
    assert torch.equal(rng_state()['torch'], state['rng']['torch'])
