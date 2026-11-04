from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from radio_moge.checkpoint import rng_state
from radio_moge.data import read_manifest, DeterministicBatches
from radio_moge.data_fork import check_data_fork
from radio_moge.prefetch import PreparedWindows
from radio_moge.rgb_input_audit import RGBInputAudit, pixels_sha
from radio_moge.corpus_admission import file_ref
from radio_moge.losses import FeatureStatistics
from radio_moge.train import make_loader, train
from run_feature_reference import ReferenceAudit, checkpoint_ref, observe_reference_model
from verify_corpus_stage_startup import equal_finite
from test_performance import feature_config, profile
from test_training import tiny_system


def setup(tmp_path):
    config = feature_config(tmp_path)
    config['data'].update(buckets=[[32, 48], [48, 32]], compact_transport=True)
    config['train'].update(eval_at_start=False, eval_every=100, log_every=1, seed=43)
    return config, read_manifest(config['data']['train_manifest'])


def test_observation_preserves_actual_pixels_grouping_rng_and_discards_prefetch(tmp_path):
    config, records = setup(tmp_path)
    config['data']['workers'] = 2
    keys = [list(k) for batch in DeterministicBatches(6, 4, config['data']['buckets'], 43, 0, 12) for k in batch]
    audit = RGBInputAudit(tmp_path, 2, 4, 4, 8, keys)
    before = rng_state()
    plain = PreparedWindows(make_loader(records, config, 0, 12), 4, 8, 3)
    observed = PreparedWindows(make_loader(records, config, 0, 12, reference=audit),
                               4, 8, 3, observer=audit.observe_window)
    try:
        for n in range(2):
            a, b = next(plain), next(observed)
            assert len(a) == len(b)
            assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
            audit.commit(3+n, (n+1)*4)
        # The third prepared window never receives an optimizer update.
    finally:
        plain.close()
        observed.close()
    report = audit.finish(4, 8)
    assert report['presentations'] == 32 and report['optimizer_updates'] == 2
    rows = [json.loads(x) for x in audit.path.read_text().splitlines()]
    assert [x['update'] for x in rows] == [3, 4]
    assert [r['key'] for x in rows for r in x['logical_images']] == keys[:32]
    assert equal_finite(before, rng_state())


@pytest.mark.parametrize('kind', ['keys', 'pixels', 'order'])
def test_rejects_changed_actual_input_or_order(tmp_path, kind):
    config, records = setup(tmp_path)
    audit = RGBInputAudit(tmp_path, 0, 4, 1, 4,
        [[999, 0, 32, 48]]*4 if kind == 'keys' else None)
    # FloatLoader.loader is the observed compact loader, prior to normalization.
    image = next(iter(make_loader(records, config, 0, 1, reference=audit).loader))
    if kind == 'pixels':
        image[0, 0, 0, 0] ^= 1
    elif kind == 'order':
        image = image.flip(0)
    with pytest.raises(ValueError, match='keys|pixels'):
        audit.observe_window([image])
    assert audit.committed == 0 and not audit.path.exists()


def test_zero_successful_updates_has_empty_audit(tmp_path):
    audit = RGBInputAudit(tmp_path, 2, 4, 4, 8)
    result = audit.finish(2, 0)
    assert result['presentations'] == 0 and audit.path.read_text() == ''


class SmallReference(RGBInputAudit):
    check_fork = staticmethod(check_data_fork)

    def before_updates(self, system, adapter, statistics, records, config, output):
        self.gate_was_called = True
        if getattr(self, 'reject_gate', False):
            raise ValueError('Injected pre-update rejection')


def test_observational_hooks_leave_actual_training_state_bitwise_unchanged(tmp_path):
    config, _ = setup(tmp_path)
    config['data'].pop('compact_transport')
    config['output_dir'] = str(tmp_path / 'parent')
    source = train(config, stop_after=2, system_adapter=tiny_system())
    config['train'].update(lr_schedule='constant', steps=4, max_seconds=600)
    performance = dict(profile(2, 0), data_pipeline='prepared', compact_transport=True,
                       gpu_prefetch=False)
    config['output_dir'] = str(tmp_path / 'plain')
    a = torch.load(train(config, fork_from=source, data_fork=True, performance=performance,
                         system_adapter=tiny_system()), weights_only=True)
    config['output_dir'] = str(tmp_path / 'observed')
    observer = SmallReference(config['output_dir'], 2, 4, 4, 8)
    b = torch.load(train(config, fork_from=source, data_fork=True, performance=performance,
                         system_adapter=tiny_system(), reference=observer), weights_only=True)
    assert observer.gate_was_called and b['step'] == a['step'] == 4
    for name in ('adapter', 'statistics', 'optimizer', 'scheduler', 'rng'):
        assert equal_finite(a[name], b[name]), name
    assert json.loads((tmp_path / 'observed/fork-state.json').read_text())['data_change']['sampling'] == 'permutation'


def test_pre_update_failure_prohibits_optimizer_progress(tmp_path):
    config, _ = setup(tmp_path)
    config['data'].pop('compact_transport')
    config['output_dir'] = str(tmp_path / 'parent')
    source = train(config, stop_after=2, system_adapter=tiny_system())
    config['train'].update(lr_schedule='constant', steps=4, max_seconds=600)
    config['output_dir'] = str(tmp_path / 'rejected')
    observer = SmallReference(config['output_dir'], 2, 4, 4, 8)
    observer.reject_gate = True
    with pytest.raises(ValueError, match='pre-update rejection'):
        train(config, fork_from=source, data_fork=True,
            performance=dict(profile(2, 0), data_pipeline='prepared', compact_transport=True),
            system_adapter=tiny_system(), reference=observer)
    initial = torch.load(tmp_path / 'rejected/initial.pt', weights_only=True)
    assert initial['step'] == 2 and initial['cursor'] == 0
    assert not (tmp_path / 'rejected/train.jsonl').exists()
    assert observer.committed == 0


def test_reference_fork_only_relaxes_two_observational_settings(tmp_path):
    config, _ = setup(tmp_path)
    parent = deepcopy(config)
    config['train'].update(lr_schedule='constant', steps=5, max_seconds=600)
    state = dict(config=parent, step=2, data_digests=dict(train='old', val='same'),
                 optimizer=dict(param_groups=[dict(lr=config['train']['lr'])]))
    reference = object.__new__(ReferenceAudit)
    reference.config = config
    reference.check_fork(config, state, dict(train='new', val='same'))
    changed = deepcopy(config)
    changed['loss']['cosine'] = .1
    reference.config = changed
    with pytest.raises(ValueError, match='identical'):
        reference.check_fork(changed, state, dict(train='new', val='same'))


def test_pixel_hash_observes_tensor_values():
    image = torch.zeros(3, 5, 4, dtype=torch.uint8)
    previous = pixels_sha(image)
    image[0, 0, 0] = 1
    assert pixels_sha(image) != previous


def test_annotated_checkpoint_reference_becomes_strict_evidence():
    assert checkpoint_ref(dict(source_checkpoint=dict(path='/x', sha256='a'*64,
                                                      payload_reread=False))) == dict(path='/x', sha256='a'*64)


@pytest.mark.parametrize('mutate', [False, True])
def test_original_feature_model_probe_preserves_rng_and_detects_state_changes(tmp_path, monkeypatch, mutate):
    config, original = setup(tmp_path)
    config['data']['buckets'] = [[32, 48], [48, 32], [32, 32]]
    records = [dict(original[i % 6], dataset=f'source{i//4}',
                    sha256=file_ref(original[i % 6]['image'])['sha256']) for i in range(32)]
    planned = [dict(index=i, augmentation_seed=43+i,
                    height=config['data']['buckets'][(i//4) % 3][0],
                    width=config['data']['buckets'][(i//4) % 3][1]) for i in range(32)]
    system, adapter = tiny_system()
    stats = FeatureStatistics(6, 6)
    if mutate:
        stats.register_buffer('counter', torch.tensor(0))
        original_loss = stats.loss
        def loss(*args):
            stats.counter.add_(1)
            return original_loss(*args)
        monkeypatch.setattr(stats, 'loss', loss)
    before = rng_state()
    def no_optimizer(*args, **kwargs):
        raise AssertionError('Zero-update gate must not construct another optimizer')
    monkeypatch.setattr(torch.optim, 'AdamW', no_optimizer)
    old_data = Path(__file__).resolve().parents[1] / 'src/radio_moge/data.py'
    if mutate:
        with pytest.raises(ValueError, match='changed model state'):
            observe_reference_model(system, adapter, stats, records, config, planned, old_data)
    else:
        proof = observe_reference_model(system, adapter, stats, records, config, planned, old_data)
        assert proof['state_before'] == proof['state_after']
        assert proof['images'] == 128 and proof['cases'] == 27
        assert proof['optimizer_updates'] == 0 and proof['original_execution_pixel_parity']
    assert equal_finite(before, rng_state())
    assert all(p.grad is None for p in adapter.parameters())
