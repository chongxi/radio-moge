"""Tiny CPU wiring/negative tests; never evidence of an actual-model preflight."""
import json

import pytest
import torch

from radio_moge.corpus_admission import (build_accounting, file_ref, readonly_db,
                                        implementation_hashes, validate_pre_model_evidence)
from radio_moge.corpus_model_probe import module_digest, observe_model, select_probe_entries
from radio_moge.losses import FeatureStatistics
from radio_moge.official_gt import OFFICIAL_GT_WEIGHTS
from radio_moge.train import validate_config
from probe_corpus_model import probe_batches
from test_corpus_admission import admission_fixture, synthetic_seal
from test_corpus_losses import small_official as small_official
from test_training import config_for, tiny_system


def fixture(tmp_path):
    root, plan, rows = admission_fixture(tmp_path)
    accounting = build_accounting(plan, root / 'account')
    config = config_for(tmp_path)
    config['data']['train_manifest'] = str(root / 'train.jsonl')
    config['geometry_supervision'] = dict(target='offline_corpus', camera='provenance_aware',
                                          objective='moge2_core')
    config['feature_replay'] = dict(manifest=str(root / 'train.jsonl'), fraction=.5, seed=100043)
    config['loss'] = dict(spatial=.1, summary=.1, cosine=.1, summary_cosine=.1, **OFFICIAL_GT_WEIGHTS)
    config['train']['batch_size'] = 1
    with readonly_db(accounting['index']['path']) as db:
        selected = select_probe_entries(db)
    return root, accounting, config, selected, rows


def test_preflight_config_is_observational_and_does_not_authorize_train(tmp_path):
    _, _, config, _, _ = fixture(tmp_path)
    validate_config(config, corpus_preflight=True)
    with pytest.raises(ValueError, match='Unsupported GT'):
        validate_config(config)
    del config['feature_replay']
    with pytest.raises(ValueError, match='RGB replay'):
        validate_config(config, corpus_preflight=True)


def test_full_nonmodel_evidence_required_without_fake_training_seal(tmp_path):
    root, accounting, config, _, _ = fixture(tmp_path)
    _, checks, _ = synthetic_seal(root, accounting, config)
    checks.pop('model_zero_update')
    result, _ = validate_pre_model_evidence(file_ref(root / 'account/report.json'), checks)
    assert result['training_admission'] is False
    with pytest.raises(ValueError, match='before model preflight'):
        validate_pre_model_evidence(file_ref(root / 'account/report.json'),
                                   {k: v for k, v in checks.items() if k != 'source_quality'})
    proof = json.loads((root / 'loss_tests.json').read_text())
    proof['implementation_sha256']['train.py'] = '0' * 64
    (root / 'loss_tests.json').write_text(json.dumps(proof))
    checks['loss_tests'] = file_ref(root / 'loss_tests.json')
    with pytest.raises(ValueError, match='different implementation'):
        validate_pre_model_evidence(file_ref(root / 'account/report.json'), checks)


def test_digest_includes_bfloat16_parameters_and_buffers():
    module = torch.nn.Linear(3, 4).to(torch.bfloat16)
    module.register_buffer('counter', torch.tensor(0))
    before = module_digest(module)
    module.counter.add_(1)
    assert module_digest(module) != before


def test_model_gate_binds_feature_objective_and_actual_joint_adapter_code():
    assert {'losses.py', 'adapters.py', 'architecture_screen.py', 'spatial_heads.py',
            'upstream.py', 'config.py'} <= set(implementation_hashes())


@pytest.mark.upstream
@pytest.mark.usefixtures('small_official')
def test_real_loss_through_tiny_models_preserves_weights_and_covers_replay(tmp_path, monkeypatch):
    _, _, config, selected, replay = fixture(tmp_path)
    system, adapter = tiny_system()
    statistics = FeatureStatistics(6, 6)
    def forbid_optimizer(*args, **kwargs):
        raise AssertionError('An observational probe must never construct an optimizer')
    monkeypatch.setattr(torch.optim, 'AdamW', forbid_optimizer)
    result = observe_model(system, adapter, statistics, probe_batches(selected, replay, config), config)
    assert len(result['rows']) == 4 and result['optimizer_updates'] == 0
    assert not result['optimizer_constructed']
    assert result['state_before'] == result['state_after']
    assert all(row['gradient']['norm'] > 0 for row in result['rows'])
    assert all(row['geometry_gradient_at_adapter_outputs'] > 0 for row in result['rows'])
    assert {r['identities'][0]['dataset'] for r in result['rows']} == {'hypersim', 'coco2017'}
    assert {r['identities'][0]['replay']['dataset'] for r in result['rows']
            if r['feature_branch'] == 'replay_rgb'} == {'hypersim', 'coco2017'}
    for row in result['rows']:
        pseudo = row['identities'][0]['label_kind'] == 'teacher_pseudo'
        assert row['losses']['gt_metric_images'] == int(not pseudo)
        assert row['total'] == pytest.approx(row['geometry_weighted'] + row['feature_weighted'])
    assert all(p.grad is None for p in adapter.parameters())
    assert 'actual_model_forward_backward' not in result  # Only actual-model CLI can attest this.


@pytest.mark.upstream
@pytest.mark.usefixtures('small_official')
def test_buffer_mutation_is_not_mistaken_for_zero_update(tmp_path, monkeypatch):
    _, _, config, selected, replay = fixture(tmp_path)
    system, adapter = tiny_system()
    statistics = FeatureStatistics(6, 6)
    statistics.register_buffer('mutating_counter', torch.tensor(0))
    original = statistics.loss
    def mutation(*args):
        statistics.mutating_counter.add_(1)
        return original(*args)
    monkeypatch.setattr(statistics, 'loss', mutation)
    with pytest.raises(ValueError, match='parameters or buffers changed'):
        observe_model(system, adapter, statistics, probe_batches(selected, replay, config), config)


@pytest.mark.upstream
@pytest.mark.usefixtures('small_official')
def test_detached_decoder_is_detected_even_when_feature_loss_has_gradients(tmp_path, monkeypatch):
    _, _, config, selected, replay = fixture(tmp_path)
    system, adapter = tiny_system()
    original = system.decoder.forward
    monkeypatch.setattr(system.decoder, 'forward',
                        lambda *args: {k: v.detach() for k, v in original(*args).items()})
    with pytest.raises(ValueError, match='Geometry loss is detached'):
        observe_model(system, adapter, FeatureStatistics(6, 6), probe_batches(selected, replay, config), config)
