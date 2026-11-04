"""Tiny CPU checks of controlled losses/streams; these never admit real models."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from radio_moge.corpus_admission import (file_ref, seal_acceptance, training_contract_hash)
from radio_moge.corpus_control import InputAudit, input_fingerprint
from radio_moge.corpus_model_probe import backward_case, observe_model
from radio_moge.geometry_cache import publish_json
from radio_moge.losses import FeatureStatistics
from radio_moge.official_gt import OFFICIAL_GT_WEIGHTS
from radio_moge.train import train, validate_config
from probe_corpus_model import probe_batches
from test_corpus_admission import synthetic_seal
from test_corpus_model_probe import fixture
from test_corpus_losses import small_official as small_official
from test_training import tiny_system


def controlled(config, multiplier):
    value = deepcopy(config)
    value['corpus_loss_control'] = dict(geometry_multiplier=multiplier, record_inputs=True)
    value['loss'].update({k: w * multiplier for k, w in OFFICIAL_GT_WEIGHTS.items()})
    return value


def test_binary_control_requires_explicit_audited_corpus_and_pinned_coefficients(tmp_path):
    _, _, original, _, _ = fixture(tmp_path)
    for multiplier in (0, 1):
        config = controlled(original, multiplier)
        validate_config(config, corpus_preflight=True)
        bad = deepcopy(config)
        bad['loss']['gt_global'] = .3
        with pytest.raises(ValueError, match='pinned'):
            validate_config(bad, corpus_preflight=True)
        for change in [dict(geometry_multiplier=.5, record_inputs=True),
                       dict(geometry_multiplier=False, record_inputs=True),
                       dict(geometry_multiplier=0, record_inputs=False),
                       dict(geometry_multiplier=0)]:
            with pytest.raises(ValueError, match='controls'):
                validate_config(dict(config, corpus_loss_control=change), corpus_preflight=True)
    disabled = controlled(original, 0)
    disabled.pop('corpus_loss_control')
    with pytest.raises(ValueError, match='pinned'):
        validate_config(disabled, corpus_preflight=True)
    with pytest.raises(ValueError, match='controls'):
        validate_config(controlled(original, 0))


@pytest.mark.upstream
@pytest.mark.usefixtures('small_official')
def test_disabled_geometry_has_zero_gradient_and_live_feature_branch(tmp_path):
    _, _, config, selected, replay = fixture(tmp_path)
    config = controlled(config, 0)
    system, adapter = tiny_system()
    result = observe_model(system, adapter, FeatureStatistics(6, 6),
                           probe_batches(selected, replay, config), config)
    assert result['disabled_geometry_exact_zero'] and result['geometry_control_multiplier'] == 0
    assert result['state_before'] == result['state_after']
    assert all(r['geometry_weighted'] == r['geometry_gradient_at_adapter_outputs'] == 0
               and r['gradient']['norm'] > 0 for r in result['rows'])


@pytest.mark.upstream
@pytest.mark.usefixtures('small_official')
def test_changing_only_geometry_does_not_change_disabled_adapter_gradient(tmp_path):
    _, _, config, selected, replay = fixture(tmp_path)
    config = controlled(config, 0)
    batch = next(probe_batches(selected, replay, config))
    modified = deepcopy(batch)
    modified['geometry_targets'][0]['points'] *= 7
    results, gradients = [], []
    for data in (batch, modified):
        system, adapter = tiny_system()
        results.append(backward_case(system, adapter, FeatureStatistics(6, 6), data, config, 0))
        gradients.append([p.grad.clone() if p.grad is not None else None for p in adapter.parameters()])
    assert results[0]['geometry_weighted'] == results[1]['geometry_weighted'] == 0
    for a, b in zip(*gradients):
        assert (a is None and b is None) or torch.equal(a, b)


def test_actual_input_hash_covers_targets_replay_and_does_not_consume_rng(tmp_path):
    _, _, config, selected, replay = fixture(tmp_path)
    batch = list(probe_batches(selected, replay, config))[1]
    rng = torch.get_rng_state().clone()
    np_rng = np.random.get_state()
    value = input_fingerprint(batch, batch['replay_image'])
    assert torch.equal(rng, torch.get_rng_state())
    assert np.array_equal(np_rng[1], np.random.get_state()[1])
    for field in ('points', 'intrinsics', 'query_uv', 'native_index'):
        changed = deepcopy(batch)
        a = changed['geometry_targets'][0][field]
        a[(0,) * a.ndim] += 1
        assert input_fingerprint(changed, changed['replay_image']) != value
    assert input_fingerprint(batch) != value
    audit = InputAudit(tmp_path / 'inputs.jsonl')
    with pytest.raises(ValueError, match='contiguous'):
        audit.add(1, batch, batch['replay_image'])
    audit.add(0, batch)
    audit.add(1, batch, batch['replay_image'])
    assert audit.report()['microbatches'] == 2
    assert audit.report()['replay_microbatches'] == 1
    audit.close()


@pytest.mark.upstream
@pytest.mark.usefixtures('small_official')
def test_fresh_train_controls_share_inputs_rng_and_initial_state_but_not_updates(tmp_path):
    root, accounting, base, _, _ = fixture(tmp_path, pseudo_source='irs')
    base['train'].update(steps=2, accumulation=2, eval_every=2, save_every=2)
    system, adapter = tiny_system()
    initial = root / 'model-fixture.pt'
    torch.save(dict(step=123, adapter_config=adapter.config, adapter=adapter.state_dict(),
        model_config=system.model_config, config=deepcopy(base),
        statistics=FeatureStatistics(6, 6).state_dict()), initial)
    _, original_checks, _ = synthetic_seal(root, accounting, base, initial)
    states, first_states, audits = [], [], []
    for multiplier in (0, 1):
        config = controlled(base, multiplier)
        config['data'].pop('geometry_corpus')
        config['output_dir'] = str(root / f'run-{multiplier}')
        proof = json.loads(Path(original_checks['model_zero_update']['path']).read_text())
        proof.update(training_contract_sha256=training_contract_hash(config),
            geometry_control_multiplier=multiplier, disabled_geometry_exact_zero=not multiplier)
        model_path = root / f'model-proof-{multiplier}.json'
        publish_json(model_path, proof)
        checks = dict(original_checks, model_zero_update=file_ref(model_path))
        gate_path = root / f'gate-{multiplier}.json'
        seal_acceptance(file_ref(root / 'account/report.json'), checks, config, gate_path)
        config['data']['geometry_corpus'] = file_ref(gate_path)
        final = train(config, init=initial, system_adapter=tiny_system())
        states.append(torch.load(final, weights_only=True))
        first_states.append(torch.load(final.parent / 'initial.pt', weights_only=True))
        audits.append(json.loads((final.parent / 'control-input-audit.json').read_text()))
        rows = [json.loads(line) for line in (final.parent / 'train.jsonl').read_text().splitlines()]
        assert all((r['geometry_weighted'] == 0) == (multiplier == 0) for r in rows)
        assert all(r['gradient_norm'] > 0 for r in rows)
        with pytest.raises(ValueError, match='fresh'):
            train(config, resume=final, system_adapter=tiny_system())
    assert audits[0]['records']['sha256'] == audits[1]['records']['sha256']
    assert audits[0]['microbatches'] == audits[1]['microbatches'] == 4
    assert first_states[0]['optimizer']['state'] == first_states[1]['optimizer']['state'] == {}
    assert first_states[0]['cursor'] == first_states[1]['cursor'] == 0
    from verify_corpus_stage_startup import equal_finite
    assert equal_finite(first_states[0]['rng'], first_states[1]['rng'])
    for section in ('adapter', 'statistics'):
        for key, value in first_states[0][section].items():
            assert torch.equal(value, first_states[1][section][key])
    assert any(not torch.equal(v, states[1]['adapter'][k]) for k,v in states[0]['adapter'].items())


def test_nonzero_geometry_model_proof_cannot_admit_disabled_control(tmp_path):
    root, accounting, base, _, _ = fixture(tmp_path)
    _, checks, _ = synthetic_seal(root, accounting, base)
    config = controlled(base, 0)
    proof = json.loads(Path(checks['model_zero_update']['path']).read_text())
    proof['training_contract_sha256'] = training_contract_hash(config)
    path = root / 'incorrect-control-proof.json'
    publish_json(path, proof)
    checks['model_zero_update'] = file_ref(path)
    with pytest.raises(ValueError, match='geometry control'):
        seal_acceptance(file_ref(root / 'account/report.json'), checks, config, root / 'bad-seal.json')
