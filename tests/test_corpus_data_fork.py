"""Tiny CPU optimizer tests; synthetic seals never authorize real training."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from launch_snapshot_training import validate_bindings
from radio_moge.corpus_admission import (build_accounting, file_ref, seal_acceptance)
from radio_moge.corpus_losses import sample_predictions, target_tensors
from radio_moge.losses import FeatureStatistics
from radio_moge.official_gt import OFFICIAL_GT_KEYS, OFFICIAL_GT_WEIGHTS
from radio_moge.train import check_corpus_data_fork, train
from test_corpus_admission import admission_fixture, synthetic_seal
from test_training import config_for, tiny_system


def prepare(tmp_path, monkeypatch):
    original = tmp_path / 'old'
    expanded = tmp_path / 'expanded'
    original.mkdir()
    expanded.mkdir()
    root, plan, _ = admission_fixture(original)
    accounting = build_accounting(plan, root / 'account')
    config = config_for(tmp_path)
    config['data']['train_manifest'] = str(root / 'train.jsonl')
    config['geometry_supervision'] = dict(target='offline_corpus', camera='provenance_aware',
                                          objective='moge2_core')
    config['feature_replay'] = dict(manifest=str(root / 'train.jsonl'), fraction=.5, seed=100043)
    config['loss'] = dict(spatial=1., summary=1., cosine=1., summary_cosine=1., **OFFICIAL_GT_WEIGHTS)
    config['train'].update(steps=2, accumulation=2)
    config['output_dir'] = str(root / 'run')
    system, adapter = tiny_system()
    initial = root / 'initial-fixture.pt'
    torch.save(dict(adapter_config=adapter.config, adapter=adapter.state_dict(),
        model_config=system.model_config, config=deepcopy(config), step=123,
        statistics=FeatureStatistics(6, 6).state_dict()), initial)
    synthetic_seal(root, accounting, config, initial)

    def test_only_loss(raw, targets, source, seed):
        losses = []
        for i, value in enumerate(targets):
            target = target_tensors(value, raw['points'].device)
            predicted = sample_predictions(raw['points'][i], target)
            losses.append((predicted[target['valid']] - target['points'][target['valid']]).square().mean())
        loss = torch.stack(losses).mean()
        return {key: loss if key == 'gt_global' else loss * 0 for key in OFFICIAL_GT_KEYS}

    monkeypatch.setattr('radio_moge.train.corpus_geometry_losses', test_only_loss)
    checkpoint = train(config, init=initial, stop_after=1, system_adapter=(system, adapter))
    parent = torch.load(checkpoint, weights_only=True)
    new_root, new_plan, _ = admission_fixture(expanded, extra_coco=True)
    new_accounting = build_accounting(new_plan, new_root / 'account')
    trial = deepcopy(config)
    trial['data'].pop('geometry_corpus')
    trial['data'].update(train_manifest=str(new_root / 'train.jsonl'),
                         mixture_weights={'hypersim': .5, 'coco2017': .5})
    trial['output_dir'] = str(new_root / 'run')
    trial['train'].update(steps=3, max_seconds=600, lr_schedule='constant')
    _, checks, _ = synthetic_seal(new_root, new_accounting, trial, checkpoint)
    proof_path = Path(checks['model_zero_update']['path'])
    proof = json.loads(proof_path.read_text())
    proof['initial_source_step'] = parent['step']
    proof_path.write_text(json.dumps(proof))
    checks['model_zero_update'] = file_ref(proof_path)
    gate_path = new_root / 'fork-acceptance.json'
    gate = seal_acceptance(file_ref(new_root / 'account/report.json'), checks, trial, gate_path)
    trial['data']['geometry_corpus'] = file_ref(gate_path)
    return checkpoint, parent, trial, gate


def test_newly_accepted_corpus_restores_adam_and_starts_new_stream(tmp_path, monkeypatch):
    checkpoint, parent, trial, gate = prepare(tmp_path, monkeypatch)
    original_bytes = file_ref(checkpoint)
    final_path = train(trial, fork_from=checkpoint, data_fork=True, system_adapter=tiny_system())
    initial = torch.load(final_path.parent / 'initial.pt', weights_only=True)
    assert initial['step'] == parent['step'] == 1 and initial['cursor'] == 0
    assert initial['fork_provenance']['data_change']['parent_cursor'] == parent['cursor'] == 2
    assert initial['data_digests']['feature_replay'] == parent['data_digests']['feature_replay']
    assert initial['data_digests']['geometry'] != parent['data_digests']['geometry']
    for key, state in parent['optimizer']['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(initial['optimizer']['state'][key][name], value, rtol=0, atol=0)
    for section in ('adapter', 'statistics'):
        for key, value in parent[section].items():
            torch.testing.assert_close(initial[section][key], value, rtol=0, atol=0)
    final = torch.load(final_path, weights_only=True)
    assert final['step'] == 3 and final['cursor'] == 4
    assert {int(v['step']) for v in final['optimizer']['state'].values()} == {3}
    assert file_ref(checkpoint) == original_bytes
    assert gate['selected_images'] == 3


def test_corpus_fork_rejects_changed_objective_replay_or_unbound_parent(tmp_path, monkeypatch):
    checkpoint, parent, trial, gate = prepare(tmp_path, monkeypatch)
    digests = dict(parent['data_digests'], train='new', geometry='new')
    check_corpus_data_fork(trial, parent, digests, gate)
    for change in ('loss', 'lr', 'replay', 'validation', 'old_gate', 'parent_digest', 'same_index', 'step'):
        config, state, actual, new_gate = deepcopy(trial), deepcopy(parent), dict(digests), deepcopy(gate)
        if change == 'loss':
            config['loss']['gt_global'] *= 2
        elif change == 'lr':
            config['train']['lr'] /= 2
        elif change == 'replay':
            config['feature_replay']['fraction'] = .25
        elif change == 'validation':
            actual['val'] = 'changed'
        elif change == 'old_gate':
            config['data']['geometry_corpus'] = state['config']['data']['geometry_corpus']
        elif change == 'parent_digest':
            state['data_digests']['geometry'] = 'unbound'
        elif change == 'same_index':
            old = json.loads(Path(state['config']['data']['geometry_corpus']['path']).read_text())
            new_gate['index'] = old['index']
        else:
            state['step'] += 1
        with pytest.raises(ValueError):
            check_corpus_data_fork(config, state, actual, new_gate)
    # A reviewed flag is not permission to bypass the exact checkpoint gate.
    with pytest.raises(ValueError, match='exact preflight initialization'):
        train(trial, fork_from=checkpoint.parent / 'initial.pt', data_fork=True,
              system_adapter=tiny_system())
    with pytest.raises(ValueError, match='implicit fork'):
        train(trial, fork_from=checkpoint, system_adapter=tiny_system())


def test_launch_readiness_cannot_silently_reset_optimizer():
    request = dict(scope_kind='reviewed_source_subset', training_authorized_after_checks=True,
        start_mode='corpus_data_fork', training_config='config', initial_checkpoint='checkpoint',
        source_step=1000, accounting_report='accounting', expected_counts={'structured3d': 1})
    ready = dict(request, checks={k: 'proof' for k in
        ('source_quality', 'coverage', 'heldout_isolation', 'loss_tests')})
    scope = dict(kind='reviewed_source_subset', expected_counts=request['expected_counts'])
    validate_bindings(request, ready, None, None, scope)
    ready.pop('start_mode')
    with pytest.raises(ValueError, match='optimizer initialization'):
        validate_bindings(request, ready, None, None, scope)
