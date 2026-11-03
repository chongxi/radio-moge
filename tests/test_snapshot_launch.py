from copy import deepcopy
import json

import pytest

from launch_snapshot_training import file_ref, publish, run, validate_bindings


def bound():
    def ref(name):
        return dict(path='/fixture/' + name, sha256='a' * 64)
    request = dict(training_authorized_after_checks=True, training_config=ref('config'),
                   initial_checkpoint=ref('checkpoint'), source_step=232288, snapshot=ref('snapshot'))
    snapshot = dict(dataset_counts={'a': 3, 'b': 2}, processed_packs=2, processed_entries=5)
    ready = {k: deepcopy(request[k]) for k in ('training_config', 'initial_checkpoint', 'source_step', 'snapshot')}
    ready['checks'] = {k: ref(k) for k in ('source_quality', 'coverage', 'heldout_isolation', 'loss_tests')}
    scope = dict(kind='frozen_candidate_subset', snapshot=ref('snapshot'), expected_counts=snapshot['dataset_counts'])
    payload = dict(stage='snapshot_payload_verified_not_accepted', mode='census',
        snapshot=ref('snapshot'), all_snapshot_packs_verified=True,
        all_snapshot_members_verified=True, failed_packs=0, verified_packs=2, verified_members=5)
    return request, ready, payload, snapshot, scope


def test_missing_readiness_waits_without_importing_or_starting_models(tmp_path, monkeypatch):
    config = tmp_path / 'config.json'
    publish(config, dict(data={}))
    snap = tmp_path / 'snapshot.json'
    publish(snap, dict(test_fixture=True))
    request = tmp_path / 'request.json'
    publish(request, dict(training_config=file_ref(config), snapshot=file_ref(snap), source_step=232288,
        training_authorized_after_checks=True, launch_directory=str(tmp_path / 'launch'),
        readiness_plan=str(tmp_path / 'absent.json')))
    def forbidden(*args, **kwargs):
        pytest.fail('No model or trainer may start before readiness exists')
    monkeypatch.setattr('subprocess.run', forbidden)
    state = run(request, once=True)
    assert state['stage'] == 'waiting_for_data_and_loss_checks'
    assert state['optimizer_updates'] == 0 and not state['gpu_used']
    assert not state['training_started'] and not state['training_admission']


def test_full_binding_passes_but_changed_checkpoint_config_scope_or_gates_fail():
    args = bound()
    validate_bindings(*args)
    for key in ('training_config', 'initial_checkpoint', 'source_step', 'snapshot'):
        changed = deepcopy(args)
        changed[1][key] = 'changed'
        with pytest.raises(ValueError, match='Readiness changed'):
            validate_bindings(*changed)
    changed = deepcopy(args)
    changed[4]['expected_counts'] = {'a': 2, 'b': 2}
    with pytest.raises(ValueError, match='Readiness changed'):
        validate_bindings(*changed)
    changed = deepcopy(args)
    changed[1]['checks'].pop('heldout_isolation')
    with pytest.raises(ValueError, match='four nonmodel gates'):
        validate_bindings(*changed)


def test_preflight_or_incomplete_census_never_starts_the_model():
    for change in (dict(mode='preflight'), dict(verified_packs=1), dict(verified_members=4),
                   dict(failed_packs=1), dict(all_snapshot_members_verified=False)):
        args = deepcopy(bound())
        args[2].update(change)
        with pytest.raises(ValueError, match='Every requested snapshot payload'):
            validate_bindings(*args)


def test_changed_request_config_is_rejected_before_queue_creation(tmp_path):
    config = tmp_path / 'config.json'
    publish(config, dict(data={}))
    request = tmp_path / 'request.json'
    publish(request, dict(training_config=file_ref(config)))
    config.write_text(json.dumps(dict(data={'geometry_corpus': 'unreviewed'})))
    with pytest.raises(ValueError, match='evidence changed'):
        run(request, once=True)
