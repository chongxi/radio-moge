import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'scripts' / (name+'.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_continuation_gate_rejects_old_domain_regression_despite_new_domain_gain():
    gate = module('audit_data_evaluations').continuation_gate
    base = {'old/abs_rel': .07, 'new/cosine': .95}
    policy = {'old/abs_rel': {'max_ratio': 1.03}, 'new/cosine': {'max_drop': .002}}
    assert gate(base, {'old/abs_rel': .071, 'new/cosine': .96}, policy)['stage'] == 'passed'
    assert gate(base, {'old/abs_rel': .08, 'new/cosine': .98}, policy)['stage'] == 'needs_inspection'
    with pytest.raises(ValueError):
        gate(base, {'old/abs_rel': float('nan'), 'new/cosine': .96}, policy)
    with pytest.raises(KeyError):
        gate(base, {'old/abs_rel': .07}, policy)


def test_controller_budget_and_output_cannot_silently_change():
    validate = module('run_data_continuation').validate_plan
    plan = dict(authorized=True, additional_updates=3000, parent_step=93367,
                learning_rate=1e-5, max_seconds=10800, effective_batch=32, remote_root='/new')
    config = dict(data={'buckets': [[480, 640], [640, 480], [400, 640]]}, output_dir='/new/train',
                  train=dict(steps=96367, lr=1e-5, lr_schedule='constant', max_seconds=10800))
    validate(plan, config, dict(effective_batch=32))
    config['output_dir'] = '/old/train'
    with pytest.raises(ValueError):
        validate(plan, config, dict(effective_batch=32))


def test_completed_epoch_source_binding_rejects_wrong_final_step(tmp_path):
    controller = module('run_data_continuation')
    checkpoint = tmp_path / 'train/latest.pt'
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b'fixture')
    completion = tmp_path / 'full-epoch-completion-verification.json'
    completion.write_text(json.dumps(dict(stage='passed', final_step=134025,
                                         normal_exit_code=0, checks={'coverage': True})))
    source = tmp_path / 'source-verification.json'
    identity = controller.identity(checkpoint)
    source.write_text(json.dumps(dict(stage='passed', files=[dict(relative='train/latest.pt',
        identity=identity, checkpoint=dict(step=134025))])))
    plan = dict(parent_step=134025, parent_identity=identity,
                source_verification=str(source), source_completion_verification=str(completion))
    assert controller.validate_source(plan, checkpoint) == tmp_path
    plan['parent_step'] = 94367
    with pytest.raises(ValueError, match='final successful step'):
        controller.validate_source(plan, checkpoint)


def test_separate_sensor_validation_uses_its_own_preparation_digest(tmp_path):
    validate = module('run_data_continuation').validate_dataset_views
    config = {'evaluation': {'dataset_variants': {}}}
    plan = {'validation_bindings': {}}
    for name in ['dslr', 'lidar']:
        manifest = tmp_path / (name + '.jsonl')
        manifest.write_text(name)
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        proof = tmp_path / (name + '-proof.json')
        proof.write_text(json.dumps(dict(stage='passed', hashes={'manifest': digest})))
        config['evaluation']['dataset_variants'][name] = dict(manifest=str(manifest))
        plan['validation_bindings'][name] = dict(manifest=str(manifest), sha256=digest,
            preparation_proof=str(proof), digest_key_path=['hashes', 'manifest'])
    prepared = dict(manifest_sha256={'val-scannetpp.jsonl': 'old-dslr-digest'})
    def hash_file(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    validate(plan, config, prepared, hash_file)
    (tmp_path / 'lidar.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='manifest changed'):
        validate(plan, config, prepared, hash_file)
    del plan['validation_bindings']['lidar']
    with pytest.raises(ValueError, match='Each validation view'):
        validate(plan, config, prepared, hash_file)
