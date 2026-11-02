"""Synthetic fixtures exercise gates; these are NOT real-corpus acceptance evidence."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image
import pytest

from radio_moge.corpus_admission import (FINAL_CHECKS, SOURCE_CHECKS, build_accounting,
    file_ref, implementation_hashes, open_training_gate, seal_acceptance,
    training_contract_hash)
from radio_moge.geometry_cache import GeometryPackWriter, publish_json, sha_file


def admission_fixture(tmp_path, *, flagged=False, missing=False):
    root = tmp_path / 'fixture'
    root.mkdir()
    rows = []
    for i, dataset in enumerate(('hypersim', 'coco2017')):
        path = root / f'rgb-{i}.png'
        Image.fromarray(np.full((24, 32, 3), 64 + i, np.uint8)).save(path)
        rows.append(dict(image=str(path), sha256=sha_file(path), dataset=dataset,
                         split='train', group=f'{dataset}:scene', height=24, width=32))
    manifest = root / 'train.jsonl'
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    manifest_hash = sha_file(manifest)
    expected = {r['dataset']: 1 for r in rows}
    scope = root / 'scope.json'
    publish_json(scope, dict(manifest_sha256=manifest_hash, expected_counts=expected))
    inventory = root / 'inventory.sqlite'
    with sqlite3.connect(inventory) as db:
        db.execute('CREATE TABLE images(ordinal INTEGER,image TEXT,sha256 TEXT,dataset TEXT,scene_group TEXT,row_json TEXT)')
        for ordinal, row in enumerate(rows):
            db.execute('INSERT INTO images VALUES (?,?,?,?,?,?)',
                       (ordinal, row['image'], row['sha256'], row['dataset'], row['group'], json.dumps(row)))
    inventory_report = root / 'inventory.json'
    publish_json(inventory_report, dict(stage='inventory_complete', manifest=str(manifest),
        manifest_sha256=manifest_hash, inventory_sha256=sha_file(inventory), dataset_counts=expected,
        images=2, duplicate_encoded_image_rows=0, heldout_checks=[]))
    evidence = root / 'synthetic-proof.json'
    publish_json(evidence, dict(test_fixture_only=True))
    exclusions = root / 'exclusions.jsonl'
    exclusions.write_text('')
    plan = dict(scope=file_ref(scope), inventory=file_ref(inventory),
                inventory_report=file_ref(inventory_report), sources=[], exclusions=file_ref(exclusions))
    for i, row in enumerate(rows[:1] if missing else rows):
        dataset = row['dataset']
        directory = root / dataset
        directory.mkdir()
        policy = dict(dataset=dataset, manifest_sha256=manifest_hash)
        writer = GeometryPackWriter(directory / 'packs', policy)
        pseudo = dataset == 'coco2017'
        shape = (24, 32) if pseudo else (12, 16)
        k = np.array([[1., 0, .5], [0, 1., .5], [0, 0, 1.]])
        kind = 'teacher_pseudo' if pseudo else 'synthetic_renderer'
        provenance = dict(image_sha256=row['sha256'], dataset=dataset, source_id='synthetic-test',
            label_kind=kind, camera_kind='teacher_estimated' if pseudo else 'synthetic_camera',
            metric_calibrated=not pseudo, conversion='synthetic test only',
            source_proofs=[dict(source=str(evidence), sha256=sha_file(evidence))],
            policy_sha256=writer.policy_hash, rgb_shape=[24, 32],
            rgb_intrinsics_normalized=k.tolist(), depth_to_rgb_uv_transform=np.eye(3).tolist(),
            source_needs_review=bool(flagged and i == 0))
        if pseudo:
            provenance['teacher_revision'] = 'test-only'
        writer.add(row, np.full(shape, 2., np.float32), np.ones(shape, bool), k, provenance)
        writer.close()
        generation = directory / 'report.json'
        publish_json(generation, dict(stage='complete', mode='generate', optimizer_updates=0,
            training_started=False, policy_sha256=writer.policy_hash, verified_cache_members=1,
            completed=1, rejected=0, passed=False, training_admission=False))
        review = directory / 'review.json'
        publish_json(review, dict(stage='source_review_complete', dataset=dataset,
            decision='eligible_with_flagged_rows_quarantined', policy_sha256=writer.policy_hash,
            manifest_sha256=manifest_hash, label_kinds=[kind],
            checks=dict.fromkeys(SOURCE_CHECKS, True), evidence=[file_ref(evidence)]))
        plan['sources'].append(dict(dataset=dataset, generation_report=file_ref(generation),
            review=file_ref(review), packs_index=file_ref(directory / 'packs/packs.json'), scope_count=1))
    return root, plan, rows


def synthetic_seal(root, accounting, config):
    """Fake proof statements ONLY for exercising integrity/negative unit tests."""
    checkpoint = root / 'test-checkpoint.bytes'
    checkpoint.write_bytes(b'Not a model; unit test fixture only')
    ref = file_ref(root / 'account/report.json')
    evidence = [file_ref(root / 'synthetic-proof.json')]
    checks = {}
    for name in FINAL_CHECKS:
        body = dict(gate=name, stage='complete', passed=True, test_fixture_only=True,
                    accounting_report_sha256=ref['sha256'], index_sha256=accounting['index']['sha256'],
                    manifest_sha256=accounting['manifest_sha256'], evidence=evidence)
        if name == 'source_quality':
            body.update(unresolved=0, reviewed_images=accounting['total_images'])
        elif name == 'coverage':
            body['reviewed_counts'] = accounting['expected_counts']
        elif name == 'heldout_isolation':
            body['overlaps'] = 0
        else:
            body['implementation_sha256'] = implementation_hashes()
        if name == 'model_zero_update':
            body.update(optimizer_updates=0, actual_model_forward_backward=True,
                frozen_backbones_unchanged=True, adapter_unchanged=True,
                finite_nonzero_adapter_gradient=True, training_contract_sha256=training_contract_hash(config),
                initial_checkpoint=file_ref(checkpoint))
        path = root / (name + '.json')
        publish_json(path, body)
        checks[name] = file_ref(path)
    gate_path = root / 'acceptance.json'
    gate = seal_acceptance(ref, checks, config, gate_path)
    config['data']['geometry_corpus'] = file_ref(gate_path)
    return gate, checks, checkpoint


def test_full_accounting_is_not_training_acceptance(tmp_path):
    root, plan, rows = admission_fixture(tmp_path)
    report = build_accounting(plan, root / 'account')
    assert report['selected_images'] == report['total_images'] == 2
    assert report['training_admission'] is False
    assert report['stage'] == 'accounting_verified_not_accepted'
    assert {r['label_kind'] for r in report['coverage']} == {'synthetic_renderer', 'teacher_pseudo'}
    with sqlite3.connect(report['index']['path']) as db:
        assert db.execute('SELECT COUNT(*) FROM samples').fetchone()[0] == 2
    config = dict(data=dict(train_manifest=str(root / 'train.jsonl'),
                            geometry_corpus=file_ref(root / 'account/report.json')))
    with pytest.raises(ValueError, match='gate is closed'):
        open_training_gate(config)
    with pytest.raises(FileExistsError):
        build_accounting(plan, root / 'account')


@pytest.mark.parametrize('problem', ['missing', 'flagged', 'duplicate', 'unreviewed', 'incomplete'])
def test_incomplete_or_ambiguous_sources_fail_closed(tmp_path, problem):
    root, plan, _ = admission_fixture(tmp_path, missing=problem == 'missing', flagged=problem == 'flagged')
    if problem == 'duplicate':
        plan['sources'].append(deepcopy(plan['sources'][0]))
    elif problem == 'unreviewed':
        path = Path(plan['sources'][0]['review']['path'])
        value = json.loads(path.read_text())
        value['checks']['geometric_quality'] = False
        path.write_text(json.dumps(value))
        plan['sources'][0]['review'] = file_ref(path)
    elif problem == 'incomplete':
        path = Path(plan['sources'][0]['generation_report']['path'])
        value = json.loads(path.read_text())
        value['stage'] = 'generating_cache'
        path.write_text(json.dumps(value))
        plan['sources'][0]['generation_report'] = file_ref(path)
    with pytest.raises((ValueError, sqlite3.IntegrityError)):
        build_accounting(plan, root / 'account')
    assert not (root / 'account/report.json').exists()
    assert (root / 'account/failure.json').exists()


def test_all_final_checks_required_and_analytic_model_not_sufficient(tmp_path):
    root, plan, _ = admission_fixture(tmp_path)
    accounting = build_accounting(plan, root / 'account')
    config = dict(data=dict(train_manifest=str(root / 'train.jsonl')), train=dict(lr=1e-5))
    _, checks, checkpoint = synthetic_seal(root, accounting, config)
    assert open_training_gate(config, init=checkpoint)['selected_images'] == 2
    with pytest.raises(ValueError, match='All final'):
        seal_acceptance(file_ref(root / 'account/report.json'), {}, config, root / 'missing.json')
    path = Path(checks['model_zero_update']['path'])
    value = json.loads(path.read_text())
    value['actual_model_forward_backward'] = False
    path.write_text(json.dumps(value))
    checks['model_zero_update'] = file_ref(path)
    with pytest.raises(ValueError, match='Actual unchanged-model'):
        seal_acceptance(file_ref(root / 'account/report.json'), checks, config, root / 'analytic.json')


@pytest.mark.parametrize('change', ['config', 'pack', 'rgb', 'index', 'review', 'initial'])
def test_seal_and_reader_detect_replacement(tmp_path, change):
    root, plan, rows = admission_fixture(tmp_path)
    accounting = build_accounting(plan, root / 'account')
    config = dict(data=dict(train_manifest=str(root / 'train.jsonl')), train=dict(lr=1e-5))
    gate, checks, checkpoint = synthetic_seal(root, accounting, config)
    if change == 'config':
        config['train']['lr'] *= 2
    elif change == 'pack':
        with (root / 'hypersim/packs/pack-000000.tar').open('ab') as file:
            file.write(b'corruption')
    elif change == 'index':
        with sqlite3.connect(accounting['index']['path']) as db:
            db.execute("UPDATE images SET disposition='excluded' WHERE ordinal=0")
    elif change == 'review':
        Path(checks['coverage']['path']).write_text('{}')
    elif change == 'initial':
        checkpoint = root / 'wrong-initial.bytes'
        checkpoint.write_bytes(b'Wrong initialization')
    else:
        from radio_moge.corpus_data import CorpusRecords, CorpusImages
        Image.fromarray(np.zeros((24, 32, 3), np.uint8)).save(rows[0]['image'])
        with pytest.raises(ValueError, match='RGB bytes changed'):
            CorpusImages(CorpusRecords(gate))[(0, 43, 20, 32)]
        return
    with pytest.raises(ValueError):
        open_training_gate(config, init=checkpoint)


def test_variable_native_grids_collate_without_padding_and_sampling_resume(tmp_path):
    from radio_moge.corpus_data import CorpusImages, CorpusRecords, collate_corpus
    from radio_moge.data import MixtureBatches, check_split, manifest_digest

    root, plan, rows = admission_fixture(tmp_path)
    accounting = build_accounting(plan, root / 'account')
    config = dict(data=dict(train_manifest=str(root / 'train.jsonl')))
    gate, _, _ = synthetic_seal(root, accounting, config)
    records = CorpusRecords(gate)
    assert len(records) == 2
    assert records[0] == rows[0]
    assert records.contains_selected(rows[0])
    assert manifest_digest(records) == records.digest()
    data = CorpusImages(records, augment=False)
    a, b = data[(0, 1, 20, 32)], data[(1, 2, 20, 32)]
    batch = collate_corpus([a, b])
    assert batch['image'].shape == (2, 3, 20, 32)
    assert len(batch['geometry_targets']) == 2
    assert a['geometry_target']['valid'].shape != b['geometry_target']['valid'].shape
    assert batch['geometry_targets'][0] is a['geometry_target']
    assert [t['supervision'] for t in batch['geometry_targets']] == ['metric_native', 'relative_pseudo']
    weights = dict(hypersim=1, coco2017=1)
    original = list(MixtureBatches(records, weights, 2, [[20, 32]], 43, 0, 10))
    resumed = list(MixtureBatches(records, weights, 2, [[20, 32]], 43, 4, 6))
    assert resumed == original[4:]
    with pytest.raises(ValueError, match='leakage'):
        check_split(records, [rows[0]])
    with pytest.raises(IndexError):
        records.geometry(2)


def test_train_rejects_candidate_before_constructing_system(tmp_path, monkeypatch):
    from test_training import config_for
    from radio_moge.train import train, validate_config
    from radio_moge.official_gt import OFFICIAL_GT_WEIGHTS

    root, plan, _ = admission_fixture(tmp_path)
    build_accounting(plan, root / 'account')
    config = config_for(tmp_path)
    config['geometry_supervision'] = dict(target='offline_corpus', camera='provenance_aware',
                                          objective='moge2_core')
    config['loss'] = dict(spatial=1., summary=1., cosine=1., summary_cosine=1., **OFFICIAL_GT_WEIGHTS)
    config['data']['geometry_corpus'] = file_ref(root / 'account/report.json')
    config['feature_replay'] = dict(manifest=config['data']['train_manifest'], fraction=.5, seed=43)
    validate_config(config)
    def forbidden(*args):
        raise AssertionError('System construction was reached before corpus acceptance')
    monkeypatch.setattr('radio_moge.train.build_system', forbidden)
    with pytest.raises(ValueError, match='gate is closed'):
        train(config)
    config['feature_replay'] = None
    with pytest.raises(ValueError, match='RGB replay'):
        validate_config(config)
