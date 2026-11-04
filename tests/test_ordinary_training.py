from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest
import torch
from accelerate import Accelerator

from radio_moge import train as trainer
from radio_moge.checkpoint import rng_state
from radio_moge.corpus_admission import file_ref
from radio_moge.data import DeterministicBatches
from radio_moge.geometry_cache import GeometryPackWriter, canonical_json
from radio_moge.losses import FEATURE_LOSS_KEYS
from radio_moge.ordinary_geometry import ORDINARY_CAMERA_CONTRACT, ORDINARY_LOCAL_RADIUS
from radio_moge.ordinary_geometry_data import GeometrySidecar
from radio_moge.ordinary_training import (validate_ordinary_config, validate_ordinary_performance,
    check_ordinary_data_fork, verify_initial_state, COUNT_KEYS)
from radio_moge.prefetch import PreparedWindows
from radio_moge.rgb_input_audit import RGBInputAudit, load_reference_records
from radio_moge.rgb_geometry_batch import RGBGeometryBatch
from test_ordinary_geometry_data import fixture, bind_fixture
from test_performance import feature_config, profile
from test_rgb_reference import SmallReference
from test_training import tiny_system
from verify_corpus_stage_startup import equal_finite


def setup(tmp_path, payloads=False):
    config = feature_config(tmp_path)
    rows, ref = fixture(tmp_path)
    if payloads:
        policy = dict(fixture=True)
        writer = GeometryPackWriter(tmp_path / 'actual-packs', policy)
        for i in (1, 3, 4):
            row = rows[i]
            h, w = (29, 19) if row['image_rotation_cw'] in (90, 270) else (19, 29)
            depth = np.full((h, w), 2., np.float32)
            valid = np.ones((h, w), bool)
            if i == 3:
                valid[:] = False
                valid[8, 9] = True
            meta = dict(image_sha256=row['sha256'], dataset=row['dataset'], source_id=str(i),
                label_kind='teacher_pseudo' if i == 4 else 'synthetic_renderer',
                camera_kind='teacher_estimated' if i == 4 else 'synthetic_camera', metric_calibrated=i != 4,
                conversion='test_only', policy_sha256=hashlib.sha256(canonical_json(policy)).hexdigest(),
                image_rotation_cw=row['image_rotation_cw'], source_proofs=[dict(source='fixture', sha256='f'*64)])
            if i == 4:
                meta['teacher_revision'] = 'b'*40
            writer.add(row, depth, valid, np.array([[.8, .03, .44], [0., 1.1, .56], [0., 0., 1.]]), meta)
        writer.close()
        receipt = json.loads((tmp_path / 'actual-packs/pack-000000.json').read_text())
        pack = tmp_path / 'actual-packs' / receipt['tar']
        with sqlite3.connect(tmp_path / 'index.sqlite') as db:
            db.execute('INSERT INTO packs VALUES (?,?)', (str(pack), file_ref(pack)['sha256']))
            for i, entry in zip((1, 3, 4), receipt['entries'], strict=True):
                db.execute('UPDATE images SET pack=?,entry_json=? WHERE ordinal=?',
                           (str(pack), json.dumps(entry), 100+i))
        ref = bind_fixture(tmp_path, tmp_path / 'index.sqlite')
    manifest = Path(config['data']['train_manifest'])
    manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    config['data'].update(val_manifest=None, buckets=[[32, 48], [48, 32], [32, 32]], compact_transport=True)
    config['train'].update(accumulation=8, eval_at_start=False, eval_every=100, log_every=1,
                           seed=43, tensorboard=False)
    return config, ref


def spec(ref, multiplier=0):
    return dict(sidecar=ref, multiplier=multiplier, camera_contract=ORDINARY_CAMERA_CONTRACT,
                local_radius_contract=ORDINARY_LOCAL_RADIUS, pixel_convention='normalized_half_center',
                admission=ref)  # Historical seal deliberately NOT a training admission.


def perf():
    return dict(profile(2, 0, effective=32, compute=8), data_pipeline='prepared', compact_transport=True,
                gpu_prefetch=False, base_effective_batch=32, schedule_ratio=1)


class SyntheticAudit(SmallReference):
    load_records = staticmethod(load_reference_records)
    check_fork = staticmethod(check_ordinary_data_fork)

    def __init__(self, config, performance):
        super().__init__(config['output_dir'], 2, 4, 8, performance['compute_batch'])
        self.sidecar = GeometrySidecar(config['ordinary_geometry']['sidecar'])


def fake_gate(config, source, performance):
    # Test-only in-memory substitution; no successful admission is written.
    validate_ordinary_config(config)
    validate_ordinary_performance(config, performance)
    return SyntheticAudit(config, performance)


@pytest.mark.parametrize('workers', [0, 2])
def test_actual_audit_preserves_rgb_grouping_rng_and_unused_prefetch(tmp_path, workers):
    cfg, ref = setup(tmp_path)
    cfg['data']['workers'] = workers
    rows = load_reference_records(cfg['data']['train_manifest'])
    planned = [list(k) for batch in DeterministicBatches(8, 4, cfg['data']['buckets'], 43, 0, 24) for k in batch]
    a_dir, b_dir = tmp_path/'plain', tmp_path/'attached'
    a_dir.mkdir()
    b_dir.mkdir()
    a, b = [RGBInputAudit(p, 2, 4, 8, 8, planned) for p in (a_dir, b_dir)]
    plain = PreparedWindows(trainer.make_loader(rows, cfg, 0, 24, reference=a), 8, 8, 3,
                            observer=a.observe_window)
    cfg['ordinary_geometry'] = spec(ref)
    before = rng_state()
    attached = PreparedWindows(trainer.make_loader(rows, cfg, 0, 24, reference=b,
                                sidecar=GeometrySidecar(ref)), 8, 8, 3, observer=b.observe_window)
    try:
        for update in range(2):
            x, y = next(plain), next(attached)
            assert all(torch.equal(left, right.image) for left, right in zip(x, y, strict=True))
            assert all(s.augment is True for group in y for s in group.slots)
            a.commit(3+update, 8*(update+1))
            b.commit(3+update, 8*(update+1))
    finally:
        plain.close()
        attached.close()
    a.finish(4, 16)
    b.finish(4, 16)
    left, right = [[json.loads(s) for s in p.path.read_text().splitlines()] for p in (a, b)]
    for group in right:
        for row in group['logical_images']:
            assert row.pop('ordinary_geometry')['augment'] is True
    assert left == right and len(left) == 2
    assert equal_finite(before, rng_state())


@pytest.mark.parametrize('change', ['order', 'reference', 'augment', 'lost'])
def test_audit_rejects_annotation_changes_even_with_identical_pixels(tmp_path, change):
    cfg, ref = setup(tmp_path)
    cfg['ordinary_geometry'] = spec(ref)
    audit = RGBInputAudit(tmp_path, 2, 4, 1, 4)
    batch = next(iter(trainer.make_loader(load_reference_records(cfg['data']['train_manifest']), cfg,
        0, 1, reference=audit, sidecar=GeometrySidecar(ref)).loader))
    slots = list(batch.slots)
    if change == 'order':
        slots[0] = replace(slots[0], key=(999, *slots[0].key[1:]))
    elif change == 'reference':
        slots[0] = replace(slots[0], geometry={'changed': True})
    elif change == 'augment':
        slots[0] = replace(slots[0], augment=False)
    mutated = batch.image if change == 'lost' else RGBGeometryBatch(batch.image, tuple(slots))
    with pytest.raises(ValueError, match='geometry|Geometry'):
        audit.observe_window([mutated])
    assert audit.committed == 0 and not audit.path.exists()


def test_production_loop_g0_exact_state_losses_gradients_rng_cursor_and_no_geometry_calls(tmp_path, monkeypatch):
    cfg, ref = setup(tmp_path)
    cfg['output_dir'] = str(tmp_path / 'parent')
    source = trainer.train(cfg, stop_after=2, system_adapter=tiny_system())
    cfg['train'].update(lr_schedule='constant', steps=4, max_seconds=600)
    performance = perf()
    captured = []
    original = Accelerator.clip_grad_norm_
    def capture(self, parameters, *args, **kwargs):
        parameters = list(parameters)
        captured.append([p.grad.clone() if p.grad is not None else None for p in parameters])
        return original(self, parameters, *args, **kwargs)
    monkeypatch.setattr(Accelerator, 'clip_grad_norm_', capture)
    cfg['output_dir'] = str(tmp_path / 'plain')
    observer = SmallReference(cfg['output_dir'], 2, 4, 8, 8)
    observer.load_records = load_reference_records
    a = torch.load(trainer.train(cfg, fork_from=source, data_fork=True, performance=performance,
        system_adapter=tiny_system(), reference=observer), weights_only=True)
    baseline_grads, captured[:] = deepcopy(captured), []
    cfg['output_dir'] = str(tmp_path / 'geometry-off')
    cfg['ordinary_geometry'] = spec(ref)
    monkeypatch.setattr(trainer, 'open_ordinary_training_gate', fake_gate)
    def forbidden(*args, **kwargs):
        raise AssertionError('Geometry-off entered geometry code')
    monkeypatch.setattr(trainer, 'ordinary_target', forbidden)
    monkeypatch.setattr(trainer, 'ordinary_geometry_losses', forbidden)
    monkeypatch.setattr('radio_moge.ordinary_geometry_targets.read_geometry_member', forbidden)
    system, adapter = tiny_system()
    monkeypatch.setattr(system.decoder, 'forward', forbidden)
    b = torch.load(trainer.train(cfg, fork_from=source, data_fork=True, performance=performance,
        system_adapter=(system, adapter)), weights_only=True)
    assert b['cursor'] == a['cursor'] == 16 and b['step'] == a['step'] == 4
    assert equal_finite(baseline_grads, captured)
    for key in ('adapter', 'statistics', 'optimizer', 'scheduler', 'rng', 'data_digests'):
        assert equal_finite(a[key], b[key]), key
    logs = [[json.loads(s) for s in (tmp_path/p/'train.jsonl').read_text().splitlines()]
            for p in ('plain', 'geometry-off')]
    for x, y in zip(*logs, strict=True):
        for k in (*FEATURE_LOSS_KEYS, 'total', 'gradient_norm', 'logical_data_cursor'):
            assert x[k] == y[k]
        assert y['geometry_weighted'] == 0 and y['feature_weighted'] == y['total']
    initial = torch.load(tmp_path/'geometry-off/initial.pt', weights_only=True)
    parent = torch.load(source, weights_only=True)
    for key in ('adapter', 'statistics', 'optimizer', 'rng'):
        assert equal_finite(initial[key], parent[key]), key
    assert initial['cursor'] == 0 and parent['cursor'] == 16
    verify_initial_state(parent, initial, cfg, 2, initial['data_digests'])
    for key in ('adapter', 'statistics', 'optimizer', 'rng'):
        changed = deepcopy(initial)
        changed[key] = {}
        with pytest.raises(ValueError, match='initial state differs'):
            verify_initial_state(parent, changed, cfg, 2, initial['data_digests'])


def test_production_loop_g1_actual_payloads_raw_features_and_integer_counts(tmp_path, monkeypatch):
    cfg, ref = setup(tmp_path, payloads=True)
    cfg['output_dir'] = str(tmp_path / 'parent')
    source = trainer.train(cfg, stop_after=2, system_adapter=tiny_system())
    cfg['output_dir'] = str(tmp_path / 'geometry-on')
    cfg['train'].update(lr_schedule='constant', steps=3, max_seconds=600)
    cfg['ordinary_geometry'] = spec(ref, 1)
    monkeypatch.setattr(trainer, 'open_ordinary_training_gate', fake_gate)
    actual_loss, observed = trainer.ordinary_geometry_losses, []
    def record(raw, targets, *args, **kwargs):
        values = actual_loss(raw, targets, *args, **kwargs)
        observed.append((len(targets), {k: v.item() for k, v in values.items()}))
        return values
    monkeypatch.setattr(trainer, 'ordinary_geometry_losses', record)
    system, adapter = tiny_system()
    raw_predictions = []
    adapter.register_forward_hook(lambda module, args, result: raw_predictions.append(result))
    def raw_only(module, args):
        assert args[0] is raw_predictions[-1]
    system.decoder.register_forward_pre_hook(raw_only)
    state = torch.load(trainer.train(cfg, fork_from=source, data_fork=True, performance=perf(),
        system_adapter=(system, adapter)), weights_only=True)
    row = json.loads((tmp_path/'geometry-on/train.jsonl').read_text())
    assert sum(n for n, _ in observed) == 32
    for key in COUNT_KEYS:
        assert type(row[key]) is int and row[key] == sum(v[key] for _, v in observed)
    assert sum(row[k] for k in ('gt_supervised_images', 'gt_skipped_sparse_images', 'gt_missing_target_images')) == 32
    assert row['gt_supervised_images'] > 0 and row['gt_skipped_sparse_images'] > 0
    assert row['gt_missing_target_images'] > 0
    assert row['geometry_weighted'] > 0 and row['gradient_norm'] > 0
    from radio_moge.official_gt import OFFICIAL_GT_WEIGHTS
    expected = sum(n/32 * sum(v[k]*w for k, w in OFFICIAL_GT_WEIGHTS.items()) for n, v in observed)
    assert row['geometry_weighted'] == pytest.approx(expected, rel=1e-6)
    assert row['total'] == pytest.approx(row['feature_weighted']+row['geometry_weighted'], rel=1e-6)
    assert state['cursor'] == 8 and state['step'] == 3
    assert all(p.grad is None and not p.requires_grad for p in system.decoder.parameters())


def test_historical_seal_cannot_admit_before_model_or_optimizer_construction(tmp_path, monkeypatch):
    cfg, ref = setup(tmp_path)
    cfg['ordinary_geometry'] = spec(ref)
    cfg['train']['lr_schedule'] = 'constant'
    def forbidden(*args, **kwargs):
        raise AssertionError('A rejected gate constructed runtime state')
    monkeypatch.setattr(trainer, 'Accelerator', forbidden)
    monkeypatch.setattr(trainer, 'build_system', forbidden)
    monkeypatch.setattr(torch.optim, 'AdamW', forbidden)
    with pytest.raises(ValueError, match='historical seals'):
        trainer.train(cfg, fork_from='/unread-parent.pt', data_fork=True, performance=perf())


@pytest.mark.parametrize('change', ['config', 'code', 'performance', 'budget', 'plan', 'cpu', 'model', 'cuda'])
def test_pending_or_stale_new_gate_fails_before_runtime(tmp_path, monkeypatch, change):
    from radio_moge import ordinary_training as gate
    cfg, ref = setup(tmp_path)
    cfg['ordinary_geometry'] = spec(ref)
    cfg['train'].update(lr_schedule='constant', steps=232538)
    performance = dict(perf(), anchor_step=232288)
    def fake_ref(name):
        return dict(path=str(tmp_path/name), sha256='a'*64)
    keys = [[0, 1, 32, 48]]
    seal = dict(stage='ordinary_rgb_geometry_training_accepted', schema_version=1, training_admission=True,
        config_sha256=gate.config_digest(cfg), performance_sha256=gate.digest(performance),
        implementation_sha256='fixture-code', source_checkpoint=fake_ref('parent.pt'),
        source_step=232288, manifest=dict(path=cfg['data']['train_manifest'], sha256='b'*64),
        sidecar=ref, rgb_images=1637908, optimizer_updates=250, planned_keys=fake_ref('keys'),
        data_digests={'train': 'fixture-digest'}, cpu_evidence=fake_ref('cpu'), model_evidence=fake_ref('model'))
    binding = {key: seal[key] for key in ('config_sha256', 'performance_sha256', 'implementation_sha256',
        'source_checkpoint', 'manifest', 'sidecar', 'planned_keys', 'data_digests')}
    cpu = dict(stage='ordinary_production_loop_cpu_verified', binding=binding, production_optimizer_updates=0,
               training_admission=False, **{k: True for k in gate.CPU_REQUIREMENTS})
    model = dict(stage='ordinary_actual_model_zero_update_verified', binding=binding, optimizer_updates=0,
                 training_admission=False, device='cuda', **{k: True for k in gate.MODEL_REQUIREMENTS})
    # In-memory synthetic pending reports only. No accepted file/model is created.
    reports = {ref['path']: seal, fake_ref('keys')['path']: keys,
               fake_ref('cpu')['path']: cpu, fake_ref('model')['path']: model}
    monkeypatch.setattr(gate, 'checked_json', lambda r: reports[r['path']])
    monkeypatch.setattr(gate, 'checked_path', lambda r: Path(r['path']))
    monkeypatch.setattr(gate, 'implementation_digest', lambda: 'fixture-code')
    monkeypatch.setattr(gate, 'DeterministicBatches', lambda *args: [[tuple(k) for k in keys]])
    def forbidden(*args, **kwargs):
        raise AssertionError('Pending gate reached sidecar/model/optimizer')
    monkeypatch.setattr(gate, 'OrdinaryTrainingAudit', forbidden)
    monkeypatch.setattr(trainer, 'Accelerator', forbidden)
    if change == 'config':
        cfg['train']['lr'] /= 2
    if change == 'code':
        seal['implementation_sha256'] = 'changed'
    if change == 'performance':
        performance['compute_batch'] *= 2
    if change == 'budget':
        seal['optimizer_updates'] = 251
    if change == 'plan':
        reports[fake_ref('keys')['path']] = []
    if change == 'cpu':
        cpu['geometry_off_no_calls'] = False
    if change == 'model':
        model['stage'] = 'pending'
    if change == 'cuda':
        model['device'] = 'cpu'
    with pytest.raises(ValueError):
        trainer.train(cfg, fork_from=fake_ref('parent.pt')['path'], data_fork=True, performance=performance)


@pytest.mark.parametrize('change', ['weights', 'mixture', 'replay', 'centered', 'baseline', 'tuning', 'batch'])
def test_structural_changes_cannot_reuse_ordinary_path(tmp_path, change):
    cfg, ref = setup(tmp_path)
    cfg['ordinary_geometry'] = spec(ref)
    cfg['train']['lr_schedule'] = 'constant'
    performance = perf()
    if change == 'weights':
        cfg['loss']['cosine'] = .1
    if change == 'mixture':
        cfg['data']['mixture_weights'] = {'fixture': 1}
    if change == 'replay':
        cfg['feature_replay'] = {}
    if change == 'centered':
        cfg['ordinary_geometry']['camera_contract'] = 'centered'
    if change == 'baseline':
        cfg['train']['eval_at_start'] = True
    if change == 'tuning':
        performance['autotune'] = True
    if change == 'batch':
        performance['effective_batch'] = 16
    with pytest.raises(ValueError):
        validate_ordinary_config(cfg)
        validate_ordinary_performance(cfg, performance)
