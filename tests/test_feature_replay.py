from contextlib import nullcontext
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import hashlib

import pytest
import torch

from radio_moge.data import read_manifest, manifest_digest, TrainingImages
from radio_moge.feature_replay import (is_replay_batch, load_replay_records, make_replay_loader,
                                       replay_feature_losses)
from radio_moge.losses import FeatureStatistics
from radio_moge.train import train, validate_config
from test_gt_geometry import gt_config
from test_training import tiny_system


def replay_config(tmp_path):
    config = gt_config(tmp_path)
    # config_for already created disjoint train/val RGB images.
    records = read_manifest(tmp_path / 'train.jsonl')
    for row in records:
        row.update(dataset='rgb', group='rgb-train', split='train')
    manifest = tmp_path / 'replay.jsonl'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in records))
    config['feature_replay'] = dict(manifest=str(manifest), fraction=.5, seed=100043)
    config['train'].update(steps=3, accumulation=2, log_every=1)
    return config


def test_replay_admission_and_budget_constraints(tmp_path):
    config = replay_config(tmp_path)
    validate_config(config)
    records = load_replay_records(config, [])
    with pytest.raises(ValueError, match='leakage'):
        load_replay_records(config, [records[:1]])
    for change in ('odd', 'no_gt', 'fraction', 'buckets'):
        bad = deepcopy(config)
        if change == 'odd':
            bad['train']['accumulation'] = 3
        elif change == 'no_gt':
            bad.pop('geometry_supervision')
        elif change == 'fraction':
            bad['feature_replay']['fraction'] = 1.
        else:
            bad['data']['buckets'].append([48, 32])
        with pytest.raises(ValueError, match='replay'):
            validate_config(bad)
    for extra in ({'split': 'val'}, {'depth': records[0]['image']}):
        row = dict(records[0], **extra)
        Path(config['feature_replay']['manifest']).write_text(json.dumps(row) + '\n')
        with pytest.raises(ValueError, match='train-only RGB'):
            load_replay_records(config, [])


def test_verified_replay_skips_global_file_scan_but_checks_each_actual_rgb(tmp_path, monkeypatch):
    config = replay_config(tmp_path)
    path = Path(config['feature_replay']['manifest'])
    rows = read_manifest(path)
    for row in rows:
        row['sha256'] = hashlib.sha256(Path(row['image']).read_bytes()).hexdigest()
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    config['feature_replay']['manifest_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    def no_global_stats(*args):
        raise AssertionError('Verified replay must not stat every source image at startup')
    monkeypatch.setattr(Path, 'is_file', no_global_stats)
    records = load_replay_records(config, [])
    assert manifest_digest(records) == config['feature_replay']['manifest_sha256']
    image = TrainingImages(records)[(0, 43, 32, 48)]
    assert image.shape == (3, 32, 48)
    p = Path(rows[0]['image'])
    p.write_bytes(p.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='RGB bytes changed'):
        TrainingImages(records)[(0, 43, 32, 48)]
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='manifest changed'):
        load_replay_records(config, [])


def test_replay_stream_has_half_the_slots_and_resumes_exactly(tmp_path):
    config = replay_config(tmp_path)
    records = load_replay_records(config, [])
    full = list(make_replay_loader(records, config, 0, 12))
    resumed = list(make_replay_loader(records, config, 4, 8))
    assert len(full) == 6 and len(resumed) == 4
    assert sum(is_replay_batch(i) for i in range(12)) == 6
    for left, right in zip(full[2:], resumed, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_replay_mixture_rejects_invalid_and_missing_sources(tmp_path):
    config = replay_config(tmp_path)
    for weights in ({}, {'rgb': 0}, {'rgb': -1}, {'rgb': float('inf')},
                    {'rgb': float('nan')}, {'rgb': True}, {'rgb': '1'}, {'': 1}, []):
        config['feature_replay']['mixture_weights'] = weights
        with pytest.raises(ValueError, match='replay mixture weights'):
            validate_config(config)
    config['feature_replay']['mixture_weights'] = {'rgb': 1}
    validate_config(config)
    assert load_replay_records(config, [])
    for weights in ({'other': 1}, {'rgb': 1, 'other': 1}):
        config['feature_replay']['mixture_weights'] = weights
        with pytest.raises(ValueError, match='every dataset exactly once'):
            load_replay_records(config, [])


def test_replay_mixture_controls_source_and_scene_mass_despite_frame_imbalance(tmp_path):
    config = replay_config(tmp_path)
    # A long capture must not dominate either the other scene or the rare source.
    records = [dict(dataset='large', group='long') for _ in range(10000)]
    records += [dict(dataset='large', group='short'), dict(dataset='small', group='only')]
    config['feature_replay']['mixture_weights'] = {'large': .4, 'small': .6}
    config['train']['batch_size'] = 4
    full = list(make_replay_loader(records, config, 0, 4000).batch_sampler)
    counts = Counter((records[i]['dataset'], records[i]['group']) for batch in full for i, *_ in batch)
    total = sum(counts.values())
    assert counts['small', 'only'] / total == pytest.approx(.6, abs=.025)
    assert counts['large', 'long'] / total == pytest.approx(.2, abs=.025)
    assert counts['large', 'short'] / total == pytest.approx(.2, abs=.025)
    for cursor in (1, 4, 5):
        resumed = list(make_replay_loader(records, config, cursor, 4000 - cursor).batch_sampler)
        assert resumed == full[cursor // 2:]


def test_replay_feature_branch_preserves_rng_and_never_calls_decoder():
    system, adapter = tiny_system()
    statistics = FeatureStatistics(6, 6)
    image = torch.rand(1, 3, 32, 48)
    before = torch.get_rng_state().clone()
    decoder_calls = []
    hook = system.decoder.register_forward_hook(lambda *args: decoder_calls.append(True))
    losses = replay_feature_losses(system, adapter, statistics, image, adapter.indices,
                                   nullcontext, seed=17)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    sum(losses.values()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.parameters())
    assert not decoder_calls
    assert all(p.grad is None for name in ('radio', 'encoder', 'decoder')
               for p in getattr(system, name).parameters())
    hook.remove()


@pytest.mark.parametrize('mixture', [False, True])
def test_replay_training_preserves_gt_stream_and_exact_resume(tmp_path, monkeypatch, mixture):
    import importlib
    trainer = importlib.import_module('radio_moge.train')
    config = replay_config(tmp_path)
    if mixture:
        config['feature_replay']['mixture_weights'] = {'rgb': 1.}
    actual = trainer.ground_truth_losses
    seen = []

    def capture(raw, target):
        seen.append({k: v.clone() for k, v in target.items()})
        return actual(raw, target)

    monkeypatch.setattr(trainer, 'ground_truth_losses', capture)
    baseline = deepcopy(config)
    baseline.pop('feature_replay')
    baseline['output_dir'] = str(tmp_path / 'control')
    control = torch.load(train(baseline, system_adapter=tiny_system()), weights_only=True)
    gt_control = seen[:]
    seen.clear()
    config['output_dir'] = str(tmp_path / 'mixed')
    full_path = train(config, system_adapter=tiny_system())
    full = torch.load(full_path, weights_only=True)
    assert len(gt_control) == len(seen) == 6
    for left, right in zip(gt_control, seen, strict=True):
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    torch.testing.assert_close(control['rng']['torch'], full['rng']['torch'], rtol=0, atol=0)
    assert any(not torch.equal(value, full['adapter'][key]) for key, value in control['adapter'].items())
    logs = [json.loads(line) for line in (full_path.parent / 'train.jsonl').read_text().splitlines()]
    assert all(row['feature_replay_fraction'] == .5 for row in logs)
    assert all(row['total'] == pytest.approx(row['feature_weighted'] + row['geometry_weighted'])
               for row in logs)
    split = deepcopy(config)
    split['output_dir'] = str(tmp_path / 'resume')
    partial = train(split, stop_after=1, system_adapter=tiny_system())
    resumed = torch.load(train(split, resume=partial, system_adapter=tiny_system()), weights_only=True)
    assert full['cursor'] == resumed['cursor'] == 6
    for key, value in full['adapter'].items():
        torch.testing.assert_close(value, resumed['adapter'][key], rtol=0, atol=0)
    torch.testing.assert_close(full['rng']['torch'], resumed['rng']['torch'], rtol=0, atol=0)
    if mixture:
        changed = deepcopy(split)
        changed['feature_replay']['mixture_weights'] = {'rgb': 2.}
        with pytest.raises(ValueError, match='identical config and manifests'):
            train(changed, resume=partial, system_adapter=tiny_system())
    # Additional RGB provenance participates in exact-resume validation.
    manifest = Path(config['feature_replay']['manifest'])
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]['source_changed'] = True
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='identical config and manifests'):
        train(split, resume=partial, system_adapter=tiny_system())
