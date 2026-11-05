from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from radio_moge.checkpoint import rng_state
from radio_moge.corpus_admission import file_ref
from radio_moge.data import DeterministicBatches, TrainingImages
from radio_moge.rgb_input_audit import ReferenceRecords, RGBInputAudit, load_reference_records
from radio_moge.rgb_replacement import ReplacementRecords, ReplacementBatches, validate_replacement
from radio_moge.train import train
from run_procthor_feature_mix import check_mix_contract, initial_state_check
from test_performance import feature_config, profile
from test_training import tiny_system
from verify_corpus_stage_startup import equal_finite
import freeze_procthor_rgb as freezer


def dummy_records(n, m):
    real, synthetic = ReferenceRecords([{} for _ in range(n)]), ReferenceRecords([{} for _ in range(m)])
    real.metadata_sha256, synthetic.metadata_sha256 = 'real', 'synthetic'
    return ReplacementRecords(real, synthetic)


def spec(cursor=18):
    return dict(manifest=dict(path='/frozen/rgb.jsonl', sha256='a'*64),
                every_images=8, real_start_cursor=cursor, seed=43)


def test_mix_preserves_every_unreplaced_key_and_resume_and_rng():
    records = dummy_records(53, 7)
    buckets = [[32, 48], [48, 32], [32, 32]]
    before = rng_state()
    full = list(ReplacementBatches(records, spec(), 4, buckets, 43, 0, 80))
    original = list(DeterministicBatches(53, 4, buckets, 43, 18, 80))
    draws = []
    for slot, (new, old) in enumerate(zip([k for b in full for k in b], [k for b in original for k in b])):
        if (slot + 1) % 8:
            assert new == old
        else:
            assert new[0] >= 53 and new[1:] == old[1:]
            draws.append(new[0])
    assert len(draws) == 40 and len(set(draws[:7])) == 7
    for cursor in (1, 5, 14, 39, 79):
        assert list(ReplacementBatches(records, spec(), 4, buckets, 43, cursor, 80-cursor)) == full[cursor:]
    assert equal_finite(before, rng_state())


@pytest.mark.parametrize('change', ['geometry', 'weight', 'mixture', 'coverage', 'fraction'])
def test_replacement_does_not_admit_other_training_changes(change):
    cfg = dict(data=dict(rgb_replacement=spec()),
               loss=dict(spatial=1., summary=1., cosine=1., summary_cosine=1.))
    if change == 'geometry':
        cfg['ordinary_geometry'] = {}
    elif change == 'weight':
        cfg['loss']['cosine'] = .1
    elif change == 'fraction':
        cfg['data']['rgb_replacement']['every_images'] = 4
    else:
        cfg['data']['mixture_weights' if change == 'mixture' else 'coverage_epochs'] = {'x': 1}
    with pytest.raises(ValueError):
        validate_replacement(cfg)


def contract():
    parent_config = dict(data=dict(buckets=[[480, 640], [640, 480], [400, 640]], train_manifest='/real'),
        loss=dict(spatial=1., summary=1., cosine=1., summary_cosine=1.),
        train=dict(steps=233538, max_seconds=3600, lr=1e-5, lr_schedule='constant', eval_at_start=False),
        evaluation=dict(dataset_variants={}), output_dir='/parent')
    parent = dict(config=parent_config, step=233538, cursor=10000, data_digests={'train': 'old'},
                  performance=dict(effective_batch=32, compute_batch=32, workers=4))
    cfg = deepcopy(parent_config)
    cfg['data']['rgb_replacement'] = spec(10000)
    cfg['train']['steps'] = 234538
    cfg['output_dir'] = '/new'
    cfg['evaluation']['dataset_variants']['procthor-empty'] = dict(manifest='/val', max_side=640, num_tokens=1200)
    plan = dict(authorized=True, additional_updates=1000,
                synthetic_manifest=cfg['data']['rgb_replacement']['manifest'], validation_manifest=dict(path='/val'))
    return cfg, parent, plan


@pytest.mark.parametrize('change', [None, 'cursor', 'real_manifest', 'real_holdout', 'lr', 'budget'])
def test_narrow_fork_contract(change):
    cfg, parent, plan = contract()
    if change == 'cursor':
        cfg['data']['rgb_replacement']['real_start_cursor'] = 0
    elif change == 'real_manifest':
        cfg['data']['train_manifest'] = '/other'
    elif change == 'real_holdout':
        cfg['evaluation']['dataset_variants']['lidar'] = {}
    elif change == 'lr':
        cfg['train']['lr'] = .01
    elif change == 'budget':
        cfg['train']['steps'] += 1
    if change is None:
        check_mix_contract(cfg, parent, plan)
    else:
        with pytest.raises(ValueError):
            check_mix_contract(cfg, parent, plan)


def test_actual_prepared_training_uses_full32_no_geometry_and_preserves_initial_adam(tmp_path, monkeypatch):
    cfg = feature_config(tmp_path)
    cfg['train'].update(eval_at_start=False, log_every=1, eval_every=100, accumulation=8)
    cfg['output_dir'] = str(tmp_path / 'parent')
    source = train(cfg, stop_after=2, system_adapter=tiny_system())
    parent = torch.load(source, weights_only=True)
    manifest = Path(cfg['data']['train_manifest'])
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    for row in rows:
        row.update(sha256=file_ref(row['image'])['sha256'], dataset='real')
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    real = load_reference_records(manifest)
    generated = tmp_path / 'synthetic.png'
    Image.fromarray(np.full((40, 60, 3), [30, 80, 150], dtype=np.uint8)).save(generated)
    path = tmp_path / 'synthetic.jsonl'
    path.write_text(json.dumps(dict(image=str(generated), sha256=file_ref(generated)['sha256'],
                                    dataset='procthor_empty', group='synthetic'))+'\n')
    records = ReplacementRecords(real, load_reference_records(path))
    cfg['data']['rgb_replacement'] = dict(spec(parent['cursor']), manifest=file_ref(path))
    cfg['train'].update(steps=4, max_seconds=600, lr_schedule='constant')
    cfg['output_dir'] = str(tmp_path / 'mixed')
    perf = dict(profile(2, 0, 32, 32), data_pipeline='prepared', compact_transport=True,
                gpu_prefetch=False, base_effective_batch=32, schedule_ratio=1.)
    keys = [list(k) for b in ReplacementBatches(records, cfg['data']['rgb_replacement'],
        4, cfg['data']['buckets'], cfg['train']['seed'], 0, 16) for k in b]

    class Audit(RGBInputAudit):
        def load_records(self, path):
            return records

        def check_fork(self, *args):
            pass  # Production config restrictions have separate rejection tests above.

        def before_updates(self, system, adapter, statistics, records, config, output):
            initial = torch.load(output / 'initial.pt', weights_only=True)
            initial_state_check(initial, parent, cfg, perf)
            for module in (system.decoder,):
                monkeypatch.setattr(module, 'forward', lambda *args: pytest.fail('Geometry decoder called'))

    audit = Audit(cfg['output_dir'], 2, 4, 8, 32, keys)
    monkeypatch.setattr('radio_moge.train.evaluate', lambda *args, **kwargs:
                        {f'feature/{k}': 0. for k in cfg['loss']})
    final = torch.load(train(cfg, fork_from=source, data_fork=True, performance=perf,
        reference=audit, system_adapter=tiny_system()), weights_only=True)
    assert final['step'] == 4 and final['cursor'] == 16
    assert {int(v['step']) for v in final['optimizer']['state'].values()} == {4}
    report = json.loads((Path(cfg['output_dir'])/'actual-rgb-input-audit.json').read_text())
    assert report['source_counts'] == {'real': 56, 'procthor_empty': 8}
    assert report['presentations'] == 64 and report['planned_keys_matched']
    assert equal_finite(parent['statistics'], final['statistics'])
    assert any(not torch.equal(final['adapter'][k], v) for k, v in parent['adapter'].items())
    plain = TrainingImages(real, True, compact_transport=True)
    joined = TrainingImages(records, True, compact_transport=True)
    for key in keys:
        if key[0] < len(real):
            assert torch.equal(plain[key], joined[key])
    generated.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='RGB bytes changed'):
        joined[next(k for k in keys if k[0] >= len(real))]


def frozen_fixture(tmp_path, monkeypatch, split, start=0):
    archive = tmp_path / 'source.jsonl.gz'
    original = dict(objects=[dict(id='chair')], rooms=[], metadata={})
    line = json.dumps(original)+'\n'
    with gzip.open(archive, 'wt') as file:
        for i in range(start):
            file.write(json.dumps(dict(original, metadata=dict(ordinal=i)))+'\n')
        file.write(line)
    digest = freezer.sha(archive)
    monkeypatch.setattr(freezer, 'source_spec', lambda root, s: (archive, digest, start + 1))
    batch = tmp_path / 'batch'
    for name in ('houses', 'house-reports', 'rgb'):
        (batch / name).mkdir(parents=True)
    house = batch / 'houses' / f'{split}-{start:05d}.json'
    house.write_text(json.dumps(dict(original, objects=[])))
    group = f'procthor10k/{split}/{start:05d}'
    rows, images = [], []
    for i in range(20):
        a = np.zeros((480, 640, 3), np.uint8)
        a[:, :320] = 100+i
        image = batch / 'rgb' / f'{i}.png'
        Image.fromarray(a).save(image)
        row = dict(image=str(image), sha256=freezer.sha(image), group=group,
            dataset='procthor_empty', split=split, projection='perspective',
            agent_visible=False, geometry_label=None, training_admission=False,
            decoded_rgb_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
            source_archive_sha256=digest, source_line_sha256=hashlib.sha256(line.encode()).hexdigest(),
            house=dict(path=str(house), sha256=freezer.sha(house)))
        rows.append(row)
        images.append(dict(image=str(image), sha256=row['sha256']))
    manifest = batch / 'rgb-candidates.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows)+'{"unfinished":')
    report = dict(house_id=group, archive_sha256=digest,
        source_line_sha256=hashlib.sha256(line.encode()).hexdigest(),
        empty_house_sha256=freezer.sha(house), images=images)
    (batch / 'house-reports' / f'{split}-{start:05d}.json').write_text(json.dumps(report))
    return batch, rows


@pytest.mark.parametrize('split,views', [('train', 20), ('val', 4)])
def test_freeze_stops_at_closed_houses_without_reading_partial_tail(tmp_path, monkeypatch, split, views):
    batch, _ = frozen_fixture(tmp_path, monkeypatch, split)
    report = freezer.freeze(tmp_path, batch, tmp_path/'frozen', split, 1, views)
    assert report['images'] == views and report['training_eligible'] == (split == 'train')
    rows = [json.loads(line) for line in (tmp_path/'frozen/rgb.jsonl').read_text().splitlines()]
    assert {r['split'] for r in rows} == {split}
    assert all(r['label_role'] == 'feature_only_rgb' for r in rows)


def test_freeze_rejects_visible_agent_before_publication(tmp_path, monkeypatch):
    batch, rows = frozen_fixture(tmp_path, monkeypatch, 'train')
    rows[0]['agent_visible'] = True
    (batch/'rgb-candidates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError, match='contract changed'):
        freezer.freeze(tmp_path, batch, tmp_path/'frozen', 'train', 1, 20)
    assert not (tmp_path/'frozen').exists()


def test_freeze_nonzero_house_range_keeps_original_ids_and_source_binding(tmp_path, monkeypatch):
    batch, expected = frozen_fixture(tmp_path, monkeypatch, 'train', start=7)
    report = freezer.freeze(tmp_path, batch, tmp_path/'frozen', 'train', 1, 20, start=7)
    rows = [json.loads(line) for line in (tmp_path/'frozen/rgb.jsonl').read_text().splitlines()]
    assert report['house_range'] == [7, 8]
    assert {r['group'] for r in rows} == {'procthor10k/train/00007'}
    assert [(r['image'], r['sha256'], r['source_line_sha256']) for r in rows] == [
        (r['image'], r['sha256'], r['source_line_sha256']) for r in expected]
    assert not (batch/'houses/train-00000.json').exists()


@pytest.mark.parametrize('start', [-1, 1])
def test_freeze_rejects_out_of_range_before_publication(tmp_path, monkeypatch, start):
    batch, _ = frozen_fixture(tmp_path, monkeypatch, 'train')
    with pytest.raises(ValueError, match='^Pinned official house source differs$'):
        freezer.freeze(tmp_path, batch, tmp_path/'frozen', 'train', 1, 20, start=start)
    assert not (tmp_path/'frozen').exists()
