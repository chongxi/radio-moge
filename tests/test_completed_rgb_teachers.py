from copy import deepcopy
import hashlib
import json
import sqlite3

import numpy as np
from PIL import Image
import pytest

from audit_completed_rgb_teachers import run, take_identity
from audit_irs_teacher_outcomes import teacher_metrics
from radio_moge.geometry_cache import GeometryPackWriter, canonical_json, publish_json, sha_file


def fixture(root, dataset):
    config = dict(root=str(root), manifest_sha256='a' * 64, expected_counts={dataset: 3},
                  teacher=dict(repo='fixture', revision='b' * 40))
    rows = []
    for i in range(3):
        path = root / f'{i}.png'
        Image.fromarray(np.full((8, 10, 3), 10 + i, np.uint8)).save(path)
        row = dict(image=str(path), sha256=sha_file(path), dataset=dataset,
                   group=f'{dataset}:scene{i}', split='train')
        row.update(dict(scene_type='office') if dataset == 'nyuv2' else dict(height=8, width=10))
        rows.append(row)
    with sqlite3.connect(root / 'inventory.sqlite') as db:
        db.execute('CREATE TABLE images(ordinal,image,row_json,dataset)')
        db.executemany('INSERT INTO images VALUES (?,?,?,?)',
                       [(i, r['image'], json.dumps(r), dataset) for i, r in enumerate(rows)])
    publish_json(root / 'inventory.json', dict(stage='inventory_complete',
        manifest_sha256=config['manifest_sha256'], dataset_counts=config['expected_counts'],
        inventory_sha256=sha_file(root / 'inventory.sqlite')))
    policy = dict(dataset=dataset, manifest_sha256=config['manifest_sha256'],
                  teacher=config['teacher'], label_kind='teacher_pseudo', native_metric_gt=False)
    policy_hash = hashlib.sha256(canonical_json(policy)).hexdigest()
    writer = GeometryPackWriter(root / 'cache/packs', policy)
    for i, row in enumerate(rows):
        meta = dict(image_sha256=row['sha256'], dataset=dataset, source_id=str(i),
            label_kind='teacher_pseudo', camera_kind='teacher_estimated', metric_calibrated=False,
            teacher_revision=config['teacher']['revision'], conversion='fixture', policy_sha256=policy_hash,
            source_proofs=[dict(source='fixture', sha256='f' * 64)])
        depth = np.full((8, 10), 2. * (i + 1), np.float32)
        depth[-1] *= 4
        writer.add(dict(row, corpus_ordinal=i), depth, np.ones((8, 10), bool), np.eye(3), meta)
    writer.close()
    publish_json(root / 'cache/rejections.json', [])
    publish_json(root / 'cache/report.json', dict(stage='complete', mode='generate', completed=3,
        verified_cache_members=3, rejected=0, teacher_frozen=True, training_admission=False,
        training_started=False, optimizer_updates=0, policy_sha256=policy_hash))
    (root / 'preflight').mkdir()
    publish_json(root / 'preflight/report.json', dict(stage='complete', passed=True,
        policy_sha256=policy_hash, examples=[dict(ordinal=i) for i in range(3)]))
    return config, rows, policy_hash


@pytest.mark.parametrize('dataset', ['coco2017', 'nyuv2'])
def test_complete_sources_preserve_legacy_rows_and_bind_actual_selected_payloads(tmp_path, dataset):
    config, rows, policy_hash = fixture(tmp_path, dataset)
    result = run(config, dataset, 'cache', 'preflight', 'audit')
    assert result['candidate_images'] == 3 and result['groups'] == 3
    assert result['relative_statistics']['p99_over_median'] == [4.] * 6
    assert result['selected_member_payloads_rechecked']
    assert not result['source_quality_accepted'] and not result['training_admission']
    views = json.loads((tmp_path / 'audit/visuals/report.json').read_text())
    assert len(views['records']) == 3
    assert all(r['original_rgb_rechecked'] for r in views['records'])
    if dataset == 'nyuv2':
        assert 'height' not in rows[0]
        receipt = json.loads((tmp_path / 'cache/packs/pack-000000.json').read_text())
        entry = receipt['entries'][0]
        with pytest.raises(ValueError, match='support or quantiles'):
            teacher_metrics(entry, policy_hash, config['teacher']['revision'], dataset, (10, 8))
        assert 'height' not in entry['image']


def test_partial_generation_and_changed_rgb_cannot_produce_audit(tmp_path):
    config, rows, _ = fixture(tmp_path, 'nyuv2')
    path = tmp_path / 'cache/report.json'
    original = json.loads(path.read_text())
    path.write_text(json.dumps(dict(original, verified_cache_members=2)))
    with pytest.raises(ValueError, match='producer readback'):
        run(config, 'nyuv2', 'cache', 'preflight', 'partial')
    assert not (tmp_path / 'partial').exists()
    path.write_text(json.dumps(original))
    Image.fromarray(np.zeros((8, 10, 3), np.uint8)).save(rows[0]['image'])
    with pytest.raises(ValueError, match='RGB bytes changed'):
        run(config, 'nyuv2', 'cache', 'preflight', 'changed')
    assert not (tmp_path / 'changed').exists()


def test_changed_duplicate_and_outside_scope_identities_are_not_silently_accepted():
    row = dict(dataset='nyuv2', image='/fixture.png', group='scene', split='train')
    for candidate in (dict(row, corpus_ordinal=9), dict(row, corpus_ordinal=1, group='other')):
        with pytest.raises(ValueError, match='Changed, duplicate or outside'):
            take_identity({1: deepcopy(row)}, candidate)
    originals = {1: deepcopy(row)}
    assert take_identity(originals, dict(row, corpus_ordinal=1)) == 1
    with pytest.raises(ValueError, match='Changed, duplicate or outside'):
        take_identity(originals, dict(row, corpus_ordinal=1))
