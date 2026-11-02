from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from audit_irs_teacher_outcomes import run, teacher_metrics
sys.path.pop(0)

from radio_moge.geometry_cache import GeometryPackWriter, canonical_json, publish_json, sha_file


def fixture(root):
    config = dict(root=str(root), manifest_sha256='a' * 64, expected_counts={'irs': 4},
                  teacher=dict(repo='test-only', revision='b' * 40))
    rows = [dict(image=str(root / f'rgb-{i}.png'), sha256=f'{i:064x}', dataset='irs',
                 split='train', group='irs:family:test', eye='left' if i % 2 == 0 else 'right',
                 width=8, height=8) for i in range(4)]
    with sqlite3.connect(root / 'inventory.sqlite') as db:
        db.execute('CREATE TABLE images(ordinal,image,sha256,scene_group,row_json,dataset)')
        for i, row in enumerate(rows):
            db.execute('INSERT INTO images VALUES (?,?,?,?,?,?)',
                (i, row['image'], row['sha256'], row['group'], json.dumps(row), 'irs'))
    publish_json(root / 'inventory.json', dict(stage='inventory_complete',
        manifest_sha256=config['manifest_sha256'], dataset_counts=config['expected_counts'],
        inventory_sha256=sha_file(root / 'inventory.sqlite')))
    cache = root / 'cache'
    cache.mkdir()
    policy = dict(dataset='irs', manifest_sha256=config['manifest_sha256'],
                  teacher=config['teacher'], label_kind='teacher_pseudo', native_metric_gt=False)
    policy_hash = hashlib.sha256(canonical_json(policy)).hexdigest()
    writer = GeometryPackWriter(cache / 'packs', policy)
    for i, row in enumerate(rows[:3]):
        meta = dict(image_sha256=row['sha256'], dataset='irs', source_id=str(i),
            label_kind='teacher_pseudo', camera_kind='teacher_estimated', metric_calibrated=False,
            teacher_revision=config['teacher']['revision'], conversion='test', policy_sha256=policy_hash,
            source_proofs=[dict(source='fixture', sha256='f' * 64)])
        depth = np.full((8, 8), 2. * (i + 1), np.float32)
        depth[-1] *= 20
        writer.add(dict(row, corpus_ordinal=i), depth, np.ones((8, 8), bool), np.eye(3), meta)
    writer.close()
    publish_json(cache / 'rejections.json', [dict(image=rows[-1], reason='teacher_has_no_valid_depth')])
    publish_json(cache / 'report.json', dict(stage='complete', mode='generate', training_admission=False,
        training_started=False, optimizer_updates=0, completed=3, rejected=1, expected_scope=4,
        verified_cache_members=3, policy_sha256=policy_hash, teacher_frozen=True, family_eye_pairs_seen=2))
    return config


def test_complete_unique_outcomes_and_scale_invariant_relative_statistics(tmp_path):
    config = fixture(tmp_path)
    result = run(config, 'cache', 'audit')
    assert result['candidate_images'] == 3 and result['rejected_images'] == 1
    summary = result['relative_statistics']
    assert summary['full_support_images'] == 3
    assert summary['p99_over_median_percentiles'] == [20.] * 6
    assert summary['p99_over_median_above'] == {'8': 3, '16': 3, '32': 0, '64': 0}
    assert not result['source_quality_accepted'] and not result['independent_array_readback_performed']
    selected = json.loads((tmp_path / 'audit/review-candidates.json').read_text())
    assert {r['corpus_ordinal'] for r in selected} == {0, 1, 2}


def test_partial_readback_and_changed_or_duplicate_rows_cannot_pass(tmp_path):
    config = fixture(tmp_path)
    path = tmp_path / 'cache/report.json'
    report = json.loads(path.read_text())
    path.write_text(json.dumps(dict(report, stage='verifying_cache')))
    with pytest.raises(ValueError, match='producer readback'):
        run(config, 'cache', 'partial')
    assert not (tmp_path / 'partial').exists()
    path.write_text(json.dumps(dict(report, verified_cache_members=2)))
    with pytest.raises(ValueError, match='producer readback'):
        run(config, 'cache', 'incomplete-readback')
    assert not (tmp_path / 'incomplete-readback').exists()
    path.write_text(json.dumps(report))
    rejection_path = tmp_path / 'cache/rejections.json'
    row = json.loads(rejection_path.read_text())[0]
    changed = deepcopy(row)
    changed['image']['eye'] = 'left'
    rejection_path.write_text(json.dumps([changed]))
    with pytest.raises(ValueError, match='Changed or duplicate'):
        run(config, 'cache', 'changed')
    receipt = json.loads((tmp_path / 'cache/packs/pack-000000.json').read_text())
    row['image'] = receipt['entries'][0]['image']
    rejection_path.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match='Changed or duplicate'):
        run(config, 'cache', 'duplicate')


def test_metric_claim_wrong_revision_and_support_mismatch_are_rejected(tmp_path):
    config = fixture(tmp_path)
    receipt = json.loads((tmp_path / 'cache/packs/pack-000000.json').read_text())
    original = receipt['entries'][0]
    for field, value in (('metric_calibrated', True), ('teacher_revision', 'bad')):
        entry = deepcopy(original)
        entry['geometry_metadata'][field] = value
        # Keep the content identity self-consistent: rejection must come from
        # the wrong supervision/revision contract, not only a stale hash.
        entry['key'] = hashlib.sha256(canonical_json(entry['geometry_metadata'])).hexdigest()
        with pytest.raises(ValueError, match='provenance'):
            teacher_metrics(entry, receipt['policy_sha256'], config['teacher']['revision'])
    entry = deepcopy(original)
    entry['valid_fraction'] = .5
    with pytest.raises(ValueError, match='support or quantiles'):
        teacher_metrics(entry, receipt['policy_sha256'], config['teacher']['revision'])
