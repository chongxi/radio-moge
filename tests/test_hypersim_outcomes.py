from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from audit_hypersim_outcomes import outcome_kind, run
sys.path.pop(0)

from radio_moge.geometry_cache import GeometryPackWriter, canonical_json, publish_json, sha_file


def test_strict_flag_policy_and_exact_one_percent_boundary():
    meta = dict(source_quality=dict(supported_pixels=100, inconsistent_supported_pixels=1),
                source_needs_review=False)
    assert outcome_kind(meta) == 'unflagged_cache'
    meta['source_quality']['inconsistent_supported_pixels'] = 2
    with pytest.raises(ValueError, match='flag differs'):
        outcome_kind(meta)
    meta['source_needs_review'] = True
    assert outcome_kind(meta) == 'flagged_cache'


def fixture(root):
    config = dict(root=str(root), manifest_sha256='a' * 64, expected_counts={'hypersim': 4})
    rows = [dict(image=str(root / f'rgb-{i}.png'), sha256=f'{i:064x}', dataset='hypersim',
                 split='train', group=f'hypersim:scene{i // 2}', frame_id=i) for i in range(4)]
    with sqlite3.connect(root / 'inventory.sqlite') as db:
        db.execute('CREATE TABLE images(ordinal,image,sha256,scene_group,row_json,dataset)')
        for i, row in enumerate(rows):
            db.execute('INSERT INTO images VALUES (?,?,?,?,?,?)',
                (i, row['image'], row['sha256'], row['group'], json.dumps(row), 'hypersim'))
    publish_json(root / 'inventory.json', dict(stage='inventory_complete',
        manifest_sha256=config['manifest_sha256'], dataset_counts=config['expected_counts'],
        inventory_sha256=sha_file(root / 'inventory.sqlite')))
    native = root / 'native'
    native.mkdir()
    policy = dict(dataset='hypersim', manifest_sha256=config['manifest_sha256'])
    policy_hash = hashlib.sha256(canonical_json(policy)).hexdigest()
    writer = GeometryPackWriter(native / 'packs', policy)
    for i, row in enumerate(rows[:2]):
        meta = dict(image_sha256=row['sha256'], dataset='hypersim', source_id=str(i),
            label_kind='synthetic_renderer', camera_kind='synthetic_camera', metric_calibrated=True,
            conversion='test', policy_sha256=policy_hash, source_proofs=[dict(source='fixture', sha256='f' * 64)],
            source_quality=dict(supported_pixels=100, inconsistent_supported_pixels=10 * i),
            source_needs_review=bool(i))
        writer.add(row, np.ones((8, 8), np.float32), np.ones((8, 8), bool), np.eye(3), meta)
    writer.close()
    publish_json(native / 'rejections.json', [dict(image=rows[2], reason='census_quarantined_native_pose'),
        dict(image=rows[3], reason='insufficient_world_verified_support',
             stats=dict(supported_pixels=100, inconsistent_supported_pixels=99))])
    publish_json(native / 'report.json', dict(stage='complete', mode='generate', training_admission=False,
        training_started=False, optimizer_updates=0, completed=2, rejected=2, expected_scope=4,
        verified_cache_members=2, policy_sha256=policy_hash, source_review_images=2))
    return config


def test_every_native_outcome_is_bound_once_and_flags_do_not_double_count_rejections(tmp_path):
    config = fixture(tmp_path)
    result = run(config, 'native', 'audit')
    assert result['outcome_counts'] == {'unflagged_cache': 1, 'flagged_cache': 1, 'rejected': 2}
    assert result['unresolved_images'] == 3 and result['source_review_counter'] == 2
    assert result['rejected_source_flags'] == 1
    assert not result['training_admission'] and not result['independent_array_readback_performed']
    pending = [json.loads(line) for line in (tmp_path / 'audit/unresolved-native.jsonl').read_text().splitlines()]
    assert [r['corpus_ordinal'] for r in pending] == [1, 2, 3]
    path = tmp_path / 'native/rejections.json'
    rejections = json.loads(path.read_text())
    rejections[1]['image'] = deepcopy(rejections[0]['image'])
    path.write_text(json.dumps(rejections))
    with pytest.raises(ValueError, match='duplicate frozen native outcome'):
        run(config, 'native', 'duplicate')


def test_generation_counters_without_complete_readback_cannot_finish_outcome_audit(tmp_path):
    config = fixture(tmp_path)
    path = tmp_path / 'native/report.json'
    report = json.loads(path.read_text())
    report['stage'] = 'generating_cache'
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='producer readback required'):
        run(config, 'native', 'not-ready')
    assert not (tmp_path / 'not-ready').exists()
