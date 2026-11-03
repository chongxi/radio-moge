import hashlib
import json
import sqlite3

import numpy as np
import pytest

import snapshot_geometry_candidates as snapshot
from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import GeometryPackWriter, canonical_json, publish_json


def fixture(root):
    config = dict(root=str(root), manifest_sha256='a' * 64, expected_counts={'hypersim': 3})
    policy = dict(dataset='hypersim', manifest_sha256=config['manifest_sha256'],
                  label_kind='synthetic_renderer')
    policy_hash = hashlib.sha256(canonical_json(policy)).hexdigest()
    rows = [dict(image=str(root / f'{i}.png'), sha256=f'{i:064x}', dataset='hypersim',
                 split='train', group=f'hypersim:scene{i}') for i in range(3)]
    with sqlite3.connect(root / 'inventory.sqlite') as db:
        db.execute('CREATE TABLE images(ordinal,image,sha256,dataset,scene_group,row_json)')
        for i, row in enumerate(rows):
            db.execute('INSERT INTO images VALUES (?,?,?,?,?,?)',
                       (i, row['image'], row['sha256'], row['dataset'], row['group'], json.dumps(row)))
    publish_json(root / 'inventory.json', dict(stage='inventory_complete',
        manifest_sha256=config['manifest_sha256'], dataset_counts=config['expected_counts'],
        inventory_sha256=file_ref(root / 'inventory.sqlite')['sha256']))
    writer = GeometryPackWriter(root / 'producer/packs', policy, max_entries=2)
    for i in range(2):
        row = rows[i]
        metadata = dict(image_sha256=row['sha256'], dataset='hypersim', source_id=str(i),
            label_kind='synthetic_renderer', camera_kind='synthetic_camera', metric_calibrated=True,
            conversion='fixture', policy_sha256=policy_hash, source_needs_review=bool(i),
            source_proofs=[dict(source='fixture', sha256='f' * 64)])
        writer.add(row, np.ones((8, 8), np.float32), np.ones((8, 8), bool), np.eye(3), metadata)
    publish_json(root / 'producer/status.json', dict(stage='generating_cache', completed=3,
        optimizer_updates=0, training_started=False, policy_sha256=policy_hash))
    # An unpublished tail must never enter the first-stage snapshot.
    (root / 'producer/packs/pack-000001.tar.incoming').write_bytes(b'incomplete')
    exclusions = root / 'exclusions.jsonl'
    exclusions.write_text('')
    plan = dict(sources=[dict(attempt='producer', dataset='hypersim', role='primary')],
                exclusions=file_ref(exclusions))
    return config, plan, writer


def test_live_producer_snapshot_has_only_published_entries_and_retains_flags(tmp_path):
    config, plan, writer = fixture(tmp_path)
    report = snapshot.run(config, plan, tmp_path / 'snapshot')
    assert report['unique_rgb'] == 2  # Counter says 3, frozen inventory has 3.
    assert report['processed_packs'] == 1 and report['processed_entries'] == 2
    assert not report['training_admission'] and not report['pack_payloads_verified']
    assert report['previously_source_reviewed_rgb'] == 0
    with sqlite3.connect(report['index']['path']) as db:
        assert db.execute('SELECT SUM(native_flag) FROM candidates').fetchone()[0] == 1
    assert not (tmp_path / 'producer/packs/packs.json').exists()
    assert json.loads((tmp_path / 'producer/status.json').read_text())['stage'] == 'generating_cache'
    writer.lock.close()


def test_receipts_published_after_capture_are_not_added(tmp_path, monkeypatch):
    config, plan, writer = fixture(tmp_path)
    capture = snapshot.capture_sources

    def with_later_receipt(*args):
        result = capture(*args)
        # Copying a receipt is intentionally invalid as a new generation pack;
        # the point is that no late pack can enter an already frozen list.
        p = tmp_path / 'producer/packs'
        (p / 'pack-000001.json').write_bytes((p / 'pack-000000.json').read_bytes())
        return result

    monkeypatch.setattr(snapshot, 'capture_sources', with_later_receipt)
    report = snapshot.run(config, plan, tmp_path / 'snapshot')
    assert report['processed_packs'] == 1 and report['unique_rgb'] == 2
    writer.lock.close()


def test_forged_candidate_rgb_and_duplicate_candidate_fail(tmp_path):
    config, plan, writer = fixture(tmp_path)
    path = tmp_path / 'producer/packs/pack-000000.json'
    receipt = json.loads(path.read_text())
    receipt['entries'][0]['image']['group'] = 'hypersim:other'
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='original frozen inventory'):
        snapshot.run(config, plan, tmp_path / 'changed')
    receipt['entries'][0]['image']['group'] = 'hypersim:scene0'
    receipt['entries'].append(receipt['entries'][0])
    path.write_text(json.dumps(receipt))
    with pytest.raises(sqlite3.IntegrityError):
        snapshot.run(config, plan, tmp_path / 'duplicate')
    writer.lock.close()


def test_exclusion_cannot_refer_to_another_original_ordinal(tmp_path):
    config, plan, writer = fixture(tmp_path)
    path = tmp_path / 'exclusions.jsonl'
    path.write_text(json.dumps(dict(sha256=f'{0:064x}', image=str(tmp_path / '0.png'),
        dataset='hypersim', corpus_ordinal=2, decision='exclude', evidence=[dict(test=True)])) + '\n')
    plan['exclusions'] = file_ref(path)
    with pytest.raises(ValueError, match='frozen ordinal'):
        snapshot.run(config, plan, tmp_path / 'bad-exclusion')
    writer.lock.close()
