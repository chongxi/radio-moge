import json
import sqlite3

import pytest

from radio_moge.cache_recovery import recover_closed_packs, recovered_rows
from radio_moge.geometry_cache import GeometryPackWriter, publish_json, sha_file
from test_geometry_cache import example


def fixture(root):
    original = root / 'failed'
    original.mkdir()
    config = dict(root=str(root), manifest_sha256='e' * 64)
    writer = GeometryPackWriter(original / 'packs', dict(dataset='nyuv2',
        manifest_sha256=config['manifest_sha256']), max_entries=1)
    row, d, v, k, p = example(writer.policy_hash)
    row.update(image=str(root / 'image.png'))
    writer.add(row, d, v, k, p)
    writer.close()
    # This uncommitted payload must not be trusted, altered, or counted.
    (original / 'packs/pack-000001.tar.incoming').write_bytes(b'uncommitted data')
    publish_json(original / 'status.json', dict(stage='failed', completed=2,
        optimizer_updates=0, policy_sha256=writer.policy_hash))
    with sqlite3.connect(root / 'inventory.sqlite') as db:
        db.execute('CREATE TABLE images(ordinal INTEGER,image TEXT,sha256 TEXT,row_json TEXT)')
        db.execute('INSERT INTO images VALUES (?,?,?,?)', (0,row['image'],row['sha256'],json.dumps(row)))
    publish_json(root / 'inventory.json', dict(manifest_sha256=config['manifest_sha256'],
                                              inventory_sha256=sha_file(root / 'inventory.sqlite')))
    return config, original, row


def test_recovery_only_publishes_verified_closed_subset_and_preserves_original(tmp_path):
    config, original, row = fixture(tmp_path)
    before = {str(p): sha_file(p) for p in original.rglob('*') if p.is_file()}
    report = recover_closed_packs(config, 'failed', 'recovered')
    assert report['completed'] == report['expected_scope'] == 1
    assert report['recovered_verified_subset'] and not report['training_admission']
    assert before == {str(p): sha_file(p) for p in original.rglob('*') if p.is_file()}
    ids, proof = recovered_rows(tmp_path, 'recovered', 'nyuv2', config['manifest_sha256'])
    assert ids == {row['sha256']} and proof['sha256'] == sha_file(tmp_path / 'recovered/report.json')
    with pytest.raises(ValueError):
        recovered_rows(tmp_path, 'recovered', 'structured3d', config['manifest_sha256'])
    (tmp_path / 'recovered/scope.jsonl').write_text('')
    with pytest.raises(ValueError, match='changed'):
        recovered_rows(tmp_path, 'recovered', 'nyuv2', config['manifest_sha256'])


def test_corrupt_closed_pack_blocks_recovery(tmp_path):
    config, original, _ = fixture(tmp_path)
    with (original / 'packs/pack-000000.tar').open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError, match='identity differs'):
        recover_closed_packs(config, 'failed', 'recovered')
    assert not (tmp_path / 'recovered/report.json').exists()
    assert json.loads((tmp_path / 'recovered/status.json').read_text())['stage'] == 'failed'
