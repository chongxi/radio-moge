from copy import deepcopy
import json
import sqlite3

import pytest

from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import publish_json
from snapshot_geometry_candidates import run as snapshot_run
from prepare_snapshot_training_scope import make_inventory, run
from test_candidate_snapshot import fixture


def captured(root):
    config, plan, writer = fixture(root)
    original = root / 'inventory.json'
    info = json.loads(original.read_text())
    manifest = root / 'parent-train.jsonl'
    manifest.write_text('identity anchor for this fixture only\n')
    info.update(manifest=str(manifest), heldout_checks=[])
    publish_json(root / 'inventory-with-manifest.json', info)
    original.write_text(json.dumps(info))
    snapshot = snapshot_run(config, plan, root / 'snapshot')
    writer.lock.close()
    return root / 'snapshot/report.json', snapshot


def test_flagged_without_alternative_is_withheld_only_from_the_frozen_stage(tmp_path):
    source, snapshot = captured(tmp_path)
    result = run(source, tmp_path / 'stage')
    assert result['images'] == snapshot['unique_rgb'] == 2
    assert result['outcome_counts'] == {'pending_source_review': 1, 'stage_quality_deferral': 1}
    assert result['proposed_geometry_images'] == 1
    assert not result['training_admission'] and not result['source_reviews_complete']
    assert not result['payload_verification_complete']
    excluded = [json.loads(v) for v in (tmp_path / 'stage/stage-exclusions.jsonl').read_text().splitlines()]
    assert len(excluded) == 1 and excluded[0]['corpus_ordinal'] == 1
    assert excluded[0]['scope'] == 'this_frozen_training_stage_only'
    assert not excluded[0]['rgb_replay_excluded']
    with sqlite3.connect(result['inventory']['path']) as db:
        assert db.execute('SELECT COUNT(*) FROM images WHERE row_json IS NOT NULL').fetchone()[0] == 2
    # Parent still has the third image; ungenerated labels are outside this stage.
    with sqlite3.connect(tmp_path / 'inventory.sqlite') as db:
        assert db.execute('SELECT COUNT(*) FROM images').fetchone()[0] == 3


def test_changed_receipt_cannot_materialize_a_stage(tmp_path):
    source, _ = captured(tmp_path)
    receipt = tmp_path / 'producer/packs/pack-000000.json'
    v = json.loads(receipt.read_text())
    v['entries'][0]['image']['group'] = 'different-scene'
    receipt.write_text(json.dumps(v))
    with pytest.raises(ValueError, match='Evidence changed'):
        run(source, tmp_path / 'changed')
    assert not (tmp_path / 'changed/report.json').exists()


def test_two_eligible_candidates_are_not_resolved_by_arbitrary_preference(tmp_path):
    _, snapshot = captured(tmp_path)
    with sqlite3.connect(snapshot['index']['path']) as db:
        row = db.execute('SELECT * FROM candidates WHERE native_flag=0').fetchone()
        duplicate = list(deepcopy(row))
        duplicate[0] = 'e' * 64
        duplicate[1] = 'd' * 64
        db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?)', duplicate)
    with pytest.raises(ValueError, match='Ambiguous duplicate'):
        make_inventory(snapshot['index']['path'], tmp_path / 'ambiguous.sqlite')


def test_missing_snapshot_members_and_mutated_ordinals_fail(tmp_path):
    source, snapshot = captured(tmp_path)
    with sqlite3.connect(snapshot['index']['path']) as db:
        db.execute('UPDATE images SET row_sha256=? WHERE ordinal=0', ('f' * 64,))
    snapshot['index'] = file_ref(snapshot['index']['path'])
    source.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match='RGB identity'):
        run(source, tmp_path / 'wrong-row')
    assert not (tmp_path / 'wrong-row/report.json').exists()
