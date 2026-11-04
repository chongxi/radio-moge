import json
import sqlite3

import pytest

from prepare_scannetpp_native_v2_validation import select


def fixture():
    db = sqlite3.connect(':memory:')
    db.execute('CREATE TABLE images(ordinal INTEGER,scene_group TEXT,row_json TEXT,outcome TEXT,'
               'policy TEXT,evidence_json TEXT,entry_key TEXT,details_json TEXT)')
    for ordinal in range(200):
        scene = f'{ordinal//3:010d}'
        group = 'scannetpp:'+scene
        row = dict(dataset='scannetpp_dslr', split='train', official_split='nvs_sem_train',
                   group=group, scene=scene)
        db.execute('INSERT INTO images VALUES (?,?,?,?,?,?,?,?)',
            (ordinal, group, json.dumps(row), 'native_candidate', 'policy', '{}', str(ordinal), '{}'))
    return db


def test_frozen_selection_is_repeatable_and_scene_disjoint_without_outcome_tuning():
    db = fixture()
    first = select(db, list(range(19)))
    assert first == select(db, list(reversed(range(19))))
    assert len(first) == len({r['corpus_ordinal'] for r in first}) == 51
    groups = [{r['group'] for r in first if r['role'] == role} for role in
              ('known_quarantine', 'ordinary_control', 'confirmation')]
    assert len(groups[1]) == len(groups[2]) == 16
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    db.execute("UPDATE images SET details_json='{}'")
    assert first == select(db, list(range(19)))


def test_missing_duplicate_or_changed_split_quarantine_fails():
    db = fixture()
    for values in [list(range(18)), [0]*19, list(range(18))+[999]]:
        with pytest.raises(ValueError):
            select(db, values)
    row = json.loads(db.execute('SELECT row_json FROM images WHERE ordinal=0').fetchone()[0])
    row['split'] = 'test'
    db.execute('UPDATE images SET row_json=? WHERE ordinal=0', (json.dumps(row),))
    with pytest.raises(ValueError, match='training-scene'):
        select(db, list(range(19)))
