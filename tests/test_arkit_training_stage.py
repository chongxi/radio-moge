import sqlite3

import pytest

from prepare_arkit_reviewed_training import extend_inventory


def fixture():
    db = sqlite3.connect(':memory:')
    db.execute("ATTACH DATABASE ':memory:' AS original")
    schema = '(ordinal PRIMARY KEY,dataset,image,sha256,scene_group,row_json)'
    db.execute('CREATE TABLE images ' + schema)
    db.execute('CREATE TABLE original.images ' + schema)
    rows = [(0, 'scannetpp_dslr', '/kept', 'a', 'scene', '{"kept":true}'),
            (1, 'scannetpp_dslr', '/unreviewed', 'b', 'scene2', '{"kept":false}'),
            (2, 'arkitscenes', '/new', 'c', 'visit', '{"new":true}')]
    db.executemany('INSERT INTO original.images VALUES(?,?,?,?,?,?)', rows)
    db.execute('INSERT INTO images VALUES(?,?,?,?,?,?)', rows[0])
    return db


def test_expansion_preserves_partial_scannet_scope_and_adds_only_arkit():
    with fixture() as db:
        assert extend_inventory(db, dict(scannetpp_dslr=1), 1) == dict(scannetpp_dslr=1, arkitscenes=1)
        assert list(db.execute('SELECT ordinal FROM images ORDER BY ordinal')) == [(0,), (2,)]


@pytest.mark.parametrize('changed', ['image', 'sha256', 'scene_group', 'row_json', 'dataset'])
def test_changed_previous_rgb_or_source_is_rejected(changed):
    with fixture() as db:
        db.execute('UPDATE images SET ' + changed + '=?', ('changed',))
        with pytest.raises(ValueError):
            extend_inventory(db, dict(scannetpp_dslr=1), 1)


def test_incomplete_added_source_and_repeated_expansion_are_rejected():
    with fixture() as db, pytest.raises(ValueError, match='Expanded source count'):
        extend_inventory(db, dict(scannetpp_dslr=1), 2)
    with fixture() as db:
        expected = extend_inventory(db, dict(scannetpp_dslr=1), 1)
        with pytest.raises(ValueError, match='previous source scope'):
            extend_inventory(db, expected, 1)
