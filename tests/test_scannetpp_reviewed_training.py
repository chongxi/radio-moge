import sqlite3

import pytest

from prepare_scannetpp_reviewed_training import add_exact_scan_scope, verify_canonical_scope


def scope_fixture():
    db = sqlite3.connect(':memory:')
    db.executescript("ATTACH DATABASE ':memory:' AS original; ATTACH DATABASE ':memory:' AS routed;"
        "CREATE TABLE original.images(ordinal,dataset,image,sha256,scene_group);"
        "CREATE TABLE routed.images(ordinal,image,sha256,scene_group,outcome);"
        "CREATE TABLE images(ordinal,dataset,image,sha256,scene_group);")
    db.executemany('INSERT INTO original.images VALUES(?,?,?,?,?)',
        [(i,'scannetpp_dslr',str(i),f'hash{i}',f'scene{i%25}') for i in range(35784)])
    db.executemany('INSERT INTO routed.images VALUES(?,?,?,?,?)',
        [(i,str(i),f'hash{i}',f'scene{i%25}','teacher_candidate') for i in range(35783)])
    return db


def test_only_exact_reviewed_scan_subset_is_added_not_the_whole_source():
    with scope_fixture() as db:
        add_exact_scan_scope(db, 25)
        assert db.execute('SELECT COUNT(*) FROM images').fetchone()[0] == 35783
        assert db.execute('SELECT COUNT(*) FROM images WHERE ordinal=35783').fetchone()[0] == 0


def test_changed_or_extra_source_identity_blocks_expansion():
    for change in ("UPDATE routed.images SET sha256='changed' WHERE ordinal=1",
                   "UPDATE routed.images SET ordinal=40000 WHERE ordinal=1",
                   "UPDATE routed.images SET outcome='rejected' WHERE ordinal=1",
                   'DELETE FROM routed.images WHERE ordinal=1'):
        with scope_fixture() as db:
            db.execute(change)
            with pytest.raises(ValueError):
                add_exact_scan_scope(db, 25)
            assert db.execute('SELECT COUNT(*) FROM images').fetchone()[0] == 0


def canonical_fixture():
    db = sqlite3.connect(':memory:')
    db.executescript("CREATE TABLE images(ordinal,dataset,scene_group,sha256,disposition);"
        "ATTACH DATABASE ':memory:' AS rgb; ATTACH DATABASE ':memory:' AS heldout;"
        "ATTACH DATABASE ':memory:' AS compared;"
        "CREATE TABLE rgb.images(ordinal,dataset,scene_group,image_sha256,raw_canonical_rgb_sha256,canonical_rgb_sha256,error);"
        "CREATE TABLE heldout.images(scene_group,scene_disjoint_required);"
        "CREATE TABLE compared.overlaps(train_ordinal,heldout_id);")
    db.executemany('INSERT INTO images VALUES(?,?,?,?,?)',
        [(1,'scannetpp_dslr','scene1','one','selected'),(2,'scannetpp_dslr','scene2','two','excluded')])
    db.executemany('INSERT INTO rgb.images VALUES(?,?,?,?,?,?,?)',
        [(1,'scannetpp_dslr','scene1','one','raw1','rgb1',None),
         (2,'scannetpp_dslr','scene2','two','raw2','rgb2',None)])
    return db


def test_complete_canonical_binding_includes_excluded_geometry():
    expected = {'scannetpp_dslr':2}
    with canonical_fixture() as db:
        assert verify_canonical_scope(db, expected) == 2
        db.execute("INSERT INTO heldout.images VALUES('scene1',0)")
        assert verify_canonical_scope(db, expected) == 2
        db.execute('INSERT INTO compared.overlaps VALUES(999,1)')
        assert verify_canonical_scope(db, expected) == 2
    for change in ("UPDATE rgb.images SET image_sha256='different' WHERE ordinal=1",
                   'DELETE FROM rgb.images WHERE ordinal=2',
                   'INSERT INTO compared.overlaps VALUES(2,1)',
                   "INSERT INTO heldout.images VALUES('scene2',1)"):
        with canonical_fixture() as db:
            db.execute(change)
            with pytest.raises(ValueError):
                verify_canonical_scope(db, expected)
