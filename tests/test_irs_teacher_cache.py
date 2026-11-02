import json
import sqlite3

import pytest

from cache_irs_teacher_geometry import rows


def inventory(root):
    with sqlite3.connect(root / 'inventory.sqlite') as db:
        db.execute('CREATE TABLE images(ordinal INTEGER,image TEXT,row_json TEXT,dataset TEXT,scene_group TEXT)')
        ordinal = 0
        for family in range(13):
            for eye in ('left', 'right'):
                for frame in range(3):
                    row = dict(dataset='irs', split='train', eye=eye, group=f'irs:family:{family}',
                               image_rotation_cw=0, frame=frame)
                    db.execute('INSERT INTO images VALUES (?,?,?,?,?)',
                               (ordinal, f'/rgb/{ordinal}.png', json.dumps(row), 'irs', row['group']))
                    ordinal += 1


def test_probe_covers_both_eyes_all_families_and_first_last(tmp_path):
    inventory(tmp_path)
    selected = list(rows(tmp_path, True))
    assert len(selected) == 52
    assert len({(r['group'], r['eye']) for r in selected}) == 26
    assert {r['frame'] for r in selected} == {0, 2}
    assert len(list(rows(tmp_path, False))) == 78
    assert len({r['corpus_ordinal'] for r in selected}) == 52


def test_no_deferred_rotation_or_heldout_rgb(tmp_path):
    inventory(tmp_path)
    with sqlite3.connect(tmp_path / 'inventory.sqlite') as db:
        row = json.loads(db.execute('SELECT row_json FROM images WHERE ordinal=0').fetchone()[0])
        row['split'] = 'test'
        db.execute('UPDATE images SET row_json=? WHERE ordinal=0', (json.dumps(row),))
    with pytest.raises(ValueError):
        list(rows(tmp_path, True))
