from copy import deepcopy
import json
import sqlite3

import numpy as np
from PIL import Image
import pytest

from cache_scannetpp_review_teacher import anonymized_support, review_scope
from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import publish_json


def fixture(root):
    scenes = []
    with sqlite3.connect(root / 'inventory.sqlite') as db:
        db.execute('CREATE TABLE images(ordinal,image,row_json,dataset,scene_group)')
        for i, (name, status) in enumerate([('flagged', 'needs_review'), ('good', 'source_checks_passed')]):
            for frame in range(3):
                row = dict(dataset='scannetpp_dslr', split='train', official_split='nvs_sem_train',
                           scene=name, group='scannetpp:' + name, view=str(frame), sha256=f'{3*i+frame:064x}')
                db.execute('INSERT INTO images VALUES (?,?,?,?,?)',
                    (i * 3 + frame, f'/rgb/{name}/{frame}.jpg', json.dumps(row), row['dataset'], row['group']))
            p = root / (name + '.json')
            publish_json(p, dict(scene=name, stage=status, images_in_training_manifest=3))
            scenes.append(dict(scene=name, stage=status, report=p.name, sha256=file_ref(p)['sha256']))
    audit = root / 'report.json'
    publish_json(audit, dict(stage='complete', manifest_sha256='a' * 64, scenes_completed=2,
        expected_scenes=2, scenes=scenes, scenes_needing_review=1))
    return dict(source_audit=file_ref(audit), scene_counts=dict(flagged=3), scope_images=3)


def test_exact_review_subset_and_two_views_per_scene(tmp_path):
    route = fixture(tmp_path)
    rows, probes = review_scope(tmp_path, route, 'a' * 64)
    assert len(rows) == 3 and [r['view'] for r in probes] == ['0', '2']
    assert {r['scene'] for r in rows} == {'flagged'}
    assert [r['corpus_ordinal'] for r in rows] == [0, 1, 2]
    changed = deepcopy(route)
    changed['scene_counts']['good'] = 3
    with pytest.raises(ValueError, match='routing differs'):
        review_scope(tmp_path, changed, 'a' * 64)


def test_changed_audit_and_heldout_rows_cannot_enter_fallback(tmp_path):
    route = fixture(tmp_path)
    with sqlite3.connect(tmp_path / 'inventory.sqlite') as db:
        row = json.loads(db.execute('SELECT row_json FROM images WHERE ordinal=0').fetchone()[0])
        row['split'] = 'test'
        db.execute('UPDATE images SET row_json=? WHERE ordinal=0', (json.dumps(row),))
    with pytest.raises(ValueError, match='frozen original training'):
        review_scope(tmp_path, route, 'a' * 64)
    (tmp_path / 'flagged.json').write_text('{}')
    with pytest.raises(ValueError, match='Evidence changed'):
        review_scope(tmp_path, route, 'a' * 64)


def test_anonymity_never_adds_support_or_resizes_uncertain_mask(tmp_path):
    mask = np.full((20, 30), 255, np.uint8)
    mask[10, 10] = 0
    mask[10, 20] = 254
    path = tmp_path / 'mask.png'
    Image.fromarray(mask).save(path)
    valid = np.ones_like(mask, bool)
    valid[5, 5] = False
    support, digest = anonymized_support(dict(mask=str(path), height=20, width=30), valid)
    assert support.sum() > 200 and not (support & ~valid).any()
    assert not support[8:13, 8:13].any() and not support[8:13, 18:23].any()
    assert not support[:2].any() and not support[:, :2].any()
    assert digest == file_ref(path)['sha256']
    with pytest.raises(ValueError, match='unchanged teacher/RGB grid'):
        anonymized_support(dict(mask=str(path), height=20, width=30), valid[::2, ::2])
