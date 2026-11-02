import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

import pytest


scripts = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('corpus_index', scripts / 'index_geometry_corpus.py')
index = importlib.util.module_from_spec(spec)
spec.loader.exec_module(index)
sys.path.remove(str(scripts))


def fixture(tmp_path, monkeypatch, rows=None, heldout=None):
    row = dict(dataset='coco2017', split='train', group='coco2017:image:1', image='one.png',
               sha256='1' * 64)
    train = tmp_path / 'train.jsonl'
    rows = rows or [row]
    train.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    validation = tmp_path / 'val.jsonl'
    validation.write_text(json.dumps(heldout or dict(image='two.png', group='coco2017:image:2',
                                                    sha256='2' * 64)) + '\n')
    monkeypatch.setattr(index, 'heldout_manifests', lambda _: [validation])
    return dict(root=str(tmp_path / 'output'), dataset_root=str(tmp_path), manifest=train.name,
        manifest_sha256=hashlib.sha256(train.read_bytes()).hexdigest(),
        expected_counts=dict(index.Counter(r['dataset'] for r in rows)),
        authorization=dict(scope_confirmed=True, training_before_acceptance=False))


def test_inventory_is_not_training_or_pixel_acceptance(tmp_path, monkeypatch):
    config = fixture(tmp_path, monkeypatch)
    result = index.create_index(config)
    assert result['images'] == 1
    assert result['train_gate'] == 'CLOSED'
    assert result['pixel_and_depth_checks'] == 'NOT_YET_PERFORMED'
    assert not result['training_started']
    with sqlite3.connect(Path(config['root']) / 'inventory.sqlite') as db:
        assert db.execute('SELECT COUNT(*) FROM images').fetchone()[0] == 1
    with pytest.raises(FileExistsError):
        index.create_index(config)


@pytest.mark.parametrize('field,value', [('image', 'one.png'),
    ('group', 'coco2017:image:1'), ('sha256', '1' * 64)])
def test_all_three_primary_leakage_checks(tmp_path, monkeypatch, field, value):
    heldout = dict(image='two.png', group='coco2017:image:2', sha256='2' * 64)
    heldout[field] = value
    config = fixture(tmp_path, monkeypatch, heldout=heldout)
    with pytest.raises(ValueError, match='overlap'):
        index.create_index(config)
    assert not (Path(config['root']) / 'inventory.sqlite').exists()


def test_changed_scope_cannot_publish(tmp_path, monkeypatch):
    config = fixture(tmp_path, monkeypatch)
    config['manifest_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='identity'):
        index.create_index(config)
    assert not (Path(config['root']) / 'inventory.sqlite').exists()


def test_irs_official_test_cannot_enter_corpus(tmp_path, monkeypatch):
    rows = [dict(dataset='irs', split='train', official_split='test', group='irs:family:a',
                 source_archive='a.tar.gz', image='one.png', sha256='1' * 64)]
    config = fixture(tmp_path, monkeypatch, rows=rows)
    with pytest.raises(ValueError, match='IRS official test'):
        index.create_index(config)
