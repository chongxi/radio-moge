import json
import sqlite3
from types import SimpleNamespace

import pytest

from radio_moge.corpus_data import CorpusRecords
from radio_moge.projection_hold import require_unheld_sources
from radio_moge.train import train
from test_training import config_for
from test_gt_geometry import gt_config


def test_corpus_checks_sample_membership_without_reading_payloads(tmp_path):
    path = tmp_path / 'index.sqlite'
    with sqlite3.connect(path) as db:
        db.executescript('CREATE TABLE images (ordinal INTEGER,dataset TEXT); '
                         'CREATE TABLE samples (sample INTEGER,ordinal INTEGER); '
                         "INSERT INTO images VALUES (10,'irs'),(11,'coco2017'); "
                         'INSERT INTO samples VALUES (0,10);')
    records = CorpusRecords(dict(stage='accepted_for_training', training_admission=True,
                                 selected_images=1, index=dict(path=str(path))))
    require_unheld_sources(records, 'primary')  # Held provenance alone is allowed.
    with sqlite3.connect(path) as db:
        db.execute('INSERT INTO samples VALUES (1,11)')
    with pytest.raises(ValueError, match='COCO projection hold'):
        require_unheld_sources(records, 'primary')


@pytest.mark.parametrize('role', ['primary', 'validation', 'replay', 'probe'])
def test_list_sources_preserved_and_held_source_rejected(role):
    rows = [dict(dataset='irs', image='unchanged'), dict(dataset='arkitscenes')]
    before = json.dumps(rows)
    require_unheld_sources(rows, role)
    assert json.dumps(rows) == before
    with pytest.raises(ValueError, match='COCO projection hold'):
        require_unheld_sources(rows + [dict(dataset='coco2017')], role)


@pytest.mark.parametrize('slot', ['train_manifest', 'val_manifest', 'train_probe_manifest',
                                'geometry_val_manifest', 'variant', 'replay'])
def test_real_trainer_rejects_before_any_encoding_or_optimizer(tmp_path, slot, monkeypatch):
    import radio_moge.train as module
    config = gt_config(tmp_path) if slot == 'geometry_val_manifest' else config_for(tmp_path)
    held = tmp_path / 'held.jsonl'
    source = json.loads((tmp_path / 'train.jsonl').read_text().splitlines()[0])
    source.update(dataset='coco2017', group='coco2017:held')
    held.write_text(json.dumps(source) + '\n')
    if slot == 'variant':
        config['evaluation']['dataset_variants'] = {
            'held': dict(manifest=str(held), num_tokens=10, max_side=32)}
    elif slot == 'replay':
        monkeypatch.setattr(module, 'load_replay_records', lambda *args: [source])
    else:
        config['data'][slot] = str(held)
    def forbidden(*args, **kwargs):
        raise AssertionError('Held inputs reached model/optimizer')
    monkeypatch.setattr(module.torch.optim, 'AdamW', forbidden)
    monkeypatch.setattr(module, 'build_system', forbidden)
    # A device sentinel tests the production path; it never initializes CUDA.
    monkeypatch.setattr(module, 'Accelerator', lambda **kwargs: SimpleNamespace(
        num_processes=1, device=SimpleNamespace(type='cuda')))
    with pytest.raises(ValueError, match='COCO projection hold'):
        train(config)
