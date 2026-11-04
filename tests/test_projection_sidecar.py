from copy import deepcopy
import json
import sqlite3

import pytest

from radio_moge.corpus_admission import file_ref
from radio_moge.ordinary_geometry_data import GeometrySidecar
from radio_moge.projection_sidecar import STAGE, validate_derivation
from test_ordinary_geometry_data import fixture


def derivative(tmp_path):
    rows, parent_ref = fixture(tmp_path)
    parent = json.loads(open(parent_ref['path']).read())
    accounting = json.loads(open(parent['accounting_report']['path']).read())
    path = tmp_path / 'index.sqlite'
    with sqlite3.connect(path) as db:
        db.execute("ALTER TABLE images ADD COLUMN dataset TEXT DEFAULT 'fixture'")
        row = dict(rows[1], dataset='coco2017')
        db.execute("UPDATE images SET disposition='projection_hold',dataset='coco2017',row_json=? WHERE ordinal=101",
                   (json.dumps(row),))
        db.execute('DELETE FROM samples WHERE ordinal=101')
    gate = dict(schema_version=1, stage=STAGE, training_admission=False,
        index=file_ref(path), manifest_sha256=parent['manifest_sha256'], selected_images=2, total_images=4)
    hold = dict(geometry=dict(input=parent['index'], output=gate['index'],
        selected_images=2, removed_selected_images=1,
        prior_selected_counts=dict(fixture=2, coco2017=1), selected_counts=dict(fixture=2),
        other_source_selection_unchanged=True, prior_exclusions_and_candidate_evidence_preserved=True,
        training_admission=False))
    return rows, row, gate, parent, accounting, hold


def test_exact_derivative_keeps_exclusions_and_never_admits_training(tmp_path, monkeypatch):
    from radio_moge import ordinary_geometry_data
    rows, held, gate, parent, accounting, hold = derivative(tmp_path)
    counts = validate_derivation(gate, parent, accounting, hold)
    assert counts == dict(selected=2, projection_hold=1, excluded=1)
    monkeypatch.setattr(ordinary_geometry_data, 'checked_derivation', lambda _: counts)
    path = tmp_path / 'derivative.json'
    path.write_text(json.dumps(gate))
    sidecar = GeometrySidecar(file_ref(path))
    assert not sidecar.training_admission
    assert sidecar.lookup(rows[3]) is not None
    assert sidecar.lookup(rows[6]) is None
    with pytest.raises(ValueError, match='Unresolved'):
        sidecar.lookup(held)


@pytest.mark.parametrize('field,value', [('training_admission', True), ('selected_images', 3),
                                        ('manifest_sha256', 'different'), ('total_images', 5)])
def test_changed_derivative_contract_is_rejected(tmp_path, field, value):
    _, _, gate, parent, accounting, hold = derivative(tmp_path)
    with pytest.raises(ValueError, match='exact source-hold'):
        validate_derivation(dict(gate, **{field: value}), parent, accounting, hold)


def test_no_other_source_selection_or_source_flags_can_be_changed(tmp_path):
    _, _, gate, parent, accounting, hold = derivative(tmp_path)
    for key, value in [('selected_counts', dict(fixture=1, coco2017=1)),
                       ('prior_exclusions_and_candidate_evidence_preserved', False),
                       ('other_source_selection_unchanged', False)]:
        altered = deepcopy(hold)
        altered['geometry'][key] = value
        with pytest.raises(ValueError, match='exact source-hold'):
            validate_derivation(gate, parent, accounting, altered)
