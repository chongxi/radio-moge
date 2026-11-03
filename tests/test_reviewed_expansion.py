from copy import deepcopy
import json
import sqlite3

import numpy as np
import pytest

from audit_training_crop_support import choose_samples, observation_metrics, run
from prepare_expanded_reviewed_training import verify_isolated_geometry
from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import publish_json
from radio_moge.geometry_sampling import centered_view, select_observations
from test_geometry_sampling import packed_fixture


def isolated_fixture():
    db = sqlite3.connect(':memory:')
    db.executescript("CREATE TABLE images(ordinal,dataset,scene_group,sha256,disposition);"
        "ATTACH DATABASE ':memory:' AS isolated;"
        "CREATE TABLE isolated.training(ordinal,dataset,scene_group,image_sha256,raw_hash,oriented_hash);")
    db.execute("INSERT INTO images VALUES(10,'structured3d','scene','rgb','selected')")
    db.execute("INSERT INTO isolated.training VALUES(10,'structured3d','scene','rgb','raw','oriented')")
    db.execute("INSERT INTO images VALUES(11,'structured3d','excluded','other','excluded')")
    return db


def test_new_geometry_must_be_exactly_inside_previously_isolated_rgb():
    with isolated_fixture() as db:
        assert verify_isolated_geometry(db, {'structured3d': 1}) == 1
        with pytest.raises(ValueError, match='counts differ'):
            verify_isolated_geometry(db, {'structured3d': 2})
    for column, value in [('ordinal', 12), ('dataset', 'coco2017'), ('scene_group', 'other'),
                          ('image_sha256', 'changed'), ('raw_hash', None), ('oriented_hash', None)]:
        with isolated_fixture() as db:
            db.execute(f'UPDATE isolated.training SET {column}=?', (value,))
            with pytest.raises(ValueError, match='outside the verified'):
                verify_isolated_geometry(db, {'structured3d': 1})


def test_crop_selection_covers_both_policies_and_keeps_extremes_deterministically():
    rows = [(i, 'scene' + str(i // 3), .05 + i / 1000, 1 + i,
             'old' if i < 60 else 'new') for i in range(120)]
    chosen = choose_samples(rows, groups_per_policy=5, support_extremes=2, depth_extremes=1)
    assert chosen == choose_samples(list(reversed(rows)), 5, 2, 1)
    assert {0, 1, 59, 60, 61, 119} <= set(chosen)
    assert sum('deterministic_scene_sample' in v for v in chosen.values()) == 10
    with pytest.raises(ValueError, match='Unique nonempty'):
        choose_samples(rows + rows[:1])


def crop_fixture():
    depth = np.full((24, 32), 2., np.float32)
    valid = np.ones(depth.shape, bool)
    valid[:2] = False
    k = np.array([[1., 0., .5], [0., 1., .5], [0., 0., 1.]])
    view = centered_view(k, depth.shape, (20, 32), (.55, .45), 1.1, True)
    target = select_observations(depth, valid, k, view)
    target['supervision'] = 'metric_native'
    return depth, valid, k, target, view


def test_crop_metrics_verify_exact_measurements_and_expose_sparse_views():
    args = crop_fixture()
    result = observation_metrics(*args)
    assert result['selected_observations'] > 32 and result['stable_edge_pairs'] > 0
    assert result['native_to_target_max_relative_error'] < 1e-6
    depth, valid, k, target, view = args
    sparse = deepcopy(target)
    sparse['valid'][:] = False
    result = observation_metrics(depth, valid, k, sparse, view)
    assert result['sparse_geometry_skipped'] and result['stable_edge_pairs'] == 0
    for kind in ('duplicate', 'invented', 'geometry', 'query'):
        changed = deepcopy(target)
        cells = np.argwhere(changed['valid'])[:2]
        a, b = (tuple(cell) for cell in cells)
        if kind == 'duplicate':
            changed['native_index'][a] = changed['native_index'][b]
        elif kind == 'invented':
            changed['native_index'][a] = 0  # Original unsupported row.
        elif kind == 'geometry':
            changed['points'][a][2] += 1.
        else:
            changed['query_uv'][a][0] += .01
        with pytest.raises(ValueError):
            observation_metrics(depth, valid, k, changed, view)


def test_real_packed_loader_crop_probe_and_changed_rgb_failure(tmp_path):
    pack, entry = packed_fixture(tmp_path)
    index = tmp_path / 'accounting.sqlite'
    with sqlite3.connect(index) as db:
        db.execute('CREATE TABLE images(ordinal,dataset,scene_group,valid_fraction,median_depth,'
                   'entry_json,pack,disposition)')
        db.execute('INSERT INTO images VALUES(?,?,?,?,?,?,?,?)', (10, 'arkitscenes', 'scene',
            1., 2., json.dumps(entry), str(pack), 'selected'))
    report_path = tmp_path / 'accounting.json'
    publish_json(report_path, dict(stage='accounting_verified_not_accepted', unresolved=0,
        selected_counts={'arkitscenes': 1}, index=file_ref(index)))
    result = run(report_path, tmp_path / 'good', 'arkitscenes', (20, 32))
    assert result['images'] == 1 and result['views'] == 3
    assert not result['training_admission'] and not result['model_forward_backward_tested']
    assert result['original_support_unchanged'] and result['actual_production_loader_used']
    from pathlib import Path
    Path(entry['image']['image']).write_bytes(b'changed RGB')
    with pytest.raises(ValueError, match='Frozen RGB bytes changed'):
        run(report_path, tmp_path / 'bad', 'arkitscenes', (20, 32))
    assert not (tmp_path / 'bad/report.json').exists()
