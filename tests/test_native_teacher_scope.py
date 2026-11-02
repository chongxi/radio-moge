from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from audit_hypersim_outcomes import run
sys.path.pop(0)

from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import publish_json, sha_file
from radio_moge.native_teacher_scope import review_scope, select_preflight, require_visual_preflight
from test_hypersim_outcomes import fixture


def route_fixture(root):
    config = fixture(root)
    audit = run(config, 'native', 'audit')
    pending = [json.loads(line) for line in (root / 'audit/unresolved-native.jsonl').read_text().splitlines()]
    route = dict(dataset='hypersim', manifest_sha256=config['manifest_sha256'],
        label_kind='teacher_pseudo', metric_calibrated=False, fill_native_holes=False,
        training_admission=False, native_outcomes=file_ref(root / 'audit/report.json'),
        scope_images=3, scene_counts=audit['unresolved_scene_counts'],
        reason_counts=dict(Counter(r['native_reason'] for r in pending)), evidence=[file_ref(root / 'audit/report.json')])
    return config, route


def test_fallback_selects_all_unresolved_rows_and_no_unflagged_native_rows(tmp_path):
    config, route = route_fixture(tmp_path)
    rows, bindings = review_scope(tmp_path, route, config)
    assert [row['corpus_ordinal'] for row in rows] == [1, 2, 3]
    assert {r['native_outcome'] for r in bindings.values()} == {'flagged_cache', 'rejected'}
    assert {r['native_reason'] for r in bindings.values()} == set(route['reason_counts'])
    probes = select_preflight(rows, bindings)
    assert probes == rows
    assert select_preflight(list(reversed(rows)), bindings) == probes
    wrong = deepcopy(route)
    wrong['scope_images'] = 2
    with pytest.raises(ValueError, match='native outcomes'):
        review_scope(tmp_path, wrong, config)


def test_fallback_cannot_change_original_rgb_or_use_heldout(tmp_path):
    config, route = route_fixture(tmp_path)
    with sqlite3.connect(tmp_path / 'inventory.sqlite') as db:
        row = json.loads(db.execute('SELECT row_json FROM images WHERE ordinal=1').fetchone()[0])
        row['split'] = 'val'
        db.execute('UPDATE images SET row_json=? WHERE ordinal=1', (json.dumps(row),))
    with pytest.raises(ValueError, match='frozen RGB identity differ'):
        review_scope(tmp_path, route, config)
    with pytest.raises(ValueError, match='nonmetric'):
        review_scope(tmp_path, dict(route, metric_calibrated=True), config)


def test_preflight_covers_every_scene_failure_class_first_last():
    rows, bindings = [], {}
    for group in range(3):
        for reason in ('pose', 'support', 'ray'):
            for frame in range(5):
                ordinal = len(rows)
                rows.append(dict(corpus_ordinal=ordinal, group=str(group), frame=frame))
                bindings[ordinal] = dict(native_reason=reason)
    probes = select_preflight(rows, bindings)
    assert len(probes) == 18
    assert {r['frame'] for r in probes} == {0, 4}
    assert len({(r['group'], bindings[r['corpus_ordinal']]['native_reason']) for r in probes}) == 9


def test_bulk_requires_exact_numeric_report_and_all_scene_visual_evidence(tmp_path):
    # Test-only fixture statements; these never approve real source data.
    previous = tmp_path / 'report.json'
    publish_json(previous, dict(stage='complete', passed=True, policy_sha256='a' * 64))
    with pytest.raises(FileNotFoundError):
        require_visual_preflight(previous, 'a' * 64, ['a', 'b'])
    proof = tmp_path / 'fixture.txt'
    proof.write_text('Synthetic fixture, not visual evidence for production')
    review = dict(stage='visual_review_complete', passed=True, policy_sha256='a' * 64,
                  preflight_report_sha256=sha_file(previous), reviewed_scenes=['a'], evidence=[file_ref(proof)])
    path = tmp_path / 'visual-review.json'
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match='all-scene visual preflight'):
        require_visual_preflight(previous, 'a' * 64, ['a', 'b'])
    review['reviewed_scenes'] = ['a', 'b']
    path.write_text(json.dumps(review))
    require_visual_preflight(previous, 'a' * 64, ['a', 'b'])
    proof.write_text('changed fixture')
    with pytest.raises(ValueError, match='Evidence changed'):
        require_visual_preflight(previous, 'a' * 64, ['a', 'b'])
