from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from audit_hypersim_teacher_routes import reconcile_review
from radio_moge.geometry_cache import GeometryPackWriter, canonical_json
from radio_moge.teacher_rgb_review import REASON


def fixture(tmp_path):
    policy = dict(dataset='hypersim', fixture_only=True)
    policy_hash = hashlib.sha256(canonical_json(policy)).hexdigest()
    rows = [dict(corpus_ordinal=i, image=str(tmp_path / f'{i}.png'), sha256=f'{i:064x}',
        dataset='hypersim', split='train', group='hypersim:test', width=8, height=8) for i in (10, 11, 12)]
    bindings = {10: dict(native_outcome='flagged_cache', native_reason='world_ray_mismatch_over_one_percent'),
                11: dict(native_outcome='rejected', native_reason='census_quarantined_native_pose'),
                12: dict(native_outcome='rejected', native_reason='insufficient_world_verified_support')}
    writer = GeometryPackWriter(tmp_path / 'packs', policy)
    for row in rows[:2]:
        native = bindings[row['corpus_ordinal']]
        meta = dict(image_sha256=row['sha256'], dataset='hypersim', source_id=str(row['corpus_ordinal']),
            label_kind='teacher_pseudo', camera_kind='teacher_estimated', metric_calibrated=False,
            teacher_revision='b' * 40, conversion='test_only', policy_sha256=policy_hash,
            native_source_review_unresolved=True, **native,
            source_proofs=[dict(source='test fixture', sha256='f' * 64)])
        writer.add(row, np.ones((8, 8), np.float32), np.ones((8, 8), bool), np.eye(3), meta)
    writer.close()
    receipt = json.loads((tmp_path / 'packs/pack-000000.json').read_text())
    proof = dict(path=str(tmp_path / 'fixture-evidence'), sha256='e' * 64)
    entries = [(e, proof) for e in receipt['entries']]
    excluded = {12: dict(channel_min=[255, 255, 255])}
    rejections = [dict(image=rows[2], reason=REASON, teacher_inference_performed=False,
                       constant_rgb=[255, 255, 255], rgb_input_review=proof)]
    return dict(rows=rows, bindings=bindings, entries=entries, rejections=rejections, excluded=excluded,
                policy_hash=policy_hash, revision='b' * 40, minimum_support=.05,
                rgb_review_ref=proof, rejection_ref=proof)


def test_one_route_per_original_retains_native_failures_and_blank_exclusion(tmp_path):
    args = fixture(tmp_path)
    values = reconcile_review(**args)
    assert set(values) == {10, 11, 12}
    assert values[10]['route'] == values[11]['route'] == 'teacher_candidate'
    assert values[10]['native_outcome'] == 'flagged_cache'
    assert values[11]['native_reason'] == 'census_quarantined_native_pose'
    assert values[12]['route'] == 'reviewed_input_exclusion'
    assert args['rows'][0]['corpus_ordinal'] == 10  # Source objects stay unchanged.


def test_duplicate_changed_missing_and_excluded_candidate_rows_fail(tmp_path):
    original = fixture(tmp_path)
    changes = []
    a = deepcopy(original)
    a['entries'].append(a['entries'][0])
    changes.append(a)
    a = deepcopy(original)
    a['entries'][0][0]['image']['group'] = 'hypersim:other'
    changes.append(a)
    a = deepcopy(original)
    a['entries'][0][0]['image']['corpus_ordinal'] = 99
    changes.append(a)
    a = deepcopy(original)
    a['entries'][0][0]['image']['corpus_ordinal'] = 12
    changes.append(a)
    changes.append(dict(original, rejections=[]))
    for args in changes:
        with pytest.raises(ValueError):
            reconcile_review(**args)


def test_teacher_cannot_clear_native_flags_or_claim_calibrated_metric_depth(tmp_path):
    original = fixture(tmp_path)
    for key, value in (('native_source_review_unresolved', False), ('native_reason', 'fixed'),
                       ('native_outcome', 'unflagged_cache'), ('metric_calibrated', True),
                       ('dataset', 'irs'), ('teacher_revision', 'wrong')):
        args = deepcopy(original)
        entry = args['entries'][0][0]
        entry['geometry_metadata'][key] = value
        entry['key'] = hashlib.sha256(canonical_json(entry['geometry_metadata'])).hexdigest()
        with pytest.raises(ValueError, match='provenance|native flags'):
            reconcile_review(**args)


def test_blank_exclusion_is_exact_and_other_rejections_stay_unreviewed(tmp_path):
    original = fixture(tmp_path)
    for key, value in (('teacher_inference_performed', True), ('constant_rgb', [0, 0, 0]),
                       ('reason', 'insufficient_teacher_support'), ('rgb_input_review', {})):
        args = deepcopy(original)
        args['rejections'][0][key] = value
        with pytest.raises(ValueError, match='blank RGB rejection'):
            reconcile_review(**args)
    args = deepcopy(original)
    args['entries'] = args['entries'][:1]
    args['rejections'].append(dict(image=args['rows'][1], reason='insufficient_teacher_support'))
    values = reconcile_review(**args)
    assert values[11]['route'] == 'teacher_rejection_needs_review'
