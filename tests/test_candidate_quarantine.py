"""Later visual findings must participate in real accounting, not just reports."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from radio_moge.corpus_admission import (build_accounting, checked_candidate_quarantine,
                                       file_ref, open_training_gate)
from radio_moge.geometry_cache import GeometryPackWriter, publish_json
from test_corpus_admission import admission_fixture, synthetic_seal


def add_flag(root, plan, ordinal=1):
    source = plan['sources'][ordinal]
    packs_path = Path(source['packs_index']['path'])
    receipt_path = packs_path.with_name('pack-000000.json')
    entry = json.loads(receipt_path.read_text())['entries'][0]
    proof = root / 'later-visual-finding.json'
    publish_json(proof, dict(test_fixture_only=True, quality_unresolved=True))
    flag = dict(corpus_ordinal=ordinal, image=entry['image']['image'],
        sha256=entry['image']['sha256'], group=entry['image']['group'],
        entry_key=entry['key'], member_sha256=entry['sha256'], receipt=file_ref(receipt_path),
        decision='quarantine_candidate', reason='Test-only unresolved surface ambiguity',
        evidence=[file_ref(proof)])
    review_path = Path(source['review']['path'])
    review = json.loads(review_path.read_text())
    register_path = Path(review['candidate_quarantine']['path'])
    register = json.loads(register_path.read_text())
    register['flags'] = [flag]
    register_path.write_text(json.dumps(register))
    review['candidate_quarantine'] = file_ref(register_path)
    review_path.write_text(json.dumps(review))
    source['review'] = file_ref(review_path)
    return flag, register_path


def add_separately_reviewed_alternative(root, plan, ordinal=1):
    source = plan['sources'][ordinal]
    old_path = Path(source['packs_index']['path'])
    old_entry = json.loads(old_path.with_name('pack-000000.json').read_text())['entries'][0]
    directory = root / 'alternative'
    directory.mkdir()
    policy = json.loads(old_path.with_name('policy.json').read_text())
    policy['test_only_separate_candidate'] = True
    writer = GeometryPackWriter(directory / 'packs', policy)
    meta = deepcopy(old_entry['geometry_metadata'])
    shape = meta['shape']
    for key in ('schema_version', 'depth_convention', 'units', 'camera_pixel_convention', 'shape', 'valid_pixels'):
        meta.pop(key, None)
    meta.update(policy_sha256=writer.policy_hash, source_needs_review=False)
    k = np.array([[1., 0, .5], [0, 1., .5], [0, 0, 1.]])
    writer.add(old_entry['image'], np.full(shape, 3., np.float32), np.ones(shape, bool), k, meta)
    writer.close()
    generation = json.loads(Path(source['generation_report']['path']).read_text())
    generation['policy_sha256'] = writer.policy_hash
    publish_json(directory / 'report.json', generation)
    publish_json(directory / 'quarantine.json', dict(schema_version=1,
        stage='candidate_quality_quarantine_recorded', dataset=source['dataset'],
        policy_sha256=writer.policy_hash, manifest_sha256=policy['manifest_sha256'],
        packs_index=file_ref(directory / 'packs/packs.json'), flags=[],
        training_admission=False, optimizer_updates=0))
    review = json.loads(Path(source['review']['path']).read_text())
    review.update(policy_sha256=writer.policy_hash,
                  candidate_quarantine=file_ref(directory / 'quarantine.json'))
    publish_json(directory / 'review.json', review)
    plan['sources'].append(dict(dataset=source['dataset'], scope_count=1,
        generation_report=file_ref(directory / 'report.json'), review=file_ref(directory / 'review.json'),
        packs_index=file_ref(directory / 'packs/packs.json')))


def test_visual_flag_without_resolution_stops_accounting(tmp_path):
    root, plan, _ = admission_fixture(tmp_path)
    flag, _ = add_flag(root, plan)
    with pytest.raises(ValueError, match='1 frozen images'):
        build_accounting(plan, root / 'account')
    assert not (root / 'account/report.json').exists()
    with sqlite3.connect(root / 'account/accounting.sqlite.incoming') as db:
        ordinal, native, encoded = db.execute(
            'SELECT ordinal,native_flag,external_flag_json FROM quarantined_candidates').fetchone()
        assert ordinal == 1 and native == 0 and json.loads(encoded) == flag
        assert db.execute('SELECT disposition FROM images WHERE ordinal=1').fetchone() == ('unresolved',)


def test_explicit_exclusion_resolves_flag_without_mutating_rgb_or_other_candidates(tmp_path):
    root, plan, rows = admission_fixture(tmp_path, extra_coco=True)
    add_flag(root, plan)
    proof = file_ref(root / 'later-visual-finding.json')
    path = Path(plan['exclusions']['path'])
    path.write_text(json.dumps(dict(image=rows[1]['image'], sha256=rows[1]['sha256'],
        decision='exclude', reason='Test-only geometry exclusion decision', evidence=[proof],
        scope='geometry_supervision', rgb_replay_excluded=False)) + '\n')
    plan['exclusions'] = file_ref(path)
    before = [file_ref(r['image']) for r in rows]
    result = build_accounting(plan, root / 'account')
    assert result['selected_counts'] == {'hypersim': 1, 'coco2017': 1}
    assert result['excluded_counts'] == {'coco2017': 1}
    assert result['external_flagged_candidate_counts'] == {'coco2017': 1}
    assert result['quarantined_candidate_counts'] == {'coco2017': 1}
    assert not result['training_admission']
    assert [file_ref(r['image']) for r in rows] == before


@pytest.mark.parametrize('alternative_first', [False, True])
def test_quarantine_is_exact_candidate_specific_and_order_independent(tmp_path, alternative_first):
    root, plan, _ = admission_fixture(tmp_path)
    add_flag(root, plan)
    add_separately_reviewed_alternative(root, plan)
    if alternative_first:
        plan['sources'].reverse()
    result = build_accounting(plan, root / 'account')
    assert result['selected_images'] == 2
    assert result['quarantined_candidate_counts'] == {'coco2017': 1}
    with sqlite3.connect(result['index']['path']) as db:
        assert '/alternative/packs/' in db.execute('SELECT pack FROM images WHERE ordinal=1').fetchone()[0]


def test_external_flag_cannot_clear_native_flag_or_double_count_candidate(tmp_path):
    root, plan, _ = admission_fixture(tmp_path, flagged=True)
    add_flag(root, plan, 0)
    add_separately_reviewed_alternative(root, plan, 0)
    result = build_accounting(plan, root / 'account')
    assert result['quarantined_candidate_counts'] == {'hypersim': 1}
    assert result['native_flagged_candidate_counts'] == {'hypersim': 1}
    assert result['external_flagged_candidate_counts'] == {'hypersim': 1}
    with sqlite3.connect(result['index']['path']) as db:
        assert db.execute('SELECT native_flag FROM quarantined_candidates').fetchone() == (1,)


@pytest.mark.parametrize('change', ['missing_register', 'policy', 'manifest', 'packs', 'duplicate',
    'image', 'sha256', 'group', 'entry_key', 'member_sha256', 'receipt', 'no_evidence', 'decision'])
def test_quarantine_requires_bound_policy_member_identity_and_evidence(tmp_path, change):
    root, plan, _ = admission_fixture(tmp_path)
    _, register_path = add_flag(root, plan)
    source = plan['sources'][1]
    review_path = Path(source['review']['path'])
    review = json.loads(review_path.read_text())
    register = json.loads(register_path.read_text())
    flag = register['flags'][0]
    if change == 'missing_register':
        review.pop('candidate_quarantine')
    else:
        if change == 'policy':
            register['policy_sha256'] = 'b' * 64
        elif change == 'manifest':
            register['manifest_sha256'] = 'c' * 64
        elif change == 'packs':
            register['packs_index'] = plan['sources'][0]['packs_index']
        elif change == 'duplicate':
            register['flags'].append(deepcopy(flag))
        elif change == 'receipt':
            flag['receipt'] = file_ref(root / 'hypersim/packs/pack-000000.json')
        elif change == 'no_evidence':
            flag['evidence'] = []
        else:
            flag[change] = 'd' * 64 if change in ('sha256', 'entry_key', 'member_sha256') else 'changed'
        register_path.write_text(json.dumps(register))
        review['candidate_quarantine'] = file_ref(register_path)
    review_path.write_text(json.dumps(review))
    source['review'] = file_ref(review_path)
    with pytest.raises(ValueError, match='quarantine|Quarantined'):
        build_accounting(plan, root / 'account')


def test_external_ordinal_must_match_frozen_inventory_even_for_legacy_receipt(tmp_path):
    root, plan, _ = admission_fixture(tmp_path)
    _, path = add_flag(root, plan)
    register = json.loads(path.read_text())
    register['flags'][0]['corpus_ordinal'] = 999
    path.write_text(json.dumps(register))
    source = plan['sources'][1]
    review_path = Path(source['review']['path'])
    review = json.loads(review_path.read_text())
    review['candidate_quarantine'] = file_ref(path)
    review_path.write_text(json.dumps(review))
    source['review'] = file_ref(review_path)
    with pytest.raises(ValueError, match='frozen ordinal'):
        build_accounting(plan, root / 'account')


@pytest.mark.parametrize('changed', ['register', 'flag_evidence'])
def test_training_gate_rechecks_quarantine_register_and_evidence(tmp_path, changed):
    root, plan, _ = admission_fixture(tmp_path)
    _, register_path = add_flag(root, plan)
    add_separately_reviewed_alternative(root, plan)
    result = build_accounting(plan, root / 'account')
    config = dict(data=dict(train_manifest=str(root / 'train.jsonl')))
    _, _, checkpoint = synthetic_seal(root, result, config)
    assert open_training_gate(config, init=checkpoint)['selected_images'] == 2
    path = register_path if changed == 'register' else root / 'later-visual-finding.json'
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='Evidence changed'):
        open_training_gate(config, init=checkpoint)


def test_flag_evidence_is_checked_before_any_source_acceptance(tmp_path):
    root, plan, _ = admission_fixture(tmp_path)
    _, register_path = add_flag(root, plan)
    source = plan['sources'][1]
    value = json.loads(register_path.read_text())
    args = (file_ref(register_path), source['dataset'], value['policy_sha256'],
            value['manifest_sha256'], source['packs_index'])
    assert len(checked_candidate_quarantine(*args)) == 1
    (root / 'later-visual-finding.json').write_text('{}')
    with pytest.raises(ValueError, match='Evidence changed'):
        checked_candidate_quarantine(*args)
