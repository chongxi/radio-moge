from copy import deepcopy
import hashlib
import json
import sqlite3

import numpy as np
import pytest

from audit_structured3d_outcomes import bind, candidate_metrics, rejection_kind, run
from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import GeometryPackWriter, canonical_json, publish_json, sha_file


def fixture(root):
    config = dict(root=str(root), manifest_sha256='a' * 64, expected_counts={'structured3d': 5})
    rows = [dict(dataset='structured3d', split='train', image=str(root / f'{i}.png'),
        sha256=f'{i:064x}', group=f'structured3d:scene:{i:05d}', scene=f'{i:05d}', room='1',
        configuration='empty', width=64, height=64) for i in range(5)]
    errata_path = root / 'native-source/structured3d-metadata/errata.txt'
    errata_path.parent.mkdir(parents=True)
    errata_path.write_text('scene_00002\n')
    proof = dict(source='fixture', sha256=sha_file(errata_path))
    publish_json(errata_path.with_name('errata.txt.receipt.json'), proof)
    with sqlite3.connect(root / 'inventory.sqlite') as db:
        db.execute('CREATE TABLE images(ordinal,image,sha256,scene_group,row_json,dataset)')
        for i, row in enumerate(rows):
            db.execute('INSERT INTO images VALUES (?,?,?,?,?,?)',
                (i, row['image'], row['sha256'], row['group'], json.dumps(row), 'structured3d'))
    publish_json(root / 'inventory.json', dict(stage='inventory_complete',
        manifest_sha256=config['manifest_sha256'], dataset_counts=config['expected_counts'],
        inventory_sha256=sha_file(root / 'inventory.sqlite')))
    for i, name in enumerate(('recovered', 'native')):
        base = root / name
        base.mkdir()
        policy = dict(dataset='structured3d', manifest_sha256=config['manifest_sha256'],
            upstream_errata=proof, minimum_image_valid_fraction=.05, native_holes_filled=False,
            label_kind='synthetic_renderer', revision=i)
        writer = GeometryPackWriter(base / 'packs', policy)
        meta = dict(dataset='structured3d', image_sha256=rows[i]['sha256'], source_id=str(i),
            label_kind='synthetic_renderer', camera_kind='synthetic_camera', metric_calibrated=True,
            conversion='fixture', policy_sha256=writer.policy_hash, source_needs_review=False,
            source_proofs=[proof], source_quality=dict(source_needs_review=False,
                valid_fraction=1., layout_verified_pixels=500, layout_conflicting_pixels=0))
        writer.add(rows[i], np.ones((64, 64), np.float32), np.ones((64, 64), bool), np.eye(3), meta)
        writer.close()
        gen = dict(stage='complete', mode='generate', training_started=False, training_admission=False,
            optimizer_updates=0, completed=1, verified_cache_members=1, policy_sha256=writer.policy_hash,
            expected_scope=1 if i == 0 else 4, rejected=0 if i == 0 else 3)
        if i == 0:
            (base / 'scope.jsonl').write_text(json.dumps(dict(ordinal=0, sha256=rows[0]['sha256'])) + '\n')
            gen.update(recovered_verified_subset=True, scope_sha256=sha_file(base / 'scope.jsonl'))
        else:
            gen.update(recovered_subset=file_ref(root / 'recovered/report.json'), recovered_skipped=1,
                       source_review_images=1)
        publish_json(base / 'report.json', gen)
    rejects = [dict(image=rows[2], reason='upstream_errata_scene_or_room', source_proofs=[proof]),
        dict(image=rows[3], reason='insufficient_verified_support', source_proofs=[proof],
             stats=dict(layout_verified_pixels=0, valid_fraction=0., source_needs_review=True)),
        dict(image=rows[4], reason='insufficient_verified_support', source_proofs=[proof],
             stats=dict(layout_verified_pixels=120, valid_fraction=.04, source_needs_review=False))]
    publish_json(root / 'native/rejections.json', rejects)
    return config, rows, proof


def test_recovered_and_remainder_cover_every_input_once_without_acceptance(tmp_path):
    config, rows, _ = fixture(tmp_path)
    before = {str(p): sha_file(p) for p in tmp_path.rglob('*') if p.is_file()}
    report = run(config, 'recovered', 'native', 'audit')
    assert report['outcome_counts'] == {'native_candidate': 2, 'rejected_needs_disposition': 3}
    assert report['rejection_categories'] == dict(upstream_errata=1,
        insufficient_layout_reference=1, insufficient_valid_fraction=1)
    assert report['rejected_source_flags'] == 1 and report['cached_source_flags'] == 0
    assert len(report['candidate_policy_counts']) == 2
    assert not report['source_quality_accepted'] and not report['training_admission']
    assert not report['independent_array_readback_performed']
    assert all(sha_file(p) == checksum for p, checksum in before.items())
    with sqlite3.connect(report['index']['path']) as db:
        assert db.execute('SELECT COUNT(DISTINCT ordinal),COUNT(DISTINCT image) FROM images').fetchone() == (5, 5)
        with pytest.raises(ValueError, match='duplicate'):
            bind(db, rows[0], 'native_candidate', '', '', {}, 'new-key')
        changed = dict(rows[1], group='wrong')
        with pytest.raises(ValueError, match='Changed'):
            bind(db, changed, 'native_candidate', '', '', {})


@pytest.mark.parametrize('change', ['incomplete_readback', 'missing_rejection', 'duplicate_rejection',
                                  'recovery_identity_changed', 'flags_differ'])
def test_missing_duplicate_or_unverified_scope_cannot_publish(tmp_path, change):
    config, rows, _ = fixture(tmp_path)
    path = tmp_path / 'native/report.json'
    gen = json.loads(path.read_text())
    if change == 'incomplete_readback':
        gen['verified_cache_members'] = 0
    elif change == 'flags_differ':
        gen['source_review_images'] = 0
    elif change in ('missing_rejection', 'duplicate_rejection'):
        p = tmp_path / 'native/rejections.json'
        values = json.loads(p.read_text())
        if change == 'missing_rejection':
            values.pop()
        else:
            values[-1]['image'] = values[-2]['image']
        p.write_text(json.dumps(values))
    else:
        p = tmp_path / 'recovered/scope.jsonl'
        p.write_text(json.dumps(dict(ordinal=1, sha256=rows[1]['sha256'])) + '\n')
        q = tmp_path / 'recovered/report.json'
        value = json.loads(q.read_text())
        value['scope_sha256'] = sha_file(p)
        q.write_text(json.dumps(value))
        gen['recovered_subset'] = file_ref(q)
    path.write_text(json.dumps(gen))
    with pytest.raises(ValueError):
        run(config, 'recovered', 'native', 'bad')
    assert not (tmp_path / 'bad/report.json').exists()


def test_source_flags_units_and_errata_cannot_be_silently_cleared(tmp_path):
    _, rows, proof = fixture(tmp_path)
    receipt = json.loads((tmp_path / 'native/packs/pack-000000.json').read_text())
    original = receipt['entries'][0]
    for name, value in [('units', 'millimeters'), ('source_needs_review', True),
                        ('metric_calibrated', False), ('depth_convention', 'ray_distance')]:
        entry = deepcopy(original)
        entry['geometry_metadata'][name] = value
        entry['key'] = hashlib.sha256(canonical_json(entry['geometry_metadata'])).hexdigest()
        with pytest.raises(ValueError, match='candidate'):
            candidate_metrics(entry, receipt['policy_sha256'], {'scene_00002'})
    item = dict(image=rows[1], reason='upstream_errata_scene_or_room', source_proofs=[proof])
    with pytest.raises(ValueError, match='published list'):
        rejection_kind(item, {'scene_00002'}, proof)
    item = dict(image=rows[3], reason='insufficient_verified_support', source_proofs=[proof],
                stats=dict(layout_verified_pixels=0, valid_fraction=.4, source_needs_review=False))
    with pytest.raises(ValueError, match='unchanged'):
        rejection_kind(item, {'scene_00002'}, proof)
