from copy import deepcopy
import hashlib
import json
import sqlite3

import numpy as np
import pytest

from audit_tartanair_outcomes import (CALIBRATION, CONVERSION, candidate_details,
                                     quality_details, rejection_details, run, source_identity)
from radio_moge.geometry_cache import GeometryPackWriter, canonical_json, publish_json, sha_file
from radio_moge.geometry_sources import normalized_opencv_intrinsics


def quality(fraction=.5):
    return dict(native_finite_positive_fraction=1., task_range_fraction=.9, stable_left_fraction=.8,
        stereo_geometric_fraction=.7, valid_fraction=fraction, task_range_m=[.1, 200.],
        task_range_is_not_publisher_invalid_mask=True, source_needs_review=fraction < .05,
        independent_physical_depth_accuracy_checked=False, rgb_stereo_mae_on_geometric=.15,
        stereo_depth_max_error_m=.01)


def fixture(root):
    config = dict(root=str(root), manifest_sha256='a' * 64, expected_counts={'tartanair': 3})
    rows = [dict(image=str(root / f'{i}.png'), sha256=f'{i:064x}', dataset='tartanair', split='train',
        group='tartanair:environment:test', environment='test', trajectory='test/Easy/P000',
        width=640, height=480, source_member=f'test/Easy/P000/image_left/{i:06d}_left.png',
        source_url='https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_fixture/tartanair/test/Easy/image_left.zip')
        for i in range(3)]
    with sqlite3.connect(root / 'inventory.sqlite') as db:
        db.execute('CREATE TABLE images(ordinal,image,sha256,scene_group,row_json,dataset)')
        for i, row in enumerate(rows):
            db.execute('INSERT INTO images VALUES(?,?,?,?,?,?)',
                (i, row['image'], row['sha256'], row['group'], json.dumps(row), 'tartanair'))
    publish_json(root / 'inventory.json', dict(stage='inventory_complete', manifest_sha256=config['manifest_sha256'],
        dataset_counts=config['expected_counts'], inventory_sha256=sha_file(root / 'inventory.sqlite')))
    native = root / 'native'
    native.mkdir()
    (native / 'rejected').mkdir()
    policy = dict(dataset='tartanair', manifest_sha256=config['manifest_sha256'], label_kind='synthetic_renderer',
        conversion=CONVERSION, calibration=CALIBRATION, task_range_m=[.1, 200.],
        task_range_is_not_publisher_invalid_mask=True, minimum_image_valid_fraction=.05,
        native_holes_filled=False, independent_physical_depth_accuracy_checked=False,
        stereo_support='both eyes 3x3 stable (.05+.02Z m), x_right=x_left-80/Z, no y shift',
        stereo_gates=dict(z_difference_m='.01+.005Z', rgb_pixel_mae_max=.08))
    checksum = hashlib.sha256(canonical_json(policy)).hexdigest()
    proofs = []
    for i, row in enumerate(rows):
        proofs.append([dict(source=row['source_url'].rsplit('/', 1)[0] + '/' + kind + '.zip',
            member=f'test/Easy/P000/{kind}/{i:06d}' + suffix, sha256='b' * 64, bytes=100, crc32=123)
            for kind, suffix in [('depth_left', '_left_depth.npy'), ('depth_right', '_right_depth.npy'),
                                 ('image_right', '_right.png')]])
    writer = GeometryPackWriter(native / 'packs', policy)
    valid = np.zeros((480, 640), bool)
    valid[:240] = True
    for i, row in enumerate(rows[:2]):
        meta = dict(image_sha256=row['sha256'], dataset='tartanair', source_id=row['source_member'],
            label_kind='synthetic_renderer', camera_kind='synthetic_camera', metric_calibrated=True,
            conversion=CONVERSION, policy_sha256=checksum, source_proofs=proofs[i],
            source_quality=quality(), source_needs_review=False)
        writer.add(row, np.full((480, 640), i + 1., np.float32), valid,
                   normalized_opencv_intrinsics(640, 480, 320, 320, 320, 240), meta)
    writer.close()
    rejection = dict(image=rows[2], reason='insufficient_stereo_verified_support',
        stats=quality(.02), source_proofs=proofs[2])
    publish_json(native / 'rejections.json', [rejection])
    publish_json(native / 'rejected' / (rows[2]['sha256'] + '.json'), rejection)
    publish_json(native / 'report.json', dict(stage='complete', mode='generate', training_admission=False,
        training_started=False, optimizer_updates=0, completed=2, rejected=1, expected_scope=3,
        verified_cache_members=2, policy_sha256=checksum, source_bundles_seen=1, environment_names_seen=1))
    entry = json.loads((native / 'packs/pack-000000.json').read_text())['entries'][0]
    return config, checksum, entry, rejection


def test_full_scope_has_one_outcome_per_input_without_admission_or_double_payload_claim(tmp_path):
    config, _, _, _ = fixture(tmp_path)
    report = run(config, 'native', 'audit')
    assert report['outcome_counts'] == {'native_candidate': 2, 'rejected': 1}
    assert report['source_bundles'] == report['environments'] == report['trajectories'] == 1
    assert not report['training_admission'] and not report['source_quality_accepted']
    assert not report['independent_array_readback_performed'] and not report['pack_payload_hashes_rechecked']
    assert report['review_candidates_count'] == 2
    with sqlite3.connect(report['index']['path']) as db:
        assert db.execute('SELECT COUNT(DISTINCT image) FROM images').fetchone()[0] == 3
        assert db.execute("SELECT evidence_json FROM images WHERE outcome='rejected'").fetchone()[0].find('/rejected/') > 0


@pytest.mark.parametrize('key,value', [('stage', 'generating_cache'), ('verified_cache_members', 1),
                                      ('training_admission', True), ('optimizer_updates', 1)])
def test_incomplete_or_training_producer_cannot_create_an_audit(tmp_path, key, value):
    config, _, _, _ = fixture(tmp_path)
    path = tmp_path / 'native/report.json'
    report = json.loads(path.read_text())
    report[key] = value
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='producer readback'):
        run(config, 'native', 'blocked')
    assert not (tmp_path / 'blocked').exists()


def test_duplicate_and_missing_outcomes_fail_even_when_generation_counters_balance(tmp_path):
    config, _, _, _ = fixture(tmp_path)
    receipt_path = tmp_path / 'native/packs/pack-000000.json'
    receipt = json.loads(receipt_path.read_text())
    original = deepcopy(receipt)
    for variant in ('duplicate', 'missing'):
        receipt = deepcopy(original)
        receipt['entries'][1] = deepcopy(receipt['entries'][0])
        if variant == 'missing':
            receipt['entries'] = receipt['entries'][:1]
        receipt_path.write_text(json.dumps(receipt))
        p = tmp_path / 'native/packs/packs.json'
        index = json.loads(p.read_text())
        index['packs'][0].update(sha256=sha_file(receipt_path), count=len(receipt['entries']))
        p.write_text(json.dumps(index))
        with pytest.raises(ValueError, match='duplicate|reconcile'):
            run(config, 'native', variant)


def test_final_rejection_must_match_its_exact_persisted_record(tmp_path):
    config, _, _, rejection = fixture(tmp_path)
    rejection['stats']['rgb_stereo_mae_on_geometric'] = .2
    p = tmp_path / 'native/rejections.json'
    p.write_text(json.dumps([rejection]))
    with pytest.raises(ValueError, match='Individual persisted rejection'):
        run(config, 'native', 'changed-rejection')


@pytest.mark.parametrize('key,value', [('minimum_image_valid_fraction', .01),
    ('stereo_support', 'no right-eye check'), ('task_range_is_not_publisher_invalid_mask', False)])
def test_rehashed_policy_cannot_change_the_task_or_stereo_rules(tmp_path, key, value):
    config, _, _, _ = fixture(tmp_path)
    p = tmp_path / 'native/packs/policy.json'
    policy = json.loads(p.read_text())
    policy[key] = value
    p.write_text(json.dumps(policy))
    checksum = hashlib.sha256(canonical_json(policy)).hexdigest()
    for name in ('report.json', 'packs/packs.json'):
        p = tmp_path / 'native' / name
        value = json.loads(p.read_text())
        value['policy_sha256'] = checksum
        p.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='unchanged native stereo policy'):
        run(config, 'native', 'changed-policy')
    assert not (tmp_path / 'changed-policy').exists()


def test_aggregate_rgb_error_is_not_the_pixelwise_gate_and_empty_support_is_explicit():
    assert quality_details(quality())['rgb_mae'] == .15  # >.08 is possible before pixel filtering.
    q = quality(0)
    q.update(stereo_geometric_fraction=0, rgb_stereo_mae_on_geometric=None,
             stereo_depth_max_error_m=None)
    assert quality_details(q)['support'] == 0
    q['rgb_stereo_mae_on_geometric'] = 0
    with pytest.raises(ValueError, match='empty-support'):
        quality_details(q)


def test_exact_support_boundary_and_nested_masks_preserve_flags():
    q = quality(.05)
    assert quality_details(q)['support'] == .05
    q['source_needs_review'] = True
    with pytest.raises(ValueError, match='source-quality flag'):
        quality_details(q)
    q = quality()
    q['stereo_geometric_fraction'] = .4
    with pytest.raises(ValueError, match='nesting'):
        quality_details(q)


def test_candidates_reject_wrong_grid_units_eye_label_flag_or_independent_accuracy(tmp_path):
    _, policy, entry, rejection = fixture(tmp_path)
    assert candidate_details(entry, policy)['support'] == .5
    for key, value in [('units', 'millimeters'), ('shape', [640, 480]), ('metric_calibrated', False),
                       ('label_kind', 'teacher_pseudo'), ('source_needs_review', True)]:
        changed = deepcopy(entry)
        changed['geometry_metadata'][key] = value
        changed['key'] = hashlib.sha256(canonical_json(changed['geometry_metadata'])).hexdigest()
        with pytest.raises(ValueError):
            candidate_details(changed, policy)
    changed = deepcopy(entry)
    changed['geometry_metadata']['source_proofs'][1]['member'] = rejection['source_proofs'][1]['member']
    with pytest.raises(ValueError, match='corresponding-eye'):
        candidate_details(changed, policy)
    changed = deepcopy(entry)
    changed['geometry_metadata']['source_quality']['independent_physical_depth_accuracy_checked'] = True
    with pytest.raises(ValueError):
        candidate_details(changed, policy)
    rejection['stats'] = quality(.05)
    with pytest.raises(ValueError, match='minimum support'):
        rejection_details(rejection)


def test_day_night_family_is_preserved_and_heldout_variants_stay_out(tmp_path):
    _, _, entry, _ = fixture(tmp_path)
    row = entry['image']
    proofs = entry['geometry_metadata']['source_proofs']
    for environment, group, allowed in [('abandonedfactory_night', 'abandonedfactory', True),
                                       ('abandonedfactory_night', 'abandonedfactory_night', False),
                                       ('office2', 'office', False), ('seasonsforest_winter', 'seasonsforest', False)]:
        new = deepcopy(row)
        new.update(environment=environment, group='tartanair:environment:' + group)
        for name in ('source_url', 'source_member', 'trajectory'):
            new[name] = row[name].replace('test/', environment + '/')
        corresponding = deepcopy(proofs)
        for proof in corresponding:
            proof['source'] = proof['source'].replace('test/', environment + '/')
            proof['member'] = proof['member'].replace('test/', environment + '/')
        if allowed:
            assert source_identity(new, corresponding) == 'Easy'
        else:
            with pytest.raises(ValueError, match='left-eye source identity'):
                source_identity(new, corresponding)
