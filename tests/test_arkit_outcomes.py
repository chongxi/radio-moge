from copy import deepcopy
import hashlib
import json
import sqlite3

import numpy as np
import pytest

from audit_arkit_outcomes import candidate_details, rejection_kind, run
from radio_moge.geometry_cache import GeometryPackWriter, canonical_json, publish_json, sha_file
from radio_moge.geometry_sources import normalized_opencv_intrinsics


def fixture(root):
    config = dict(root=str(root), manifest_sha256='a' * 64, expected_counts={'arkitscenes': 4})
    rows = [dict(image=str(root / f'{i}.png'), sha256=f'{i:064x}', dataset='arkitscenes', split='train',
        group='arkitscenes:visit:1', video_id='10', timestamp=float(i), image_rotation_cw=0,
        width=640, height=480, camera_width_height_fx_fy_cx_cy=[640, 480, 500, 500, 320, 240],
        source_member=f'vga_wide/10_{i}.png',
        source_url='https://docs-assets.developer.apple.com/test/raw/Training/10/vga_wide.zip') for i in range(4)]
    with sqlite3.connect(root / 'inventory.sqlite') as db:
        db.execute('CREATE TABLE images(ordinal,image,sha256,scene_group,row_json,dataset)')
        for i, row in enumerate(rows):
            db.execute('INSERT INTO images VALUES(?,?,?,?,?,?)',
                (i, row['image'], row['sha256'], row['group'], json.dumps(row), 'arkitscenes'))
    publish_json(root / 'inventory.json', dict(stage='inventory_complete',
        manifest_sha256=config['manifest_sha256'], dataset_counts=config['expected_counts'],
        inventory_sha256=sha_file(root / 'inventory.sqlite')))
    native = root / 'native'
    native.mkdir()
    policy = dict(dataset='arkitscenes', manifest_sha256=config['manifest_sha256'],
        minimum_image_valid_fraction=.05, native_holes_filled=False,
        photometric_gate=dict(pixel_mae_max=.08, image_mae_max=.04, image_p90_max=.08))
    checksum = hashlib.sha256(canonical_json(policy)).hexdigest()
    writer = GeometryPackWriter(native / 'packs', policy)
    for i, row in enumerate(rows[:2]):
        proofs = []
        for kind in ('lowres_depth', 'confidence', 'lowres_wide', 'lowres_wide_intrinsics', 'vga_wide_intrinsics'):
            suffix = '.pincam' if kind.endswith('intrinsics') else '.png'
            proofs.append(dict(source=row['source_url'].rsplit('/', 1)[0]+'/'+kind+'.zip',
                member=f'{kind}/10_{i}{suffix}', sha256='b'*64, bytes=100, crc32=123))
        meta = dict(image_sha256=row['sha256'], dataset='arkitscenes', source_id=row['source_member'],
            label_kind='measured_sensor', camera_kind='calibrated_sensor', metric_calibrated=True,
            conversion='fixture', policy_sha256=checksum, source_proofs=proofs, native_depth_not_resized=True,
            source_needs_review=bool(i), image_rotation_cw=0, rgb_shape=[480, 640],
            rgb_intrinsics_normalized=normalized_opencv_intrinsics(*row['camera_width_height_fx_fy_cx_cy']).tolist(),
            depth_to_rgb_uv_transform=np.eye(3).tolist(), source_quality=dict(image_registration_pass=True,
                independent_physical_depth_accuracy_checked=False, native_confident_stable_pixels=49152,
                paired_rgb_mae=.01, paired_rgb_error_p90=.02, valid_fraction=1.))
        writer.add(row, np.ones((192, 256), np.float32), np.ones((192, 256), bool), np.eye(3), meta)
    writer.close()
    publish_json(native / 'rejections.json', [dict(image=rows[2], reason='missing_exact_timestamp_native_member',
        stats=dict(reason='missing_exact_timestamp_native_member', detail='confidence/10_2.png')),
        dict(image=rows[3], reason='insufficient_registered_native_support', stats=dict(valid_fraction=.01,
            image_registration_pass=True, independent_physical_depth_accuracy_checked=False,
            paired_rgb_mae=.01, paired_rgb_error_p90=.02))])
    publish_json(native / 'report.json', dict(stage='complete', mode='generate', training_admission=False,
        training_started=False, optimizer_updates=0, completed=2, rejected=2, expected_scope=4,
        verified_cache_members=2, policy_sha256=checksum, videos_seen=1))
    return config, checksum, json.loads((native / 'packs/pack-000000.json').read_text())['entries'][0]


def test_full_scope_counts_preserve_flags_and_missing_sources_remain_unresolved(tmp_path):
    config, _, _ = fixture(tmp_path)
    report = run(config, 'native', 'audit')
    assert report['outcome_counts'] == {'native_candidate': 1, 'flagged_candidate': 1, 'rejected': 2}
    assert report['rejection_reasons'] == {'low_registered_support': 1, 'missing_exact_timestamp_native_member': 1}
    assert report['native_flags_preserved'] and report['local_temporary_index']
    assert not report['source_quality_accepted'] and not report['training_admission']
    assert not report['independent_array_readback_performed']
    with sqlite3.connect(report['index']['path']) as db:
        assert db.execute('SELECT COUNT(DISTINCT image) FROM images').fetchone()[0] == 4
    path = tmp_path / 'native/rejections.json'
    rejections = json.loads(path.read_text())
    rejections[1] = deepcopy(rejections[0])
    path.write_text(json.dumps(rejections))
    with pytest.raises(ValueError, match='duplicate'):
        run(config, 'native', 'duplicate')


def test_incomplete_producer_readback_is_rejected_before_output_creation(tmp_path):
    config, _, _ = fixture(tmp_path)
    path = tmp_path / 'native/report.json'
    report = json.loads(path.read_text())
    report['verified_cache_members'] = 1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='producer readback'):
        run(config, 'native', 'incomplete')
    assert not (tmp_path / 'incomplete').exists()


def test_candidate_cannot_change_units_grid_timestamp_or_independent_accuracy_claim(tmp_path):
    _, policy, entry = fixture(tmp_path)
    assert candidate_details(entry, policy)['support'] == 1
    for key, value in [('units', 'millimeters'), ('shape', [480, 640]), ('metric_calibrated', False),
                       ('image_rotation_cw', 90), ('label_kind', 'teacher_pseudo')]:
        changed = deepcopy(entry)
        changed['geometry_metadata'][key] = value
        changed['key'] = hashlib.sha256(canonical_json(changed['geometry_metadata'])).hexdigest()
        with pytest.raises(ValueError):
            candidate_details(changed, policy)
    changed = deepcopy(entry)
    changed['geometry_metadata']['source_proofs'][0]['member'] = 'lowres_depth/10_999.png'
    changed['key'] = hashlib.sha256(canonical_json(changed['geometry_metadata'])).hexdigest()
    with pytest.raises(ValueError, match='source proof'):
        candidate_details(changed, policy)
    changed = deepcopy(entry)
    changed['geometry_metadata']['source_quality']['independent_physical_depth_accuracy_checked'] = True
    changed['key'] = hashlib.sha256(canonical_json(changed['geometry_metadata'])).hexdigest()
    with pytest.raises(ValueError):
        candidate_details(changed, policy)


def test_rejection_requires_exact_timestamp_and_unchanged_support_threshold():
    record = dict(reason='missing_exact_timestamp_native_member',image=dict(source_member='vga_wide/10_2.png'),
        stats=dict(reason='missing_exact_timestamp_native_member', detail='confidence/10_99.png'))
    with pytest.raises(ValueError, match='timestamp'):
        rejection_kind(record)
    record = dict(reason='insufficient_registered_native_support', stats=dict(valid_fraction=.05))
    with pytest.raises(ValueError, match='unchanged minimum'):
        rejection_kind(record)
