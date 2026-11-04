from copy import deepcopy
import hashlib
import json
import sqlite3

import numpy as np
import pytest

from audit_scannetpp_native_outcomes import (
    CONVERSION, LARGE_MESH, VISIBILITY, candidate_details, quality, receipt_metadata, rejection_details, run,
)
from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import GeometryPackWriter, canonical_json, publish_json, sha_file
from radio_moge.geometry_sources import normalized_opencv_intrinsics


def checksum(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def support(pixels, large=True):
    result = dict(verified_pixels=pixels, cpu_hits=pixels, cpu_rays_checked=pixels,
        retained_pixels=pixels, initial_valid_pixels=5120, cpu_missing_hits=0,
        rejected_inconsistent_hits=0, depth_filled_or_averaged=False,
        independent_physical_depth_accuracy_checked=False,
        maximum_allowed_gpu_cpu_error_m=.002, maximum_accepted_gpu_cpu_error_m=.001 if pixels else None,
        final_depth='CPU nearest-hit camera Z', visibility_policy=VISIBILITY,
        valid_fraction=pixels/5120, retained_fraction=pixels/5120)
    if large:
        result.update(mesh_faces=2097153, max_faces_per_pass=1048576, raster_passes_per_ray_grid=3)
    return result


def fixture(root):
    data = root/'data'
    config = dict(root=str(root), dataset_root=str(data), manifest_sha256='a'*64,
                  expected_counts={'scannetpp_dslr': 4})
    audit = root/'scannetpp-source-audit-01'
    audit.mkdir()
    publish_json(audit/'policy.json', dict(fixture=True))
    scenes, refs = {}, {}
    for name in ('aaa', 'bbb'):
        scene = dict(scene=name, stage='source_checks_passed' if name == 'aaa' else 'needs_review',
            mesh_sha256='b'*64, faces=2097153, dslr=dict(colmap_images_sha256='c'*64),
            review_scans=[] if name == 'aaa' else ['scan01'])
        publish_json(audit/(name+'.json'), scene)
        scenes[name], refs[name] = scene, file_ref(audit/(name+'.json'))
    publish_json(audit/'report.json', dict(stage='complete', manifest_sha256=config['manifest_sha256'],
        expected_scenes=2, scenes_completed=2, scenes_needing_review=1, training_admission=False,
        scenes=[dict(scene=n, report=n+'.json', sha256=refs[n]['sha256'], stage=s['stage']) for n,s in scenes.items()]))
    rows = []
    for i in range(4):
        name = 'bbb' if i == 3 else 'aaa'
        base = data/'scannetpp-v2-full/data'/name
        rows.append(dict(dataset='scannetpp_dslr', split='train', official_split='nvs_sem_train',
            official_view_split='test' if i == 0 else 'train', scene=name, group='scannetpp:'+name,
            view=f'view{i}.JPG', image=str(base/'dslr/resized_undistorted_images'/f'view{i}.JPG'),
            mask=str(base/'dslr/resized_undistorted_masks'/f'publisher-mask{i}.png'), width=640, height=8,
            camera_width_height_fx_fy_cx_cy=[640, 8, 300, 300, 320, 4],
            transform_matrix=np.eye(4).tolist(), sha256=f'{i+1:064x}'))
    with sqlite3.connect(root/'inventory.sqlite') as db:
        db.execute('CREATE TABLE images(ordinal,image,sha256,scene_group,row_json,dataset)')
        for i, row in enumerate(rows):
            db.execute('INSERT INTO images VALUES(?,?,?,?,?,?)',
                (i, row['image'], row['sha256'], row['group'], json.dumps(row), 'scannetpp_dslr'))
    publish_json(root/'inventory.json', dict(stage='inventory_complete', manifest_sha256=config['manifest_sha256'],
        dataset_counts=config['expected_counts'], inventory_sha256=sha_file(root/'inventory.sqlite')))

    def proofs(row, mask=False):
        ref = refs[row['scene']]
        result = [dict(source=ref['path'], sha256=ref['sha256'])]
        if row['scene'] == 'aaa':
            base = data/'scannetpp-v2-full/data'/row['scene']
            result += [dict(source=str(base/p), sha256=h*64) for p,h in
                [('scans/mesh_aligned_0.05.ply', 'b'), ('dslr/colmap/images.txt', 'c'),
                 ('dslr/nerfstudio/transforms_undistorted.json', 'd')]]
            if mask:
                result.append(dict(source=row['mask'], sha256='e'*64))
        return result

    entries, policies = [], []
    for i, name in enumerate(('recovered', 'native')):
        base = root/name
        policy = dict(dataset='scannetpp_dslr', manifest_sha256=config['manifest_sha256'],
            label_kind='laser_mesh_derived', conversion=CONVERSION, not_direct_per_dslr_pixel_measurements=True,
            render_width=640, minimum_image_valid_fraction=.05, gpu_cpu_max_difference_m=.002,
            source_audit_policy_sha256=sha_file(audit/'policy.json'),
            anonymization='native mask==255, float area support, 2-pixel erosion on output grid',
            visibility='GPU 5x5/quarter-pixel stability AND EVERY candidate verified by CPU closest hit',
            source_review_policy='quarantine entire scene pending explicit source review; no automatic repairs')
        if i:
            policy['large_mesh_handling'] = LARGE_MESH
        writer = GeometryPackWriter(base/'packs', policy)
        row = rows[i]
        meta = dict(dataset='scannetpp_dslr', image_sha256=row['sha256'], source_id=row['scene']+'/'+row['view'],
            label_kind='laser_mesh_derived', camera_kind='calibrated_sensor', metric_calibrated=True,
            conversion=CONVERSION, policy_sha256=checksum(policy), source_needs_review=False,
            not_direct_per_dslr_pixel_measurements=True, rgb_shape=[8, 640],
            rgb_intrinsics_normalized=normalized_opencv_intrinsics(*row['camera_width_height_fx_fy_cx_cy']).tolist(),
            depth_to_rgb_uv_transform=np.eye(3).tolist(), source_quality=support(5120, large=bool(i)),
            source_proofs=proofs(row, mask=True))
        writer.add(row, np.full((8, 640), 2, dtype=np.float32), np.ones((8, 640), bool), np.eye(3), meta)
        writer.close()
        if not i:
            (base/'scope.jsonl').write_text(json.dumps(dict(ordinal=0, sha256=row['sha256']))+'\n')
        gen = dict(stage='complete', mode='generate', passed=False, training_admission=False, training_started=False,
            optimizer_updates=0, completed=1, verified_cache_members=1, policy_sha256=checksum(policy),
            expected_scope=3 if i else 1, rejected=2 if i else 0, scenes_seen=2 if i else 1)
        if i:
            gen.update(recovered_subset=file_ref(root/'recovered/report.json'), recovered_skipped=1)
        else:
            gen.update(dataset='scannetpp_dslr', manifest_sha256=config['manifest_sha256'],
                recovered_verified_subset=True, scope_sha256=sha_file(base/'scope.jsonl'), packs_verified=1)
        publish_json(base/'report.json', gen)
        entries.append(json.loads((base/'packs/pack-000000.json').read_text())['entries'][0])
        policies.append(policy)
    rejections = [dict(image=rows[2], reason='insufficient_verified_visible_support', stats=support(100),
        source_proofs=proofs(rows[2])), dict(image=rows[3], reason='source_scene_pending_review',
        stats=dict(source_audit_stage='needs_review', review_scans=['scan01']), source_proofs=proofs(rows[3]))]
    publish_json(root/'native/rejections.json', rejections)
    return dict(config=config, rows=rows, entries=entries, policies=policies, scenes=scenes, refs=refs,
                rejections=rejections)


def candidate(args, entry, i=1):
    return candidate_details(entry, checksum(args['policies'][i]), args['policies'][i],
        args['scenes'][entry['image']['scene']], args['refs'][entry['image']['scene']], args['config']['dataset_root'])


def reject(args, item):
    return rejection_details(item, args['policies'][1], args['scenes'][item['image']['scene']],
        args['refs'][item['image']['scene']], args['config']['dataset_root'])


def rekey(entry):
    entry['key'] = checksum(entry['geometry_metadata'])
    entry['member'] = entry['key']+'.npz'
    return entry


def test_complete_partition_retains_distinct_policies_flags_and_no_acceptance(tmp_path):
    args = fixture(tmp_path)
    report = run(args['config'], 'recovered', 'native', 'audit')
    assert report['outcome_counts'] == {'native_candidate': 2, 'native_rejected': 2}
    assert report['rejection_reasons'] == {'insufficient_verified_visible_support': 1, 'source_scene_pending_review': 1}
    assert len(report['candidate_policy_counts']) == 2
    assert report['source_flags_preserved'] and report['source_review_scenes'] == 1
    assert not report['teacher_replacements_counted_as_additional_native_rows']
    assert not report['training_admission'] and not report['source_quality_accepted']
    assert report['receipt_metadata_only'] and not report['independent_array_readback_performed']
    assert report['automatic_exclusions'] == 0
    with sqlite3.connect(report['index']['path']) as db:
        assert db.execute('SELECT COUNT(DISTINCT image) FROM images').fetchone()[0] == 4
    assert candidate(args, args['entries'][0], 0)['valid_fraction'] == 1  # Legacy has no large-mesh counters.


@pytest.mark.parametrize('key,value', [('verified_cache_members', 0), ('stage', 'generating_cache'),
    ('recovered_skipped', 0), ('expected_scope', 4)])
def test_incomplete_readback_or_wrong_partition_fails_before_output(tmp_path, key, value):
    args = fixture(tmp_path)
    path = tmp_path/'native/report.json'
    obj = json.loads(path.read_text())
    obj[key] = value
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError):
        run(args['config'], 'recovered', 'native', 'audit')
    assert not (tmp_path/'audit').exists()


@pytest.mark.parametrize('change', ['duplicate', 'changed_row', 'missing', 'recovery_scope'])
def test_missing_duplicate_changed_and_outside_recovery_rows_fail(tmp_path, change):
    args = fixture(tmp_path)
    items = deepcopy(args['rejections'])
    if change == 'duplicate':
        items[1] = deepcopy(items[0])
    elif change == 'changed_row':
        items[0]['image']['transform_matrix'][0][3] = 1
    elif change == 'missing':
        items.pop()
    else:
        path = tmp_path/'recovered/scope.jsonl'
        path.write_text(json.dumps(dict(ordinal=1, sha256=args['rows'][1]['sha256']))+'\n')
        report_path = tmp_path/'recovered/report.json'
        report = json.loads(report_path.read_text())
        report['scope_sha256'] = sha_file(path)
        report_path.write_text(json.dumps(report))
        native_path = tmp_path/'native/report.json'
        report = json.loads(native_path.read_text())
        report['recovered_subset'] = file_ref(report_path)
        native_path.write_text(json.dumps(report))
    (tmp_path/'native/rejections.json').write_text(json.dumps(items))
    with pytest.raises(ValueError):
        run(args['config'], 'recovered', 'native', 'audit')


@pytest.mark.parametrize('key,value', [('units', 'millimeters'), ('label_kind', 'measured_sensor'),
    ('source_needs_review', True), ('not_direct_per_dslr_pixel_measurements', False), ('shape', [640, 8]),
    ('depth_to_rgb_uv_transform', [[1,0,1], [0,1,0], [0,0,1]])])
def test_candidate_cannot_change_source_meaning_camera_or_units(tmp_path, key, value):
    args = fixture(tmp_path)
    entry = deepcopy(args['entries'][1])
    entry['geometry_metadata'][key] = value
    with pytest.raises(ValueError):
        candidate(args, rekey(entry))


def test_bound_source_proofs_and_source_flags_are_required(tmp_path):
    args = fixture(tmp_path)
    for index, field, value in [(0, 'sha256', 'f'*64), (1, 'sha256', 'f'*64),
                                (3, 'source', '/other/calibration.json'), (4, 'source', '/other/mask.png')]:
        entry = deepcopy(args['entries'][1])
        entry['geometry_metadata']['source_proofs'][index][field] = value
        with pytest.raises(ValueError):
            candidate(args, rekey(entry))
    entry = deepcopy(args['entries'][1])
    entry['geometry_metadata']['source_quality'].update(mesh_faces=1, raster_passes_per_ray_grid=1)
    with pytest.raises(ValueError):
        candidate(args, rekey(entry))
    args['scenes']['aaa']['stage'] = 'needs_review'
    with pytest.raises(ValueError):
        candidate(args, args['entries'][1])


def test_visibility_counts_error_limits_and_no_independent_accuracy_claim():
    for key, value in [('cpu_hits', 99), ('valid_fraction', .5), ('maximum_accepted_gpu_cpu_error_m', .0021),
        ('depth_filled_or_averaged', True), ('independent_physical_depth_accuracy_checked', True),
        ('raster_passes_per_ray_grid', 2), ('cpu_missing_hits', 1)]:
        q = support(100)
        q[key] = value
        with pytest.raises(ValueError):
            quality(q, [8, 640], True)
    assert quality(support(0), [8, 640], True) == 0


def test_rejections_preserve_support_cutoff_and_original_scene_flags(tmp_path):
    args = fixture(tmp_path)
    assert reject(args, args['rejections'][0]) == 'insufficient_verified_visible_support'
    assert reject(args, args['rejections'][1]) == 'source_scene_pending_review'
    for change in ('threshold', 'flags', 'reason', 'mask'):
        item = deepcopy(args['rejections'][int(change == 'flags')])
        if change == 'threshold':
            item['stats'] = support(256)  # Exactly 5% must not be called below the cutoff.
        elif change == 'flags':
            item['stats']['review_scans'] = []
        elif change == 'reason':
            item['reason'] = 'repaired_native_source'
        else:
            item['source_proofs'].append(args['entries'][1]['geometry_metadata']['source_proofs'][-1])
        with pytest.raises(ValueError):
            reject(args, item)


def test_global_source_flags_cannot_be_omitted_from_count(tmp_path):
    args = fixture(tmp_path)
    path = tmp_path/'scannetpp-source-audit-01/report.json'
    report = json.loads(path.read_text())
    report['scenes_needing_review'] = 0
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='Source-review scene count'):
        run(args['config'], 'recovered', 'native', 'audit')
    assert not (tmp_path/'audit').exists()


@pytest.mark.parametrize('change', ['schema', 'admission', 'stage', 'sha', 'offset', 'overflow'])
def test_receipt_checks_layout_and_provenance_without_reading_payload(tmp_path, change):
    args = fixture(tmp_path)
    path = tmp_path/'native/packs/pack-000000.json'
    receipt = json.loads(path.read_text())
    if change == 'schema':
        receipt['schema_version'] = 2
    elif change == 'admission':
        receipt['training_admission'] = True
    elif change == 'stage':
        receipt['stage'] = 'accepted'
    elif change == 'sha':
        receipt['entries'][0]['sha256'] = 'unbound'
    elif change == 'offset':
        receipt['entries'][0]['offset'] += 512
    else:
        receipt['entries'][0]['size'] = receipt['bytes']
    with pytest.raises(ValueError):
        receipt_metadata(receipt, dict(count=1), checksum(args['policies'][1]), path.with_suffix('.tar'))
