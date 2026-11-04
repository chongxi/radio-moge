from copy import deepcopy
import hashlib
import json

import numpy as np
from PIL import Image
import pytest
import torch

from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import GeometryPackWriter, canonical_json
from radio_moge.geometry_sources import pinhole_rays
from radio_moge.data import ordinary_crop_parameters
from radio_moge.ordinary_geometry import ORDINARY_CAMERA_CONTRACT, ORDINARY_LOCAL_RADIUS
from radio_moge.ordinary_geometry_targets import ordinary_target
from radio_moge.rgb_geometry_batch import GeometrySlot


def fixture(tmp_path, rotation=0, multigrid=False, pseudo=False, **extra):
    path = tmp_path / 'rgb.png'
    shape = (53, 73)
    Image.fromarray(np.random.default_rng(43).integers(0, 256, (*shape, 3), dtype=np.uint8)).save(path)
    oriented = shape[::-1] if rotation in (90, 270) else shape
    row = dict(image=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), split='train',
               dataset='fixture', group='fixture:scene', image_rotation_cw=rotation,
               height=oriented[0], width=oriented[1])
    native = (31, 47) if multigrid else oriented
    kd = np.array([[.8, .03, .44], [0., 1.1, .56], [0., 0., 1.]])
    kr = np.array([[.9, .021, .4], [0., 1.2, .6], [0., 0., 1.]]) if multigrid else kd
    depth = 2 + np.arange(np.prod(native), dtype=np.float32).reshape(native)/10000
    valid = np.ones(native, bool)
    valid[4:9, 6:14] = False
    policy = dict(fixture=True)
    meta = dict(image_sha256=row['sha256'], dataset='fixture', source_id='fixture',
        label_kind='teacher_pseudo' if pseudo else 'synthetic_renderer',
        camera_kind='teacher_estimated' if pseudo else 'synthetic_camera', metric_calibrated=not pseudo,
        conversion='test_only', policy_sha256=hashlib.sha256(canonical_json(policy)).hexdigest(),
        image_rotation_cw=rotation, source_proofs=[dict(source='fixture', sha256='f'*64)], **extra)
    if pseudo:
        meta.update(teacher_revision='b'*40, native_source_review_unresolved=True,
                    native_reason='retained_fixture_flag')
    if multigrid:
        meta.update(rgb_shape=list(oriented), rgb_intrinsics_normalized=kr.tolist(),
                    depth_to_rgb_uv_transform=(kr @ np.linalg.inv(kd)).tolist())
    writer = GeometryPackWriter(tmp_path/'packs', policy)
    writer.add(row, depth, valid, kd, meta)
    writer.close()
    receipt = json.loads((tmp_path/'packs/pack-000000.json').read_text())
    geometry = dict(corpus_ordinal=123, pack=file_ref(tmp_path/'packs'/receipt['tar']),
                    entry=receipt['entries'][0], source_index=dict(path='fixture-index', sha256='e'*64))
    return GeometrySlot((7, 43, 40, 64), row, geometry), depth, valid, kd, kr


@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
@pytest.mark.parametrize('multigrid', [False, True])
def test_payload_original_crop_and_exact_native_observations(tmp_path, rotation, multigrid):
    slot, depth, valid, kd, kr = fixture(tmp_path, rotation, multigrid)
    original = deepcopy(slot)
    rng = torch.get_rng_state().clone()
    target = ordinary_target(slot, augment=True)
    assert torch.equal(torch.get_rng_state(), rng) and slot == original
    view = target['ordinary_view']
    shape = slot.rgb_row['height'], slot.rgb_row['width']
    box, flip = ordinary_crop_parameters(shape, slot.key[2:], slot.key[1])
    assert view['crop_box'] == box and view['flipped'] == flip
    assert target['camera_contract'] == ORDINARY_CAMERA_CONTRACT
    assert target['local_radius_contract'] == ORDINARY_LOCAL_RADIUS
    mask = target['valid']
    ids = target['native_index'][mask]
    assert len(ids) > 32 and len(ids) == len(set(ids)) and valid.ravel()[ids].all()
    native = (pinhole_rays(kd, *depth.shape)*depth[..., None]).reshape(-1, 3)[ids]
    np.testing.assert_allclose(target['points'][mask], native @ view['camera_transform'].T, atol=3e-7)
    projection = native @ kr.T @ view['rgb_to_view_uv'].T
    np.testing.assert_allclose(target['query_uv'][mask], projection[:, :2]/projection[:, 2:], atol=1e-7)
    assert target['source_metadata'] == slot.geometry['entry']['geometry_metadata']
    assert target['member_sha256'] == slot.geometry['entry']['sha256']


def test_missing_target_never_opens_rgb_or_pack(tmp_path):
    slot = GeometrySlot((9, 43, 40, 64), dict(image='missing'), None)
    assert ordinary_target(slot, augment=True) is None


def test_changed_rgb_member_or_identity_fail_closed(tmp_path):
    slot, *_ = fixture(tmp_path)
    changed = deepcopy(slot.geometry)
    changed['entry']['image']['group'] = 'other'
    with pytest.raises(ValueError, match='identity'):
        ordinary_target(GeometrySlot(slot.key, slot.rgb_row, changed), augment=True)
    pack = slot.geometry['pack']['path']
    with open(pack, 'r+b') as file:
        file.seek(slot.geometry['entry']['offset'])
        file.write(b'xxxx')
    with pytest.raises(ValueError, match='hash mismatch'):
        ordinary_target(slot, augment=True)


def test_changed_rgb_bytes_rejected(tmp_path):
    slot, *_ = fixture(tmp_path)
    with open(slot.rgb_row['image'], 'ab') as file:
        file.write(b'changed')
    with pytest.raises(ValueError, match='RGB bytes'):
        ordinary_target(slot, augment=True)


def test_unresolved_flags_fail_and_relative_pseudo_flags_preserved(tmp_path):
    bad = tmp_path/'bad'
    bad.mkdir()
    slot, *_ = fixture(bad, source_needs_review=True)
    with pytest.raises(ValueError, match='source-quality'):
        ordinary_target(slot, augment=True)
    good = tmp_path/'good'
    good.mkdir()
    slot, *_ = fixture(good, pseudo=True)
    target = ordinary_target(slot, augment=True)
    assert target['supervision'] == 'relative_pseudo'
    assert target['source_metadata']['native_source_review_unresolved'] is True
    assert target['source_metadata']['native_reason'] == 'retained_fixture_flag'


def test_bounded_real_input_verifier_reads_bound_payloads_and_rejects_changed_prior_key(tmp_path):
    import sqlite3
    from verify_ordinary_geometry_targets import run
    slot, *_ = fixture(tmp_path)
    index = tmp_path/'index.sqlite'
    with sqlite3.connect(index) as db:
        db.executescript('''CREATE TABLE images (ordinal INTEGER PRIMARY KEY,image TEXT UNIQUE,
            sha256 TEXT,row_json TEXT,disposition TEXT,pack TEXT,entry_json TEXT);
            CREATE TABLE samples (sample INTEGER PRIMARY KEY,ordinal INTEGER UNIQUE);
            CREATE TABLE packs (path TEXT PRIMARY KEY,sha256 TEXT);''')
        db.execute('INSERT INTO images VALUES (?,?,?,?,?,?,?)', (123, slot.rgb_row['image'],
            slot.rgb_row['sha256'], json.dumps(slot.rgb_row), 'selected',
            slot.geometry['pack']['path'], json.dumps(slot.geometry['entry'])))
        db.execute('INSERT INTO samples VALUES (0,123)')
        db.execute('INSERT INTO packs VALUES (?,?)', tuple(slot.geometry['pack'][k] for k in ('path','sha256')))
    accounting = dict(stage='accounting_verified_not_accepted', training_admission=False, unresolved=0,
                      index=file_ref(index), manifest_sha256='c'*64, selected_images=1, total_images=1)
    p = tmp_path/'accounting.json'
    p.write_text(json.dumps(accounting))
    gate = dict(schema_version=1, stage='accepted_for_training', training_admission=True,
        accounting_report=file_ref(p), **{k:accounting[k] for k in ('index','manifest_sha256','selected_images')})
    p = tmp_path/'gate.json'
    p.write_text(json.dumps(gate))
    manifest = tmp_path/'rgb.jsonl'
    manifest.write_text(json.dumps(slot.rgb_row)+'\n')
    prior = dict(stage='ordinary_geometry_sidecar_cpu_verified_not_admitted', training_admission=False,
        optimizer_updates=0, all_rgb_slots_retained=True, prepared_rgb_bitwise_equal=True,
        original_observed_pixel_hashes_equal=True, torch_rng_unchanged=True, presentations=32,
        manifest_rows=1, job=dict(manifest=file_ref(manifest), historical_acceptance=file_ref(p)),
        records=[dict(key=[0,i,40,64], image=slot.rgb_row['image'], sha256=slot.rgb_row['sha256'],
            dataset=slot.rgb_row['dataset'], geometry_available=True, corpus_ordinal=123,
            entry_key=slot.geometry['entry']['key'], pack=slot.geometry['pack']) for i in range(32)])
    path = tmp_path/'prior.json'
    path.write_text(json.dumps(prior))
    report = run(file_ref(path), tmp_path/'report.json')
    assert report['counts'] == {'supervised':32} and not report['training_admission']
    assert report['geometry_member_payloads_read'] == 32 and report['camera_rays_verified']
    prior['records'][0]['entry_key'] = 'e'*64
    path.write_text(json.dumps(prior))
    with pytest.raises(ValueError, match='saved sidecar'):
        run(file_ref(path), tmp_path/'bad.json')
