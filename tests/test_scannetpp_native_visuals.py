from copy import deepcopy
import hashlib
import json
import sqlite3

import numpy as np
from PIL import Image
import pytest

from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import GeometryPackWriter, canonical_json, publish_json, read_geometry_member, sha_file
from radio_moge.geometry_sources import normalized_opencv_intrinsics
from visualize_scannetpp_native_outcomes import (
    aligned_rgb, anonymity_support, checked_display, render, render_entry,
)


def fixture(root):
    rgb = np.full((32, 1280, 3), [50, 100, 150], np.uint8)
    mask = np.full((32, 1280), 255, np.uint8)
    mask[14, 600] = 0  # A single unsupported native pixel must survive float area support.
    Image.fromarray(rgb).save(root/'rgb.png')
    Image.fromarray(mask).save(root/'mask.png')
    row = dict(image=str(root/'rgb.png'), mask=str(root/'mask.png'), sha256=sha_file(root/'rgb.png'),
        dataset='scannetpp_dslr', split='train', scene='fixture', group='scannetpp:fixture', view='rgb.png',
        width=1280, height=32, camera_width_height_fx_fy_cx_cy=[1280, 32, 700, 702, 650, 18])
    valid = anonymity_support(mask, rgb.shape[:2], (16, 640))
    depth = np.full((16, 640), 2, np.float32)
    depth[~valid] = 0
    kd = normalized_opencv_intrinsics(*row['camera_width_height_fx_fy_cx_cy'])
    policy = dict(dataset='scannetpp_dslr', fixture_only=True)
    checksum = hashlib.sha256(canonical_json(policy)).hexdigest()
    meta = dict(image_sha256=row['sha256'], dataset=row['dataset'], source_id='fixture/rgb.png',
        label_kind='laser_mesh_derived', camera_kind='calibrated_sensor', metric_calibrated=True,
        conversion='fixture', policy_sha256=checksum, source_needs_review=False,
        not_direct_per_dslr_pixel_measurements=True, rgb_shape=[32, 1280],
        rgb_intrinsics_normalized=kd.tolist(), depth_to_rgb_uv_transform=np.eye(3).tolist(),
        source_quality=dict(verified_pixels=int(valid.sum())),
        source_proofs=[dict(source=row['mask'], sha256=sha_file(root/'mask.png'))])
    writer = GeometryPackWriter(root/'packs', policy)
    writer.add(row, depth, valid, kd, meta)
    writer.close()
    receipt_path = root/'packs/pack-000000.json'
    entry = json.loads(receipt_path.read_text())['entries'][0]
    pack = receipt_path.with_suffix('.tar')
    chosen = dict(corpus_ordinal=7, image=row, entry_key=entry['key'], receipt=file_ref(receipt_path),
        policy_sha256=checksum, metrics=dict(valid_fraction=float(valid.mean()), median_depth_m=2., maximum_depth_m=2.),
        selection_reasons=['valid_fraction:ASC'])
    return entry, pack, chosen, rgb, mask


def test_pixel_centers_map_between_different_grids_without_renormalizing_K():
    rgb = np.zeros((8, 12, 3), np.uint8)
    rgb[..., 0] = np.arange(12)*10
    rgb[..., 1] = np.arange(8)[:, None]*10
    k = normalized_opencv_intrinsics(12, 8, 8, 7, 5.2, 3.8)
    sampled, uv = aligned_rgb(rgb, k, k, (4, 6))
    x, y = np.meshgrid((np.arange(6)+.5)/6, (np.arange(4)+.5)/4)
    np.testing.assert_allclose(uv, np.stack((x, y), -1), atol=1e-14, rtol=0)
    # Center mapping: native index0 -> original coordinate0.5, then increments2.
    np.testing.assert_allclose(sampled[..., 0], np.broadcast_to([5,25,45,65,85,105], (4,6)), atol=1)
    np.testing.assert_allclose(sampled[..., 1], np.broadcast_to(np.array([5,25,45,65])[:,None], (4,6)), atol=1)


def test_mask_uses_float_area_and_cross_erosion_not_teacher_square():
    mask = np.full((12, 12), 255, np.uint8)
    mask[6, 6] = 0
    support = anonymity_support(mask, (12, 12), (12, 12))
    assert support[4, 4] and not support[4, 6]  # L1-distance2 cross, not a5x5 square.
    large = np.full((24, 24), 255, np.uint8)
    large[12, 12] = 254  # Require exactly255 and retain a single-pixel partial-area hole.
    assert not anonymity_support(large, large.shape, (12, 12))[6, 6]


def test_anonymity_support_matches_actual_producer_function(tmp_path):
    from cache_scannetpp_geometry import conservative_anonymization

    for i, shape in enumerate(((24, 36), (31, 47), (32, 48))):
        mask = np.full(shape, 255, np.uint8)
        mask[shape[0]//2, shape[1]//2] = 254
        mask[3:5, 4:8] = 0
        path = tmp_path/f'{i}.png'
        Image.fromarray(mask).save(path)
        assert np.array_equal(anonymity_support(mask, shape, (12, 18)),
                              conservative_anonymization(path, shape, (12, 18)))


@pytest.mark.parametrize('change', ['K_rescaled', 'K_nan', 'units', 'range', 'support', 'quantiles', 'RGB_shape'])
def test_wrong_projection_support_and_depth_contracts_are_rejected(tmp_path, change):
    entry, pack, chosen, rgb, mask = fixture(tmp_path)
    depth, valid, kd, meta = read_geometry_member(pack, entry)
    if change == 'K_rescaled':
        kd[:2] *= 2
    elif change == 'K_nan':
        kd[0, 0] = np.nan
    elif change in ('units', 'range'):
        meta['units' if change == 'units' else 'depth_convention'] = 'millimeters' if change == 'units' else 'radial_range'
        entry['geometry_metadata'] = deepcopy(meta)
    elif change == 'support':
        valid[0, 0] = True
    elif change == 'quantiles':
        entry['depth_quantiles_m'][2] = 3
    else:
        rgb = rgb[:-1]
    with pytest.raises(ValueError):
        checked_display(entry, (depth, valid, kd, meta), rgb, mask, chosen)


def test_selected_payload_rgb_and_mask_are_read_without_altering_arrays(tmp_path):
    entry, pack, chosen, rgb, mask = fixture(tmp_path)
    before = read_geometry_member(pack, entry)
    sampled, allowed, q = checked_display(entry, before, rgb, mask, chosen)
    assert sampled.shape == (16, 640, 3) and q.tolist() == [2]*5
    assert np.array_equal(allowed, before[1])
    page, identity = render_entry(pack, entry, chosen)
    assert page.size == (1440, 1000) and identity['grid'] == [16, 640]
    after = read_geometry_member(pack, entry)
    for a,b in zip(before[:3], after[:3]):
        assert np.array_equal(a,b)
    Image.fromarray(np.zeros_like(mask)).save(tmp_path/'mask.png')
    with pytest.raises(ValueError, match='byte identity'):
        render_entry(pack, entry, chosen)


def test_optional_receipt_ordinal_must_match_frozen_selection(tmp_path):
    entry, pack, chosen, _, _ = fixture(tmp_path)
    entry['image']['corpus_ordinal'] = chosen['corpus_ordinal']
    _, identity = render_entry(pack, entry, chosen)
    assert identity['corpus_ordinal'] == chosen['corpus_ordinal']
    entry['image']['corpus_ordinal'] += 1
    with pytest.raises(ValueError, match='Selected native camera'):
        render_entry(pack, entry, chosen)


def test_complete_fixture_render_binds_selected_outcome_and_never_accepts(tmp_path):
    entry, pack, chosen, _, _ = fixture(tmp_path)
    index = tmp_path/'index.sqlite'
    with sqlite3.connect(index) as db:
        db.execute('CREATE TABLE images(ordinal,row_json,outcome,policy,entry_key,evidence_json,details_json)')
        db.execute('INSERT INTO images VALUES(?,?,?,?,?,?,?)', (7,json.dumps(chosen['image']),
            'native_candidate',chosen['policy_sha256'],entry['key'],json.dumps(chosen['receipt']),json.dumps(chosen['metrics'])))
    publish_json(tmp_path/'selected.json', [chosen])
    audit = dict(stage='native_outcomes_accounted_not_accepted', dataset='scannetpp_dslr',
        producer_array_readback_complete=True, source_flags_preserved=True, training_admission=False,
        source_quality_accepted=False, review_candidates=file_ref(tmp_path/'selected.json'),
        review_candidate_count=1, index=file_ref(index), candidate_policy_counts={chosen['policy_sha256']:1})
    publish_json(tmp_path/'audit.json', audit)
    report = render(tmp_path/'audit.json', tmp_path/'visuals')
    assert report['rendered_images'] == 1 and report['selected_payloads_rechecked']
    assert not report['source_quality_accepted'] and not report['training_admission']
    assert not report['mesh_rerendered'] and not report['independent_physical_accuracy_checked']
    assert not report['gpu_used'] and not report['depth_or_mask_edited']
    altered = deepcopy(audit)
    altered['stage'] = 'generating_cache'
    publish_json(tmp_path/'incomplete.json', altered)
    with pytest.raises(ValueError, match='Completed native outcome audit'):
        render(tmp_path/'incomplete.json', tmp_path/'blocked')
    assert not (tmp_path/'blocked').exists()
