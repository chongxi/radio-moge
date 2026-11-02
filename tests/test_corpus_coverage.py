import json
import sqlite3

import numpy as np
import pytest

from radio_moge.corpus_admission import build_accounting, readonly_db
from radio_moge.corpus_coverage import geometry_coverage, metadata_coverage, selected_geometry_coverage
from audit_geometry_coverage import run
from test_corpus_admission import admission_fixture


def test_native_histogram_ignores_invalid_values_and_counts_exact_boundaries():
    depth = np.array([[.1, .5, 1., 2., 5., 10., 20., 50., 200., np.nan]], np.float32)
    valid = np.ones_like(depth, bool)
    valid[0, -1] = False
    value = geometry_coverage(depth, valid, 'metric_native')
    assert value['histogram_counts'] == [1] * 9
    assert value['valid_pixels'] == 9 and value['grid_pixels'] == 10
    assert value['median_depth_m'] == 5 and value['valid_fraction'] == .9


def test_pseudo_histogram_discards_nominal_scale_and_rejects_supported_invalids():
    depth = np.array([[.125, .25, .5, 1., 2., 4., 8.]], np.float32)
    valid = np.ones_like(depth, bool)
    a = geometry_coverage(depth, valid, 'relative_pseudo')
    b = geometry_coverage(depth * 1024, valid, 'relative_pseudo')
    assert a == b and a['median_depth_m'] is None
    assert a['histogram_counts'] == [1] * 7
    assert a['depth_units'] == 'relative_to_image_valid_median'
    depth[0, 0] = np.nan
    with pytest.raises(ValueError, match='Invalid supported'):
        geometry_coverage(depth, valid, 'relative_pseudo')


def test_group_concentration_and_pixel_duplicates_are_not_independence_claims(tmp_path):
    with sqlite3.connect(tmp_path / 'metadata.sqlite') as db:
        db.execute('CREATE TABLE images(dataset,scene_group,row_json,disposition)')
        for dataset, group, checksum, outcome in [
                ('hypersim', 'scene1', 'a' * 64, 'selected'),
                ('hypersim', 'scene1', 'b' * 64, 'selected'),
                ('hypersim', 'scene1', None, 'selected'),
                ('hypersim', 'scene2', 'a' * 64, 'excluded'),
                ('coco2017', 'image1', 'a' * 64, 'selected')]:
            db.execute('INSERT INTO images VALUES (?,?,?,?)',
                (dataset, group, json.dumps(dict(decoded_rgb_sha256=checksum)), outcome))
        groups = []
        result = metadata_coverage(db, accounting=True, write_group=groups.append)
    values = {v['dataset']: v for v in result['groups']}
    assert values['hypersim']['scope_groups'] == 2
    assert values['hypersim']['fully_excluded_groups'] == 1
    assert values['hypersim']['effective_group_count'] == 1
    assert 'unknown' in values['coco2017']['group_meaning']
    assert result['exact_decoded_duplicates']['clusters'] == 1
    assert result['exact_decoded_duplicates']['excess_rows'] == 2
    assert result['exact_decoded_duplicates']['cross_dataset_clusters'] == 1
    assert not result['perceptual_near_duplicates_checked']
    assert sum(sum(v['counts'].values()) for v in groups) == 5


def test_full_accounting_records_native_pixel_counts_and_never_approves_coverage(tmp_path):
    root, plan, _ = admission_fixture(tmp_path)
    accounting = build_accounting(plan, root / 'account')
    with readonly_db(accounting['index']['path']) as db:
        by_source = {v['dataset']: v for v in selected_geometry_coverage(db)}
    assert by_source['hypersim']['valid_pixels'] == 12 * 16
    assert by_source['coco2017']['valid_pixels'] == 24 * 32
    assert by_source['hypersim']['depth_units'] == 'meters_camera_z'
    assert by_source['coco2017']['depth_units'] == 'relative_to_image_valid_median'
    result = run('accounting', root / 'account/report.json', root / 'coverage')
    assert result['stage'] == 'coverage_audit_complete_not_reviewed'
    assert result['training_admission'] is False and 'passed' not in result
    assert result['scope_images'] == 2 and result['outstanding_review']
    with pytest.raises(FileExistsError):
        run('accounting', root / 'account/report.json', root / 'coverage')


def test_pixel_weighted_and_image_weighted_coverage_are_distinct(tmp_path):
    with sqlite3.connect(tmp_path / 'counts.sqlite') as db:
        db.execute('CREATE TABLE images(ordinal,dataset,label_kind,disposition,coverage_json)')
        for i, shape in enumerate([(2, 2), (10, 10)]):
            mask = np.ones(shape, bool)
            if i:
                mask[0] = False
            value = geometry_coverage(np.full(shape, .25 if i == 0 else 20.), mask, 'metric_native')
            db.execute('INSERT INTO images VALUES (?,?,?,?,?)',
                       (i, 'hypersim', 'synthetic_renderer', 'selected', json.dumps(value)))
        result = selected_geometry_coverage(db)[0]
        assert result['image_mean_valid_fraction'] == pytest.approx(.95)
        assert result['pixel_weighted_valid_fraction'] == pytest.approx(94 / 104)
        assert result['mean_per_image_bin_fraction'][0] == .5
        assert result['histogram_counts'][0] == 4
        db.execute('UPDATE images SET coverage_json=NULL WHERE ordinal=0')
        with pytest.raises(ValueError, match='lacks verified'):
            selected_geometry_coverage(db)
