import json
import os
from pathlib import Path
import sqlite3
import time

import pytest

from test_corpus_admission import admission_fixture
from radio_moge.corpus_admission import build_accounting, file_ref
from radio_moge.producer_readback import MODE


def prepare(tmp_path):
    root, plan, _ = admission_fixture(tmp_path)
    p = Path(plan['scope']['path'])
    value = json.loads(p.read_text())
    value['kind'] = 'reviewed_source_subset'
    p.write_text(json.dumps(value))
    plan['scope'] = file_ref(p)
    for source in plan['sources']:
        p = Path(source['generation_report']['path'])
        value = json.loads(p.read_text())
        value['completed_at'] = time.time()
        p.write_text(json.dumps(value))
        source['generation_report'] = file_ref(p)
        source['payload_verification'] = MODE
    return root, plan


def test_completed_reviewed_producer_reuses_readback_without_fabricating_histogram(tmp_path, monkeypatch):
    root, plan = prepare(tmp_path)
    def no_second_read(*args):
        raise AssertionError('Unexpected second array census')
    monkeypatch.setattr('radio_moge.corpus_admission.read_geometry_member', no_second_read)
    result = build_accounting(plan, root / 'account')
    assert result['selected_images'] == 2 and not result['training_admission']
    assert {s['payload_verification'] for s in result['sources']} == {MODE}
    with sqlite3.connect(result['index']['path']) as db:
        values = [json.loads(x) for x, in db.execute('SELECT coverage_json FROM images')]
    assert all(v['histogram_counts'] is None and not v['per_pixel_histogram_recomputed'] for v in values)
    assert {v['depth_units'] for v in values} == {'meters_camera_z', 'relative_to_image_valid_median'}


def test_changed_pack_or_incomplete_producer_cannot_reuse_old_readback(tmp_path):
    root, plan = prepare(tmp_path)
    source = plan['sources'][0]
    p = Path(source['packs_index']['path']).parent / 'pack-000000.tar'
    t = time.time() + 60
    os.utime(p, (t, t))
    with pytest.raises(ValueError, match='changed after completed readback'):
        build_accounting(plan, root / 'changed')
    p = Path(source['generation_report']['path'])
    value = json.loads(p.read_text())
    value['verified_cache_members'] = 0
    p.write_text(json.dumps(value))
    source['generation_report'] = file_ref(p)
    with pytest.raises(ValueError, match='fully reread'):
        build_accounting(plan, root / 'incomplete')


def test_producer_reuse_is_explicitly_scoped_and_preserves_quality_checks(tmp_path):
    root, plan = prepare(tmp_path)
    p = Path(plan['scope']['path'])
    value = json.loads(p.read_text())
    value.pop('kind')
    p.write_text(json.dumps(value))
    plan['scope'] = file_ref(p)
    with pytest.raises(ValueError, match='explicit reviewed-source stage'):
        build_accounting(plan, root / 'wrong-scope')
    value['kind'] = 'reviewed_source_subset'
    p.write_text(json.dumps(value))
    plan['scope'] = file_ref(p)
    p = Path(plan['sources'][0]['review']['path'])
    review = json.loads(p.read_text())
    review['checks']['visual_quality'] = False
    p.write_text(json.dumps(review))
    plan['sources'][0]['review'] = file_ref(p)
    with pytest.raises(ValueError, match='source quality review'):
        build_accounting(plan, root / 'unreviewed')
