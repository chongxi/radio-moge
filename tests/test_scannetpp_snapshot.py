import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

spec = importlib.util.spec_from_file_location('snapshot', Path(__file__).parents[1] /
                                             'scripts/prepare_scannetpp_snapshot.py')
snapshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot)


def test_requires_both_receipts_before_cutoff_and_checks_metadata(tmp_path):
    dataset, control = tmp_path / 'data', tmp_path / 'control'
    dataset.mkdir()
    for name in ['receipts', 'extractions']:
        (control / name).mkdir(parents=True)
    relative = 'images.zip'
    receipt = control / 'receipts' / (snapshot.sha(relative.encode()) + '.json')
    receipt.write_text(json.dumps(dict(path=relative, completed_at=10)))
    assert snapshot.receipt_ready(control, dataset, relative, 20, True) is None
    extraction = control / 'extractions' / (snapshot.sha(str(dataset / relative).encode()) + '.json')
    extraction.write_text(json.dumps(dict(stage='passed', crc_checked=True, completed_at=30)))
    assert snapshot.receipt_ready(control, dataset, relative, 20, True) is None
    assert snapshot.receipt_ready(control, dataset, relative, 40, True)
    relative = 'views.json'
    (dataset / relative).write_text('[]')
    (control / 'receipts' / (snapshot.sha(relative.encode()) + '.json')).write_text(json.dumps(
        dict(path=relative, completed_at=10, bytes=2, sha256=snapshot.sha(b'[]'))))
    assert snapshot.receipt_ready(control, dataset, relative, 20)
    (dataset / relative).write_text('{}')
    with pytest.raises(ValueError, match='differs'):
        snapshot.receipt_ready(control, dataset, relative, 20)


def test_scene_balanced_validation_and_quality_filter(tmp_path):
    rows = [dict(image=f'{scene}/{i}', group=scene) for scene in ['a', 'b', 'c']
            for i in range(10)]
    chosen = snapshot.fixed_validation(rows, 6)
    assert [r['group'] for r in chosen] == ['a', 'b', 'c'] * 2
    image, mask = tmp_path / 'image.jpg', tmp_path / 'mask.png'
    Image.new('RGB', (640, 480), 'red').save(image)
    Image.new('L', (640, 480), 255).save(mask)
    row = dict(image=str(image), mask=str(mask), width=640, height=480)
    result, reason = snapshot.inspect_image(row)
    assert reason is None and result['valid_fraction'] == 1
    values = np.full((480, 640), 255, dtype=np.uint8)
    values[:100] = 0
    Image.fromarray(values).save(mask)
    assert snapshot.inspect_image(row) == (None, 'valid_fraction_below_0.98')
    image.write_bytes(image.read_bytes()[:100])
    with pytest.raises(OSError):
        snapshot.inspect_image(row)


def test_train_views_exclude_official_test_and_bad_camera(tmp_path):
    directory = tmp_path / 'data/scene/dslr'
    (directory / 'nerfstudio').mkdir(parents=True)
    (directory / 'train_test_lists.json').write_text(json.dumps(
        dict(train=['one.jpg', 'bad.jpg'], test=['test.jpg'])))
    camera = dict(camera_model='PINHOLE', w=640, h=480, fl_x=200, fl_y=200, cx=320, cy=240,
                  frames=[dict(file_path=n, is_bad=n=='bad.jpg', transform_matrix=np.eye(4).tolist())
                          for n in ['one.jpg', 'bad.jpg', 'test.jpg']])
    (directory / 'nerfstudio/transforms_undistorted.json').write_text(json.dumps(camera))
    rows = snapshot.candidate_views(tmp_path, 'scene', 'nvs_sem_train')
    assert [r['view'] for r in rows] == ['one.jpg']


def test_holdout_rule_is_stable_and_bounded_workers_preserve_order(monkeypatch):
    scenes = [f'{i:010x}' for i in range(856)]
    heldout = {s for s in scenes if snapshot.internal_holdout(s)}
    assert 20 < len(heldout) < 70
    assert heldout == {s for s in reversed(scenes) if snapshot.internal_holdout(s)}
    monkeypatch.setattr(snapshot, 'inspect_image', lambda r: (dict(r, checked=True), None))
    rows = [dict(index=i) for i in range(41)]
    result = list(snapshot.bounded_inspection(rows, 4))
    assert [r[0] for r in result] == rows
    assert all(r[1][0]['checked'] for r in result)
