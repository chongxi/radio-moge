import hashlib
import json

import pytest

from combine_procthor_rgb import TRAIN_SHA, VAL_SHA, combine


def frozen(tmp_path, name, split, start):
    root = tmp_path / name
    root.mkdir()
    n = 20 if split == 'train' else 4
    source = TRAIN_SHA if split == 'train' else VAL_SHA
    group = f'procthor10k/{split}/{start:05d}'
    rows = [dict(image=f'/{name}/{i}.png', group=group, dataset='procthor_empty',
        split=split, source_archive_sha256=source, projection='perspective',
        agent_visible=False, geometry_label=None, label_role='feature_only_rgb',
        training_admission=split == 'train', decoded_rgb_sha256=f'{name}-pixel-{i}',
        original_house_sha256=f'{name}-original', house=dict(sha256=f'{name}-empty'))
        for i in range(n)]
    report = dict(stage='frozen_verified_feature_rgb', split=split,
        source=dict(sha256=source), every_selected_rgb_decoded_and_hashed=True,
        original_houses_preserved_except_empty_objects=True, perspective_only=True,
        agent_visible=False, geometry_labels=False, exact_pixel_duplicates=0,
        training_eligible=split == 'train', house_range=[start, start + 1],
        houses=1, images=n, views_per_house=n)
    save(root, report, rows)
    return root, report, rows


def save(root, report, rows):
    raw = ''.join(json.dumps(row)+'\n' for row in rows).encode()
    (root/'rgb.jsonl').write_bytes(raw)
    report['manifest'] = dict(path=str(root/'rgb.jsonl'), sha256=hashlib.sha256(raw).hexdigest())
    (root/'report.json').write_text(json.dumps(report))


def fixtures(tmp_path):
    return [frozen(tmp_path, 'one', 'train', 0), frozen(tmp_path, 'two', 'train', 500),
            frozen(tmp_path, 'val', 'val', 0)]


def run(values, output):
    return combine([v[0]/'report.json' for v in values[:2]], values[2][0]/'report.json', output)


def test_combination_preserves_bytes_without_accessing_verified_image_files(tmp_path):
    values = fixtures(tmp_path)
    report = run(values, tmp_path/'output')
    expected = b''.join((v[0]/'rgb.jsonl').read_bytes() for v in values[:2])
    assert (tmp_path/'output/rgb.jsonl').read_bytes() == expected
    assert report['images'] == 40 and report['houses'] == 2
    assert not report['training_launched'] and not report['current_training_manifest_changed']


@pytest.mark.parametrize('field', ['image', 'decoded_rgb_sha256', 'original_house_sha256', 'empty'])
@pytest.mark.parametrize('target', [1, 2])
def test_cross_batch_and_validation_overlap_fail_before_publication(tmp_path, field, target):
    values = fixtures(tmp_path)
    root, report, rows = values[target]
    for row in rows if field in ('original_house_sha256', 'empty') else rows[:1]:
        if field == 'empty':
            row['house']['sha256'] = values[0][2][0]['house']['sha256']
        else:
            row[field] = values[0][2][0][field]
    save(root, report, rows)
    with pytest.raises(ValueError, match='overlap'):
        run(values, tmp_path/'output')
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('change', ['hash', 'group', 'count', 'panorama', 'agent', 'source'])
def test_changed_frozen_contract_fails_before_publication(tmp_path, change):
    values = fixtures(tmp_path)
    root, report, rows = values[1]
    if change == 'group':
        rows[0]['group'] = 'procthor10k/train/00499'
    elif change == 'count':
        rows.pop()
    elif change == 'panorama':
        rows[0]['projection'] = 'equirectangular'
    elif change == 'agent':
        rows[0]['agent_visible'] = True
    elif change == 'source':
        rows[0]['source_archive_sha256'] = VAL_SHA
    save(root, report, rows)
    if change == 'hash':
        with (root/'rgb.jsonl').open('a') as f:
            f.write('\n')
    with pytest.raises(ValueError):
        run(values, tmp_path/'output')
    assert not (tmp_path/'output').exists()
