from copy import deepcopy
import hashlib
import json
import sqlite3

import numpy as np
from PIL import Image
import pytest

from audit_corpus_rgb import run
from radio_moge.geometry_cache import publish_json, sha_file
from radio_moge.rgb_audit import (choose_groups, choose_rows, decoded_hash, dhash128,
                                   inspect_rgb, pair_diagnostics)


def row_for(path, dataset='hypersim', **extra):
    with Image.open(path) as image:
        width, height = image.size
    return dict(image=str(path), dataset=dataset, group=dataset + ':scene', split='train',
        sha256=sha_file(path), sha256_kind='encoded_file_bytes', width=width, height=height, **extra)


def test_canonical_decode_bridges_source_hash_conventions_and_file_containers(tmp_path):
    image = Image.fromarray(np.arange(18 * 24 * 3, dtype=np.uint8).reshape(18, 24, 3))
    a, b = tmp_path / 'a.png', tmp_path / 'b.png'
    image.save(a, compress_level=0)
    image.save(b, compress_level=9)
    assert sha_file(a) != sha_file(b)
    nyu_hash = hashlib.sha256(image.tobytes()).hexdigest()
    archive_hash = decoded_hash(image)
    assert nyu_hash != archive_hash
    _, nyu = inspect_rgb(row_for(a, 'nyuv2', decoded_rgb_sha256=nyu_hash))
    _, archive = inspect_rgb(row_for(b, 'structured3d', decoded_rgb_sha256=archive_hash))
    assert nyu['canonical_rgb_sha256'] == archive['canonical_rgb_sha256'] == archive_hash
    with pytest.raises(ValueError, match='documented convention'):
        inspect_rgb(row_for(a, 'nyuv2', decoded_rgb_sha256=archive_hash))


def test_only_explicit_frozen_rotation_and_oriented_dimensions_are_used(tmp_path):
    image = Image.fromarray(np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3))
    path = tmp_path / 'rgb.png'
    image.save(path)
    row = row_for(path, image_rotation_cw=90)
    with pytest.raises(ValueError, match='dimensions differ'):
        inspect_rgb(row)
    row.update(width=8, height=12)
    rotated, info = inspect_rgb(row)
    np.testing.assert_array_equal(np.asarray(rotated), np.rot90(np.asarray(image), -1))
    assert info['canonical_rgb_sha256'] != info['raw_canonical_rgb_sha256']
    for rotation in (True, 1, -90):
        with pytest.raises(ValueError, match='rotation'):
            inspect_rgb(dict(row, image_rotation_cw=rotation))


def test_changed_encoded_pixels_and_exif_are_not_silently_transformed(tmp_path):
    path = tmp_path / 'rgb.png'
    Image.new('RGB', (16, 12), 'red').save(path)
    row = row_for(path)
    Image.new('RGB', (16, 12), 'blue').save(path)
    with pytest.raises(ValueError, match='encoded RGB hash differs'):
        inspect_rgb(row)
    image = Image.new('RGB', (16, 12), 'blue')
    exif = Image.Exif()
    exif[274] = 3
    image.save(path, exif=exif)
    with pytest.raises(ValueError, match='EXIF orientation'):
        inspect_rgb(row_for(path))


def test_sampling_is_stable_and_retains_dense_scene_windows():
    groups = [dict(dataset=d, group=f'{d}:{i}', images=i + 1)
              for d, count in [('irs', 13), ('hypersim', 100)] for i in range(count)]
    chosen = choose_groups(groups)
    assert chosen == choose_groups(list(reversed(groups)))
    assert sum(g['dataset'] == 'irs' for g in chosen) == 13
    selected = {g['group'] for g in chosen}
    assert {'hypersim:0', 'hypersim:99'} <= selected
    rows = [dict(corpus_ordinal=i, sha256=f'{99 - i:064x}') for i in range(100)]
    sample = choose_rows(rows)
    assert sample == choose_rows(list(reversed(rows)))
    assert {0, 1, 2, 3, 48, 49, 50, 51, 96, 97, 98, 99} <= {r['corpus_ordinal'] for r in sample}


def test_descriptor_collision_is_only_a_retrieval_candidate():
    a, b = Image.new('RGB', (20, 20), 'red'), Image.new('RGB', (20, 20), 'blue')
    assert dhash128(a) == dhash128(b) and decoded_hash(a) != decoded_hash(b)
    records = [dict(corpus_ordinal=i, dataset='coco2017', group=str(i),
                    dhash128=dhash128(im), canonical_rgb_sha256=decoded_hash(im)) for i, im in enumerate([a, b])]
    result = pair_diagnostics(records)
    assert result['candidate_pairs']['coco2017:cross_group'] == 1
    assert not result['canonical_exact_clusters'] and result['automatic_exclusions'] == 0
    assert result['requires_visual_review'] and not result['full_corpus_near_duplicate_scan']


def census_fixture(root):
    path = root / 'rgb.png'
    Image.new('RGB', (16, 12), 'green').save(path)
    counts, groups = {}, []
    with sqlite3.connect(root / 'inventory.sqlite') as db:
        db.execute('CREATE TABLE images(ordinal,image,row_json,dataset,scene_group)')
        for ordinal, dataset in enumerate(('hypersim', 'coco2017', 'nyuv2')):
            row = row_for(path, dataset)
            db.execute('INSERT INTO images VALUES (?,?,?,?,?)',
                (ordinal, str(path), json.dumps(row), dataset, row['group']))
            counts[dataset] = 1
            groups.append(dict(dataset=dataset, group=row['group'], images=1))
    publish_json(root / 'inventory.json', dict(stage='inventory_complete', manifest_sha256='a' * 64,
        images=3, dataset_counts=counts, source_groups=groups, inventory_sha256=sha_file(root / 'inventory.sqlite')))
    return dict(root=str(root), manifest_sha256='a' * 64, expected_counts=counts)


def test_census_finds_same_pixels_across_sources_without_excluding_or_accepting(tmp_path):
    config = census_fixture(tmp_path)
    preflight = run(config, 'preflight', 'probe', 'unused')
    assert preflight['processed'] == 3 and preflight['failed_images'] == 0
    census = run(config, 'census', 'census', 'probe')
    assert census['exact_duplicate_summary']['canonical_rgb_sha256']['cross_dataset_clusters'] == 1
    assert census['automatic_exclusions'] == 0 and census['training_admission'] is False
    assert 'passed' not in census and not census['semantic_content_reviewed']
    with pytest.raises(FileExistsError):
        run(config, 'preflight', 'probe', 'unused')
    changed = deepcopy(config)
    changed['expected_counts']['hypersim'] = 2
    with pytest.raises(ValueError, match='frozen inventory'):
        run(changed, 'preflight', 'bad-counts', 'unused')


def test_failed_preflight_retains_error_and_prevents_full_census(tmp_path):
    config = census_fixture(tmp_path)
    (tmp_path / 'rgb.png').write_bytes(b'changed bytes')
    preflight = run(config, 'preflight', 'probe', 'unused')
    assert preflight['failed_images'] == 3 and preflight['visuals'] == []
    with pytest.raises(ValueError, match='successful RGB identity preflight'):
        run(config, 'census', 'blocked', 'probe')
