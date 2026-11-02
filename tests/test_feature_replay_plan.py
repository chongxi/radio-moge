import json

import pytest

from scripts.run_feature_replay_ablation import (check_heldout_metadata, quotas, select_rgb_pool,
                                                write_new_jsonl)


def test_replay_selection_is_reproducible_source_balanced_and_rgb_only(tmp_path):
    rows = [dict(image=f'{source}-{i}.png', dataset=source, group=f'{source}-{i // 4}',
                 split='train', depth='must-not-read.npy', image_rotation_cw=90)
            for source in ('a', 'b') for i in range(20)]
    manifest = tmp_path / 'train.jsonl'
    write_new_jsonl(rows, manifest)
    with pytest.raises(FileExistsError):
        write_new_jsonl(rows, manifest)
    selected, proof = select_rgb_pool(manifest, {'a': .3, 'b': .7}, 10, 43)
    repeated, repeat_proof = select_rgb_pool(manifest, {'a': .3, 'b': .7}, 10, 43)
    assert selected == repeated and proof == repeat_proof
    assert proof['selected_sources'] == {'a': 3, 'b': 7}
    assert all('depth' not in row and row['image_rotation_cw'] == 90 for row in selected)
    assert len({row['image'] for row in selected}) == 10
    assert sum(quotas({'a': .33, 'b': .67}, 7).values()) == 7
    heldout = tmp_path / 'val.jsonl'
    heldout.write_text(json.dumps(dict(image='val.png', group='val')) + '\n')
    assert check_heldout_metadata(selected, [heldout])[str(heldout)]['overlap'] == 0
    heldout.write_text(json.dumps(dict(image='val.png', group=selected[0]['group'])) + '\n')
    with pytest.raises(ValueError, match='overlaps heldout'):
        check_heldout_metadata(selected, [heldout])


def test_replay_source_rejects_validation_rows(tmp_path):
    manifest = tmp_path / 'bad.jsonl'
    write_new_jsonl([dict(image='a.png', dataset='a', group='a', split='val')], manifest)
    with pytest.raises(ValueError, match='training scenes'):
        select_rgb_pool(manifest, {'a': 1.}, 1, 43)


def test_official_irs_audit_allows_shared_scene_but_never_shared_test_frames(tmp_path):
    selected = [dict(image=str(tmp_path / 'train.png'), group='irs:scene', sha256='train-hash')]
    audit = tmp_path / 'official.jsonl'
    row = dict(image='test.png', group='irs:scene', sha256='test-hash', dataset='irs',
               split='official-test', official_split='test')
    audit.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='overlaps heldout'):
        check_heldout_metadata(selected, [audit])
    proof = check_heldout_metadata(selected, [audit], [audit])[str(audit)]
    assert not proof['scene_disjoint_required'] and proof['shared_scene_rows'] == 1
    for change in ({'image': 'train.png'}, {'sha256': 'train-hash'}):
        audit.write_text(json.dumps(dict(row, **change)) + '\n')
        with pytest.raises(ValueError, match='overlaps heldout'):
            check_heldout_metadata(selected, [audit], [audit])
    audit.write_text(json.dumps(dict(row, dataset='another-source')) + '\n')
    with pytest.raises(ValueError, match='restricted to the IRS'):
        check_heldout_metadata(selected, [audit], [audit])


def test_irs_training_pool_rejects_official_test_frames(tmp_path):
    manifest = tmp_path / 'bad-irs.jsonl'
    write_new_jsonl([dict(image='a.png', dataset='irs', group='irs:a', split='train',
                          official_split='test')], manifest)
    with pytest.raises(ValueError, match='exclude every official test frame'):
        select_rgb_pool(manifest, {'irs': 1.}, 1, 43)
