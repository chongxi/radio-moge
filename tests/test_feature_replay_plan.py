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
