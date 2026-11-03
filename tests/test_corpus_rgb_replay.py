from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3

from PIL import Image
import pytest

from prepare_corpus_rgb_replay import run
from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import publish_json, sha_file


def fixture(root):
    rows = []
    for i, dataset in enumerate(['hypersim'] * 3 + ['tartanair'] * 3 + ['nyuv2']):
        path = root / f'rgb-{i}.png'
        Image.new('RGB', (8, 6), (255, 255, 255) if i == 0 else (i, 2, 3)).save(path)
        rows.append(dict(image=path.name, sha256=sha_file(path), dataset=dataset,
            group=dataset + ':scene', split='train', width=8, height=6,
            depth='missing-depth-do-not-read.npy', geometry='missing-geometry-do-not-read.tar',
            camera='not-RGB', mask='not-RGB', image_rotation_cw=90 if i == 3 else 0))
    # The original legacy NYU hash protocol must survive projection unchanged.
    rows[-1].update(sha256_kind='decoded_rgb', decoded_rgb_sha256=hashlib.sha256(b'legacy').hexdigest())
    manifest = root / 'original.jsonl'
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with sqlite3.connect(root / 'inventory.sqlite') as db:
        db.execute('CREATE TABLE images(ordinal,dataset,scene_group,image,sha256,row_json)')
        for i, row in enumerate(rows):
            db.execute('INSERT INTO images VALUES(?,?,?,?,?,?)', (i, row['dataset'], row['group'],
                str(root / row['image']), row['sha256'], json.dumps(row)))
    counts = dict(hypersim=3, tartanair=3, nyuv2=1)
    config = dict(root=str(root), manifest_sha256=sha_file(manifest), expected_counts=counts)
    publish_json(root / 'inventory.json', dict(stage='inventory_complete', images=7,
        dataset_counts=counts, duplicate_encoded_image_rows=0, manifest=str(manifest),
        manifest_sha256=config['manifest_sha256'], inventory_sha256=sha_file(root / 'inventory.sqlite')))
    proof_path = root / 'rgb-review.json'
    blank = dict(corpus_ordinal=0, image=str(root / rows[0]['image']), sha256=rows[0]['sha256'],
        group=rows[0]['group'], channel_min=[255] * 3, channel_max=[255] * 3, spatially_constant=True)
    publish_json(proof_path, dict(stage='rgb_input_review_complete', passed=True,
        decision='exclude_exactly_spatially_constant_rgb', training_admission=False,
        excluded_images=[blank], excluded_corpus_ordinals=[0]))
    geometry_proof = root / 'geometry-review.json'
    publish_json(geometry_proof, dict(reason='low geometry support, RGB unjudged'))
    decisions = []
    for i in (0, 3, 4):
        rgb_excluded = i == 0
        proof = file_ref(proof_path if rgb_excluded else geometry_proof)
        item = dict(corpus_ordinal=i, dataset=rows[i]['dataset'], image=str(root / rows[i]['image']),
            sha256=rows[i]['sha256'], decision='exclude', reason='reviewed test decision',
            scope='geometry_and_rgb_replay' if rgb_excluded else 'geometry_supervision',
            rgb_replay_excluded=rgb_excluded, evidence=[proof])
        if rgb_excluded:
            item['rgb_input_review'] = proof
        decisions.append(item)
    return config, rows, decisions


def save_decisions(root, decisions, name='decisions.jsonl'):
    path = root / name
    path.write_text(''.join(json.dumps(r) + '\n' for r in decisions))
    return file_ref(path)


def test_scope_filter_preserves_geometry_only_rgb_orientation_hash_protocol_and_never_reads_labels(tmp_path):
    config, rows, decisions = fixture(tmp_path)
    result = run(config, save_decisions(tmp_path, decisions), 'pool')
    actual = [json.loads(line) for line in Path(result['manifest']['path']).read_text().splitlines()]
    assert [r['corpus_ordinal'] for r in actual] == list(range(1, 7))
    assert result['candidate_counts'] == dict(hypersim=2, tartanair=3, nyuv2=1)
    assert result['rgb_excluded_ordinals'] == [0]
    assert result['geometry_only_retained_ordinals'] == [3, 4]
    assert actual[2]['image_rotation_cw'] == 90
    assert actual[-1]['decoded_rgb_sha256'] == rows[-1]['decoded_rgb_sha256']
    assert actual[-1]['sha256_kind'] == rows[-1]['sha256_kind']
    assert all(not {'depth', 'geometry', 'mask', 'camera'} & r.keys() for r in actual)
    assert not result['sampling_policy_applied'] and not result['training_admission']
    assert not result['canonical_heldout_comparison_complete'] and not result['all_rgb_pixels_reread']
    assert result['constant_rgb_exclusions_reverified'] == 1
    assert sum(x['images'] for x in json.loads(Path(result['groups']['path']).read_text())) == 6
    with pytest.raises(FileExistsError):
        run(config, save_decisions(tmp_path, decisions), 'pool')


def test_geometry_exclusion_cannot_silently_remove_rgb_or_substitute_for_rgb_review(tmp_path):
    config, _, original = fixture(tmp_path)
    for i, mode in enumerate(('missing_scope', 'conflicting_flag', 'promote_without_rgb_review')):
        decisions = deepcopy(original)
        if mode == 'missing_scope':
            decisions[1].pop('scope')
        else:
            decisions[1]['rgb_replay_excluded'] = True
            if mode == 'promote_without_rgb_review':
                decisions[1]['scope'] = 'geometry_and_rgb_replay'
        with pytest.raises(ValueError, match='scoped|own reviewed'):
            run(config, save_decisions(tmp_path, decisions), f'bad-{i}')
        assert not (tmp_path / f'bad-{i}/candidates-rgb.jsonl').exists()


def test_duplicate_outside_scope_or_changed_identity_decisions_fail_without_publishing(tmp_path):
    config, _, original = fixture(tmp_path)
    for i, mode in enumerate(('duplicate', 'outside', 'changed_image')):
        decisions = deepcopy(original)
        if mode == 'duplicate':
            decisions.append(deepcopy(decisions[1]))
        elif mode == 'outside':
            decisions[1]['corpus_ordinal'] = 100
        else:
            decisions[1]['image'] += '.wrong'
        with pytest.raises(ValueError, match='Unique|accounted exactly once|frozen RGB identity'):
            run(config, save_decisions(tmp_path, decisions), f'bad-{i}')
        assert not (tmp_path / f'bad-{i}/candidates-rgb.jsonl').exists()


def test_stale_evidence_and_changed_inventory_are_rejected(tmp_path):
    config, _, decisions = fixture(tmp_path)
    ref = save_decisions(tmp_path, decisions)
    proof = tmp_path / 'geometry-review.json'
    proof.write_text('{}')
    with pytest.raises(ValueError, match='Evidence changed'):
        run(config, ref, 'stale-proof')
    decisions = [decisions[0]]
    with sqlite3.connect(tmp_path / 'inventory.sqlite') as db:
        db.execute("UPDATE images SET dataset='unknown' WHERE ordinal=2")
    with pytest.raises(ValueError, match='Evidence changed'):
        run(config, save_decisions(tmp_path, decisions), 'stale-inventory')


def test_blank_review_is_checked_against_actual_rgb_pixels(tmp_path):
    config, _, decisions = fixture(tmp_path)
    proof_path = tmp_path / 'rgb-review.json'
    proof = json.loads(proof_path.read_text())
    proof['excluded_images'][0].update(channel_min=[0] * 3, channel_max=[0] * 3)
    proof_path.write_text(json.dumps(proof))
    new_ref = file_ref(proof_path)
    decisions[0].update(rgb_input_review=new_ref, evidence=[new_ref])
    with pytest.raises(ValueError, match='exactly spatially constant reviewed RGB'):
        run(config, save_decisions(tmp_path, decisions), 'wrong-color')
    assert not (tmp_path / 'wrong-color/candidates-rgb.jsonl').exists()
