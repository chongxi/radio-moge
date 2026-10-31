import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


scripts = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('expansion', scripts / 'expand_scannetpp_snapshot.py')
expansion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(expansion)
sys.path.remove(str(scripts))


def fixture(tmp_path, monkeypatch):
    previous = tmp_path / 'previous'
    previous.mkdir()
    heldouts = tmp_path / 'heldouts'
    heldouts.mkdir()
    original = b'{"image":"old.jpg", "sha256":"existing"}\n'
    validation = b''.join((json.dumps(dict(scene='holdout', sha256=f'val-{n}')) + '\n').encode()
                          for n in range(128))
    (previous / 'train.jsonl').write_bytes(original)
    (previous / 'val-scannetpp.jsonl').write_bytes(validation)
    (heldouts / 'val-original.jsonl').write_text('{"sha256":"old-heldout"}\n')
    old_scene = dict(scene='old', split='nvs_sem_train')
    (previous / 'snapshot.json').write_text(json.dumps(dict(
        scenes=[old_scene], cutoff=10, internal_validation_rule='fixed')))
    proof = dict(stage='passed', decoded_all_accepted_images=True,
                 counts=dict(original_train=1, scannetpp_train=0),
                 manifest_sha256={'train.jsonl': hashlib.sha256(original).hexdigest(),
                                  'val-scannetpp.jsonl': hashlib.sha256(validation).hexdigest()})
    (previous / 'verification.json').write_text(json.dumps(proof))
    scenes = [old_scene, dict(scene='new', split='nvs_sem_train'),
              dict(scene='holdout', split='nvs_sem_train'),
              dict(scene='official-val', split='nvs_sem_val')]
    monkeypatch.setattr(expansion, 'select_scenes', lambda *args: (scenes, {}))
    monkeypatch.setattr(expansion, 'internal_holdout', lambda s: s == 'holdout')
    called = []

    def candidates(dataset, scene, split):
        called.append(scene)
        return [dict(image=f'{k}.jpg', scene=scene, sha256=k)
                for k in ['existing', 'val-0', 'old-heldout', 'accepted', 'rejected']]

    def inspect(rows, workers):
        assert workers == 2
        for row in rows:
            yield row, ((None, 'valid_fraction_below_0.98') if row['sha256'] == 'rejected'
                        else (row, None))

    monkeypatch.setattr(expansion, 'candidate_views', candidates)
    monkeypatch.setattr(expansion, 'bounded_inspection', inspect)
    args = dict(dataset=tmp_path / 'dataset', control=tmp_path / 'control',
                previous=previous, heldout_root=heldouts, output=tmp_path / 'expanded',
                cutoff=20, workers=2)
    return args, original, validation, called


def test_expands_only_new_train_scenes_preserving_prefix_and_validation(tmp_path, monkeypatch):
    args, original, validation, called = fixture(tmp_path, monkeypatch)
    expansion.expand(**args)
    output = args['output']
    merged = (output / 'train.jsonl').read_bytes()
    assert called == ['new']
    assert merged.startswith(original) and len(merged.splitlines()) == 2
    assert (output / 'val-scannetpp.jsonl').read_bytes() == validation
    assert (args['previous'] / 'train.jsonl').read_bytes() == original
    proof = json.loads((output / 'verification.json').read_text())
    assert proof['counts']['duplicate_filtered'] == 3
    assert proof['counts']['scannetpp_train'] == 1
    assert proof['counts']['valid_fraction_below_0.98'] == 1
    assert proof['manifest_sha256']['train.jsonl'] == hashlib.sha256(merged).hexdigest()
    assert proof['fixed_validation_byte_identical']
    assert not proof['parent_accepted_images_redecoded']
    with pytest.raises(ValueError, match='new sibling'):
        expansion.expand(**args)


@pytest.mark.parametrize('filename', ['train.jsonl', 'val-scannetpp.jsonl'])
def test_parent_corruption_fails_before_decoding(tmp_path, monkeypatch, filename):
    args, _, _, called = fixture(tmp_path, monkeypatch)
    with (args['previous'] / filename).open('ab') as file:
        file.write(b'\n')
    with pytest.raises((ValueError, json.JSONDecodeError)):
        expansion.expand(**args)
    assert called == []
    assert not (args['output'] / 'verification.json').exists()
    assert json.loads((args['output'] / 'status.json').read_text())['stage'] == 'failed'
