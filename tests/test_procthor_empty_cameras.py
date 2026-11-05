import importlib.util
from pathlib import Path
import sys
import gzip
import json
from types import SimpleNamespace

import numpy as np
import pytest


scripts = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('empty_batch', scripts / 'render_procthor_empty_batch.py')
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def room(name, vertices):
    return dict(id=name, floorPolygon=[dict(x=x, y=0., z=z) for x, z in vertices])


def test_concave_floor_poses_stay_inside_with_clearance():
    polygon = [(0, 0), (4, 0), (4, 1.5), (1.5, 1.5), (1.5, 4), (0, 4)]
    positions = batch.room_positions(room('L', polygon))
    assert len(positions) == 4
    for point in positions:
        assert batch.inside(point, polygon)
        assert not (point[0] > 1.5 and point[1] > 1.5)
        assert batch.clearance(point, polygon) >= 0.5
    reversed_positions = batch.room_positions(room('L', polygon[::-1]))
    assert positions == reversed_positions


def test_rooms_share_views_without_forcing_unsafe_narrow_floor():
    rooms = [room('a', [(0, 0), (4, 0), (4, 4), (0, 4)]),
             room('b', [(5, 0), (8, 0), (8, 3), (5, 3)]),
             room('narrow', [(10, 0), (10.7, 0), (10.7, 3), (10, 3)])]
    house = dict(rooms=rooms, metadata=dict(agent=dict(position=dict(y=0.95))))
    slots, omitted = batch.camera_slots(house, 20)
    assert len(slots) == 20 and len(omitted) == 1 and omitted[0]['room'] == 'narrow'
    assert sum(s['room'] == 'a' for s in slots) == 10
    assert sum(s['room'] == 'b' for s in slots) == 10
    assert all(s['position']['y'] == 0.95 for s in slots)
    assert {s['horizon'] for s in slots} == {-10, 0, 10}
    assert len({(s['position']['x'], s['position']['z']) for s in slots}) > 2


def test_empty_floor_selection_is_explicit():
    house = dict(rooms=[room('tiny', [(0, 0), (.5, 0), (.5, .5), (0, .5)])],
                 metadata=dict(agent=dict(position=dict(y=.95))))
    slots, omitted = batch.camera_slots(house, 20)
    assert slots == [] and len(omitted) == 1


def test_single_room_does_not_repeat_the_camera_cycle_before_twenty_images():
    house = dict(rooms=[room('one', [(0, 0), (4, 0), (4, 4), (0, 4)])],
                 metadata=dict(agent=dict(position=dict(y=.95))))
    slots, _ = batch.camera_slots(house, 60)
    keys = [(s['position']['x'], s['position']['z'], s['rotation']['y'], s['horizon']) for s in slots]
    assert len(keys) == len(set(keys)) == 60


@pytest.mark.parametrize('split', ['train', 'val'])
def test_actual_published_rows_keep_official_split(tmp_path, monkeypatch, split):
    house = dict(objects=[{'id': 'chair'}], rooms=[room('one', [(0, 0), (4, 0), (4, 4), (0, 4)])],
                 metadata=dict(agent=dict(position=dict(y=.95))))
    archive = tmp_path / 'houses.jsonl.gz'
    with gzip.open(archive, 'wt') as file:
        file.write(json.dumps(house) + '\n')
    digest = batch.sha(archive)
    monkeypatch.setattr(batch, 'source_spec', lambda root, chosen: (archive, digest, 1))
    monkeypatch.setattr(batch.os, 'sched_setaffinity', lambda *args: None)

    class Controller:
        initialization_parameters = dict(makeAgentsVisible=False)
        last_event = SimpleNamespace(metadata=dict(lastActionSuccess=True))
        index = 0

        def step(self, **pose):
            self.index += 1
            rgb = np.zeros((480, 640, 3), np.uint8)
            rgb[:, :320] = 100 + self.index
            return SimpleNamespace(frame=rgb, metadata=dict(lastActionSuccess=True, agent=pose,
                                                           cameraPosition=pose['position']))

        def stop(self):
            pass

    monkeypatch.setattr(batch, 'create_controller', lambda *args, **kwargs: Controller())
    output = tmp_path / 'render'
    batch.run(tmp_path, output, 0, 1, 20, 60, split=split)
    rows = [json.loads(line) for line in (output / 'rgb-candidates.jsonl').read_text().splitlines()]
    assert len(rows) == 20
    assert {r['split'] for r in rows} == {split}
    assert {r['group'] for r in rows} == {f'procthor10k/{split}/00000'}
    assert all(r['source_archive_sha256'] == digest and not r['training_admission'] for r in rows)
    assert json.loads((output / 'houses' / f'{split}-00000.json').read_text())['objects'] == []
    assert (output / 'house-reports' / f'{split}-00000.json').exists()
    assert json.loads((output / 'report.json').read_text())['dataset_split'] == split


def test_source_split_cannot_alias_training_houses(tmp_path):
    train = batch.source_spec(tmp_path, 'train')
    val = batch.source_spec(tmp_path, 'val')
    assert train[0] != val[0] and train[1] != val[1]
    assert train[2] == 10000 and val[2] == 1000
    with pytest.raises(ValueError, match='official'):
        batch.source_spec(tmp_path, 'test')
