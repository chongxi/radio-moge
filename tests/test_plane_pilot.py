import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'plane_pilot', Path(__file__).parents[1] / 'scripts/run_plane_subset_pilot.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_scene_sampling_is_reproducible_without_duplicate_frames():
    rows = [dict(id=f'{g}-{i}', group=str(g)) for g in range(50) for i in range(3)]
    selected = runner.select_scenes(rows, 32, 2, 42)
    assert selected == runner.select_scenes(list(reversed(rows)), 32, 2, 42)
    assert len(selected) == len({r['id'] for r in selected}) == 64
    assert len({r['group'] for r in selected}) == 32


def test_control_matches_budget_limited_plane_steps_and_duplicate_is_rejected(tmp_path, monkeypatch):
    base = tmp_path / 'base'
    (base / 'attention-24h').mkdir(parents=True)
    root = base / 'pilot'
    root.mkdir()
    parent = base / 'parent.pt'
    parent.write_bytes(b'immutable')
    (root / 'plan.json').write_text(json.dumps(dict(
        max_updates=300, max_seconds_per_arm=1800, revision='revision', parent=str(parent),
        parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest())))
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **kw: '')
    calls = []

    class Child:
        pid = 123
        returncode = 0

        def poll(self):
            return 0

    def launch(command, **kwargs):
        calls.append(command)
        if '--config' in command:
            arm = 'plane' if len(calls) == 1 else 'control'
            (root / arm).mkdir()
            (root / arm / 'stop-report.json').write_text(json.dumps(
                dict(step=123, final_evaluation='complete')))
        return Child()

    monkeypatch.setattr(runner.subprocess, 'Popen', launch)
    runner.run(base, root, tmp_path, 'revision')
    assert calls[0][-1] == '300' and calls[1][-1] == '123'
    assert '--init' in calls[0] and '--init' in calls[1]
    assert json.loads((root / 'status.json').read_text())['stage'] == 'complete'
    with pytest.raises(FileExistsError):
        runner.run(base, root, tmp_path, 'revision')
    assert len(calls) == 3
