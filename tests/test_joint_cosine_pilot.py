import importlib.util
import json
from pathlib import Path

import pytest
import torch

from radio_moge.adapters import GeometryAdapter, RadioFeatures
from radio_moge.config import load_config


spec = importlib.util.spec_from_file_location(
    'joint_pilot', Path(__file__).resolve().parents[1] / 'scripts/run_joint_cosine_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_warm_start_retains_all_base_weights_and_exact_initial_function():
    torch.manual_seed(3)
    base = GeometryAdapter('attention', 8, 12, 6, 6, indices=[0, 1, 2, 3],
                           spatial_width=16, attention_heads=4, summary_width=8, summary_heads=2)
    parent = dict(adapter_config=base.config, adapter=base.state_dict())
    maps = tuple(torch.randn(1, 8, 2, 3) for _ in range(4))
    inputs = RadioFeatures(maps[-1], torch.randn(1, 12), maps)
    control, joint = [pilot.make_adapter(parent, arm) for arm in pilot.ARMS]
    with torch.autocast('cpu', dtype=torch.bfloat16):
        a, b = control(inputs), joint(inputs)
    torch.testing.assert_close(a.spatial, b.spatial, rtol=0, atol=0)
    torch.testing.assert_close(a.summary, b.summary, rtol=0, atol=0)
    assert sum(p.numel() for p in joint.parameters()) > sum(p.numel() for p in control.parameters())


def test_protocol_changes_only_extension_and_is_feature_only(tmp_path):
    # Small configuration fixtures, no weights, data reads, or GPU work.
    template = load_config(Path(__file__).parents[1] / 'configs/attention-full-day.json')
    previous = tmp_path / 'teacher-plane-ownflat512-pilot-20261028'
    previous.mkdir()
    (previous / 'completion-verification.json').write_text('{"stage":"passed"}')
    (previous / 'plane-config.json').write_text(json.dumps(template))
    for name in ('train', 'validation', 'train-probe', 'screenshot'):
        (previous / f'{name}.jsonl').write_text('{"id":"fixture"}\n')
    original = tmp_path / 'attention-24h'
    original.mkdir()
    (original / 'config.json').write_text(json.dumps(template))
    root = tmp_path / 'new'
    plan = pilot.prepare(tmp_path, root, 'test-revision')
    configs = [json.loads((root / f'{a}-config.json').read_text()) for a in pilot.ARMS]
    for c in configs:
        assert c['train']['steps'] == 600 and c['train']['max_seconds'] == 1200
        assert c['loss']['spatial'] == c['loss']['summary'] == 1
        assert c['loss']['cosine'] == .1
        assert all(v == 0 for k, v in c['loss'].items() if k not in ('spatial', 'summary', 'cosine'))
        for k in ('output_dir', 'adapter', 'tracking'):
            c.pop(k)
    assert configs[0] == configs[1]
    assert plan['evaluation_steps'] == list(range(0, 601, 100))
    with pytest.raises(FileExistsError):
        pilot.prepare(tmp_path, root, 'test-revision')
