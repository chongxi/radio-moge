import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from radio_moge.architecture_screen import make_candidate
from radio_moge.capacity_screen import capacity_candidates, initialization_seed


def test_grid_changes_only_capacity_and_initialization_seed():
    rows = capacity_candidates()
    assert len(rows) == len({r["name"] for r in rows}) == 12
    for row in rows:
        assert row["position"] == "rope" and row["gate"] and row["registers"] == 4
        assert row["activation"] == "swiglu" and row["layer_scale"] == 0.01
        assert row["heads"] * 64 == row["width"] and not row["local"]
    assert {(r["blocks"], r["width"], r["seed"]) for r in rows} == {
        (b, w, s) for b in (4, 8) for w in (512, 768, 1024) for s in (42, 43)
    }
    counts = {}
    with torch.device("meta"):
        for row in rows[:6]:
            counts[row["blocks"], row["width"]] = sum(
                p.numel() for p in make_candidate(row).parameters()
            )
    assert counts[4, 512] == 17638628
    for b in (4, 8):
        assert counts[b, 512] < counts[b, 768] < counts[b, 1024]
    for w in (512, 768, 1024):
        assert counts[4, w] < counts[8, w]


def test_seed_override_repeats_init_but_changes_replicate():
    plan = {"seed": 42}
    spec = dict(capacity_candidates()[0], width=16, heads=2, blocks=1)
    states = []
    for seed in (42, 43, 42):
        spec["seed"] = seed
        torch.manual_seed(initialization_seed(spec, plan))
        states.append(
            make_candidate(
                spec, dict(spatial_dim=8, summary_dim=12, target_dim=6, target_summary_dim=10)
            ).state_dict()
        )
    assert all(torch.equal(states[0][k], states[2][k]) for k in states[0])
    assert any(not torch.equal(states[0][k], states[1][k]) for k in states[0])
    assert initialization_seed({}, plan) == 42


def test_wide_grid_preserves_blocks_components_and_seed_pairing():
    rows = capacity_candidates(((4, 1024), (4, 1536), (4, 2048)))
    assert len(rows) == 6
    assert {(r["blocks"], r["width"], r["seed"]) for r in rows} == {
        (4, w, s) for w in (1024, 1536, 2048) for s in (42, 43)
    }
    counts = []
    with torch.device("meta"):
        for row in rows[:3]:
            assert row["heads"] * 64 == row["width"]
            assert (
                row["position"] == "rope"
                and row["gate"]
                and row["registers"] == 4
                and row["activation"] == "swiglu"
            )
            counts.append(sum(p.numel() for p in make_candidate(row).parameters()))
    assert counts[0] == 60408196 and counts[0] < counts[1] < counts[2]
    assert 3 < counts[2] / counts[0] < 4


def test_reused_cache_checksum_failure_never_starts_gpu_child(tmp_path, monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "capacity_queue", scripts / "run_architecture_queue.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "new"
    root.mkdir()
    (tmp_path / "attention-24h").mkdir()
    (root / "plan.json").write_text(json.dumps({"cache_reuse": {"index_sha256": "wrong"}}))
    (root / "cache-index.json").write_text('{"stage":"passed"}')

    def forbidden(*args, **kwargs):
        raise RuntimeError("Child must never launch")

    monkeypatch.setattr(module.subprocess, "Popen", forbidden)
    with pytest.raises(AssertionError):
        module.run(root, tmp_path, "test")
    assert json.loads((root / "queue-status.json").read_text())["stage"] == "needs_inspection"


def test_4096_grid_keeps_exact_2048_control_and_counts_parameters_without_allocation():
    previous = capacity_candidates(((4, 1024), (4, 1536), (4, 2048)))
    rows = capacity_candidates(((4, 2048), (4, 4096)))
    assert len(rows) == 4
    for seed in (42, 43):
        control = next(r for r in rows if r["width"] == 2048 and r["seed"] == seed)
        assert control == next(r for r in previous if r["width"] == 2048 and r["seed"] == seed)
        treatment = next(r for r in rows if r["width"] == 4096 and r["seed"] == seed)
        assert {k: v for k, v in treatment.items() if k not in ("name", "width", "heads")} == {
            k: v for k, v in control.items() if k not in ("name", "width", "heads")
        }
        assert treatment["heads"] == 64
    with torch.device("meta"):
        counts = [sum(p.numel() for p in make_candidate(r).parameters()) for r in rows[:2]]
    assert counts == [221739844, 846392900]
