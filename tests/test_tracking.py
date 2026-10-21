import json
from types import SimpleNamespace

import torch

from radio_moge.checkpoint import rng_state
from radio_moge.tracking import ExperimentTracker
from test_training import config_for, tiny_system
from radio_moge.train import train


class FakeRun:
    def __init__(self, kwargs):
        self.id, self.entity, self.project = kwargs["id"], "test-user", kwargs["project"]
        self.url = f"https://wandb.ai/test-user/{self.project}/runs/{self.id}"
        self.logs, self.definitions = [], []
        self.summary = {}
        self.exit_code = None

    def define_metric(self, *args, **kwargs):
        self.definitions.append((args, kwargs))

    def log(self, values):
        torch.rand(1)  # SDK activity must not alter model/training randomness.
        self.logs.append(values)

    def finish(self, exit_code):
        self.exit_code = exit_code


def fake_sdk(monkeypatch):
    calls, runs = [], []

    def init(**kwargs):
        torch.rand(1)
        calls.append(kwargs)
        runs.append(FakeRun(kwargs))
        return runs[-1]

    monkeypatch.setitem(__import__("sys").modules, "wandb", SimpleNamespace(
        init=init, util=SimpleNamespace(generate_id=lambda: "same-run"),
        Settings=lambda **kwargs: kwargs,
        Image=lambda path, caption: {"path": path, "caption": caption},
    ))
    return calls, runs


def test_tracker_images_steps_identity_and_rng(tmp_path, monkeypatch):
    calls, runs = fake_sdk(monkeypatch)
    config = {"tracking": {"enabled": True, "images": 1}}
    before = rng_state()
    tracker = ExperimentTracker(config, tmp_path)
    (tmp_path / "0000.png").touch()
    (tmp_path / "0001.png").touch()
    tracker.log("train", {"total": 1.0}, 10)
    tracker.log("validation", {"feature/total": 0.9}, 10, tmp_path)
    torch.testing.assert_close(before["torch"], rng_state()["torch"], rtol=0, atol=0)
    assert len(runs[0].logs[1]["validation/decoded_depth"]) == 1
    assert runs[0].logs[0]["optimizer_step"] == runs[0].logs[1]["optimizer_step"] == 10
    tracker.finish()
    resumed = ExperimentTracker(config, tmp_path, resume=True)
    assert calls[1]["resume"] == "must" and calls[1]["id"] == "same-run"
    assert calls[1]["entity"] == "test-user"
    resumed.finish(False)
    assert runs[0].exit_code == 0 and runs[1].exit_code == 1
    assert json.loads((tmp_path / "wandb-run.json").read_text())["url"] == runs[0].url


def test_training_logs_validation_and_resumes_same_run(tmp_path, monkeypatch):
    calls, runs = fake_sdk(monkeypatch)
    config = config_for(tmp_path)
    config["output_dir"] = str(tmp_path / "tracked")
    config["tracking"] = {"enabled": True}
    config["train"].update(steps=4, eval_every=2, eval_at_start=True)

    def evaluate(*args):
        args[5].mkdir(parents=True, exist_ok=True)
        return {"feature/spatial": 1.0, "feature/summary": 0.5,
                "feature/cosine": 0.1, "teacher_depth/abs_rel": 0.2, "postprocess_failed": 0}

    monkeypatch.setattr("radio_moge.train.evaluate", evaluate)
    checkpoint = train(config, stop_after=2, system_adapter=tiny_system())
    train(config, resume=checkpoint, system_adapter=tiny_system())
    steps = [row["optimizer_step"] for run in runs for row in run.logs]
    assert steps == [0, 1, 2, 2, 3, 4, 4]
    assert runs[0].logs[0]["validation/feature/total"] == 1.51
    assert calls[1]["resume"] == "must"
    assert (tmp_path / "tracked/best.pt").is_file()
