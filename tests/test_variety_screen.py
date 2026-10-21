import importlib.util
import json
from pathlib import Path


def module(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_all_architectures_have_same_data_budget_and_objectives(tmp_path):
    (tmp_path / "dataset_info.json").write_text(
        json.dumps({"mixture_weights": {"a": 0.5, "b": 0.5}})
    )
    runner = module("run_variety_architectures")
    for batch in [16, 8, 4]:
        configs = runner.configurations(tmp_path, tmp_path / "run", "test-revision", batch)
        for stage in ["features", "geometry"]:
            reference = configs[("fusion", stage)]
            for head in ["dpt", "attention"]:
                candidate = configs[(head, stage)]
                for key in ["data", "train", "evaluation", "loss", "models"]:
                    assert candidate[key] == reference[key]
                assert candidate["train"]["batch_size"] * candidate["train"]["accumulation"] == 32
                assert candidate["train"]["steps"] == 6000
        assert configs[("fusion", "features")]["loss"]["points"] == 0
        assert configs[("fusion", "geometry")]["loss"]["points"] == 1


def test_report_keeps_macro_selection_and_does_not_rank_incomplete_runs(tmp_path):
    report = module("summarize_variety_screen")
    criterion = report.CRITERION
    directory = tmp_path / "fusion/features/train"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps({"train": {"steps": 6000}}))
    rows = [
        dict(step=0, postprocess_failed=0, **{criterion: 0.4, "teacher_depth/rmse": 1.0}),
        dict(step=1000, postprocess_failed=0, **{criterion: 0.2, "teacher_depth/rmse": 1.1}),
        dict(step=2000, postprocess_failed=0, **{criterion: 0.3, "teacher_depth/rmse": 0.8}),
    ]
    (directory / "validation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (directory / "train.jsonl").write_text(json.dumps({"step": 2000, "seconds": 1.0}) + "\n")
    result = report.summarize(tmp_path)
    assert result["heads"]["fusion"]["features"]["selected"]["step"] == 1000
    assert result["heads"]["fusion"]["features"]["last"]["teacher_depth/rmse"] == 0.8
    assert result["completed_stage_rankings"] == {}
