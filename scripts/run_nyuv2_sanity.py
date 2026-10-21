"""Run the fixed-budget NYUv2 sanity experiment, then test and export exactly once."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import yaml

from radio_moge.checkpoint import atomic_json
from radio_moge.config import ROOT, load_config
from radio_moge.data import check_split, read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args()
    dataset, run = Path(args.dataset).resolve(), Path(args.run_dir).resolve()
    training, test, subset = (
        read_manifest(dataset / f"{name}.jsonl") for name in ("train", "test", "overfit")
    )
    check_split(training, test)
    if not {r["image"] for r in subset} <= {r["image"] for r in training}:
        raise ValueError("Overfit images must be a subset of the training split")
    run.mkdir(parents=True, exist_ok=False)
    state = {"code_revision": args.code_revision, "dataset": str(dataset)}

    def status(stage, **extra):
        state.update(stage=stage, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
        atomic_json(state, run / "status.json")
        print(json.dumps(state), flush=True)

    def execute(stage, command):
        status(stage)
        with (run / f"{stage}-console.log").open("x") as log:
            subprocess.run([sys.executable, "-u", *command], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        for stage, source, manifest in (
            ("overfit", "nyuv2-overfit.yaml", "overfit.jsonl"),
            ("pilot", "nyuv2.yaml", "train.jsonl"),
        ):
            config = load_config(ROOT / "configs" / source)
            config["output_dir"] = str(run / stage)
            config["data"]["train_manifest"] = str(dataset / manifest)
            config["data"]["val_manifest"] = None
            config_path = run / f"{stage}.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            execute(stage, ["-m", "radio_moge.train", "--config", str(config_path)])
            if stage == "overfit":
                rows = [json.loads(line) for line in (run / stage / "train.jsonl").read_text().splitlines()]
                initial = sum(r["total"] for r in rows[:3]) / 3
                final = sum(r["total"] for r in rows[-3:]) / 3
                if not final < initial:
                    raise RuntimeError("32-image learning check did not reduce feature loss")
                state["learning_check"] = {"initial_loss": initial, "final_loss": final}
        checkpoint = str(run / "pilot/latest.pt")
        execute("test", ["-m", "radio_moge.evaluate", "--checkpoint", checkpoint,
                         "--manifest", str(dataset / "test.jsonl"),
                         "--output", str(run / "test-final")])
        execute("export", ["-m", "radio_moge.export", "--checkpoint", checkpoint,
                           "--output", str(run / "geometry-head.pt")])
        status("complete")
    except Exception as error:
        status("failed", failed_stage=state.get("stage"), error=str(error))
        raise


if __name__ == "__main__":
    main()
