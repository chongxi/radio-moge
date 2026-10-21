"""Run the authorized indoor feature/geometry stages only after dataset preparation."""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from radio_moge.checkpoint import atomic_json
from radio_moge.config import ROOT, load_config


def preflight(config_path, output, checkpoint, geometry):
    import torch
    from preflight_ablation import check

    try:
        result = check(load_config(config_path), checkpoint, geometry)
    except torch.cuda.OutOfMemoryError:
        atomic_json({"stage": "out_of_memory"}, Path(output))
        return 42
    atomic_json(result, Path(output))
    return 0


def run(args):
    authorization = json.loads(args.authorization.read_text())
    if authorization.get("authorized") is not True:
        raise ValueError("No authorization for training")
    dataset = json.loads((args.dataset / "dataset_info.json").read_text())
    if dataset.get("stage") != "ready" or dataset.get("pilot"):
        raise ValueError("Full dataset must pass preparation before training")
    args.root.mkdir(parents=True, exist_ok=True)
    lock = (args.root / ".training.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.root / "status.json").exists():
        raise FileExistsError("Existing training state requires inspection; do not duplicate jobs")
    status = dict(stage="preflight", started=time.time(), revision=args.revision, stages=[])
    atomic_json(status, args.root / "status.json")
    atomic_json(
        dict(
            authorization=authorization,
            dataset=dataset,
            feature_steps=30000,
            geometry_steps=10000,
            effective_batch=32,
            selection="validation/teacher_depth/abs_rel",
            initialization="Fresh cleaned fusion adapter; geometry initializes feature best",
            gpu_policy="Keep H200 running after completion",
        ),
        args.root / "plan.json",
    )
    try:
        for stage, template in [
            ("features", "indoor-fusion-features.json"),
            ("geometry", "indoor-fusion-geometry.json"),
        ]:
            if json.loads(args.authorization.read_text()).get("authorized") is not True:
                raise ValueError("Training authorization was withdrawn")
            directory = args.root / stage
            directory.mkdir()
            config = load_config(ROOT / "configs" / template)
            config["code_revision"] = args.revision
            config["output_dir"] = str(directory / "train")
            config["data"]["train_manifest"] = str(args.dataset / "train.jsonl")
            config["data"]["val_manifest"] = str(args.dataset / "val.jsonl")
            checkpoint = str(args.root / "features/train/best.pt") if stage == "geometry" else None
            for batch_size, accumulation in [(32, 1), (16, 2)]:
                config["train"].update(batch_size=batch_size, accumulation=accumulation)
                config_path = directory / "config.json"
                atomic_json(config, config_path)
                command = [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "--preflight-config",
                    str(config_path),
                    "--preflight-output",
                    str(directory / f"preflight-b{batch_size}.json"),
                ]
                if checkpoint:
                    command += ["--checkpoint", checkpoint, "--geometry"]
                status.update(stage=f"{stage}_preflight", batch_size=batch_size)
                atomic_json(status, args.root / "status.json")
                with (directory / f"preflight-b{batch_size}.log").open("w") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode == 0:
                    break
                if result.returncode != 42 or batch_size == 16:
                    raise RuntimeError(f"{stage} preflight failed; inspect saved logs")
            command = [sys.executable, "-u", "-m", "radio_moge.train", "--config", str(config_path)]
            if checkpoint:
                command += ["--init", checkpoint]
            with (directory / "console.log").open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                status.update(
                    stage=f"{stage}_training",
                    training_pid=process.pid,
                    batch_size=batch_size,
                    accumulation=accumulation,
                )
                atomic_json(status, args.root / "status.json")
                result = process.wait()
            if result:
                raise RuntimeError(f"{stage} training exited with code {result}")
            import torch

            state = torch.load(
                directory / "train/latest.pt", map_location="cpu", mmap=True, weights_only=True
            )
            if state["step"] != config["train"]["steps"]:
                raise RuntimeError(f"{stage} ended early at step {state['step']}")
            del state
            status["stages"].append(stage)
        status.update(stage="complete", finished=time.time())
    except Exception as error:
        status.update(stage="failed", error=f"{type(error).__name__}: {error}", updated=time.time())
        atomic_json(status, args.root / "status.json")
        raise
    atomic_json(status, args.root / "status.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--preflight-config")
    parser.add_argument("--preflight-output")
    parser.add_argument("--checkpoint")
    parser.add_argument("--geometry", action="store_true")
    args = parser.parse_args()
    if args.preflight_config:
        raise SystemExit(
            preflight(args.preflight_config, args.preflight_output, args.checkpoint, args.geometry)
        )
    if not all((args.dataset, args.root, args.authorization, args.revision)):
        parser.error("dataset, root, authorization and revision are required")
    run(args)


if __name__ == "__main__":
    main()
