"""Queue matched adapter trials after the existing run; never manage GPU lifecycle."""

import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from radio_moge.checkpoint import atomic_json
from radio_moge.config import ROOT, load_config


HEADS = {
    "fusion": {"kind": "fusion"},
    "dpt": {"kind": "dpt", "spatial_width": 384, "summary_width": 512, "summary_heads": 8},
    "attention": {
        "kind": "attention",
        "spatial_width": 1280,
        "attention_blocks": 2,
        "attention_heads": 16,
        "summary_width": 512,
        "summary_heads": 8,
    },
}


def run_child(command, log):
    with log.open("a") as file:
        process = subprocess.Popen(command, stdout=file, stderr=subprocess.STDOUT)
        return process.wait()


def configurations(dataset, root, revision, batch_size=16):
    info = json.loads((dataset / "dataset_info.json").read_text())
    base = load_config(ROOT / "configs/variety-screen.json")
    base["code_revision"] = revision
    base["data"].update(
        train_manifest=str(dataset / "train.jsonl"),
        val_manifest=str(dataset / "val.jsonl"),
        mixture_weights=info["mixture_weights"],
    )
    base["train"].update(batch_size=batch_size, accumulation=32 // batch_size)
    configs = {}
    for head, adapter in HEADS.items():
        for stage in ["features", "geometry"]:
            config = deepcopy(base)
            config["adapter"].update(adapter)
            config["output_dir"] = str(root / head / stage / "train")
            config["tracking"].update(
                name=f"variety-{head}-{stage}", tags=["variety", head, stage, "screen-6000-2000"]
            )
            if stage == "geometry":
                config["train"].update(steps=2000, lr=0.00003, warmup=200)
                config["loss"].update(points=1.0, mask=0.1, scale=0.1, gradient=0.05, normal=0.1)
            configs[(head, stage)] = config
    return configs


def run(args):
    args.root.mkdir(parents=True, exist_ok=True)
    lock = (args.root / ".queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.root / "status.json").exists():
        raise FileExistsError("Inspect existing queue state; do not duplicate trials")
    state = dict(
        stage="waiting",
        pid=os.getpid(),
        started=time.time(),
        revision=args.revision,
        predecessor=str(args.predecessor),
        completed=[],
    )

    def record(**changes):
        state.update(changes, updated=time.time())
        atomic_json(state, args.root / "status.json")

    record()
    try:
        while True:
            prior = json.loads((args.predecessor / "status.json").read_text())
            data_path = args.coco / "status.json"
            data = json.loads(data_path.read_text()) if data_path.exists() else {"stage": "pending"}
            if prior["stage"] == "failed" or data["stage"] == "failed":
                raise RuntimeError(
                    "Predecessor or data preparation failed; inspect before continuing"
                )
            if prior["stage"] == "complete" and data["stage"] == "complete":
                break
            record(stage="waiting", predecessor_stage=prior["stage"], data_stage=data["stage"])
            time.sleep(30)
        # The old supervisor may still be exiting. Its own lock is the authoritative gate.
        prior_lock = (args.predecessor / ".training.lock").open("r+")
        fcntl.flock(prior_lock, fcntl.LOCK_EX)
        fcntl.flock(prior_lock, fcntl.LOCK_UN)
        record(stage="preparing_data")
        from prepare_variety_mix import prepare

        info = prepare(args.base, args.dataset, args.coco)
        atomic_json(info, args.root / "dataset_info.json")
        atomic_json(
            dict(
                heads=HEADS,
                features_steps=6000,
                geometry_steps=2000,
                effective_batch=32,
                selection="macro/teacher_depth/abs_rel",
                data_weights=info["mixture_weights"],
                ordering="All feature stages, then all geometry stages from their own best.",
                purpose="Matched-budget architecture screen, not a convergence comparison.",
                gpu_policy="Keep existing GPU running",
            ),
            args.root / "plan.json",
        )
        # Find ONE common microbatch size before any training; loss and sample stream
        # remain matched across all three heads, including geometry normalization.
        for batch in [16, 8, 4]:
            configs = configurations(args.dataset, args.root, args.revision, batch)
            oom = False
            for (head, stage), config in configs.items():
                directory = args.root / head / stage
                directory.mkdir(parents=True, exist_ok=True)
                config_path = directory / "config.json"
                atomic_json(config, config_path)
                report = directory / f"preflight-b{batch}.json"
                command = [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "--preflight-config",
                    str(config_path),
                    "--preflight-output",
                    str(report),
                ]
                if stage == "geometry":
                    command += ["--geometry"]
                record(stage="preflight", head=head, phase=stage, batch_size=batch)
                result = run_child(command, report.with_suffix(".log"))
                if result == 42:
                    oom = True
                    break
                if result:
                    raise RuntimeError(f"{head}/{stage} preflight exited {result}")
            if not oom:
                break
        else:
            raise RuntimeError("No common safe batch size passed preflight")
        for stage in ["features", "geometry"]:
            for head in HEADS:
                directory = args.root / head / stage
                config = configs[(head, stage)]
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "radio_moge.train",
                    "--config",
                    str(directory / "config.json"),
                ]
                if stage == "geometry":
                    command += ["--init", str(args.root / head / "features/train/best.pt")]
                with (directory / "console.log").open("w") as file:
                    process = subprocess.Popen(command, stdout=file, stderr=subprocess.STDOUT)
                    record(
                        stage="training",
                        head=head,
                        phase=stage,
                        training_pid=process.pid,
                        batch_size=batch,
                        accumulation=32 // batch,
                    )
                    result = process.wait()
                if result:
                    raise RuntimeError(f"{head}/{stage} training exited {result}")
                import torch

                checkpoint = torch.load(
                    directory / "train/latest.pt", map_location="cpu", mmap=True, weights_only=True
                )
                if checkpoint["step"] != config["train"]["steps"]:
                    raise RuntimeError(f"{head}/{stage} ended early")
                del checkpoint
                state["completed"].append(f"{head}/{stage}")
                record()
        record(stage="complete", finished=time.time())
    except BaseException as error:
        record(stage="failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["root", "predecessor", "base", "dataset", "coco"]:
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--preflight-config")
    parser.add_argument("--preflight-output")
    parser.add_argument("--geometry", action="store_true")
    args = parser.parse_args()
    if args.preflight_config:
        from run_indoor_training import preflight

        raise SystemExit(
            preflight(args.preflight_config, args.preflight_output, None, args.geometry)
        )
    if not all([args.root, args.predecessor, args.base, args.dataset, args.coco, args.revision]):
        parser.error("root, predecessor, base, dataset, coco and revision are required")
    run(args)
