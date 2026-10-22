"""Preflight and run the authorized 24h Attention experiment on prepared data.

Downloads/extraction must already be finished. No dataset or GPU lifecycle
operations occur here. The trainer owns the cumulative wall-clock limit.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from radio_moge.checkpoint import atomic_json
from radio_moge.config import load_config
from radio_moge.train import validate_config

from prepare_full_attention_mix import HELDOUT, SOURCES, WEIGHTS, digest


def ready(plan, config):
    if plan.get("authorized") is not True:
        raise ValueError("Training authorization is missing or withdrawn")
    if config["adapter"]["kind"] != "attention" or config["data"]["mixture_weights"] != WEIGHTS:
        raise ValueError("Unexpected architecture or source mixture")
    if config["train"].get("max_seconds") != 86400:
        raise ValueError("This plan requires the cumulative 24h budget")
    if any(config["loss"].get(k, 0) for k in ("points", "mask", "scale", "gradient", "normal")):
        raise ValueError("This experiment retains the successful feature objective")
    if not config["tracking"]["enabled"] or config["tracking"]["mode"] != "online":
        raise ValueError("Online W&B tracking is required")
    dataset = Path(plan["prepared_dataset"]).resolve()
    if Path(config["data"]["train_manifest"]).resolve() != dataset / "train.jsonl":
        raise ValueError("Unexpected training manifest")
    if Path(config["data"]["val_manifest"]).resolve() != dataset / "val-selection.jsonl":
        raise ValueError("Unexpected selection manifest")
    info = json.loads((dataset / "dataset_info.json").read_text())
    if info.get("stage") != "ready" or set(info["train_sources"]) != set(WEIGHTS):
        raise ValueError("All seven prepared sources must be ready")
    if any(info["train_sources"][n] <= 0 for n in WEIGHTS):
        raise ValueError("Every source needs eligible training RGB")
    if info.get("selection_images") != plan.get("selection_images", 1268):
        raise ValueError("Prepared selection panel differs from the declared full validation")
    if set(info.get("new_counts", {})) != set(SOURCES):
        raise ValueError("Three newly queued source audits are required")
    if info.get("mixture_weights") != WEIGHTS:
        raise ValueError("Prepared mixture weights differ from training configuration")
    checks = info.get("verification", {})
    if not all(
        checks.get(k) is True
        for k in (
            "all_paths_exist",
            "split_groups_paths_hashes_disjoint",
            "old_holdouts_byte_identical",
        )
    ):
        raise ValueError("Data preparation checks did not pass")
    expected_manifests = {"train.jsonl", "val-selection.jsonl", "test-irs-official.jsonl"}
    expected_manifests.update(f"{s}.jsonl" for s in HELDOUT)
    expected_manifests.update(f"{s}-{n}.jsonl" for s in ("val", "test") for n in SOURCES)
    if set(info["manifest_sha256"]) != expected_manifests:
        raise ValueError("Incomplete prepared manifest inventory")
    for name, sha in info["manifest_sha256"].items():
        if digest(dataset / name) != sha:
            raise ValueError(f"Prepared manifest changed: {name}")
    for split in HELDOUT:
        if (
            digest(Path(plan["base_dataset"]) / f"{split}.jsonl")
            != info["preserved_heldout_sha256"][split]
        ):
            raise ValueError("Original held-out manifests changed")
    roots = plan["download_roots"]
    expected = {
        "tartanair": {"complete"},
        "structured3d": {"archives_complete", "archives_complete_with_missing_assets"},
        "irs": {"archives_complete"},
    }
    for name in SOURCES:
        source = Path(roots[name])
        status = json.loads((source / "status.json").read_text())
        if status["stage"] not in expected[name]:
            raise ValueError(f"{name} download is not complete")
        with (source / ".download.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if name == "structured3d":
            missing = status.get("missing", [])
            if missing not in ([], ["Structured3D_perspective_full_09.zip"]):
                raise ValueError("Unexpected missing Structured3D archives")
            available = {
                f"Structured3D_perspective_{kind}_{i:02d}.zip"
                for kind in ("full", "empty")
                for i in range(18)
            } | {"Structured3D_annotation_3d.zip", "Structured3D_bbox.zip"}
            available -= set(missing)
            if (
                len(status["completed"]) != len(available)
                or {r["name"] for r in status["completed"]} != available
            ):
                raise ValueError("Structured3D archive inventory incomplete")
        if name == "irs" and set(status["completed"]) != {
            "Store.tar.gz",
            "Home-1.tar.gz",
            "Home-2.tar.gz",
            "Office-1.tar.gz",
            "Office-2.tar.gz",
            "Restaurant.tar.gz",
        }:
            raise ValueError("IRS full archive inventory incomplete")
    if digest(plan["initial_checkpoint"]) != plan["initial_sha256"]:
        raise ValueError("Attention6000 initialization checkpoint changed")
    validate_config(config)
    return info


def run(args):
    plan = json.loads(args.plan.read_text())
    config = load_config(args.config)
    info = ready(plan, config)
    root = Path(plan["output_root"]).resolve()
    if Path(config["output_dir"]).resolve() != root / "train":
        raise ValueError("Unexpected run output directory")
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / ".training.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "status.json").exists() or (root / "train").exists():
        raise FileExistsError("Existing run needs inspection; never launch a duplicate")
    repo = args.training_repo.resolve()
    config["code_revision"] = args.revision
    atomic_json(config, root / "config.json")
    state = dict(
        stage="preflight",
        pid=os.getpid(),
        started=time.time(),
        revision=args.revision,
        dataset_images=info["train_images"],
        wandb_url=None,
        gpu_policy="Never stop GPU",
    )
    environment = dict(os.environ, PYTHONPATH=str(repo / "src"))

    def record(**changes):
        state.update(changes, updated=time.time())
        atomic_json(state, root / "status.json")

    record()
    try:
        command = [
            sys.executable,
            "-u",
            str(repo / "scripts/run_indoor_training.py"),
            "--preflight-config",
            str(root / "config.json"),
            "--preflight-output",
            str(root / "preflight.json"),
            "--checkpoint",
            plan["initial_checkpoint"],
        ]
        with (root / "preflight.log").open("w") as log:
            result = subprocess.run(
                command, cwd=repo, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
        if result.returncode:
            raise RuntimeError("Preflight failed; inspect logs before launch")
        # Recheck authorization and hashes after preflight; do not consume training
        # time while downloads/preparation are incomplete or files are changing.
        ready(json.loads(args.plan.read_text()), config)
        command = [
            sys.executable,
            "-u",
            "-m",
            "radio_moge.train",
            "--config",
            str(root / "config.json"),
            "--init",
            plan["initial_checkpoint"],
        ]
        with (root / "console.log").open("w") as log:
            child = subprocess.Popen(
                command,
                cwd=repo,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            record(stage="training", training_pid=child.pid, command=command)
            while child.poll() is None:
                identity = root / "train/wandb-run.json"
                if identity.exists() and not state["wandb_url"]:
                    wandb = json.loads(identity.read_text())
                    record(wandb_url=wandb["url"], wandb=wandb)
                    print(json.dumps({"wandb_url": wandb["url"]}), flush=True)
                time.sleep(10)
        if child.returncode:
            raise RuntimeError(f"Trainer exited {child.returncode}; inspect logs/checkpoint")
        stop = json.loads((root / "train/stop-report.json").read_text())
        if stop["reason"] == "signal" or stop["final_evaluation"] != "complete":
            record(stage="interrupted", stop=stop)
            return
        record(stage="training_complete_export_pending", stop=stop, finished=time.time())
    except BaseException as error:
        record(stage="needs_inspection", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-repo", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    run(parser.parse_args())
