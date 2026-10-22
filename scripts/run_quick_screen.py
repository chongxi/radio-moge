"""Run one bounded architecture-screen rung using the unchanged training code.

All numerical configs and the full learning-rate horizon remain unchanged.
Stopping uses the trainer's existing --stop-after and continuation uses --resume.
This supervisor never signals processes or manages the GPU lifecycle.
"""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


HEADS = ("fusion", "dpt", "attention")
MATCHED = ("data", "models", "train", "loss", "evaluation")


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_rows(path):
    if not path.exists():
        return []
    rows = []
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return rows


def validate_matched_configs(root, stop_after):
    configs = {
        head: json.loads((root / head / "features/config.json").read_text()) for head in HEADS
    }
    base = configs["fusion"]
    for head, config in configs.items():
        if any(config[key] != base[key] for key in MATCHED):
            raise ValueError(f"Unmatched data/training/evaluation configuration: {head}")
        train = config["train"]
        if not 0 < stop_after <= train["steps"] or stop_after % train["eval_every"]:
            raise ValueError("Rung must end at a validation boundary within the original horizon")
        expected = root / head / "features/train"
        if Path(config["output_dir"]).resolve() != expected.resolve():
            raise ValueError(f"Unexpected output directory for {head}")
    return configs


def checkpoint_step(path):
    if not path.exists():
        return 0
    import torch

    torch.set_num_threads(2)
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if not all(k in state for k in ("optimizer", "scheduler", "cursor", "rng")):
        raise ValueError(f"Checkpoint cannot resume a training run: {path}")
    return int(state["step"])


def command_for(root, head, stop_after, python, step):
    directory = root / head / "features"
    latest = directory / "train/latest.pt"
    if step >= stop_after:
        return None
    command = [
        python,
        "-u",
        "-m",
        "radio_moge.train",
        "--config",
        str(directory / "config.json"),
        "--stop-after",
        str(stop_after),
    ]
    if latest.exists():
        command += ["--resume", str(latest)]
    elif (directory / "train").exists() and any((directory / "train").iterdir()):
        raise ValueError(f"Partial run without resumable checkpoint needs inspection: {head}")
    return command


def feedback(root, stop_after):
    by_head = {}
    criterion = "macro/teacher_depth/abs_rel"
    for head in HEADS:
        path = root / head / "features/train/validation.jsonl"
        rows = [r for r in read_rows(path) if r["step"] == stop_after]
        if len(rows) > 1:
            raise ValueError(f"Duplicate validation entries at rung {stop_after}: {head}")
        by_head[head] = rows[0] if rows else None
    comparable = all(
        row is not None
        and math.isfinite(row.get(criterion, math.nan))
        and row.get("postprocess_failed") == 0
        for row in by_head.values()
    )
    return {
        "checked_at": time.time(),
        "step": stop_after,
        "phase": "features",
        "same_step_validation": by_head,
        "all_heads_evaluated_without_failures": comparable,
        "descriptive_order": sorted(HEADS, key=lambda h: by_head[h][criterion])
        if comparable
        else None,
        "scope": "Common 884-image internal validation; one seed, same updates and examples.",
        "decision": "Review domain metrics and learning curves before expansion. No automatic "
        "elimination at 1000; no held-out-test ranking or generalization winner claim here.",
        "next_work": "Compare paired scenes and same fixed visual samples; expand promising "
        "architectures at a matched budget. Do not launch all geometry runs automatically.",
    }


def run(args):
    root = args.root.resolve()
    lock = (root / ".queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    configs = validate_matched_configs(root, args.stop_after)
    state = {
        "stage": "quick_screen",
        "pid": os.getpid(),
        "started": time.time(),
        "revision": args.revision,
        "training_repo": str(args.training_repo),
        "training_revision": configs["fusion"].get("code_revision"),
        "rung": args.stop_after,
        "heads": args.heads,
        "completed": [],
        "gpu_policy": "Never stop GPU",
        "new_data_policy": "Candidate table only; no downloads/cleaning",
    }

    def record(**changes):
        state.update(changes, updated=time.time())
        atomic_json(state, root / "quick-screen-status.json")
        atomic_json(state, root / "status.json")

    environment = dict(os.environ, PYTHONPATH=str(args.training_repo / "src"))
    record()
    try:
        for head in args.heads:
            directory = root / head / "features"
            latest = directory / "train/latest.pt"
            step = checkpoint_step(latest)
            command = command_for(root, head, args.stop_after, args.python, step)
            if command:
                with (directory / "console.log").open("a") as log:
                    process = subprocess.Popen(
                        command,
                        cwd=args.training_repo,
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    record(
                        stage="training",
                        head=head,
                        phase="features",
                        training_pid=process.pid,
                        batch_size=configs[head]["train"]["batch_size"],
                        accumulation=configs[head]["train"]["accumulation"],
                        command=command,
                    )
                    result = process.wait()
                if result:
                    raise RuntimeError(f"{head} exited {result}; inspect logs before continuing")
                step = checkpoint_step(latest)
                if step != args.stop_after:
                    raise RuntimeError(f"{head} ended at {step}, expected {args.stop_after}")
            rows = read_rows(directory / "train/validation.jsonl")
            if not any(r["step"] == args.stop_after for r in rows):
                raise RuntimeError(f"{head} has no validation at the requested rung")
            # Training saves by atomic replacement, so these links retain the
            # exact rung weights even after a later continuation replaces latest/best.
            if step == args.stop_after:
                archive = directory / "rungs" / str(args.stop_after)
                archive.mkdir(parents=True, exist_ok=True)
                for name in ("latest.pt", "best.pt"):
                    source, target = directory / "train" / name, archive / name
                    if source.exists() and not target.exists():
                        os.link(source, target)
            state["completed"].append(f"{head}/features@{args.stop_after}")
            atomic_json(
                feedback(root, args.stop_after), root / f"quick-feedback-{args.stop_after}.json"
            )
            record(stage="between_trials", training_pid=None)
        record(stage="review_ready", training_pid=None, finished=time.time())
    except BaseException as error:
        record(stage="failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--training-repo", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--stop-after", type=int, required=True)
    parser.add_argument("--heads", nargs="+", choices=HEADS, default=["dpt", "attention"])
    options = parser.parse_args()
    if len(options.heads) != len(set(options.heads)):
        parser.error("Repeated heads would duplicate work")
    run(options)
