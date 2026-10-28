"""Evaluate one frozen trained adapter on the screen's exact confirmation cache."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import torch

from radio_moge.adapters import GeometryAdapter
from radio_moge.checkpoint import atomic_json
from radio_moge.losses import FeatureStatistics
from run_architecture_screen import evaluate, load_cache


def identity(path):
    value = path.stat()
    return dict(inode=value.st_ino, bytes=value.st_size, mtime_ns=value.st_mtime_ns)


def compare(screen, output, source):
    torch.set_num_threads(2)
    lock = (screen.parent / "attention-24h/.training.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip(), "GPU occupied; no concurrent launch"
    output.mkdir(exist_ok=False)
    started = time.time()

    def progress(stage, **extra):
        atomic_json(
            dict(stage=stage, pid=os.getpid(), started_at=started, updated_at=time.time(),
                 optimizer_updates=0, **extra), output / "status.json"
        )

    def timeout(*_):
        raise TimeoutError("600-second read-only comparison budget exceeded")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(600)
    try:
        progress("loading_checkpoint")
        proof = json.loads(source.read_text())
        checkpoint = Path(proof["path"])
        before = identity(checkpoint)
        assert before == proof["identity"]
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        assert state["step"] == proof["step"] == 156000
        plan = json.loads((screen / "plan.json").read_text())
        assert state["config"]["models"] == plan["models"]
        adapter = GeometryAdapter(**state["adapter_config"]).cuda().eval()
        adapter.load_state_dict(state["adapter"], strict=True)
        adapter.requires_grad_(False)
        assert list(adapter.indices) == plan["indices"]
        stats = FeatureStatistics(adapter.config["target_dim"],
                                  adapter.config["target_summary_dim"]).cuda()
        stats.load_state_dict(state["statistics"], strict=True)
        stats.requires_grad_(False)
        metadata = dict(step=state["step"], cursor=state["cursor"],
                        adapter_config=state["adapter_config"], source_proof=proof,
                        checkpoint_identity_before=before)
        del state
        progress("loading_confirmation_cache")
        cache, records = load_cache(screen, "confirmation")
        assert len(records) == 224
        progress("evaluating_frozen_adapter")
        with torch.inference_mode():
            result = evaluate(adapter, {"family": "legacy"}, cache, records,
                              list(range(len(records))), stats, output / "features", 156000,
                              1, dict(optimizer_seconds=0.0, elapsed_seconds=time.time()-started),
                              seed=903)
        assert identity(checkpoint) == before
        metadata.update(
            checkpoint_identity_after=identity(checkpoint), optimizer_updates=0,
            cache_index_sha256=hashlib.sha256((screen / "cache-index.json").read_bytes()).hexdigest(),
            manifest_sha256=hashlib.sha256((screen / "confirmation.jsonl").read_bytes()).hexdigest(),
            inputs="Exact same existing BF16 cache and FP32 targets as attention screen",
            new_encoder_inference=False, batch=16, precision="bf16", images=len(records),
            checkpoint_hash_reused_from_verified_source=True,
        )
        atomic_json(metadata, output / "provenance.json")
        progress("complete", elapsed_seconds=time.time()-started,
                 spatial_cosine=result["macro/spatial/cosine"],
                 summary_cosine=result["macro/summary/cosine"])
        print(json.dumps(dict(stage="complete", spatial=result["macro/spatial/cosine"],
                              summary=result["macro/summary/cosine"], optimizer_updates=0)))
    except BaseException as error:
        progress("failed", error=repr(error))
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    compare(args.screen, args.output, args.source)
