"""Own the existing GPU training lock across cache creation and a bounded screen."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def atomic(path, value):
    temporary = path.with_suffix(".incoming")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def run(root, repo, revision):
    if (root / "queue-status.json").exists():
        raise FileExistsError("Inspect existing queue, do not duplicate")
    with (root.parent / "attention-24h/.training.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(
            stage="starting",
            pid=os.getpid(),
            revision=revision,
            started_at=time.time(),
            child_pid=None,
        )
        atomic(root / "queue-status.json", state)
        try:
            plan = json.loads((root / "plan.json").read_text())
            if "cache_reuse" in plan:
                payload = (root / "cache-index.json").read_bytes()
                assert hashlib.sha256(payload).hexdigest() == plan["cache_reuse"]["index_sha256"]
                assert json.loads(payload)["stage"] == "passed"
            for stage, command in [
                (
                    "cache",
                    [
                        sys.executable,
                        "-u",
                        str(repo / "scripts/prepare_architecture_screen.py"),
                        "--root",
                        str(root),
                        "--cache",
                    ],
                ),
                (
                    "screen",
                    [
                        sys.executable,
                        "-u",
                        str(repo / "scripts/run_architecture_screen.py"),
                        "--root",
                        str(root),
                        "--revision",
                        revision,
                    ],
                ),
            ]:
                if stage == "cache" and "cache_reuse" in plan:
                    continue
                with (root / f"{stage}-console.log").open("w") as log:
                    child = subprocess.Popen(
                        command, cwd=repo, stdout=log, stderr=subprocess.STDOUT
                    )
                    state.update(stage=stage, child_pid=child.pid, updated_at=time.time())
                    atomic(root / "queue-status.json", state)
                    code = child.wait()
                state.update(child_pid=None, child_exit=code, updated_at=time.time())
                atomic(root / "queue-status.json", state)
                if code:
                    raise RuntimeError(f"{stage} exited {code}; inspect before any retry")
            state.update(
                stage="screen_complete_pending_review", finished_at=time.time(), child_pid=None
            )
        except Exception as error:
            state.update(stage="needs_inspection", error=str(error), updated_at=time.time())
            raise
        finally:
            atomic(root / "queue-status.json", state)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--revision", required=True)
    a = p.parse_args()
    run(a.root, a.repo, a.revision)
