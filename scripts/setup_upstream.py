"""Fetch pinned source only; pretrained weights download when loading real models."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from radio_moge.config import MOGE_COMMIT, RADIO_COMMIT  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-utils", action="store_true")
    args = parser.parse_args()
    for name, url, revision in (
        ("MoGe", "https://github.com/microsoft/MoGe.git", MOGE_COMMIT),
        ("RADIO", "https://github.com/NVlabs/RADIO.git", RADIO_COMMIT),
    ):
        path = ROOT / "vendor" / name
        if path.exists():
            actual = subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
            ).strip()
            if actual != revision:
                raise RuntimeError(f"{path} has a different revision; leaving it unchanged")
            continue
        path.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "fetch", "--depth=1", url, revision], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", "FETCH_HEAD"], check=True)
    if args.install_utils:
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "utils3d-moge @ git+https://github.com/EasternJournalist/utils3d-moge.git@"
                "62f09d58509485564e24d5d9f6aac9ee9ebc0c37",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
