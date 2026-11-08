"""Build the minimal point-cloud runtime wheel (no training/data/PPD dependencies)."""

import argparse
from pathlib import Path
import shutil
import subprocess

from build_inference_release import MODULES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(root / "deployment/pointcloud/example.py", args.output / "example.py")
    package = args.output / "package"
    code = package / "src/radio_moge"
    code.mkdir(parents=True)
    for name in MODULES:
        shutil.copy2(root / f"src/radio_moge/{name}.py", code / f"{name}.py")
    shutil.copy2(root / "deployment/pointcloud/pyproject.toml", package / "pyproject.toml")
    shutil.copy2(root / "docs/pointcloud-api.md", package / "README.md")
    shutil.copy2(root / "THIRD_PARTY.md", package / "THIRD_PARTY.md")
    subprocess.run(["uv", "build", "--wheel", "--no-create-gitignore", "--out-dir",
                    str(args.output / "wheels"), str(package)], check=True)


if __name__ == "__main__":
    main()
