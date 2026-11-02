"""Build a small inference wheel and stage explicit, reviewed HF upload contents."""

import argparse
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("__init__", "adapters", "architecture_screen", "spatial_heads", "feature_refinement",
           "config", "upstream", "decoder", "checkpoint", "deployment")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="release staging folder, outside git")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    package, model, space = (output / x for x in ("package", "model", "space"))
    code = package / "src/radio_moge"
    for path in (code, model, space):
        path.mkdir(parents=True, exist_ok=True)
    for name in MODULES:
        shutil.copy2(ROOT / f"src/radio_moge/{name}.py", code / f"{name}.py")
    shutil.copy2(ROOT / "deployment/huggingface/runtime-pyproject.toml", package / "pyproject.toml")
    shutil.copy2(ROOT / "THIRD_PARTY.md", package / "THIRD_PARTY.md")
    subprocess.run(["uv", "build", "--wheel", "--no-create-gitignore", "--out-dir", str(model),
                    str(package)], check=True)
    shutil.copy2(ROOT / "deployment/huggingface/model-card.md", model / "README.md")
    shutil.copy2(ROOT / "THIRD_PARTY.md", model / "THIRD_PARTY.md")
    for path in (ROOT / "deployment/huggingface/space").iterdir():
        if path.is_file() and path.suffix in (".py", ".md", ".txt"):
            shutil.copy2(path, space / path.name)
    shutil.copy2(model / "radio_moge_inference-0.1.0-py3-none-any.whl", space)
    shutil.copy2(ROOT / "THIRD_PARTY.md", space / "THIRD_PARTY.md")
    print(f"Staged model: {model}\nStaged Space: {space}")


if __name__ == "__main__":
    main()
