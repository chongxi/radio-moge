"""Publish the staged adapter and comparison Space using normal HF authentication.

Defaults to private and free ZeroGPU. No paid hardware is ever requested here.
Run only after reviewing the staging files and completing the parity report.
Credentials are passed to HF's secret API in memory, never printed or written.
"""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, get_token

MODEL_REPO = "chongxi/radio-moge-adapter-step232288"
SPACE_REPO = "chongxi/radio-moge-vs-moge2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--public", action="store_true", help="Only use with owner's approval")
    args = parser.parse_args()
    root = Path(args.release)
    report = json.loads((root / "verification/parity.json").read_text())
    if not report.get("original_weights_exact") or not report.get("samples"):
        raise ValueError("Missing successful real-weight parity report")
    digest = hashlib.sha256()
    with (root / "model/adapter.pt").open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != report["adapter_sha256"]:
        raise ValueError("Adapter differs from the GPU-verified artifact")
    api = HfApi()
    if api.whoami()["name"] != "chongxi":
        raise ValueError("Authenticate as chongxi before publishing this release")
    private = not args.public
    api.create_repo(MODEL_REPO, private=private, exist_ok=True)
    if api.model_info(MODEL_REPO).private != private:
        raise ValueError("Existing model visibility differs; refusing to change or upload")
    commit = api.upload_folder(
        repo_id=MODEL_REPO, folder_path=root / "model",
        allow_patterns=["adapter.pt", "config.json", "*.whl", "README.md", "THIRD_PARTY.md"],
        commit_message="Release pre-GT step 232288 adapter and perception inference API",
    )
    print(f"Model: https://huggingface.co/{MODEL_REPO}\nRevision: {commit.oid}", flush=True)
    api.create_repo(SPACE_REPO, repo_type="space", space_sdk="gradio",
                    space_hardware="zero-a10g", private=private, exist_ok=True)
    if api.space_info(SPACE_REPO).private != private:
        raise ValueError("Existing Space visibility differs; refusing to change or upload")
    api.add_space_variable(SPACE_REPO, key="ADAPTER_REVISION", value=commit.oid)
    if private:
        token = get_token()
        if not token:
            raise ValueError("No HF token available to authorize private model loading")
        api.add_space_secret(SPACE_REPO, key="HF_TOKEN", value=token)
    api.upload_folder(repo_id=SPACE_REPO, repo_type="space", folder_path=root / "space",
                      allow_patterns=["app.py", "requirements.txt", "README.md", "*.whl",
                                      "THIRD_PARTY.md"],
                      commit_message="Add matched-grid RADIO adapter vs MoGe-2 comparison")
    print(f"Space: https://huggingface.co/spaces/{SPACE_REPO}", flush=True)
    print("Files uploaded. Check Space build/runtime status before claiming it is running.")


if __name__ == "__main__":
    main()
