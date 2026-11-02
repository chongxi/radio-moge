"""Exercise the running Gradio upload -> comparison -> NPZ download workflow."""

import argparse
import json
from pathlib import Path
import tempfile

from gradio_client import Client, handle_file
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7861")
    parser.add_argument("--reference", required=True, help="Private NPZ containing a real RGB")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    client = Client(args.url, verbose=False)
    with tempfile.TemporaryDirectory(prefix="radio-moge-smoke-") as folder:
        image_path = Path(folder) / "input.png"
        with np.load(args.reference, allow_pickle=False) as sample:
            Image.fromarray(sample["rgb"]).save(image_path)
        response = client.predict(handle_file(str(image_path)), "400×640", api_name="/compare")
    info = response[4]
    assert info["adapter_step"] == 232288, info
    assert info["processed_hw"] == [400, 640], info
    with np.load(response[5], allow_pickle=False) as outputs:
        assert outputs["rgb"].shape == (400, 640, 3)
        for prefix in ("adapter", "moge2"):
            assert outputs[f"{prefix}_depth"].shape == (400, 640)
            assert outputs[f"{prefix}_points"].shape == (400, 640, 3)
            assert outputs[f"{prefix}_intrinsics"].shape == (3, 3)
            assert outputs[f"{prefix}_mask"].dtype == np.bool_
        info["download_keys"] = sorted(outputs.files)
        info["download_bytes"] = Path(response[5]).stat().st_size
    from radio_moge.checkpoint import atomic_json
    atomic_json(dict(success=True, **info), args.output)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
