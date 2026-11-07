"""Package a verified PPD cloud export and local Three.js into an offline page."""

import argparse
import base64
import gzip
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from probe_ppd_feature_swap import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads((args.root / "report.json").read_text())
    data = json.loads((args.root / "data.json").read_text())
    assert report["completed"] and data["completed"] and len(data["samples"]) == 32
    assert report["data_sha256"] == digest(args.root / "data.json")
    assert not report["ground_truth_used"] and not report["diffusion_inference_repeated"]
    for sample in data["samples"]:
        n = ((sample["height"] + 3) // 4) * ((sample["width"] + 3) // 4)
        binary = gzip.decompress(base64.b64decode(sample["payload"]))
        assert len(binary) == n * 25
        for i in range(5):
            z = np.frombuffer(binary, dtype="<f4", count=n, offset=i * n * 4)
            reference = 0 if i in (0, 2, 4) else 1
            valid = np.frombuffer(binary, dtype="u1", count=n, offset=n * (20 + reference)).astype(bool)
            assert np.isfinite(z[valid]).all() and (z[valid] > 0).all()
        for i in range(2):
            mask = np.frombuffer(binary, dtype="u1", count=n, offset=n * (20 + i))
            assert set(np.unique(mask)) <= {0, 1} and mask.sum() > 1000
        for k in sample["cameras"].values():
            k = np.asarray(k)
            assert k.shape == (3, 3) and np.isfinite(k).all()
            assert k[0, 0] > 0 and k[1, 1] > 0
            assert np.allclose(k[[0, 1], [1, 0]], 0)
    repo = Path(__file__).resolve().parents[1]
    bundle = args.root / "viewer.bundle.js"
    subprocess.run(["bun", "build", str(repo / "web/ppd-clouds.js"), "--target=browser", "--minify", "--outfile=" + str(bundle)], check=True)
    html = (repo / "web/ppd-clouds.html").read_text()
    html = html.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    html = html.replace("__BUNDLE__", bundle.read_text().replace("</script", "<\\/script"))
    for path in (args.output, args.root / "index.html"):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".new")
        temporary.write_text(html)
        os.replace(temporary, path)
    print(json.dumps({"scenes": 16, "cases": 32, "bytes": args.output.stat().st_size, "page": str(args.output)}))


if __name__ == "__main__":
    main()
