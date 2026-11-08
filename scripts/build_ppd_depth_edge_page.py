"""Build a standalone three-panel viewer for saved depth-edge filtering results."""

import argparse
import base64
import gzip
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from export_ppd_depth_edges import digest, publish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads((args.root / "report.json").read_text())
    data = json.loads((args.root / "data.json").read_text())
    assert report["completed"] and report["data_sha256"] == digest(args.root / "data.json")
    expected = []
    for s in data["samples"]:
        n = ((s["height"] + s["stride"] - 1) // s["stride"]) * ((s["width"] + s["stride"] - 1) // s["stride"])
        binary = gzip.decompress(base64.b64decode(s["payload"]))
        assert len(binary) == n * 14
        base = np.frombuffer(binary, "u1", n, offset=n*8).astype(bool)
        bits = [np.frombuffer(binary, "u1", n, offset=n*(9+i)) for i in range(2)]
        for i in range(2):
            depth = np.frombuffer(binary, "<f4", n, offset=i*n*4)
            assert np.isfinite(depth[base]).all() and (depth[base] > 0).all()
        counts = []
        with np.load(args.root / "masks" / (s["id"] + ".npz")) as masks:
            assert np.array_equal(base, masks["base_mask"][::s["stride"], ::s["stride"]].ravel())
            for index in range(4):
                pair = []
                for method, bits_map in zip(["moge_adapter", "ppd_adapter"], bits):
                    keep = base & ((bits_map & (1 << index)) == 0)
                    assert np.array_equal(keep, masks[f"{method}_keep_{index}"][::s["stride"], ::s["stride"]].ravel())
                    pair.append(int(keep.sum()))
                counts.append(pair)
        expected.append({"id": s["id"], "base_count": int(base.sum()), "filtered_counts": counts})
    publish(args.root / "preview-verification.json", {"passed": True, "cases": expected,
        "payload_masks_equal_full_resolution_masks_at_preview_pixels": True})
    repo = Path(__file__).resolve().parents[1]
    bundle = args.root / "viewer.bundle.js"
    subprocess.run(["bun", "build", str(repo / "web/ppd-depth-edges.js"), "--target=browser", "--minify", "--outfile="+str(bundle)], check=True)
    html = (repo / "web/ppd-depth-edges.html").read_text()
    html = html.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    html = html.replace("__BUNDLE__", bundle.read_text().replace("</script", "<\\/script"))
    for path in [args.output, args.root / "index.html"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".new")
        temporary.write_text(html)
        os.replace(temporary, path)
    print(json.dumps({"cases": len(expected), "bytes": args.output.stat().st_size, "page": str(args.output)}))


if __name__ == "__main__":
    main()
