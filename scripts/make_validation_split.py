"""Reserve validation scenes inside an existing training manifest; never open test data."""

import argparse
import json
import os
from pathlib import Path
import tempfile

from radio_moge.checkpoint import atomic_json
from radio_moge.data import check_split, read_manifest
from radio_moge.nyuv2 import sha256_file, split_scenes


def prepare(train_manifest, output, fraction=0.1, seed=43):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    records = read_manifest(train_manifest)
    train, validation = split_scenes(records, test_fraction=fraction, seed=seed)
    check_split(train, validation)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".validation-") as temporary:
        temporary = Path(temporary)
        for name, rows in (("train", train), ("val", validation)):
            relative = []
            for row in rows:
                row = dict(row)
                for key in ("image", "depth"):
                    if key in row:
                        row[key] = os.path.relpath(row[key], output)
                relative.append(row)
            (temporary / f"{name}.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in relative)
            )
        atomic_json({
            "source_sha256": sha256_file(train_manifest), "seed": seed,
            "requested_validation_fraction": fraction,
            "train_images": len(train), "validation_images": len(validation),
            "train_groups": len({r["group"] for r in train}),
            "validation_groups": len({r["group"] for r in validation}),
            "protocol": "Validation reserved from original training scenes; test set untouched.",
        }, temporary / "split.json")
        temporary.rename(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    print(prepare(args.train_manifest, args.output, args.fraction, args.seed))
