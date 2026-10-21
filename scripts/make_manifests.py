"""Split images by parent directory (scene/sequence), never individual video frames."""

import argparse
import hashlib
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        parser.error("val-fraction must be between zero and one")
    root = args.root.resolve()
    groups, seen = {}, set()
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"} or not path.is_file():
            continue
        # Remove byte-identical copies before assigning split groups.
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        group = str(path.parent.relative_to(root))
        groups.setdefault(group, []).append({"image": str(path), "group": group, "sha256": digest})
    keys = sorted(groups)
    if len(keys) < 2:
        parser.error("Need at least two scene/sequence folders for a disjoint validation split")
    random.Random(args.seed).shuffle(keys)
    nval = max(1, min(len(keys) - 1, round(len(keys) * args.val_fraction)))
    val = [row for key in keys[:nval] for row in groups[key]]
    train = [row for key in keys[nval:] for row in groups[key]]
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val), ("overfit", train[:32])):
        with (args.output / f"{name}.jsonl").open("x") as file:
            for row in rows:
                file.write(json.dumps(row) + "\n")
    print(f"{len(train)} train / {len(val)} val images, {len(keys)} disjoint scene groups")


if __name__ == "__main__":
    main()
