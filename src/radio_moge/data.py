import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, Sampler


def read_manifest(path):
    path = Path(path).resolve()
    records = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("image", "depth"):
            if key in row:
                row[key] = str((path.parent / row[key]).resolve())
                if not Path(row[key]).is_file():
                    raise FileNotFoundError(row[key])
        if "image" not in row:
            raise ValueError("Every manifest row requires an image path")
        records.append(row)
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    if len({r["image"] for r in records}) != len(records):
        raise ValueError(f"Duplicate image paths in {path}")
    return records


def check_split(train, validation):
    overlap = {r["image"] for r in train} & {r["image"] for r in validation}
    groups = {r["group"] for r in train if "group" in r}
    overlap_groups = groups & {r["group"] for r in validation if "group" in r}
    hashes = {r["sha256"] for r in train if "sha256" in r}
    overlap_hashes = hashes & {r["sha256"] for r in validation if "sha256" in r}
    if overlap or overlap_groups or overlap_hashes:
        raise ValueError("Train/validation leakage: image paths or scene groups overlap")


def manifest_digest(records):
    # Include size and mtime to detect accidental file replacement on resume.
    metadata = [
        (r, Path(r["image"]).stat().st_size, Path(r["image"]).stat().st_mtime_ns) for r in records
    ]
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()


def tensor_from_pil(image):
    return torch.from_numpy(np.array(image, dtype=np.float32, copy=True)).permute(2, 0, 1) / 255


def image_rotation(row):
    rotation = row.get("image_rotation_cw", 0)
    if type(rotation) is not int or rotation not in {0, 90, 180, 270}:
        raise ValueError("image_rotation_cw must be an integer quarter turn")
    return rotation


def oriented_rgb(row):
    with Image.open(row["image"]) as file:
        image = ImageOps.exif_transpose(file).convert("RGB")
    rotation = image_rotation(row)
    if rotation:
        image = image.transpose(
            {
                90: Image.Transpose.ROTATE_270,
                180: Image.Transpose.ROTATE_180,
                270: Image.Transpose.ROTATE_90,
            }[rotation]
        )
    return image


class TrainingImages(Dataset):
    """A seeded crop is shared by teacher and student, including horizontal flips."""

    def __init__(self, records, augment=True):
        self.records, self.augment = records, augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, key):
        index, seed, h, w = key
        rng = random.Random(seed)
        image = oriented_rgb(self.records[index])
        iw, ih = image.size
        # Largest crop at the chosen aspect ratio; random location, no geometric warping.
        cw, ch = min(iw, max(1, round(ih * w / h))), min(ih, max(1, round(iw * h / w)))
        left = rng.randint(0, iw - cw) if self.augment else (iw - cw) // 2
        top = rng.randint(0, ih - ch) if self.augment else (ih - ch) // 2
        image = image.crop((left, top, left + cw, top + ch))
        if self.augment and rng.random() < 0.5:
            image = ImageOps.mirror(image)
        return tensor_from_pil(image.resize((w, h), Image.Resampling.BILINEAR))


class DeterministicBatches(Sampler):
    """Recreate any microbatch from its absolute index, independent of prefetch/RNG.

    DataLoader prefetch never advances the checkpoint cursor. Every batch has the
    full configured size; the last batch of an epoch wraps within that epoch.
    """

    def __init__(self, size, batch_size, buckets, seed, start, count):
        if min(size, batch_size) < 1 or not buckets:
            raise ValueError("Dataset, batch size and resolution buckets must be nonempty")
        self.size, self.batch_size, self.buckets = size, batch_size, buckets
        self.seed, self.start, self.count = seed, start, count

    def __len__(self):
        return self.count

    def __iter__(self):
        per_epoch = math.ceil(self.size / self.batch_size)
        previous_epoch, order = None, None
        for absolute in range(self.start, self.start + self.count):
            epoch, batch = divmod(absolute, per_epoch)
            if epoch != previous_epoch:
                order = list(range(self.size))
                random.Random(self.seed + epoch).shuffle(order)
                previous_epoch = epoch
            rng = random.Random(self.seed + absolute * 104729)
            h, w = rng.choice(self.buckets)
            yield [
                (order[(batch * self.batch_size + j) % self.size], rng.randrange(2**63), h, w)
                for j in range(self.batch_size)
            ]


def evaluation_image(row, max_side=768):
    image = oriented_rgb(row)
    original_size = image.size
    ratio = min(1, max_side / max(image.size))
    size = tuple(max(1, round(v * ratio)) for v in image.size)
    image = image.resize(size, Image.Resampling.BILINEAR)
    depth = None
    if "depth" in row:
        if Path(row["depth"]).suffix == ".npy":
            array = np.load(row["depth"], allow_pickle=False)
        else:
            with Image.open(row["depth"]) as file:
                array = np.array(file)
        rotation = image_rotation(row)
        native_size = original_size[::-1] if rotation in {90, 270} else original_size
        if array.shape != native_size[::-1]:
            raise ValueError("Depth must match the EXIF-oriented RGB dimensions")
        array = array.astype(np.float32) * float(row.get("depth_scale", 1.0))
        if rotation:
            array = np.rot90(array, -(rotation // 90)).copy()
        depth = torch.from_numpy(
            np.array(Image.fromarray(array).resize(size, Image.Resampling.NEAREST), copy=True)
        )
    return tensor_from_pil(image).unsqueeze(0), depth
