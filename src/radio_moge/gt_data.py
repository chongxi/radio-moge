"""Calibrated RGB/metric-GT pairs with shared deterministic crop and flip."""
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .data import tensor_from_pil


GT_SOURCES = ('scannetpp_panocam_laser', 'scannetpp_dslr_mesh')


def validate_gt_records(records, split='train'):
    if split not in ('train', 'val'):
        raise ValueError('Unsupported GT split')
    for row in records:
        scene = row.get('scene', '')
        heldout = int(hashlib.sha256(('scannetpp-internal-val-v1:' + scene).encode())
                      .hexdigest()[:8], 16) % 20 == 0
        if (row.get('supervision') not in GT_SOURCES or row.get('split') != split
                or row.get('official_split') != 'nvs_sem_train' or heldout != (split == 'val')
                or row.get('group') != 'scannetpp:' + scene or not scene
                or not row.get('geometry') or not row.get('geometry_sha256')):
            raise ValueError('GT training requires verified measured geometry and eligible training scenes')
        if row.get('image_rotation_cw', 0) != 0:
            raise ValueError('GT pairs must be published in their final camera orientation')


def geometry_digest(records):
    """Detect replacement of paired labels as well as RGB on exact resume."""
    rows = []
    for row in records:
        path = Path(row['geometry'])
        info = path.stat()
        rows.append((row['geometry'], row['geometry_sha256'], info.st_size, info.st_mtime_ns))
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


class GroundTruthImages(Dataset):
    def __init__(self, records, augment=True):
        self.records, self.augment = records, augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, key):
        index, seed, height, width = key
        row, rng = self.records[index], random.Random(seed)
        with Image.open(row['image']) as image_file:
            if image_file.getexif().get(274, 1) != 1:
                raise ValueError('GT RGB must already be EXIF-oriented')
            image = image_file.convert('RGB')
        with np.load(row['geometry'], allow_pickle=False) as packed:
            depth = packed['depth'].astype(np.float32)
            valid = packed['valid'].astype(bool)
            intrinsics = packed['intrinsics'].astype(np.float32)
            labels = packed['plane_labels'].astype(np.uint8)
            planes = packed['planes'].astype(np.float32)
        iw, ih = image.size
        if (depth.shape != (ih, iw) or valid.shape != depth.shape or labels.shape != depth.shape
                or intrinsics.shape != (3, 3) or planes.shape != (6, 4)):
            raise ValueError('RGB, GT and calibration dimensions differ')
        if (not np.isfinite(intrinsics).all() or intrinsics[0, 0] <= 0
                or intrinsics[1, 1] <= 0 or not np.allclose(intrinsics[2], [0, 0, 1])
                or intrinsics[0, 1] != 0 or intrinsics[1, 0] != 0):
            raise ValueError('GT requires finite normalized pinhole intrinsics without skew')
        if not np.isfinite(planes).all() or np.any(labels > 6):
            raise ValueError('Invalid GT plane data')
        occupied = np.unique(labels[labels > 0]).astype(int) - 1
        if len(occupied) and not np.allclose(np.linalg.norm(planes[occupied, :3], axis=1),
                                            1., atol=1e-4):
            raise ValueError('GT plane normals must have unit length')
        cw, ch = min(iw, max(1, round(ih * width / height))), min(ih, max(1, round(iw * height / width)))
        left = rng.randint(0, iw - cw) if self.augment else (iw - cw) // 2
        top = rng.randint(0, ih - ch) if self.augment else (ih - ch) // 2
        flip = self.augment and rng.random() < .5
        image = image.crop((left, top, left + cw, top + ch))
        # Normalized pixel-center K: resize cancels after normalizing by new W,H.
        intrinsics[0, 0] *= iw / cw
        intrinsics[1, 1] *= ih / ch
        intrinsics[0, 2] = (intrinsics[0, 2] * iw - left) / cw
        intrinsics[1, 2] = (intrinsics[1, 2] * ih - top) / ch
        arrays = [depth, valid.astype(np.uint8), labels]
        arrays = [a[top:top + ch, left:left + cw] for a in arrays]
        if flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            arrays = [a[:, ::-1].copy() for a in arrays]
            intrinsics[0, 2] = 1 - intrinsics[0, 2]
            planes[:, 0] *= -1  # Reflect camera x, preserving positive focal length.
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        depth, valid, labels = [np.array(Image.fromarray(a).resize(
            (width, height), Image.Resampling.NEAREST), copy=True) for a in arrays]
        valid = valid.astype(bool) & np.isfinite(depth) & (depth > 0)
        if not valid.any():
            raise ValueError('GT crop has no valid measurements')
        return {'image': tensor_from_pil(image), 'depth': torch.from_numpy(depth),
                'valid': torch.from_numpy(valid), 'intrinsics': torch.from_numpy(intrinsics),
                'plane_labels': torch.from_numpy(labels), 'planes': torch.from_numpy(planes)}
