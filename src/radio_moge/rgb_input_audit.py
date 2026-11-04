"""Observe real RGB samples and prepared compute groups without changing tensors.

Only windows followed by a successful optimizer update are published. Loader and
CUDA prefetch can run ahead; their unused records stay unpublished. Hashes cover
the actual compact CPU tensors before the unchanged normalization/H2D pipeline.
"""
from collections import Counter, deque
import hashlib
from pathlib import Path
from threading import Lock

import torch

from .checkpoint import atomic_json
from .corpus_admission import file_ref
from .geometry_cache import canonical_json


def pixels_sha(tensor):
    if tensor.device.type != 'cpu' or tensor.dtype != torch.uint8:
        raise ValueError('Expected actual compact CPU RGB')
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


class AuditedImages:
    def __init__(self, dataset):
        self.dataset = dataset
        if not dataset.compact_transport:
            raise ValueError('Reference audit requires unchanged compact transport')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        image = self.dataset[key]
        row = self.dataset.records[key[0]]
        identity = dict(key=list(key), image=row['image'], sha256=row.get('sha256'),
                        dataset=row.get('dataset'), group=row.get('group'),
                        image_rotation_cw=row.get('image_rotation_cw', 0),
                        compact_rgb_sha256=pixels_sha(image))
        return image, identity


def collate_audited(samples):
    return torch.stack([x for x, _ in samples]), [r for _, r in samples]


class ObservedLoader:
    def __init__(self, loader, audit):
        self.loader, self.audit = loader, audit

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return ObservedIterator(iter(self.loader), self.audit)


class ObservedIterator:
    def __init__(self, iterator, audit):
        self.iterator, self.audit = iterator, audit

    def __iter__(self):
        return self

    def __next__(self):
        image, identities = next(self.iterator)
        if len(image) != self.audit.logical_batch:
            raise ValueError('Changed logical batch size')
        self.audit.pending_logical.append(identities)
        return image

    def _shutdown_workers(self):
        shutdown = getattr(self.iterator, '_shutdown_workers', None)
        if shutdown:
            shutdown()


class RGBInputAudit:
    collate = staticmethod(collate_audited)
    wrap_dataset = staticmethod(AuditedImages)

    def __init__(self, output, source_step, logical_batch, accumulation, compute_batch,
                 planned_keys=None):
        self.output = Path(output)
        self.source_step, self.logical_batch = source_step, logical_batch
        self.accumulation, self.compute_batch = accumulation, compute_batch
        self.planned_keys = planned_keys
        self.pending_logical, self.pending_windows = deque(), deque()
        self.lock = Lock()
        self.prepared = self.committed = self.cursor = 0
        self.source_counts, self.resolution_counts = Counter(), Counter()
        self.path = self.output / 'actual-rgb-inputs.jsonl'
        self.file = None

    def wrap_loader(self, loader):
        return ObservedLoader(loader, self)

    def observe_window(self, tensors):
        rows = [self.pending_logical.popleft() for _ in range(self.accumulation)]
        flat = [r for batch in rows for r in batch]
        start = self.prepared * self.accumulation * self.logical_batch
        if self.planned_keys is not None and [r['key'] for r in flat] != self.planned_keys[start:start+len(flat)]:
            raise ValueError('Actual dataset keys differ from the frozen reference plan')
        grouped = {}
        for i, row in enumerate(flat):
            grouped.setdefault(tuple(row['key'][2:]), []).append(i)
        expected = [ids[j:j+self.compute_batch] for ids in grouped.values()
                    for j in range(0, len(ids), self.compute_batch)]
        if len(tensors) != len(expected):
            raise ValueError('Prepared compute grouping changed')
        groups = []
        for tensor, slots in zip(tensors, expected, strict=True):
            if len(tensor) != len(slots):
                raise ValueError('Prepared compute batch size changed')
            for image, slot in zip(tensor, slots, strict=True):
                if pixels_sha(image) != flat[slot]['compact_rgb_sha256']:
                    raise ValueError('Prepared pixels changed or image identity was reordered')
            groups.append(dict(logical_image_slots=slots, shape=list(tensor.shape),
                               compact_tensor_sha256=pixels_sha(tensor)))
        value = dict(update=self.source_step+self.prepared+1, logical_images=flat,
                     compute_groups=groups)
        with self.lock:
            self.pending_windows.append(value)
        self.prepared += 1

    def commit(self, step, cursor):
        if (step != self.source_step+self.committed+1
                or cursor != (self.committed+1)*self.accumulation):
            raise ValueError('Successful update or logical cursor changed')
        with self.lock:
            value = self.pending_windows.popleft()
        if value['update'] != step:
            raise ValueError('Prepared and consumed windows differ')
        if self.file is None:
            self.file = self.path.open('x')
        self.file.write(canonical_json(value).decode() + '\n')
        self.file.flush()
        self.committed += 1
        self.cursor = cursor
        for row in value['logical_images']:
            self.source_counts[row['dataset']] += 1
            self.resolution_counts['x'.join(map(str, row['key'][2:]))] += 1

    def finish(self, step, cursor):
        if step != self.source_step+self.committed or cursor != self.cursor:
            raise ValueError('Final checkpoint differs from consumed input audit')
        self.close()
        if not self.path.exists():
            self.path.touch(exist_ok=False)
        report = dict(stage='actual_rgb_inputs_accounted', optimizer_updates=self.committed,
            cursor=cursor, presentations=cursor*self.logical_batch, source_counts=dict(self.source_counts),
            resolution_counts=dict(self.resolution_counts), records=file_ref(self.path),
            planned_keys_matched=self.planned_keys is not None, unused_prefetch_excluded=True,
            actual_compute_groups_verified=True, gpu_float_payload_rehashed=False,
            pixel_hash_scope='Actual uint8 CPU pixels before unchanged float normalization and H2D')
        atomic_json(report, self.output / 'actual-rgb-input-audit.json')
        return report

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None
