"""Acceptance-gated, disk-backed records and variable native-grid GT batches."""
import hashlib
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .corpus_admission import readonly_db
from .geometry_cache import canonical_json
from .geometry_sampling import load_candidate


class CorpusRecords:
    """Only opened with an already verified seal, before model construction.

    Read-only connections are per-process, not shared across DataLoader forks.
    Original frozen RGB records are exposed to existing calibration/sampling code;
    the large cache references stay in SQLite until the image is actually used.
    """
    def __init__(self, gate):
        if gate.get('stage') != 'accepted_for_training' or gate.get('training_admission') is not True:
            raise ValueError('A verified corpus acceptance gate is required')
        self.gate = gate
        self.path = gate['index']['path']
        self.size = gate['selected_images']
        self._connection, self._pid = None, None

    @property
    def db(self):
        if self._connection is None or self._pid != os.getpid():
            if self._connection is not None:
                self._connection.close()
            self._connection = readonly_db(self.path)
            self._pid = os.getpid()
        return self._connection

    def __getstate__(self):
        return dict(self.__dict__, _connection=None, _pid=None)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if not 0 <= index < self.size:
            raise IndexError(index)
        encoded = self.db.execute('SELECT i.row_json FROM samples s JOIN images i '
                                 'ON i.ordinal=s.ordinal WHERE s.sample=?', (int(index),)).fetchone()
        if encoded is None:
            raise ValueError('Accepted sample index is incomplete')
        return json.loads(encoded[0])

    def geometry(self, index):
        if not 0 <= index < self.size:
            raise IndexError(index)
        value = self.db.execute('SELECT i.pack,i.entry_json,i.disposition FROM samples s JOIN images i '
                               'ON i.ordinal=s.ordinal WHERE s.sample=?', (int(index),)).fetchone()
        if value is None or value[2] != 'selected' or not value[0] or not value[1]:
            raise ValueError('Only selected, accepted geometry may enter the loader')
        return value[0], json.loads(value[1])

    def sampling_records(self):
        # Avoid decoding millions of full image/calibration JSON rows at startup.
        for sample, dataset, group in self.db.execute('SELECT s.sample,i.dataset,i.scene_group '
                'FROM samples s JOIN images i ON s.ordinal=i.ordinal ORDER BY s.sample'):
            yield sample, dict(dataset=dataset, group=group)

    def check_heldout(self, records):
        # Includes excluded corpus images: exclusion cannot launder heldout data.
        for row in records:
            for field, value in (('image', row['image']), ('sha256', row.get('sha256')),
                                 ('scene_group', row.get('group'))):
                if value is not None and self.db.execute(
                        f'SELECT 1 FROM images WHERE {field}=? LIMIT 1', (value,)).fetchone():
                    raise ValueError('Train/validation leakage: accepted corpus overlap')

    def contains_selected(self, row):
        value = self.db.execute("SELECT sha256 FROM images WHERE image=? AND disposition='selected'",
                                (row['image'],)).fetchone()
        return value is not None and row.get('sha256', value[0]) == value[0]

    def digest(self):
        # Sealed index/evidence/config + strict per-read RGB and payload hashes.
        return hashlib.sha256(canonical_json(self.gate)).hexdigest()


class CorpusImages(Dataset):
    def __init__(self, records, augment=True):
        if not isinstance(records, CorpusRecords):
            raise TypeError('CorpusImages requires acceptance-gated CorpusRecords')
        self.records, self.augment = records, augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, key):
        index, seed, height, width = key
        pack, entry = self.records.geometry(index)
        rgb, target, _ = load_candidate(pack, entry, (height, width), seed, self.augment)
        image = torch.from_numpy(np.array(rgb, dtype=np.float32, copy=True)).permute(2, 0, 1) / 255
        return dict(image=image, geometry_target=target)


def collate_corpus(samples):
    """Stack only RGB; never pad/resize/replicate different native GT grids."""
    if not samples or any(set(s) != {'image', 'geometry_target'} for s in samples):
        raise ValueError('Expected RGB plus one provenance-aware target per sample')
    if len({tuple(s['image'].shape) for s in samples}) != 1:
        raise ValueError('RGB images must share the selected training resolution')
    return dict(image=torch.stack([s['image'] for s in samples]),
                geometry_targets=[s['geometry_target'] for s in samples])
