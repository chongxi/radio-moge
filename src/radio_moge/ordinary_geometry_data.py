"""Read historical accepted target references without resampling original RGB.

The historical acceptance binds DATA only here. It cannot admit this new camera,
batching or training implementation. The ordinary trainer requires its own gate.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import threading

import torch
from torch.utils.data import Dataset

from .corpus_admission import checked_json, checked_path, readonly_db
from .data import TrainingImages, image_rotation
from .geometry_cache import HEX64
from .rgb_geometry_batch import GeometrySlot, RGBGeometryBatch
from .projection_sidecar import STAGE as PROJECTION_STAGE, checked_derivation


def image_identity(row):
    if (not isinstance(row.get('sha256'), str) or not HEX64.fullmatch(row['sha256'])
            or row.get('split') != 'train'):
        raise ValueError('Geometry lookup requires an exact training RGB identity')
    return (str(Path(row['image']).resolve()), row['sha256'], row['dataset'], row['group'],
            image_rotation(row))


def fingerprint(path):
    # Some deployment filesystems expose timestamps too coarse for back-to-back
    # commits. SQLite's header also contains its commit change counter. A sealed
    # index must use the ordinary rollback journal, never an unbound WAL sidecar.
    with Path(path).open('rb') as file:
        info = os.fstat(file.fileno())
        header = file.read(100)
    if len(header) != 100 or header[:16] != b'SQLite format 3\x00' or header[18:20] != b'\x01\x01':
        raise ValueError('Expected a closed non-WAL SQLite geometry index')
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, header


class GeometrySidecar:
    """Sparse lookup by original path + SHA; never a dataset or sample index.

    Connections are process AND thread local because preparation can run in a
    background thread when DataLoader has zero workers. Pickling/forking cannot
    transfer a live connection. Payload verification remains mandatory when the
    returned geometry is actually read.
    """
    training_admission = False

    def __init__(self, acceptance_ref):
        self.acceptance_ref = deepcopy(acceptance_ref)
        gate = checked_json(self.acceptance_ref)
        filtered = gate.get('stage') == PROJECTION_STAGE
        if filtered:
            expected_counts = checked_derivation(gate)
            total_images = gate['total_images']
        elif (gate.get('schema_version') != 1 or gate.get('stage') != 'accepted_for_training'
                or gate.get('training_admission') is not True):
            raise ValueError('Expected historical accepted geometry evidence')
        else:
            accounting = checked_json(gate['accounting_report'])
            if (accounting.get('stage') != 'accounting_verified_not_accepted'
                    or accounting.get('training_admission') is not False
                    or accounting.get('unresolved') != 0
                    or any(gate[k] != accounting[k] for k in ('index', 'manifest_sha256', 'selected_images'))):
                raise ValueError('Historical acceptance and accounting identity differ')
            total_images = accounting['total_images']
            expected_counts = dict(selected=gate['selected_images'],
                                   excluded=total_images - gate['selected_images'])
        self.index_ref = deepcopy(gate['index'])
        self.path = self.index_ref['path']
        self._fingerprint = fingerprint(self.path)
        checked_path(self.index_ref)
        self._assert_unchanged()
        self._local = threading.local()
        with readonly_db(self.path) as db:
            counts = dict(db.execute('SELECT disposition,COUNT(*) FROM images GROUP BY disposition'))
            samples = db.execute('SELECT COUNT(*) FROM samples').fetchone()[0]
            bad = db.execute("SELECT COUNT(*) FROM samples s LEFT JOIN images i ON i.ordinal=s.ordinal "
                             "WHERE i.ordinal IS NULL OR i.disposition!='selected'").fetchone()[0]
            if ({k: v for k, v in expected_counts.items() if v} != counts or bad
                    or counts.get('selected', 0) != gate['selected_images']
                    or samples != gate['selected_images'] or sum(counts.values()) != total_images):
                raise ValueError('Historical geometry index counts or dispositions differ')
            if filtered and db.execute("SELECT COUNT(*) FROM images WHERE "
                    "(disposition='selected' AND dataset='coco2017') OR "
                    "(disposition='projection_hold' AND dataset!='coco2017')").fetchone()[0]:
                raise ValueError('Projection sidecar retained a selected held source or held another source')
        if fingerprint(self.path) != self._fingerprint:
            raise ValueError('Historical geometry index changed while opening')

    @property
    def db(self):
        self._assert_unchanged()
        if getattr(self._local, 'pid', None) != os.getpid():
            old = getattr(self._local, 'connection', None)
            if old is not None:
                old.close()
            self._local.connection = readonly_db(self.path)
            self._local.pid = os.getpid()
        return self._local.connection

    def _assert_unchanged(self):
        if fingerprint(self.path) != self._fingerprint:
            raise ValueError('Historical geometry index changed after verification')

    def __getstate__(self):
        return dict(self.__dict__, _local=None)

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._local = threading.local()

    def close(self):
        if self._local is not None:
            connection = getattr(self._local, 'connection', None)
            if connection is not None:
                connection.close()
                del self._local.connection
                del self._local.pid

    def lookup(self, row):
        identity = image_identity(row)
        value = self.db.execute('SELECT ordinal,sha256,row_json,disposition,pack,entry_json '
                                'FROM images WHERE image=?', (identity[0],)).fetchone()
        self._assert_unchanged()
        if value is None:
            return None  # Outside the historical accepted scope; RGB still participates.
        ordinal, sha, encoded, disposition, pack, entry_json = value
        original = json.loads(encoded)
        if sha != identity[1] or image_identity(original) != identity:
            raise ValueError('Known RGB path has changed hash, source, group or orientation')
        if disposition == 'excluded':
            return None
        if disposition != 'selected' or not pack or not entry_json:
            raise ValueError('Unresolved or incomplete geometry must not enter a sidecar')
        entry = json.loads(entry_json)
        if image_identity(entry['image']) != identity:
            raise ValueError('Selected geometry receipt has a different RGB identity')
        checksum = self.db.execute('SELECT sha256 FROM packs WHERE path=?', (pack,)).fetchone()
        self._assert_unchanged()
        if checksum is None or not HEX64.fullmatch(checksum[0]):
            raise ValueError('Selected geometry has no bound pack identity')
        return dict(corpus_ordinal=ordinal, pack=dict(path=pack, sha256=checksum[0]),
                    entry=entry, source_index=deepcopy(self.index_ref))


class GeometryAttachedImages(Dataset):
    """Delegate ALL RGB sampling and pixels to the original TrainingImages."""
    def __init__(self, images, sidecar):
        if (type(images) is not TrainingImages or not images.compact_transport
                or not getattr(images.records, 'verify_on_read', False)
                or not isinstance(sidecar, GeometrySidecar)):
            raise TypeError('Expected original compact hash-verified RGB and bound geometry sidecar')
        self.images, self.sidecar = images, sidecar

    @property
    def records(self):
        return self.images.records

    @property
    def compact_transport(self):
        return self.images.compact_transport

    def __len__(self):
        return len(self.images)

    def __getitem__(self, key):
        row = deepcopy(self.images.records[key[0]])
        geometry = self.sidecar.lookup(row)
        image = self.images[key]
        return image, GeometrySlot(tuple(key), row, geometry, self.images.augment)


def collate_geometry_attached(samples):
    if not samples:
        raise ValueError('Empty original RGB batch')
    return RGBGeometryBatch(torch.stack([image for image, _ in samples]),
                            tuple(slot for _, slot in samples))
