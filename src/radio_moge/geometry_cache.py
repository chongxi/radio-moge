"""Immutable, checksummed offline depth packs; publication is NOT acceptance.

Only completed tar+receipt pairs are visible. A failed attempt is never silently
resumed, overwritten, or admitted to training. No torch or network dependency.
"""
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile

import numpy as np


LABEL_KINDS = {'measured_sensor', 'laser_mesh_derived', 'synthetic_renderer', 'teacher_pseudo'}
CAMERA_KINDS = {'calibrated_sensor', 'synthetic_camera', 'teacher_estimated'}
HEX64 = re.compile(r'^[0-9a-f]{64}$')


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha_file(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def publish_json(path, value):
    """A new atomic file, refusing to replace an already published receipt."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as file:
            file.write(canonical_json(value) + b'\n')
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)  # Only this function's exact temporary file.


def encode_geometry(depth, valid, intrinsics, provenance):
    depth, valid, intrinsics = np.asarray(depth), np.asarray(valid), np.asarray(intrinsics)
    if (depth.ndim != 2 or valid.shape != depth.shape or valid.dtype != np.bool_
            or not valid.any() or not np.isfinite(depth[valid]).all() or not (depth[valid] > 0).all()
            or intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()
            or not np.allclose(intrinsics[2], [0, 0, 1])
            or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0):
        raise ValueError('Invalid depth/support/pinhole camera; no implicit validity repair')
    required = {'image_sha256', 'dataset', 'source_id', 'label_kind', 'camera_kind',
                'metric_calibrated', 'conversion', 'policy_sha256', 'source_proofs'}
    if required - provenance.keys():
        raise ValueError('Incomplete geometry provenance')
    if set(provenance) & {'schema_version', 'depth_convention', 'units',
                          'camera_pixel_convention', 'shape', 'valid_pixels'}:
        raise ValueError('Provenance cannot override the geometry format contract')
    if (provenance['label_kind'] not in LABEL_KINDS or provenance['camera_kind'] not in CAMERA_KINDS
            or type(provenance['metric_calibrated']) is not bool
            or not HEX64.fullmatch(provenance['image_sha256'])
            or not HEX64.fullmatch(provenance['policy_sha256'])
            or not provenance['source_proofs']):
        raise ValueError('Invalid geometry provenance')
    if provenance['label_kind'] == 'teacher_pseudo':
        if provenance['metric_calibrated'] or not provenance.get('teacher_revision'):
            raise ValueError('Teacher pseudo depth cannot claim calibrated metric ground truth')
    elif provenance['camera_kind'] == 'teacher_estimated':
        raise ValueError('Native labels cannot silently use an estimated teacher camera')
    for proof in provenance['source_proofs']:
        if not proof.get('source') or not HEX64.fullmatch(proof.get('sha256', '')):
            raise ValueError('Each native/teacher source needs immutable byte provenance')
    # Preserve EXACT support; invalidate nothing and fill no holes here.
    clean = np.zeros(depth.shape, np.float32)
    clean[valid] = depth[valid]
    if not np.isfinite(clean).all() or not (clean[valid] > 0).all():
        raise ValueError('Depth cannot be represented as float32')
    metadata = dict(schema_version=1, depth_convention='camera_z', units='meters',
                    camera_pixel_convention='normalized_half_center',
                    shape=list(depth.shape), valid_pixels=int(valid.sum()),
                    **provenance)
    key = hashlib.sha256(canonical_json(metadata)).hexdigest()
    buffer = io.BytesIO()
    np.savez_compressed(buffer, depth=clean, valid=valid, intrinsics=intrinsics.astype(np.float64),
                        metadata_json=np.frombuffer(canonical_json(metadata), np.uint8))
    payload = buffer.getvalue()
    if len(payload) > 64 * 1024**2:
        raise ValueError('Geometry member exceeds bounded-read limit')
    return key, payload, metadata


class GeometryPackWriter:
    def __init__(self, directory, policy, max_entries=256):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / '.writer.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if list(self.directory.glob('pack-*')) or (self.directory / 'policy.json').exists():
                raise FileExistsError('Cache attempt exists; inspect its receipts before explicit recovery')
            if not 1 <= max_entries <= 1024:
                raise ValueError('Pack size outside bound')
            self.policy_hash = hashlib.sha256(canonical_json(policy)).hexdigest()
            publish_json(self.directory / 'policy.json', policy)
        except BaseException:
            self.lock.close()
            raise
        self.max_entries, self.number = max_entries, 0
        self.entries, self.archive, self.stream, self.temporary = [], None, None, None
        self.packs = []

    def add(self, image_record, depth, valid, intrinsics, provenance):
        if (image_record.get('split') != 'train'
                or image_record.get('sha256') != provenance.get('image_sha256')
                or image_record.get('dataset') != provenance.get('dataset')
                or provenance.get('policy_sha256') != self.policy_hash):
            raise ValueError('Geometry does not match this training image/policy')
        key, payload, metadata = encode_geometry(depth, valid, intrinsics, provenance)
        if any(entry['key'] == key for entry in self.entries):
            raise ValueError('Repeated geometry key in a pack')
        if self.archive is None:
            self.temporary = self.directory / f'pack-{self.number:06d}.tar.incoming'
            self.stream = self.temporary.open('xb')
            self.archive = tarfile.open(fileobj=self.stream, mode='w', format=tarfile.USTAR_FORMAT)
        info = tarfile.TarInfo(key + '.npz')
        info.size, info.mtime, info.mode = len(payload), 0, 0o644
        offset = self.stream.tell() + 512  # Fixed USTAR header; the name fits 100 bytes.
        self.archive.addfile(info, io.BytesIO(payload))
        self.entries.append(dict(key=key, member=info.name, offset=offset, size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(), image=image_record, geometry_metadata=metadata,
            valid_fraction=float(valid.mean()), depth_quantiles_m=np.quantile(
                depth[valid], [0, .01, .5, .99, 1]).tolist()))
        if len(self.entries) >= self.max_entries:
            self.flush()

    def flush(self):
        if self.archive is None:
            return
        self.archive.close()
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        final = self.temporary.with_suffix('')
        if final.exists():
            raise FileExistsError(final)
        checksum = sha_file(self.temporary)
        os.replace(self.temporary, final)
        receipt = dict(schema_version=1, stage='cache_written_not_accepted', training_admission=False,
                       tar=final.name, sha256=checksum, bytes=final.stat().st_size,
                       policy_sha256=self.policy_hash, entries=self.entries)
        receipt_path = self.directory / f'pack-{self.number:06d}.json'
        publish_json(receipt_path, receipt)
        self.packs.append(dict(receipt=receipt_path.name, sha256=sha_file(receipt_path),
                               count=len(self.entries)))
        self.entries, self.archive, self.stream, self.temporary = [], None, None, None
        self.number += 1

    def close(self):
        self.flush()
        publish_json(self.directory / 'packs.json', dict(stage='cache_written_not_accepted',
            training_admission=False, policy_sha256=self.policy_hash, packs=self.packs,
            count=sum(pack['count'] for pack in self.packs)))
        self.lock.close()


def read_geometry_member(pack_path, entry):
    """Bounded random read, SHA verification, and safe non-pickle array decoding."""
    if (entry['offset'] < 512 or entry['offset'] % 512 or not 0 < entry['size'] <= 64 * 1024**2
            or not HEX64.fullmatch(entry['key']) or entry['member'] != entry['key'] + '.npz'):
        raise ValueError('Invalid packed geometry reference')
    with Path(pack_path).open('rb') as file:
        file.seek(entry['offset'])
        data = file.read(entry['size'])
    if len(data) != entry['size'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
        raise ValueError('Packed geometry hash mismatch')
    with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
        depth, valid, k = arrays['depth'], arrays['valid'], arrays['intrinsics']
        metadata = json.loads(arrays['metadata_json'].tobytes())
    if (hashlib.sha256(canonical_json(metadata)).hexdigest() != entry['key']
            or metadata != entry['geometry_metadata'] or list(depth.shape) != metadata['shape']
            or depth.dtype != np.float32 or valid.dtype != np.bool_ or valid.shape != depth.shape
            or int(valid.sum()) != metadata['valid_pixels']
            or not np.isfinite(depth).all() or not (depth[valid] > 0).all()
            or not np.all(depth[~valid] == 0) or k.shape != (3, 3)):
        raise ValueError('Packed geometry content violates its receipt')
    return depth, valid, k, metadata
