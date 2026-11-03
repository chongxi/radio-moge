"""Explicit corpus objective controls and hashes of actual paired training inputs."""
import hashlib
from pathlib import Path

import numpy as np
import torch

from .geometry_cache import canonical_json, sha_file


def geometry_multiplier(config, *, corpus_preflight=False):
    control = config.get('corpus_loss_control')
    if control is None:
        return 1
    if (not isinstance(control, dict) or set(control) != {'geometry_multiplier', 'record_inputs'}
            or type(control['geometry_multiplier']) is not int
            or control['geometry_multiplier'] not in (0, 1) or control['record_inputs'] is not True
            or not (config['data'].get('geometry_corpus') or corpus_preflight)
            or config.get('geometry_supervision') != dict(target='offline_corpus',
                camera='provenance_aware', objective='moge2_core')
            or any(config['loss'].get(k, 0) <= 0 for k in
                   ('spatial', 'summary', 'cosine', 'summary_cosine'))):
        raise ValueError('Corpus controls require an explicit binary geometry multiplier, input audit and feature loss')
    return control['geometry_multiplier']


def input_fingerprint(batch, replay=None):
    """Hash actual CPU RGB/native targets without changing data or random state."""
    identities = [{k: target[k] for k in ('dataset', 'label_kind', 'supervision',
                   'cache_key', 'image_sha256', 'output_shape', 'native_shape')}
                  for target in batch['geometry_targets']]
    digest = hashlib.sha256(canonical_json(identities))
    arrays = [('image', batch['image'])]
    if replay is not None:
        arrays.append(('replay_image', replay))
    for i, target in enumerate(batch['geometry_targets']):
        arrays.extend((f'target{i}/{key}', target[key]) for key in
                      ('points', 'query_uv', 'valid', 'intrinsics', 'native_index'))
    for name, value in arrays:
        if torch.is_tensor(value):
            if value.device.type != 'cpu' or value.requires_grad:
                raise ValueError('Input audit requires detached original CPU tensors')
            value = value.numpy()
        value = np.ascontiguousarray(value)
        digest.update(canonical_json(dict(name=name, dtype=str(value.dtype), shape=list(value.shape))))
        digest.update(memoryview(value).cast('B'))
    return dict(view_sha256=digest.hexdigest(), identities=identities, replay_slot=replay is not None)


class InputAudit:
    """Fresh experiment only; a partial audit never authorizes a resume."""
    def __init__(self, path):
        self.path = Path(path)
        self.file = self.path.open('x')
        self.digest, self.count, self.replay_count = hashlib.sha256(), 0, 0

    def add(self, cursor, batch, replay=None):
        if cursor != self.count or len(batch['image']) != 1 or (replay is not None) != bool(cursor % 2):
            raise ValueError('Controlled comparison requires contiguous one-image paired/replay slots')
        row = dict(cursor=cursor, **input_fingerprint(batch, replay))
        encoded = canonical_json(row) + b'\n'
        self.digest.update(encoded)
        self.file.write(encoded.decode())
        self.file.flush()
        self.count += 1
        self.replay_count += replay is not None

    def report(self):
        self.file.flush()
        if sha_file(self.path) != self.digest.hexdigest():
            raise ValueError('Actual input audit changed')
        return dict(microbatches=self.count, replay_microbatches=self.replay_count,
            records=dict(path=str(self.path.resolve()), sha256=self.digest.hexdigest()),
            algorithm='SHA256 canonical cursor/metadata and exact CPU RGB/native arrays/replay pixels')

    def close(self):
        self.file.close()
