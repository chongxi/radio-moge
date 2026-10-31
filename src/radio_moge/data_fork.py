"""Explicit data changes retain model/Adam state but start a new sample stream."""
from copy import deepcopy
import hashlib
import math
import random


def check_data_fork(config, state, digests):
    from .train import check_lr_fork
    from .lr_adjustment import reviewed_source_lr
    normalized = deepcopy(config)
    previous = state['config']
    for key in ['train_manifest', 'mixture_weights', 'coverage_epochs']:
        normalized['data'].pop(key, None)
        if key in previous['data']:
            normalized['data'][key] = previous['data'][key]
    old_views = previous.get('evaluation', {}).get('dataset_variants', {})
    new_views = config.get('evaluation', {}).get('dataset_variants', {})
    if any(new_views.get(k) != v for k, v in old_views.items()):
        raise ValueError('Data fork must preserve previously established validation views')
    normalized['evaluation'].pop('dataset_variants', None)
    if old_views:
        normalized['evaluation']['dataset_variants'] = old_views
    for key, value in state['data_digests'].items():
        if key != 'train' and digests.get(key) != value:
            raise ValueError('Data fork must preserve all existing validation/probe identities')
    check_lr_fork(normalized, state, state['data_digests'])
    if config['train']['lr'] > reviewed_source_lr(state):
        raise ValueError('Data fork cannot increase the source learning rate')
    coverage = config['data'].get('coverage_epochs')
    if coverage is not None and (coverage != 1 or config['data'].get('mixture_weights')):
        raise ValueError('One coverage epoch requires permutation sampling without mixture weights')


class CoverageTracker:
    """Account only batches consumed by successful updates, never prefetch.

    Uses the same seed/epoch permutation as DeterministicBatches. Records actual
    sample-index coverage rather than inferring it from a number of random draws.
    """
    def __init__(self, size, batch, seed, cursor=0):
        self.size, self.batch, self.seed = size, batch, seed
        self.cursor = 0
        self.seen = bytearray(size)
        self.unique = 0
        self.digest = hashlib.sha256()
        self.epoch = None
        self.order = None
        self.advance(cursor)

    def advance(self, cursor):
        if cursor < self.cursor:
            raise ValueError('Coverage cursor moved backwards')
        per_epoch = math.ceil(self.size / self.batch)
        for absolute in range(self.cursor, cursor):
            epoch, batch = divmod(absolute, per_epoch)
            if self.epoch != epoch:
                self.order = list(range(self.size))
                random.Random(self.seed + epoch).shuffle(self.order)
                self.epoch = epoch
            for j in range(self.batch):
                index = self.order[(batch * self.batch + j) % self.size]
                if not self.seen[index]:
                    self.seen[index] = 1
                    self.unique += 1
                self.digest.update(index.to_bytes(8, 'little'))
        self.cursor = cursor

    def report(self):
        count = self.cursor * self.batch
        return dict(sampling='DeterministicBatches permutation', records=self.size,
                    consumed_microbatches=self.cursor, consumed_presentations=count,
                    unique_indices=self.unique, missing_indices=self.size - self.unique,
                    repeated_presentations=count - self.unique,
                    consumed_index_sha256=self.digest.hexdigest(),
                    complete=self.unique == self.size, prefetch_excluded=True)
