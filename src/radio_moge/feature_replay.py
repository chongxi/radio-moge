"""Replace half of GT-stage feature slots with train-only, RGB-only replay.

GT geometry still consumes every original microbatch. Feature supervision keeps
the same number of images and weights; it is replaced, never added twice.
"""
import math
import hashlib
import json
from pathlib import Path
import re

import torch
from torch.utils.data import DataLoader

from .data import DeterministicBatches, MixtureBatches, TrainingImages, check_split, read_manifest
from .tracking import preserve_rng


def validate_feature_replay(config):
    replay = config.get('feature_replay')
    if replay is None:
        return
    required = {'manifest', 'fraction', 'seed'}
    if (not isinstance(replay, dict) or not required <= set(replay)
            or set(replay) - required - {'mixture_weights', 'manifest_sha256'}
            or replay['fraction'] != .5 or not isinstance(replay['manifest'], str)
            or not replay['manifest'] or type(replay['seed']) is not int or replay['seed'] < 0):
        raise ValueError('Feature replay requires manifest, fraction=0.5 and a nonnegative integer seed')
    if 'manifest_sha256' in replay and not re.fullmatch('[0-9a-f]{64}', str(replay['manifest_sha256'])):
        raise ValueError('Replay manifest SHA256 must be exact')
    if 'mixture_weights' in replay:
        weights = replay['mixture_weights']
        if (not isinstance(weights, dict) or not weights
                or any(not isinstance(k, str) or not k or type(v) not in (int, float)
                       or not math.isfinite(v) or v <= 0 for k, v in weights.items())
                or not math.isfinite(sum(weights.values()))):
            raise ValueError('Feature replay mixture weights must be finite positive source weights')
    if not config.get('geometry_supervision'):
        raise ValueError('Feature replay requires explicit GT geometry supervision')
    if config['train']['accumulation'] % 2 or len(config['data']['buckets']) != 1:
        raise ValueError('Feature replay requires even accumulation and one fixed resolution bucket')


class VerifiedReplayRecords(list):
    """A frozen RGB index with payload identity checked on every actual read."""
    verify_on_read = True

    def __init__(self, rows, checksum):
        super().__init__(rows)
        self.checksum = checksum

    def digest(self):
        return self.checksum


def load_verified_replay(path, checksum):
    rows, digest, seen = [], hashlib.sha256(), set()
    with Path(path).open('rb') as file:
        for line in file:
            digest.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            if (not Path(row['image']).is_absolute() or row['image'] in seen
                    or not re.fullmatch('[0-9a-f]{64}', str(row.get('sha256', '')))):
                raise ValueError('Verified replay requires unique absolute RGB paths and hashes')
            seen.add(row['image'])
            rows.append(row)
    if not rows or digest.hexdigest() != checksum:
        raise ValueError('Verified replay manifest changed')
    return VerifiedReplayRecords(rows, checksum)


def load_replay_records(config, heldout_sets):
    if config.get('feature_replay') is None:
        return []
    settings = config['feature_replay']
    records = (load_verified_replay(settings['manifest'], settings['manifest_sha256'])
               if 'manifest_sha256' in settings else read_manifest(settings['manifest']))
    if any(row.get('split') != 'train' or not row.get('dataset') or not row.get('group')
           or 'depth' in row or 'geometry' in row for row in records):
        raise ValueError('Replay must contain train-only RGB records with source/scene and no GT labels')
    for heldout in heldout_sets:
        check_split(records, heldout)
    weights = config['feature_replay'].get('mixture_weights')
    if weights is not None and set(weights) != {r['dataset'] for r in records}:
        raise ValueError('Feature replay mixture weights must name every dataset exactly once')
    return records


def is_replay_batch(cursor):
    return cursor % 2 == 1


def make_replay_loader(records, config, cursor, microbatches):
    seed = config['feature_replay']['seed']
    count = (cursor + microbatches) // 2 - cursor // 2
    weights = config['feature_replay'].get('mixture_weights')
    if weights is None:
        sampler = DeterministicBatches(len(records), config['train']['batch_size'],
                                       config['data']['buckets'], seed, cursor // 2, count)
    else:
        # The replay cursor counts only replay slots. Source/scene probabilities
        # are independent of frame counts and do not advance the geometry RNG.
        sampler = MixtureBatches(records, weights, config['train']['batch_size'],
                                 config['data']['buckets'], seed, cursor // 2, count)
    return DataLoader(TrainingImages(records, config['data']['augment']),
                      batch_sampler=sampler, num_workers=0, pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(seed))


def replay_feature_losses(system, adapter, statistics, image, indices, amp, seed):
    # Keep the GT/model random stream unchanged, including with stochastic adapters.
    # The replay branch never calls the decoder or consumes depth/pseudo-depth labels.
    with preserve_rng():
        torch.manual_seed(seed)
        with amp(), torch.no_grad():
            features, target = system.encode(image, indices)
        with amp():
            prediction = adapter(features)
        return statistics.loss(prediction, target)
