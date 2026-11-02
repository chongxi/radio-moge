"""Replace half of GT-stage feature slots with train-only, RGB-only replay.

GT geometry still consumes every original microbatch. Feature supervision keeps
the same number of images and weights; it is replaced, never added twice.
"""
import torch
from torch.utils.data import DataLoader

from .data import DeterministicBatches, TrainingImages, check_split, read_manifest
from .tracking import preserve_rng


def validate_feature_replay(config):
    replay = config.get('feature_replay')
    if replay is None:
        return
    if (not isinstance(replay, dict) or set(replay) != {'manifest', 'fraction', 'seed'}
            or replay['fraction'] != .5 or not isinstance(replay['manifest'], str)
            or not replay['manifest'] or type(replay['seed']) is not int or replay['seed'] < 0):
        raise ValueError('Feature replay requires manifest, fraction=0.5 and a nonnegative integer seed')
    if not config.get('geometry_supervision'):
        raise ValueError('Feature replay requires explicit GT geometry supervision')
    if config['train']['accumulation'] % 2 or len(config['data']['buckets']) != 1:
        raise ValueError('Feature replay requires even accumulation and one fixed resolution bucket')


def load_replay_records(config, heldout_sets):
    if config.get('feature_replay') is None:
        return []
    records = read_manifest(config['feature_replay']['manifest'])
    if any(row.get('split') != 'train' or not row.get('dataset') or not row.get('group')
           or 'depth' in row or 'geometry' in row for row in records):
        raise ValueError('Replay must contain train-only RGB records with source/scene and no GT labels')
    for heldout in heldout_sets:
        check_split(records, heldout)
    return records


def is_replay_batch(cursor):
    return cursor % 2 == 1


def make_replay_loader(records, config, cursor, microbatches):
    seed = config['feature_replay']['seed']
    count = (cursor + microbatches) // 2 - cursor // 2
    sampler = DeterministicBatches(len(records), config['train']['batch_size'],
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
