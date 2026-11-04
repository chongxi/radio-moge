"""Check the immutable reference loader's sampler and full plan without RGB I/O.

This is an independent construction check, not an observation inside the trainer.
The trainer additionally binds each consumed key/pixel/group before committing it.
"""
from copy import deepcopy

from radio_moge.data import DeterministicBatches
from radio_moge.prefetch import FloatLoader


def verify(config, performance, images, planned, make_loader, start_step=232288, updates=250):
    if 'mixture_weights' in config['data']:
        raise ValueError('Reference must have no mixture weights')
    if (config['train']['steps'] - start_step != updates
            or performance['effective_batch'] != 32 or config['train']['batch_size'] != 4):
        raise ValueError('Independent update/effective-batch budget differs')
    if not performance['compact_transport'] or not performance['authorized']:
        raise ValueError('Authorized original compact performance required')
    accumulation = performance['effective_batch'] // config['train']['batch_size']
    count = updates * accumulation
    loader_config = deepcopy(config)
    # Exactly the effective override applied in train.py before make_loader.
    loader_config['data']['workers'] = performance['workers']
    loader_config['data']['compact_transport'] = performance['compact_transport']

    class OriginalRowCount:
        def __len__(self):
            return images

    loader = make_loader(OriginalRowCount(), loader_config, 0, count)
    if not isinstance(loader, FloatLoader):
        raise ValueError('Expected original compact loader')
    sampler = loader.loader.batch_sampler
    if type(sampler) is not DeterministicBatches:
        raise ValueError('Actual loader construction did not select DeterministicBatches')
    actual = [list(key) for batch in sampler for key in batch]
    expected = [[r['index'], r['augmentation_seed'], r['height'], r['width']] for r in planned]
    if actual != expected or len(actual) != updates * performance['effective_batch']:
        raise ValueError('Full planned input keys differ')
    return dict(sampler_class=type(sampler).__name__, mixture_weights_absent=True,
        full_planned_keys_equal=True, planned_presentations=len(actual),
        logical_microbatches=count, updates_checked_separately=updates,
        coverage_epochs_used_to_infer_sampler=False,
        effective_loader_overrides=dict(workers=performance['workers'], compact_transport=True),
        actual_training_loader_type_observed_in_process=False,
        actual_consumed_key_checks_required=True)
