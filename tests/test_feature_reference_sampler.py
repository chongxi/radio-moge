from copy import deepcopy

import pytest

from radio_moge.data import DeterministicBatches
from radio_moge.train import make_loader
from verify_feature_reference_sampler import verify


def settings():
    config = dict(data=dict(workers=0, augment=True, buckets=[[32, 48], [48, 32]]),
                  train=dict(batch_size=4, seed=43, steps=232538))
    performance = dict(authorized=True, effective_batch=32, workers=4, compact_transport=True)
    planned = [dict(index=i, augmentation_seed=s, height=h, width=w)
               for batch in DeterministicBatches(97, 4, config['data']['buckets'], 43, 0, 2000)
               for i, s, h, w in batch]
    return config, performance, planned


def test_full_plan_and_effective_loader_override_without_images_or_workers():
    config, performance, planned = settings()
    result = verify(config, performance, 97, planned, make_loader)
    assert result['planned_presentations'] == 8000
    assert result['sampler_class'] == 'DeterministicBatches'
    assert not result['coverage_epochs_used_to_infer_sampler']
    assert config['data']['workers'] == 0 and 'compact_transport' not in config['data']


@pytest.mark.parametrize('change', ['weights', 'budget', 'key'])
def test_wrong_sampler_budget_or_any_planned_key_rejected(change):
    config, performance, planned = settings()
    if change == 'weights':
        config['data']['mixture_weights'] = {}
    elif change == 'budget':
        config['train']['steps'] += 1
    else:
        planned = deepcopy(planned)
        planned[-1]['augmentation_seed'] += 1
    with pytest.raises(ValueError):
        verify(config, performance, 97, planned, make_loader)
