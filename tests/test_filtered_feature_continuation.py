from copy import deepcopy

import pytest
import torch

from run_filtered_feature_continuation import initial_checks, validate_config


def case():
    config = dict(output_dir='/old', tracking={}, models={},
        data=dict(buckets=[[480, 640], [640, 480], [400, 640]]),
        loss=dict(spatial=1., summary=1., cosine=1., summary_cosine=1.),
        train=dict(steps=232538, max_seconds=2700, lr=1e-5, lr_schedule='constant'),
        evaluation={})
    parent = dict(config=config, step=232538, cursor=2000, data_digests={'train':'frozen'},
                  performance=dict(effective_batch=32, workers=4))
    for name in ('adapter', 'statistics', 'optimizer', 'rng', 'adapter_config', 'model_config', 'revisions'):
        parent[name] = {'test': torch.tensor([1., 2.])}
    new = deepcopy(config)
    new.update(output_dir='/new', tracking={'name':'continuation'})
    new['train'].update(steps=233538, max_seconds=3600)
    return new, parent, dict(authorized=True, additional_updates=1000)


def test_exact_state_and_bounded_same_pipeline():
    config, parent, plan = case()
    validate_config(config, parent, plan)
    initial_checks(deepcopy(parent), parent)


@pytest.mark.parametrize('change', ['loss', 'manifest', 'lr', 'budget', 'batch'])
def test_reject_changed_training_contract(change):
    config, parent, plan = case()
    if change == 'loss':
        config['loss']['cosine'] = .1
    elif change == 'manifest':
        config['data']['train_manifest'] = '/unfiltered'
    elif change == 'lr':
        config['train']['lr'] = 1e-4
    elif change == 'budget':
        config['train']['steps'] += 1
    else:
        parent['performance']['effective_batch'] = 16
    with pytest.raises(ValueError):
        validate_config(config, parent, plan)


@pytest.mark.parametrize('section', ['adapter', 'statistics', 'optimizer', 'rng', 'cursor'])
def test_initial_state_verifier_rejects_changes(section):
    _, parent, _ = case()
    new = deepcopy(parent)
    if section == 'cursor':
        new[section] = 0
    else:
        new[section]['test'][0] += 1
    with pytest.raises(ValueError):
        initial_checks(new, parent)
