from copy import deepcopy

import pytest
import torch

from prepare_corpus_controls import configurations, FEATURE_KEYS
from run_corpus_controls import verify_initial_state
from radio_moge.official_gt import OFFICIAL_GT_WEIGHTS
from test_corpus_model_probe import fixture


def specification(tmp_path):
    _, _, config, _, _ = fixture(tmp_path)
    config['data'].update(geometry_corpus={'path': '/fixture/gate', 'sha256': 'a'*64},
                          mixture_weights={'hypersim': 1.})
    config['tracking'] = dict(enabled=False)
    train = dict(config['train'], steps=250, seed=43, batch_size=1, accumulation=16,
                 lr=1e-5, max_seconds=2700)
    plan = dict(stage='controlled_experiment_plan_not_launched',
        common_initialization=dict(step=232288), optimizer_step_origin=0, statistics_refit=False,
        fresh_optimizer_all_arms=True, data_config=deepcopy(config['data']),
        feature_replay=deepcopy(config['feature_replay']), source_mixture=config['data']['mixture_weights'],
        base_geometry_loss=OFFICIAL_GT_WEIGHTS, updates_per_arm=250, train=train,
        variants={f'f{f}_g{g}':dict(feature_coefficients={k:f for k in FEATURE_KEYS}, geometry_multiplier=g)
                  for f in (.1, 1.) for g in (0, 1)})
    return plan, config


def test_four_configs_differ_only_in_predeclared_factors_and_run_metadata(tmp_path):
    plan, base = specification(tmp_path)
    values = configurations(plan, base, tmp_path / 'controls')
    assert len(values) == 4
    for config in values.values():
        assert 'geometry_corpus' not in config['data']
        assert config['feature_replay'] == base['feature_replay']
        assert config['models'] == base['models'] and config['train'] == plan['train']
        multiplier = config['corpus_loss_control']['geometry_multiplier']
        assert all(config['loss'][k] == multiplier * w for k,w in OFFICIAL_GT_WEIGHTS.items())
    assert 'geometry_corpus' in base['data']


def test_changed_sampling_budget_or_incomplete_factorial_is_rejected(tmp_path):
    plan, base = specification(tmp_path)
    for change in ('seed', 'source', 'budget', 'initial', 'missing', 'duplicate'):
        bad = deepcopy(plan)
        if change == 'seed':
            bad['train']['seed'] += 1
        elif change == 'source':
            bad['source_mixture'] = {'coco2017': 1.}
        elif change == 'budget':
            bad['train']['steps'] = 100
        elif change == 'initial':
            bad['common_initialization']['step'] = 3892
        elif change == 'missing':
            bad['variants'].pop(next(iter(bad['variants'])))
        else:
            bad['variants']['duplicate'] = deepcopy(next(iter(bad['variants'].values())))
        with pytest.raises(ValueError):
            configurations(bad, base, tmp_path / change)


def initial_fixture():
    config = dict(train=dict(seed=43))
    old = dict(step=232288, adapter={'x':torch.tensor([1.,2.])}, statistics={'rms':torch.ones(2)},
               adapter_config={'dims':2}, model_config={'fixture':True})
    new = dict(old, step=0, cursor=0, config=config, optimizer=dict(state={},param_groups=[]),
        scheduler={'last_epoch':0}, rng={'torch':torch.tensor([1,2],dtype=torch.uint8)},
        revisions={'fixture':'a'},data_digests={'train':'gate-A','geometry':'gate-A','feature_replay':'rgb'})
    return old, new, config


def test_separate_trainer_initials_must_reset_optimizer_and_match_rng_statistics():
    old, first, config = initial_fixture()
    new = deepcopy(first)
    new['data_digests'].update(train='gate-B',geometry='gate-B')
    verify_initial_state(old, new, config, first)
    for problem in ('step','cursor','optimizer','rng','statistics','replay'):
        changed = deepcopy(new)
        if problem in ('step','cursor'):
            changed[problem] = 1
        elif problem == 'optimizer':
            changed['optimizer']['state'] = {0:{'step':3892}}
        elif problem == 'rng':
            changed['rng']['torch'][0] += 1
        elif problem == 'statistics':
            changed['statistics']['rms'][0] *= 2
        else:
            changed['data_digests']['feature_replay'] = 'wrong'
        with pytest.raises(ValueError):
            verify_initial_state(old, changed, config, first)
