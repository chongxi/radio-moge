import math
from copy import deepcopy
import json
import sqlite3

import pytest
import torch

from radio_moge.corpus_gradient_diagnostic import component_gram, combined_direction, observe_components
from radio_moge.corpus_model_probe import module_digest
from radio_moge.losses import FeatureStatistics
from test_corpus_model_probe import fixture
from test_corpus_losses import small_official as small_official
from test_training import tiny_system
from probe_corpus_model import probe_batches
from diagnose_corpus_gradients import (batch_identity, bind_entries, choose_entries,
                                      replay_representatives)
from radio_moge.corpus_admission import file_ref


def test_opposing_components_predict_direction_change_without_parameter_update():
    parameter = torch.nn.Parameter(torch.tensor([2., 3.]))
    original = parameter.detach().clone()
    result = component_gram(dict(feature=parameter[0], geometry=-2 * parameter[0] + parameter[1]),
                            [parameter])
    assert result['gram'] == {'feature': {'feature': 1., 'geometry': -2.},
                              'geometry': {'feature': -2., 'geometry': 5.}}
    assert result['pairwise_cosine']['feature']['geometry'] == pytest.approx(-2 / math.sqrt(5))
    weak = combined_direction(result['gram'], dict(feature=.1, geometry=1.))
    strong = combined_direction(result['gram'], dict(feature=3., geometry=1.))
    assert weak['predicted_loss_change_per_unit_small_sgd_step']['feature'] == pytest.approx(1.9)
    assert strong['predicted_loss_change_per_unit_small_sgd_step']['feature'] == -1
    assert torch.equal(parameter, original) and parameter.grad is None


def test_disjoint_and_zero_components_do_not_invent_alignment():
    a, b = torch.nn.Parameter(torch.tensor(2.)), torch.nn.Parameter(torch.tensor(3.))
    result = component_gram(dict(a=a**2, b=b**2, zero=a * 0), [a, b])
    assert result['gram']['a']['b'] == 0 and result['norms']['zero'] == 0
    assert result['pairwise_cosine']['zero']['a'] is None
    assert result['parameters_with_gradient'] == {'a': 1, 'b': 1, 'zero': 1}
    assert combined_direction(result['gram'], {'zero': 1})['component_cosine_to_total']['a'] is None


def test_detached_nonfinite_and_populated_gradients_fail():
    p = torch.nn.Parameter(torch.tensor(1.))
    for loss in [torch.tensor(1.), p * float('nan')]:
        with pytest.raises(ValueError, match='scalar'):
            component_gram(dict(bad=loss), [p])
    p.grad = torch.ones_like(p)
    with pytest.raises(ValueError, match='empty'):
        component_gram(dict(good=p**2), [p])


@pytest.mark.upstream
@pytest.mark.usefixtures('small_official')
def test_production_loss_tiny_model_replay_sparse_and_state_invariance(tmp_path, monkeypatch):
    _, _, config, selected, replay = fixture(tmp_path)
    batches = list(probe_batches(selected, replay, config))
    batches[0]['geometry_targets'][0]['valid'][:] = False
    system, adapter = tiny_system()
    statistics = FeatureStatistics(6, 6)
    before = module_digest(adapter)
    def forbid(*args, **kwargs):
        raise AssertionError('No optimizer may be constructed')
    monkeypatch.setattr(torch.optim, 'AdamW', forbid)
    report = observe_components(system, adapter, statistics, batches, config)
    assert report['optimizer_updates'] == 0 and not report['optimizer_constructed']
    assert report['state_before'] == report['state_after'] and module_digest(adapter) == before
    assert {r['feature_branch'] for r in report['rows']} == {'paired_rgb', 'replay_rgb'}
    assert report['rows'][0]['gradients']['norms']['geometry_weighted'] == 0
    for row in report['rows'][1:]:
        assert row['gradients']['norms']['geometry_weighted'] > 0
        assert row['scenarios']['current']['coefficients']['feature_unit'] == .1
    assert all(p.grad is None for p in adapter.parameters())


def test_selection_covers_strata_and_is_order_independent_with_distinct_scenes():
    entries = [dict(ordinal=i, group=f'{dataset}:{i}', entry=dict(image=dict(
        dataset=dataset, split='train'), geometry_metadata=dict(label_kind=kind)))
        for i, (dataset, kind) in enumerate([('a', 'native')] * 3 + [('a', 'pseudo')] * 3
                                             + [('b', 'pseudo')] * 3)]
    selected = choose_entries(entries)
    assert selected == choose_entries(list(reversed(entries)))
    assert len(selected) == 6 and len({v['group'] for v in selected}) == 6
    bad = deepcopy(entries)
    bad[0]['entry']['image']['split'] = 'val'
    with pytest.raises(ValueError, match='training'):
        choose_entries(bad)


def test_accounting_binding_rejects_altered_excluded_and_absent_members():
    db = sqlite3.connect(':memory:')
    db.execute('CREATE TABLE images(ordinal,pack,entry_json,disposition)')
    item = dict(ordinal=1, pack='pack.tar', entry=dict(key='a', image=dict(sha256='b')))
    db.execute('INSERT INTO images VALUES (?,?,?,?)',
               (1, item['pack'], json.dumps(item['entry']), 'selected'))
    bind_entries([item], db)
    for key, value in [('ordinal', 2), ('pack', 'wrong.tar'), ('entry', {})]:
        with pytest.raises(ValueError, match='accounting'):
            bind_entries([dict(item, **{key: value})], db)
    db.execute("UPDATE images SET disposition='excluded'")
    with pytest.raises(ValueError, match='accounting'):
        bind_entries([item], db)
    db.close()


def test_replay_hash_and_sources_are_bound_without_depth_inputs(tmp_path):
    path = tmp_path / 'replay.jsonl'
    rows = [dict(dataset=s, group=s + ':scene', split='train', image=s + '.png') for s in ('a', 'b', 'a')]
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    ref = file_ref(path)
    chosen, count = replay_representatives(ref, {'a': .5, 'b': .5})
    assert count == 3 and [v['dataset'] for v in chosen] == ['a', 'b']
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='identity'):
        replay_representatives(ref, {'a': .5, 'b': .5})
    rows[0]['depth'] = 'forbidden.npy'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='RGB'):
        replay_representatives(file_ref(path), {'a': .5, 'b': .5})


def test_paired_input_digest_includes_rgb_native_geometry_and_replay(tmp_path):
    _, _, config, selected, replay = fixture(tmp_path)
    original = list(probe_batches(selected, replay, config))[1]
    checksum = batch_identity(original)
    assert checksum == batch_identity(deepcopy(original))
    for key in ('image', 'replay_image'):
        changed = deepcopy(original)
        changed[key][(0,) * changed[key].ndim] += .1
        assert batch_identity(changed) != checksum
    changed = deepcopy(original)
    valid = changed['geometry_targets'][0]['valid']
    valid[(0,) * valid.ndim] ^= True
    assert batch_identity(changed) != checksum
