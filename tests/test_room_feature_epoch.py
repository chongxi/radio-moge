from copy import deepcopy
from collections import Counter
import json

import pytest

from radio_moge.data import DeterministicBatches
from radio_moge.rgb_replacement import ReplacementBatches
from unittest.mock import patch
import radio_moge.train as trainer
from run_room_feature_epoch import (Coverage, EpochAudit, IMAGES, UPDATES, SECONDS,
    TAIL_POLICY, SingleEpochBatches, check_epoch_contract)
from test_procthor_rgb_training import contract, dummy_records


def epoch_contract():
    old, parent, _ = contract()
    old['evaluation']['dataset_variants']['structural-rooms'] = dict(
        manifest='/structural-val', max_side=640, num_tokens=1200)
    parent.update(config=deepcopy(old), step=234538, cursor=8000)
    current = deepcopy(old)
    current['output_dir'] = '/epoch'
    current['data']['rgb_replacement'].update(real_start_cursor=18000,
        manifest=dict(path='/epoch-pool', sha256='b'*64))
    current['train'].update(steps=234538+UPDATES, max_seconds=SECONDS, eval_every=1000, save_every=250)
    plan = dict(authorized=True, coverage_goal='every_frozen_synthetic_rgb_once',
        synthetic_images=IMAGES, additional_updates=UPDATES, maximum_training_seconds=SECONDS,
        source_step=234538, source_parent_cursor=8000, source_cursor=18000, tail_policy=TAIL_POLICY,
        synthetic_manifest=current['data']['rgb_replacement']['manifest'],
        validation_manifest=dict(path='/val'), structural_validation_manifest=dict(path='/structural-val'))
    return current, parent, plan


@pytest.mark.parametrize('mutation', [None, 'short_budget', 'reset_cursor', 'forget_parent_cursor',
    'drop_geometry_guard', 'feature_weight', 'resolution', 'old_pool', 'validation',
    'start_inference', 'unapproved', 'wrong_parent', 'fraction', 'lr', 'tail'])
def test_full_epoch_contract_binds_parent_stream_and_budget(mutation):
    config, parent, plan = epoch_contract()
    if mutation == 'short_budget': config['train']['steps'] = parent['step']+1000
    elif mutation == 'reset_cursor': config['data']['rgb_replacement']['real_start_cursor'] = 0
    elif mutation == 'forget_parent_cursor': config['data']['rgb_replacement']['real_start_cursor'] = 8000
    elif mutation == 'drop_geometry_guard': config['ordinary_geometry'] = {}
    elif mutation == 'feature_weight': config['loss']['cosine'] = .1
    elif mutation == 'resolution': config['data']['buckets'] = [[400, 640]]
    elif mutation == 'old_pool': config['data']['rgb_replacement']['manifest'] = parent['config']['data']['rgb_replacement']['manifest']
    elif mutation == 'validation': config['evaluation']['dataset_variants']['structural-rooms']['manifest'] = '/train'
    elif mutation == 'start_inference': config['train']['eval_at_start'] = True
    elif mutation == 'unapproved': plan['authorized'] = False
    elif mutation == 'wrong_parent': plan['source_parent_cursor'] = 0
    elif mutation == 'fraction': config['data']['rgb_replacement']['every_images'] = 4
    elif mutation == 'lr': config['train']['lr'] = 1e-4
    elif mutation == 'tail': plan['tail_policy'] = 'repeat_to_pad'
    if mutation is None:
        check_epoch_contract(config, parent, plan)
    else:
        with pytest.raises(ValueError): check_epoch_contract(config, parent, plan)


def test_actual_50787_sampler_permutation_covers_all_without_wrapping_tail():
    config, _, _ = epoch_contract()
    records = dummy_records(53, IMAGES)
    spec = config['data']['rgb_replacement']
    mixed = SingleEpochBatches(records, spec, 4, config['data']['buckets'], 43, 0, UPDATES*8)
    real = DeterministicBatches(53, 4, config['data']['buckets'], 43, 18000, UPDATES*8)
    counts = Counter()
    for number, (batch, before) in enumerate(zip(mixed, real, strict=True)):
        for offset, (key, old) in enumerate(zip(batch, before, strict=True)):
            if (number*4+offset+1) % 8 == 0 and number*4+offset < IMAGES*8:
                counts[key[0]-53] += 1
                assert key[1:] == old[1:]
            else:
                assert key == old
    assert len(counts) == IMAGES and set(counts) == set(range(IMAGES))
    assert set(counts.values()) == {1} and UPDATES == 12697


@pytest.mark.parametrize('n', [1, 2, 3, 4, 7, 8])
def test_tail_and_resume_keep_all_other_original_keys_and_do_not_wrap(n):
    config, _, _ = epoch_contract()
    records = dummy_records(53, n)
    spec, buckets = config['data']['rgb_replacement'], config['data']['buckets']
    total = ((n+3)//4)*8 + 8  # Include unused loader prefetch.
    full = list(SingleEpochBatches(records, spec, 4, buckets, 43, 0, total))
    original = list(DeterministicBatches(53, 4, buckets, 43, 18000, total))
    assert sum(key[0] >= 53 for batch in full for key in batch) == n
    for start in [1, total-9, total-4]:
        assert list(SingleEpochBatches(records, spec, 4, buckets, 43, start, total-start)) == full[start:]
    assert full[-8:] == original[-8:]
    for i, (batch, old) in enumerate(zip(full, original, strict=True)):
        for j, (key, prior) in enumerate(zip(batch, old, strict=True)):
            if key[0] < 53:
                assert key == prior
            else:
                assert key[1:] == prior[1:] and i*4+j < n*8


def test_production_loader_selects_bounded_sampler_only_inside_explicit_runner_patch():
    config, _, _ = epoch_contract()
    records = dummy_records(53, 7)
    config['data'].update(workers=0, augment=True)
    config['train'].update(batch_size=4, seed=43)
    old = trainer.ReplacementBatches
    with patch.object(trainer, 'ReplacementBatches', SingleEpochBatches):
        loader = trainer.make_loader(records, config, 0, 16)
        keys = [key for batch in loader.batch_sampler for key in batch]
        assert len(keys) == 64 and sum(k[0] >= 53 for k in keys) == 7
        assert isinstance(loader.batch_sampler, SingleEpochBatches)
    assert trainer.ReplacementBatches is old is ReplacementBatches


def test_coverage_tail_rejects_padding_and_early_short_update():
    tracker = coverage(7)
    with pytest.raises(ValueError, match='remaining unique'):
        tracker.validate(images([0, 1, 2]))
    tracker.commit(tracker.validate(images([0, 1, 2, 3])))
    with pytest.raises(ValueError, match='repeated'):
        tracker.validate(images([4, 5, 6, 0]))
    tracker.commit(tracker.validate(images([4, 5, 6])))
    assert tracker.report()['each_image_exactly_once']
    with pytest.raises(ValueError):
        tracker.validate([])


def images(indices, real_size=5):
    return [dict(image=f'/s{i}', sha256=f'h{i}', dataset='structural_rooms',
                 key=[real_size+i, 43, 32, 32]) for i in indices]


def coverage(n=8):
    return Coverage(5, [dict(image=f'/s{i}', sha256=f'h{i}', dataset='structural_rooms') for i in range(n)])


def test_coverage_counts_only_committed_exact_identities_and_rejects_duplicates():
    tracker = coverage()
    first = tracker.validate(images(range(4)))
    assert tracker.report()['committed'] == 0  # Prepared but not optimizer-committed.
    tracker.commit(first)
    assert tracker.report()['unique'] == 4 and not tracker.report()['each_image_exactly_once']
    with pytest.raises(ValueError, match='repeated'): tracker.validate(images([0, 5, 6, 7]))
    with pytest.raises(ValueError, match='repeated'): tracker.validate(images([4, 4, 6, 7]))
    wrong = images(range(4, 8)); wrong[0]['sha256'] = 'different-payload'
    with pytest.raises(ValueError, match='identity differs'): tracker.validate(wrong)
    tracker.commit(tracker.validate(images(range(4, 8))))
    assert tracker.report() == dict(expected=8, committed=8, unique=8, missing=0,
                                    repeated=0, each_image_exactly_once=True)


@pytest.mark.parametrize('updates,full,missing', [(1, False, 4), (2, True, 0)])
@pytest.mark.parametrize('size', [7, 8])
def test_production_audit_closure_retains_missing_and_excludes_unused_prefetch(tmp_path, monkeypatch,
                                                                           updates, full, missing, size):
    monkeypatch.setattr('run_room_feature_epoch.UPDATES', 2)
    output = tmp_path/'training'; output.mkdir()
    audit = EpochAudit(tmp_path, {}, dict(output_dir=str(output)), dict(step=100), lambda *a, **k: None)
    audit.coverage = coverage(size)
    audit.planned_keys = []
    for update in range(2):
        logical = images(range(update*4, min(size, update*4+4)))
        logical += [dict(image='/real', sha256='r', dataset='real', key=[0,43,32,32])]*(32-len(logical))
        audit.pending_windows.append(dict(update=101+update, logical_images=logical, compute_groups=[]))
    for update in range(updates): audit.commit(101+update, (update+1)*8)
    audit.finish(100+updates, updates*8)
    result = json.loads((tmp_path/'coverage-final.json').read_text())
    assert result['full_epoch_completed'] is full
    assert result['missing'] == max(0, size-updates*4)
    assert result['committed'] == min(size, updates*4)
    assert result['unused_prefetch_excluded']
    assert len((output/'actual-rgb-inputs.jsonl').read_text().splitlines()) == updates
