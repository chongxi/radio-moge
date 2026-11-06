from copy import deepcopy
import json
import math

import pytest

from review_room_feature_epoch import VIEWS, check_identity, committed_prefix, endpoints
from run_room_feature_epoch import IMAGES, UPDATES, SOURCES
from verify_room_epoch_finish import check_closure
from verify_room_epoch_online import exact_values, metadata_differences


def identity():
    old = dict(revisions={'teacher':'fixed'}, adapter='fixed',
               data_digests={name:name for name in VIEWS}|{'train':'old'})
    new = deepcopy(old); new['data_digests']['train'] = 'new'
    parent = dict(step=234538, views={name:dict(files={'bound':name}) for name in VIEWS})
    plan = dict(source_step=234538, additional_updates=UPDATES, synthetic_images=IMAGES,
        coverage_goal='every_frozen_synthetic_rgb_once', source_cursor=18000,
        source_parent_cursor=8000, source_results={'bound':'parent'})
    config = dict(loss=dict(spatial=1., summary=1., cosine=1., summary_cosine=1.))
    initial = dict(step=234538, cursor=0, optimizer_states=100, real_stream_start_cursor=18000,
                   parent_mixed_cursor=8000, data_digests=new['data_digests'])
    meta = dict(synthetic_images=IMAGES, planned_synthetic_unique=IMAGES,
                planned_synthetic_repeats=0, geometry_loss=False, mixed_digest='new')
    baseline = dict(source=plan['source_results'], step=234538, model_inference=False,
                    views={k:v['files'] for k,v in parent['views'].items()})
    return plan, config, parent, old, new, initial, meta, baseline


@pytest.mark.parametrize('mutation', [None, 'heldout', 'missing_view', 'baseline', 'adam', 'cursor',
                                     'geometry', 'short_budget'])
def test_continued_six_view_identity_accepts_only_new_training_population(mutation):
    args = identity()
    plan, config, parent, old, new, initial, meta, baseline = args
    if mutation == 'heldout': new['data_digests']['original'] = 'new-images'
    elif mutation == 'missing_view': del parent['views']['dataset-structural-rooms']
    elif mutation == 'baseline': baseline['model_inference'] = True
    elif mutation == 'adam': initial['optimizer_states'] = 0
    elif mutation == 'cursor': initial['real_stream_start_cursor'] = 8000
    elif mutation == 'geometry': config['ordinary_geometry'] = {}
    elif mutation == 'short_budget': plan['additional_updates'] = 1000
    if mutation is None:
        check_identity(*args)
    else:
        with pytest.raises(ValueError): check_identity(*args)


@pytest.mark.parametrize('mutation', [None, 'duplicate', 'identity', 'key', 'missing', 'geometry'])
def test_saved_prefix_counts_committed_identities_not_total_steps(tmp_path, mutation):
    synthetic = [dict(image=f's{i}', sha256=f'h{i}', dataset='structural_rooms') for i in range(8)]
    rows = []
    for update in range(2):
        images = [dict(image='real', sha256='r', dataset='real', key=[0,43,32,32]) for _ in range(28)]
        images += [dict(**synthetic[i], key=[5+i,43,32,32]) for i in range(update*4, update*4+4)]
        rows.append(dict(update=101+update, logical_images=images))
    planned = [r['key'] for row in rows for r in row['logical_images']]
    if mutation == 'duplicate':
        rows[1]['logical_images'][-1] = deepcopy(rows[0]['logical_images'][-1])
        planned[-1] = rows[1]['logical_images'][-1]['key']
    elif mutation == 'identity': rows[1]['logical_images'][-1]['sha256'] = 'wrong'
    elif mutation == 'key': planned[-1] = [100,43,32,32]
    elif mutation == 'missing': rows.pop()
    elif mutation == 'geometry': rows[0]['logical_images'][0]['ordinary_geometry'] = {}
    file = tmp_path/'audit.jsonl'
    file.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    if mutation is None:
        result = committed_prefix(file, planned, synthetic, 5, 100, 2)
        assert result['synthetic_coverage']['each_image_exactly_once']
        assert result['presentations'] == 64 and result['synthetic_coverage']['unique'] == 8
        assert result['source_counts'] == dict(real=56, structural_rooms=8)
    else:
        with pytest.raises(ValueError): committed_prefix(file, planned, synthetic, 5, 100, 2)


def closed():
    coverage = dict(expected=IMAGES, committed=IMAGES, unique=IMAGES, missing=0,
                    repeated=0, each_image_exactly_once=True)
    sources = dict(**SOURCES, original=UPDATES*32-IMAGES)
    audit = dict(optimizer_updates=UPDATES, cursor=UPDATES*8, presentations=UPDATES*32,
        planned_keys_matched=True, unused_prefetch_excluded=True, actual_compute_groups_verified=True,
        source_counts=sources, resolution_counts={'three_resolutions':UPDATES*32}, records={'sha256':'full'})
    actual = dict(presentations=UPDATES*32, actual_planned_keys_matched=True,
        source_counts=sources, synthetic_identity_and_no_repeats_verified=True,
        synthetic_coverage=deepcopy(coverage), actual_input_prefix={'sha256':'full'})
    result = dict(step=234538+UPDATES, updates=UPDATES, full_epoch_completed=True,
        unused_prefetch_excluded=True, matched_encoded_image_identity=True,
        actual_inputs={'sha256':'full'}, **coverage)
    plan = dict(source_step=234538, additional_updates=UPDATES, synthetic_images=IMAGES)
    config = dict(train={'steps':234538+UPDATES},
                  loss=dict(spatial=1., summary=1., cosine=1., summary_cosine=1.))
    completion = dict(new_updates=UPDATES, full_epoch_complete=True,
        stop_report=dict(reason='step_limit', step=234538+UPDATES, final_evaluation='complete'))
    review = dict(step=234538+UPDATES, new_updates=UPDATES, full_epoch_complete=True, actual_inputs=actual)
    return plan, config, completion, review, audit, result, [1]*IMAGES


@pytest.mark.parametrize('mutation', [None, 'partial', 'count', 'source', 'hash', 'geometry'])
def test_final_epoch_requires_independent_full_count_and_same_closed_audit(mutation):
    args = closed()
    if mutation == 'partial': args[2]['full_epoch_complete'] = False
    elif mutation == 'count': args[-1][3] = 0
    elif mutation == 'source': args[4]['source_counts']['structural_rooms'] -= 1
    elif mutation == 'hash': args[3]['actual_inputs']['actual_input_prefix']['sha256'] = 'different'
    elif mutation == 'geometry': args[1]['geometry_supervision'] = {}
    if mutation is None:
        assert check_closure(*args) == 234538+UPDATES
    else:
        with pytest.raises(ValueError): check_closure(*args)


def test_validation_schedule_includes_short_final_interval():
    assert endpoints(234538) == [234538+n for n in range(1000, 13000, 1000)] + [234538+UPDATES]


def test_online_keeps_logged_values_exact_and_bounds_only_metadata_serialization():
    config = dict(evaluation={'checkpoint_metrics':{'spatial':{'baseline':.96}}}, loss={'cosine':1.})
    provenance = dict(performance={'original_started_at':1793919981.6150732})
    actual = deepcopy(config)
    actual['evaluation']['checkpoint_metrics']['spatial']['baseline'] = math.nextafter(.96, math.inf)
    diffs = metadata_differences(actual, provenance, config, provenance)
    assert len(diffs) == 1 and diffs[0]['absolute_difference'] == math.ulp(.96)
    actual['loss']['cosine'] = math.nextafter(1., math.inf)
    with pytest.raises(ValueError): metadata_differences(actual, provenance, config, provenance)
    assert exact_values({'loss':1}, {'loss':1.}, 'train')[0]['absolute_difference'] == 0
    with pytest.raises(ValueError): exact_values({'loss':math.nextafter(1., math.inf)}, {'loss':1.}, 'train')
