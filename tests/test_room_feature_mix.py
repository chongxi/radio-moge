from copy import deepcopy

import pytest

from prepare_room_rgb_pool import check_rows, overlap
from run_room_feature_mix import check_mix_contract
from test_procthor_rgb_training import contract


def room(projection='perspective', split='train'):
    return dict(dataset='structural_rooms', projection=projection,
        camera_model='pinhole' if projection == 'perspective' else 'parallel',
        user_approved_projection=True, geometry_admission=False,
        frame_policy='strict-rgb-frame-filter-v2', window_count=0, door_count=0,
        exterior_background_pixels=0, split=split, agent_visible=False,
        geometry_label=None, label_role='feature_only_rgb', training_admission=split == 'train',
        group=f'structural-room-rgb-multiview-v1/{split}/000008')


@pytest.mark.parametrize('projection', ['perspective', 'orthographic'])
@pytest.mark.parametrize('split', ['train', 'val'])
def test_explicit_projection_and_split_are_preserved(projection, split):
    row = room(projection, split)
    before = deepcopy(row)
    check_rows([row], split)
    assert row == before


@pytest.mark.parametrize('key,value', [
    ('projection', 'equirectangular'), ('camera_model', 'panorama'),
    ('window_count', 1), ('exterior_background_pixels', 1), ('agent_visible', True),
    ('depth', '/not-a-feature-target'), ('training_admission', False),
    ('user_approved_projection', False), ('geometry_admission', True),
    ('dataset', 'coco2017')])
def test_undeclared_or_held_inputs_rejected(key, value):
    row = room()
    row[key] = value
    with pytest.raises(ValueError):
        check_rows([row], 'train')


def test_cross_split_detects_pixels_even_with_different_names():
    a = [dict(image='a', group='train', decoded_rgb_sha256='same')]
    b = [dict(image='b', group='val', decoded_rgb_sha256='same')]
    assert overlap(a, b)['decoded_rgb_sha256'] == 1
    assert overlap(a, b)['group'] == 0


@pytest.mark.parametrize('mutation', [None, 'population', 'geometry', 'weight', 'budget', 'real'])
def test_room_fork_only_adds_declared_population(mutation):
    config, parent, plan = contract()
    plan['structural_validation_manifest'] = dict(path='/structural-val')
    config['evaluation']['dataset_variants']['structural-rooms'] = dict(
        manifest='/structural-val', max_side=640, num_tokens=1200)
    if mutation == 'population':
        config['evaluation']['dataset_variants']['structural-rooms']['manifest'] = '/train'
    elif mutation == 'geometry':
        config['ordinary_geometry'] = {}
    elif mutation == 'weight':
        config['loss']['cosine'] = .1
    elif mutation == 'budget':
        config['train']['steps'] += 1
    elif mutation == 'real':
        config['data']['rgb_replacement']['real_start_cursor'] = 0
    if mutation is None:
        check_mix_contract(config, parent, plan)
    else:
        with pytest.raises(ValueError):
            check_mix_contract(config, parent, plan)
