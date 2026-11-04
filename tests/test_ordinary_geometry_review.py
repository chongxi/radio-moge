from copy import deepcopy

import pytest

from review_ordinary_geometry_training import compare_input_rows, reference_descriptor


def inputs():
    old = [dict(update=232289, logical_images=[dict(key=[3, 0, 400, 640], image='/rgb.png',
        compact_rgb_sha256='a'*64)], compute_groups=[dict(logical_image_slots=[0], shape=[1, 3, 400, 640],
        compact_tensor_sha256='b'*64)])]
    new = deepcopy(old)
    new[0]['logical_images'][0]['ordinary_geometry'] = dict(augment=True, reference_sha256='c'*64)
    return old, new


def test_actual_input_pairing_ignores_only_the_new_geometry_annotation():
    old, new = inputs()
    compare_input_rows(old, new, 1)
    assert 'ordinary_geometry' in new[0]['logical_images'][0]


@pytest.mark.parametrize('change', ['pixel', 'key', 'group', 'missing_annotation', 'short_budget'])
def test_changed_pixels_keys_prepared_groups_and_short_run_cannot_pass(change):
    old, new = inputs()
    if change == 'pixel':
        new[0]['logical_images'][0]['compact_rgb_sha256'] = 'd'*64
    elif change == 'key':
        new[0]['logical_images'][0]['key'][1] += 1
    elif change == 'group':
        new[0]['compute_groups'][0]['logical_image_slots'] = [1]
    elif change == 'missing_annotation':
        new[0]['logical_images'][0].pop('ordinary_geometry')
    else:
        new = []
    with pytest.raises(ValueError):
        compare_input_rows(old, new, 1)


def test_reference_descriptor_uses_reference_final_values_without_mutating_history():
    archive = dict(images=2, manifest={'path':'fixed'}, datasets={'a':dict(images=2,
        metrics={'cosine':dict(before=.9, step3892=.8)})})
    reference = dict(images=2, files=dict(metrics={'path':'new'}, per_image={'path':'rows'},
        protocol={'path':'protocol'}), protocol={'crop':'fixed'},
        datasets={'a':dict(images=2, metrics={'cosine':dict(before=.9, final=.91)})})
    value = reference_descriptor(archive, reference)
    assert value['manifest'] == archive['manifest']
    assert value['before'] == reference['files']['metrics']
    assert value['datasets']['a']['metrics']['cosine'] == {'before':.91}
    assert archive['datasets']['a']['metrics']['cosine']['before'] == .9
    reference['images'] = 3
    with pytest.raises(ValueError, match='scope differs'):
        reference_descriptor(archive, reference)
