from copy import deepcopy
import json

import pytest

from radio_moge.corpus_admission import file_ref
from review_procthor_feature_mix import compare_saved
from review_projection_pair_arm import metrics_for


@pytest.mark.parametrize('change', [None, 'order', 'teacher', 'protocol', 'aggregate', 'log'])
def test_saved_new_house_comparison_keeps_protocol_and_independent_metrics(tmp_path, change):
    rows = [dict(image=f'{i}.png', dataset='procthor_empty', **{
        'spatial/cosine': .9+i*.01, 'summary/cosine': .8, 'gt_teacher/abs_rel': .1}) for i in range(4)]
    before = dict(per_image=rows, metrics=metrics_for(rows), protocol=dict(images=4, max_side=640))
    now = deepcopy(before)
    for row in now['per_image']:
        row['spatial/cosine'] += .001
    now['metrics'] = metrics_for(now['per_image'])
    if change == 'order':
        now['per_image'].reverse()
    elif change == 'teacher':
        now['per_image'][0]['gt_teacher/abs_rel'] += .01
    elif change == 'protocol':
        now['protocol']['max_side'] = 480
    elif change == 'aggregate':
        now['metrics']['macro/spatial/cosine'] += .01
    old_dir, new_dir = tmp_path/'old', tmp_path/'new'
    old_dir.mkdir()
    new_dir.mkdir()
    refs = {}
    for key in before:
        old_file, new_file = old_dir/(key+'.json'), new_dir/(key+'.json')
        old_file.write_text(json.dumps(before[key]))
        new_file.write_text(json.dumps(now[key]))
        refs[key] = file_ref(old_file)
    prefix = 'dataset/procthor-empty/'
    log = {prefix+k:v for k,v in now['metrics'].items()}
    if change == 'log':
        log[prefix+'macro/spatial/cosine'] += .1
    if change is None:
        result = compare_saved(refs, new_dir, log, prefix)
        assert result['images'] == 4 and result['exact_images_protocol_teacher_support']
        assert result['metrics']['macro/spatial/cosine']['difference'] == pytest.approx(.001)
    else:
        with pytest.raises(ValueError):
            compare_saved(refs, new_dir, log, prefix)
