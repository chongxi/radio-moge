import json

from scripts.report_loss_weight_ablation import compare


def test_paired_report_respects_metric_directions(tmp_path):
    records = [{'image': str(i), 'dataset': 'nyuv2', 'group': str(i // 2),
                'depth': 'measured.npy'} for i in range(4)]
    paths = []
    for name, cosine, error in [('before', .9, .2), ('after', .95, .1)]:
        path = tmp_path / name
        (path / 'gt-laser-known-camera').mkdir(parents=True)
        rows = [dict(image=r['image'], dataset='nyuv2', postprocess_failed=0,
                     **{'spatial/cosine': cosine, 'summary/cosine': cosine,
                        'gt_student/abs_rel': error, 'gt_student/rmse': error}) for r in records]
        (path / 'per_image.json').write_text(json.dumps(rows))
        laser = [dict(id=r['image'], scene=r['group'], abs_rel=error,
                      rmse=error, plane_rmse_m=error) for r in records]
        (path / 'gt-laser-known-camera/per_image.json').write_text(json.dumps(laser))
        paths.append(path)
    result = compare(*paths, records)
    assert result['feature']['metrics']['spatial/cosine']['macro_difference'] > 0
    assert result['feature']['metrics']['spatial/cosine']['sources']['nyuv2']['scenes_improved'] == 2
    assert result['nyu']['metrics']['gt_student/abs_rel']['macro_difference'] < 0
    assert result['laser']['metrics']['rmse']['macro_difference'] < 0
    assert 'postprocess_failures' not in result['laser']
    assert result['laser']['camera_protocol'].startswith('Known measured intrinsics')
    assert result['nyu']['postprocess_failures'] == {'baseline': 0, 'candidate': 0}
