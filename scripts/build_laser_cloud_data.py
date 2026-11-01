"""Audit cached physical geometry and make a measured-GT four-way cloud payload."""
import argparse
import base64
import gzip
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from radio_moge.checkpoint import atomic_json
from radio_moge.gt_geometry_metrics import metric_camera_points, physical_geometry_metrics
from export_gt_pilot_comparison import flatten_metric_tree


def build(root, manifest):
    report = json.loads((root / 'geometry.json').read_text())
    assert report['stage'] == 'complete' and report['images'] == 18 and report['scenes'] == 7
    names = ['student', 'parent', 'teacher', 'ground_truth']
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert [r['id'] for r in rows] == [r['id'] for r in report['per_image']['student']]
    samples = []
    for i, row in enumerate(rows):
        with np.load(root / f"{row['id']}.npz", allow_pickle=False) as data:
            a = {k: data[k].copy() for k in data.files}
        with np.load(manifest.parent / row['geometry'], allow_pickle=False) as gt:
            for key in ('depth', 'valid', 'intrinsics', 'plane_labels', 'planes'):
                assert np.array_equal(a[key], gt[key]), (row['id'], key)
        assert np.array_equal(a['rgb'], np.asarray(Image.open(manifest.parent / row['image']).convert('RGB')))
        truth = metric_camera_points(a['depth'], a['intrinsics'])
        target = {k: a[k] for k in ('depth', 'valid', 'intrinsics', 'plane_labels', 'planes')}
        physical = {}
        metrics = {}
        depths = [a[n + '_depth'] for n in names[:3]] + [a['depth']]
        for j, name in enumerate(names):
            raw = a[name + '_raw_xyz'] if name != 'ground_truth' else truth
            current = physical_geometry_metrics(depths[j], raw, target)
            if name != 'ground_truth':
                stored = report['per_image'][name][i]
                assert stored['id'] == row['id']
                for key, value in current.items():
                    assert value == stored[key], (row['id'], name, key)
                # Independent direct depth and raw3D arithmetic on exactly GT support.
                mask = a['valid']
                delta = raw[mask] - truth[mask]
                assert abs(np.sqrt(np.sum(delta * delta, axis=-1).mean()) - stored['all']['raw_xyz_rmse_m']) < 1e-10
                assert abs(np.sqrt(((depths[j][mask].astype(float) - a['depth'][mask]) ** 2).mean()) - stored['all']['depth_rmse_m']) < 1e-10
            physical[name] = flatten_metric_tree(current)
            mask = a['valid']
            z = depths[j][mask].astype(float)
            gt_z = a['depth'][mask].astype(float)
            metrics[name] = {'sensor': {'abs_rel': float(np.mean(abs(z - gt_z) / gt_z)),
                                       'rmse': float(np.sqrt(np.mean((z - gt_z) ** 2)))},
                             'teacher': {'abs_rel': float(np.mean(abs(z - a['teacher_depth'][mask]) / np.maximum(a['teacher_depth'][mask], 1e-6)))}}
        valid_masks = [a['valid'] & np.isfinite(z) & (z > 0) for z in depths]
        buffer = io.BytesIO()
        Image.fromarray(a['rgb']).save(buffer, format='PNG')
        stride = 2
        blob = b''.join(np.asarray(z[::stride, ::stride], dtype='<f4').tobytes() for z in depths)
        blob += b''.join(np.asarray(m[::stride, ::stride], dtype=np.uint8).tobytes() for m in valid_masks)
        samples.append(dict(id=row['id'], scene=row['scene'], scene_type='激光验证', width=640, height=400,
                            stride=stride, rgb='data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode(),
                            cameras=[a['intrinsics'].tolist()] * 4, metrics=metrics, physical=physical,
                            full_valid_points=[int(m.sum()) for m in valid_masks], projection_error_m=0.,
                            range=np.quantile(a['depth'][a['valid']], [.02, .98]).tolist(),
                            payload=base64.b64encode(gzip.compress(blob, compresslevel=6, mtime=0)).decode()))
    summary = {n: {'sensor': {k: float(np.mean([s['metrics'][n]['sensor'][k] for s in samples]))
                              for k in ('abs_rel', 'rmse')}} for n in names}
    data = dict(status='complete', names=names, labels=['GT训练后 · 1000步', '训练前归档 · 232288步', 'MoGe-2', '激光测量GT'],
                samples=samples, selected_ids=[s['id'] for s in samples], summary=summary,
                checkpoint_step=1000, selection_metric='fixed final1000',
                selection_value=summary['student']['sensor']['rmse'], known_camera=True,
                view=dict(title='三模型 × 激光测量GT', subtitle='18幅验证图 · 7个隔离场景 · 相同测量相机',
                          badge_label='GT训练后 · 激光深度RMSE（米）', checkpoint_label='第1000步最终权重',
                          weights='橙色为GT试训最终模型，蓝色为训练前归档，绿色为原始MoGe-2，灰色为激光测量。三模型都使用同一已知测量相机，不对GT拟合尺度或平移。',
                          protocol='仅显示测量GT有效像素，空洞不填充；评估固定GT支持，不允许预测掩码隐藏误差。非正深度无法绘成正常点云，但仍计入数值误差。原始XYZ误差保留解码器XY，点云展示则通过已知相机投影。',
                          caveat='这是18图/7场景的反复使用验证集，不是新独立测试。自身平整度不等于位置、方向或尺度准确；需结合XYZ和固定GT平面误差。未标注平面区域包含曲面、物体和未被识别的平面。'))
    atomic_json(data, root / 'data.json')
    atomic_json(dict(stage='passed', images=18, branches=3, fixed_target_exact=True,
                     saved_metrics_exact=True, independent_depth_xyz_rmse=True,
                     no_inference=True), root / 'data-verification.json')
    print(json.dumps({'stage': 'passed', 'images': 18, 'branches': 4}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    build(args.root, args.manifest)
