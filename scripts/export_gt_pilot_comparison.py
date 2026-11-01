"""Post-pilot evaluation only: fixed20 cache comparison and physical laser metrics."""
import argparse
import base64
import gc
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from radio_moge.checkpoint import atomic_json
from radio_moge.decoder import postprocess
from radio_moge.data import read_manifest
from radio_moge.gt_data import GroundTruthImages, validate_gt_records
from radio_moge.gt_geometry import calibrated_geometry
from radio_moge.gt_geometry_metrics import physical_geometry_metrics
from radio_moge.metrics import depth_metrics
from radio_moge.pointcloud import verify_geometry
from radio_moge.system import build_system
from radio_moge.upstream import extract_radio, radio_input
from scannet_cloud_cache import cached_reference


def identity(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def fixed20(system, adapter, config, reference, output, checkpoint, final_cosine):
    output.mkdir()
    raw = (reference / 'data.json').read_bytes()
    old = json.loads(raw)
    proof = json.loads((reference / 'completion.json').read_text())
    assert proof['stage'] == 'complete' and old['status'] == 'complete'
    assert old['checkpoint_step'] == 232288 and old['names'] == ['student', 'parent', 'teacher']
    assert len(old['samples']) == len(set(old['selected_ids'])) == 20
    assert [s['id'] for s in old['samples']] == old['selected_ids']
    assert all(not any(c['overlap'].values()) for c in old['split_checks'])
    names = ['student', 'parent', 'teacher']
    report = dict(status='initializing', started_at=time.time(), checkpoint_step=1000,
                  checkpoint=str(checkpoint), selection_metric='fixed final1000, not selected on20scenes',
                  selection_value=final_cosine, names=names,
                  labels=['GT几何训练 · 1,000步', '训练前归档 · 232,288步', 'MoGe-2 · 原始模型'],
                  selected_ids=old['selected_ids'], samples=[], backend='pytorch',
                  model_config=config['models'], revisions=old['revisions'],
                  split_checks=old['split_checks'], parent_checkpoint_step=232288,
                  cached_reference_sha256=hashlib.sha256(raw).hexdigest(),
                  parent_teacher_reused=True, new_model_inference_only=True, production_updates=0,
                  evaluation=old['evaluation'], coordinates=old['coordinates'], projection=old['projection'])
    report['view'] = dict(title='GT几何训练后 × 训练前归档 × MoGe-2',
        subtitle='相同20个场景 · 同步三维视角 · 640×400',
        badge_label='完整验证集 Spatial cosine · 640×400',
        checkpoint_label='GT试训最终模型 · 第1,000步',
        weights='左侧为1000步GT试训最终权重，中间为训练前232288步归档模型，右侧为原始MoGe-2。展示最终权重，没有根据这20张图挑选模型。',
        protocol='相同640×400中心裁剪RGB；使用各自恢复的相机、预测掩码和米制尺度，不做点云配准或GT尺度拟合。训练前模型和MoGe-2复用已验证缓存。',
        caveat='此前已查看的固定20个NYU演示场景，不是新的独立测试集。请结合激光验证集与跨数据集指标判断改善和退步；cosine不能代表几何质量。')
    for i, original in enumerate(old['samples']):
        saved = cached_reference(reference, original)
        rgb = saved['rgb']
        image = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div_(255).cuda()
        gt = torch.from_numpy(saved['sensor_depth'])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            features = extract_radio(system.radio, radio_input(image, system.radio, None), adapter.indices)
            encoded = adapter(features)
        assert list(features.spatial.shape[-2:]) == list(encoded.spatial.shape[-2:]) == [25, 40]
        assert all(list(t.shape[-2:]) == [25, 40] for t in features.intermediates)
        pred = postprocess(system.decoder(encoded, (400, 640)), image, config['models'].get('moge_source'))
        depth = pred['depth'][0].cpu().numpy()
        camera, mask, error = verify_geometry(depth, pred)
        depths = [depth, saved['student_depth'], saved['teacher_depth']]
        masks = [mask, saved['student_mask'], saved['teacher_mask']]
        cameras = [camera, original['cameras'][0], original['cameras'][2]]
        metrics = {}
        for name, d, valid in zip(names, depths, masks, strict=True):
            assert valid.any() and np.isfinite(d[valid]).all() and (d[valid] > 0).all()
            metrics[name] = dict(sensor=depth_metrics(torch.from_numpy(d), gt, gt > 0),
                                 teacher=depth_metrics(torch.from_numpy(d), torch.from_numpy(depths[2]), torch.from_numpy(masks[2])))
            assert all(np.isfinite(v) for ref in metrics[name].values() for v in ref.values())
        for new_name, old_name in [('parent', 'student'), ('teacher', 'teacher')]:
            assert metrics[new_name] == original['metrics'][old_name]
        stride = original['stride']
        blob = b''.join(np.asarray(d[::stride, ::stride], dtype='<f4').tobytes() for d in depths)
        blob += b''.join(np.asarray(m[::stride, ::stride], dtype=np.uint8).tobytes() for m in masks)
        valid_depths = np.concatenate([d[m] for d, m in zip(depths, masks)])
        sample = dict(original, cameras=cameras, full_valid_points=[int(m.sum()) for m in masks],
                      range=np.quantile(valid_depths, [.02, .98]).tolist(), metrics=metrics,
                      projection_error_m=max(error, original['projection_error_m']),
                      payload=base64.b64encode(gzip.compress(blob, compresslevel=6, mtime=0)).decode(),
                      cached_parent_teacher_verified=True)
        np.savez_compressed(output / f"{sample['id']}.npz", rgb=rgb, sensor_depth=gt.numpy(),
                            student_depth=depths[0], parent_depth=depths[1], teacher_depth=depths[2],
                            student_mask=masks[0], parent_mask=masks[1], teacher_mask=masks[2],
                            cameras=np.asarray(cameras))
        report['samples'].append(sample)
        atomic_json(dict(status='exporting', completed=i + 1, total=20, updated_at=time.time()), output / 'status.json')
        print(f'Exported GT1000 {i + 1}/20; cachedH232288 and teacher unchanged', flush=True)
        del features, encoded, pred
    report['summary'] = {name: {ref: {metric: float(np.mean([s['metrics'][name][ref][metric] for s in report['samples']]))
                                    for metric in ('abs_rel', 'rmse', 'delta1')}
                               for ref in ('sensor', 'teacher')} for name in names}
    report.update(status='complete', finished_at=time.time())
    atomic_json(report, output / 'data.json')
    return report


def flatten_metric_tree(row):
    result = {}
    for region in ('all', 'plane_region', 'nonplane_region'):
        for key, value in row[region].items():
            if key != 'pixels':
                result[f'{region}/{key}'] = value
    for region, values in row['projected_normals'].items():
        for key, value in values.items():
            if key != 'stencils':
                result[f'projected_normals/{region}/{key}'] = value
    for key in ('fixed_gt_plane_rmse_m', 'gt_to_fixed_plane_rmse_m', 'self_fit_rmse_m',
                'gt_self_fit_rmse_m', 'self_fit_rmse_over_gt_median_depth',
                'fit_normal_error_deg', 'fit_offset_error_m'):
        planes = [p for p in row['planes'] if p[key] is not None]
        if planes:
            result['planes/' + key] = sum(p[key] * p['pixels'] for p in planes) / sum(p['pixels'] for p in planes)
    return result


def means(rows):
    flat = [flatten_metric_tree(r) for r in rows]
    keys = set().union(*(set(r) for r in flat))
    return {k: float(np.mean([r[k] for r in flat if k in r])) for k in sorted(keys)}


def laser(system, adapter, config, records, output, archive, expected_final_rows, expected_initial_rows):
    output.mkdir()
    data = GroundTruthImages(records, augment=False)
    results = {name: [] for name in ('student', 'parent', 'teacher')}
    # First pass uses new adapter and original teacher; second loads the archived
    # adapter once solely for this new physical evaluation. No optimizer exists.
    for phase in ('current_and_teacher', 'archive'):
        if phase == 'archive':
            state = torch.load(archive, map_location='cpu', weights_only=True, mmap=True)
            assert state['step'] == 232288
            assert state['config']['models'] == config['models']
            assert state['config']['adapter'] == config['adapter']
            adapter.load_state_dict(state['adapter'], strict=True)
            del state
            gc.collect()
        for i, record in enumerate(records):
            h, w = record['height'], record['width']
            item = data[i, 0, h, w]
            target = {k: v.numpy() for k, v in item.items() if k != 'image'}
            batch = {k: v[None].cuda() for k, v in item.items()}
            image = batch.pop('image')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                if phase == 'current_and_teacher':
                    features, teacher = system.encode(image, adapter.indices)
                    encoded = [('student', adapter(features)), ('teacher', teacher)]
                else:
                    features = extract_radio(system.radio, radio_input(image, system.radio, None), adapter.indices)
                    encoded = [('parent', adapter(features))]
            path = output / f"{record['id']}.npz"
            arrays = {'rgb': (item['image'].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8),
                      **target}
            if phase == 'archive':
                with np.load(path, allow_pickle=False) as packed:
                    arrays = {k: packed[k].copy() for k in packed.files}
            for name, features_out in encoded:
                raw = system.decoder(features_out, (h, w))
                decoded = calibrated_geometry(raw, batch['intrinsics'], batch['valid'])
                depth = decoded['depth'][0].float().cpu().numpy()
                raw_xyz = decoded['points'][0].float().cpu().numpy()
                physical = physical_geometry_metrics(depth, raw_xyz, target)
                legacy = depth_metrics(decoded['depth'], batch['depth'], batch['valid'])
                expected = expected_final_rows[i] if name == 'student' else expected_initial_rows[i]
                if name != 'teacher':
                    assert expected['id'] == record['id']
                    for key in ('abs_rel', 'rmse', 'invalid_fraction'):
                        assert abs(legacy[key] - expected[key]) < 1e-6, (name, record['id'], key)
                results[name].append({'id': record['id'], 'scene': record['scene'],
                                      'legacy_depth': legacy, **physical})
                arrays.update({name + '_depth': depth, name + '_raw_xyz': raw_xyz,
                               name + '_projected_xyz': decoded['projected_points'][0].float().cpu().numpy()})
            np.savez_compressed(path, **arrays)
            print(f'Laser {phase} {i + 1}/{len(records)}', flush=True)
    for name in results:
        assert [r['id'] for r in results[name]] == [r['id'] for r in records]
    scenes = sorted({r['scene'] for r in records})
    summary = {name: means(rows) for name, rows in results.items()}
    per_scene = {scene: {name: means([r for r in rows if r['scene'] == scene])
                         for name, rows in results.items()} for scene in scenes}
    report = dict(stage='complete', images=len(records), scenes=len(scenes),
                  per_image=results, summary=summary, per_scene=per_scene,
                  protocol={'camera': 'known measured K with own-output z-shift recovery',
                            'GT_scale_shift_alignment': False, 'support': 'all measured valid GT; no prediction mask',
                            'normal': 'projected XYZ, oriented five-point stride-one GT-valid stencil,3% GTedge filter',
                            'aggregation': 'equal image; each image plane metrics weighted by fixed GT pixel count',
                            'scope': 'repeated heldout validation, not independent test; plane0 includes unannotated planes',
                            'flatness': 'self-fit RMS on fixed GT plane ROIs; also inspect metric scale/offset/XYZ errors'},
                  production_updates=0)
    atomic_json(report, output / 'geometry.json')
    return report


@torch.inference_mode()
def export(args):
    torch.set_num_threads(1)
    output, pilot = args.output, args.pilot
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / 'inference-attempt.json').exists(), 'Do not repeat this inference'
    proof = json.loads((pilot / 'completion-verification.json').read_text())
    assert proof['stage'] == 'passed' and proof['final_step'] == 1000
    checkpoint = pilot / 'train/latest.pt'
    assert identity(checkpoint) == proof['checkpoint_identity']
    archive_proof = json.loads(args.archive_proof.read_text())
    assert archive_proof['stage'] == 'verified' and archive_proof['source_step'] == 232288
    archive = Path(archive_proof['archive'])
    assert archive.stat().st_size == archive_proof['bytes']
    archive_before = identity(archive)
    for step in (0, 250, 500, 750, 1000):
        for kind in ('data', 'wandb'):
            assert json.loads((pilot / f'eval-{step:07d}-{kind}-verification.json').read_text())['stage'] == 'passed'
    atomic_json(dict(started_at=time.time(), checkpoint=str(checkpoint), checkpoint_step=1000,
                     archive=str(archive), source_identity=identity(checkpoint),
                     archive_identity=archive_before, production_updates=0), output / 'inference-attempt.json')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    assert state['step'] == 1000 and state['cursor'] == 16000
    config = state['config']
    assert config == json.loads((pilot / 'train/config.json').read_text())
    assert state['runtime_limits']['steps'] == 1000 and state['runtime_limits']['max_seconds'] == 7200
    provenance = json.loads((pilot / 'train/provenance.json').read_text())
    assert state['data_digests'] == provenance['data_digests']
    assert state['revisions'] == provenance['revisions']
    records = read_manifest(config['data']['geometry_val_manifest'])
    validate_gt_records(records, split='val')
    assert len(records) == 18 and len({r['scene'] for r in records}) == 7
    system, adapter = build_system(config, 'cuda')
    adapter.load_state_dict(state['adapter'], strict=True)
    adapter.eval().requires_grad_(False)
    del state
    gc.collect()
    review = json.loads((pilot / 'eval-0001000-feature-alignment-review.json').read_text())
    cosine = review['views']['native-640x400']['metrics']['spatial/cosine']['current']
    if args.completed_fixed20 is None:
        fixed = fixed20(system, adapter, config, args.reference_export, output / 'fixed20', checkpoint, cosine)
    else:
        fixed = json.loads((args.completed_fixed20 / 'data.json').read_text())
        assert fixed['status'] == 'complete' and fixed['checkpoint_step'] == 1000
        assert len(fixed['samples']) == 20
        assert Path(fixed['checkpoint']) == checkpoint
    def rows(step):
        return json.loads((pilot / f'train/eval-{step:07d}/gt-laser-known-camera/per_image.json').read_text())
    measured = laser(system, adapter, config, records, output / 'laser', archive, rows(1000), rows(0))
    assert identity(checkpoint) == proof['checkpoint_identity']
    assert identity(archive) == archive_before
    atomic_json(dict(stage='complete', finished_at=time.time(), checkpoint_step=1000,
                     fixed20_images=len(fixed['samples']), laser_images=measured['images'],
                     source_identity_unchanged=True, archive_unchanged=True,
                     production_updates=0, fixed20_parent_teacher_reused=True,
                     fixed20_existing_export=str(args.completed_fixed20) if args.completed_fixed20 else None),
                output / 'completion.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot', 'reference-export', 'archive-proof', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--completed-fixed20', type=Path)
    export(parser.parse_args())
