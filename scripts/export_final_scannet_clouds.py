"""One final232288 fixed20 inference; reuse the exact94367 and MoGe-2 outputs."""

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
from radio_moge.metrics import depth_metrics
from radio_moge.pointcloud import verify_geometry
from radio_moge.system import build_system
from radio_moge.upstream import extract_radio, radio_input
from scannet_cloud_cache import cached_reference


@torch.inference_mode()
def export(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / 'inference-attempt.json').exists(), 'Do not repeat an inference'
    assert not (args.output / 'data.json').exists()
    old_path = args.reference_export / 'data.json'
    raw = old_path.read_bytes()
    old = json.loads(raw)
    old_proof = json.loads((args.reference_export / 'completion.json').read_text())
    assert old_proof['stage'] == 'complete' and old['status'] == 'complete'
    assert old['checkpoint_step'] == 94367 and old['names'] == ['student', 'parent', 'teacher']
    assert len(old['samples']) == len(set(old['selected_ids'])) == 20
    assert [s['id'] for s in old['samples']] == old['selected_ids']
    assert all(not any(c['overlap'].values()) for c in old['split_checks'])
    complete = json.loads((args.training_root / 'full-epoch-completion-verification.json').read_text())
    assert complete['all_checks_passed'] and complete['final_step'] == 232288
    assert complete['normal_exit_code'] == 0 and complete['coverage']['complete']
    expected = complete['latest_checkpoint_metadata_only']
    before = args.checkpoint.stat()
    assert [before.st_size, before.st_mtime_ns] == [expected['bytes'], expected['mtime_ns']]
    assert str(args.checkpoint) == expected['remote_path']
    atomic_json(dict(started_at=time.time(), checkpoint=str(args.checkpoint),
                     checkpoint_step=232288, production_updates=0), args.output / 'inference-attempt.json')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    provenance = json.loads((args.training_root / 'train/provenance.json').read_text())
    assert state['step'] == 232288 and state['cursor'] == 409480
    assert state['data_digests'] == provenance['data_digests']
    assert state['revisions'] == provenance['revisions'] == old['revisions']
    config = state['config']
    assert config == json.loads((args.training_root / 'train/config.json').read_text())
    assert config['data']['buckets'] == [[480, 640], [640, 480], [400, 640]]
    preflight = json.loads((args.training_root / 'preflight.json').read_text())
    assert preflight['stage'] == 'passed'
    names = ['student', 'parent', 'teacher']
    report = dict(status='initializing', started_at=time.time(), checkpoint_step=232288,
                  checkpoint=str(args.checkpoint), selection_metric='completed final epoch',
                  selection_value=0.9601821339163893,
                  labels=['当前模型 · 232,288步', '上次展示 · 94,367步', 'MoGe-2 · 原始模型'],
                  names=names, selected_ids=old['selected_ids'], samples=[], backend='pytorch',
                  model_config=config['models'], revisions=state['revisions'],
                  data_digests=state['data_digests'], split_checks=old['split_checks'],
                  split_proof_reused_from=str(args.reference_export / 'completion.json'),
                  heldout_preservation_proof=str(args.training_root / 'preflight.json'),
                  fixed20_scope='Previously viewed NYU test scenes, preserved by verified dataset lineage; not an unseen benchmark.',
                  cached_reference_sha256=hashlib.sha256(raw).hexdigest(),
                  parent_teacher_reused=True, parent_checkpoint_step=94367,
                  new_model_inference_only=True, production_updates=0,
                  evaluation=dict(num_tokens=None, native_size_wh=[640, 400], spatial_tokens=1000),
                  coordinates=old['coordinates'], projection=old['projection'])
    report['view'] = dict(
        title='当前模型 × 上次模型 × MoGe-2',
        subtitle='同样20个场景 · 同步三维视角 · 640×400',
        badge_label='完整验证集 Spatial cosine · 640×400',
        checkpoint_label='当前模型 · 第232,288步',
        weights='左侧是完成本轮全数据训练的232,288步最终权重，中间是上次页面展示的94,367步权重，右侧是固定版本的原始MoGe-2。这里按当前最终权重展示，不按这20张图挑选模型；94,367步也不是本轮的直接父权重。',
        protocol='三者使用完全相同的640×400中心裁剪RGB。RADIO输入640×400、网格25×40。上次模型与MoGe-2的完整深度、相机和掩码直接复用已验证输出；只有当前模型运行新推理。默认使用各自恢复的内参、掩码和米制尺度，不做点云配准或尺度拟合。',
        caveat='以下是此前已查看的20个固定NYU演示场景，排除在训练与1268张选权重验证集之外；不能替代完整独立测试。右上角cosine来自1268张验证图，本页深度指标仅对应这20张演示图。')
    atomic_json({k: v for k, v in report.items() if k != 'samples'}, args.output / 'status.json')
    system, adapter = build_system(config, 'cuda')
    adapter.load_state_dict(state['adapter'], strict=True)
    adapter.eval().requires_grad_(False)
    del state
    gc.collect()
    for i, original in enumerate(old['samples']):
        saved = cached_reference(args.reference_export, original)
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
        np.savez_compressed(args.output / f"{sample['id']}.npz", rgb=rgb, sensor_depth=gt.numpy(),
                            student_depth=depths[0], parent_depth=depths[1], teacher_depth=depths[2],
                            student_mask=masks[0], parent_mask=masks[1], teacher_mask=masks[2],
                            cameras=np.asarray(cameras))
        report['samples'].append(sample)
        atomic_json(dict(status='exporting', completed=i + 1, total=20, updated_at=time.time()), args.output / 'status.json')
        print(f'Exported final232288 {i + 1}/20; cached94367 and teacher unchanged', flush=True)
        del features, encoded, pred
    report['summary'] = {name: {ref: {metric: float(np.mean([s['metrics'][name][ref][metric] for s in report['samples']]))
                                    for metric in ('abs_rel', 'rmse', 'delta1')}
                               for ref in ('sensor', 'teacher')} for name in names}
    after = args.checkpoint.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    report.update(status='complete', finished_at=time.time(), checkpoint_unchanged=True)
    atomic_json(report, args.output / 'data.json')
    atomic_json(dict(status='complete', completed=20, total=20, checkpoint_step=232288,
                     finished_at=report['finished_at'], summary=report['summary'], backend='pytorch',
                     max_projection_error_m=max(s['projection_error_m'] for s in report['samples']),
                     parent_teacher_reused=True, cached_outputs_exact=True, new_model_inference_only=True,
                     production_updates=0, checkpoint_unchanged=True), args.output / 'status.json')
    print(json.dumps(dict(status='complete', summary=report['summary'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'training-root', 'reference-export', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    export(parser.parse_args())
