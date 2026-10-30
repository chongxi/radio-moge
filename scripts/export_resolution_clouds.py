"""New cosine-best inference with unchanged parent and MoGe-2 geometry reused."""

import argparse
import base64
import gc
import gzip
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from radio_moge.checkpoint import atomic_json
from radio_moge.decoder import postprocess
from radio_moge.metrics import depth_metrics
from radio_moge.pointcloud import verify_geometry
from radio_moge.system import build_system
from radio_moge.upstream import extract_radio, radio_input


def cached_sample(root, sample):
    """Reject changed cached RGB/geometry instead of silently mixing comparisons."""
    sid = sample['id']
    assert Path(sid).name == sid
    with np.load(root / f'{sid}.npz', allow_pickle=False) as data:
        saved = {k: data[k].copy() for k in data.files}
    rgb = np.array(Image.open(io.BytesIO(base64.b64decode(sample['rgb'].split(',', 1)[1]))))
    assert np.array_equal(rgb, saved['rgb']) and rgb.shape == (400, 640, 3)
    assert saved['sensor_depth'].shape == (400, 640)
    assert np.isfinite(saved['sensor_depth']).all()
    assert sample['feature_grid_hw'] == [25, 40]
    assert sample['preprocessing']['crop_xyxy'] == [0, 40, 640, 440]
    assert np.array_equal(saved['cameras'], np.asarray(sample['cameras']))
    stride = sample['stride']
    blob = b''.join(np.asarray(saved[n+'_depth'][::stride, ::stride], dtype='<f4').tobytes()
                    for n in ('student', 'teacher'))
    blob += b''.join(np.asarray(saved[n+'_mask'][::stride, ::stride], dtype=np.uint8).tobytes()
                     for n in ('student', 'teacher'))
    assert blob == gzip.decompress(base64.b64decode(sample['payload']))
    return saved


@torch.inference_mode()
def export(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output/'data.json').exists(), 'Completed export must never be repeated'
    old = json.loads((args.parent_export/'data.json').read_text())
    old_proof = json.loads((args.parent_export/'completion.json').read_text())
    assert old_proof['stage'] == old['status'] == 'complete' and old['checkpoint_step'] == 91367
    assert old['names'] == ['student', 'teacher']
    assert len(old['samples']) == len(set(old['selected_ids'])) == 20
    assert [s['id'] for s in old['samples']] == old['selected_ids']
    assert all(not any(c['overlap'].values()) for c in old['split_checks'])
    proof = json.loads((args.training_root/'final-review/backup-verification.json').read_text())
    assert proof['stage'] == 'passed'
    source = json.loads((args.training_root/'final-review/source-verification.json').read_text())
    selected = source['independent_selections']
    assert selected['spatial']['step'] == selected['640x400-spatial']['step'] == 94367
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    assert state['step'] == 94367 and state['cursor'] == 743996
    assert state['data_digests'] == json.loads((args.training_root/'train/provenance.json').read_text())['data_digests']
    assert state['revisions'] == old['revisions']
    assert state['fork_provenance']['step'] == old['checkpoint_step']
    assert Path(state['fork_provenance']['checkpoint']) == Path(old['checkpoint'])
    config = state['config']
    assert config['data']['buckets'] == [[480, 640], [640, 480], [400, 640]]
    names = ['student', 'parent', 'teacher']
    report = dict(status='initializing', started_at=time.time(), checkpoint_step=94367,
                  checkpoint=str(args.checkpoint), selection_metric=selected['640x400-spatial']['metric'],
                  selection_value=selected['640x400-spatial']['value'],
                  labels=['混合分辨率 · 94,367步', '训练前 · 91,367步', 'MoGe-2 · 原始模型'],
                  names=names, selected_ids=old['selected_ids'], samples=[], backend='pytorch',
                  model_config=config['models'], revisions=state['revisions'],
                  data_digests=state['data_digests'], split_checks=old['split_checks'],
                  split_proof_reused_from=str(args.parent_export/'completion.json'),
                  parent_teacher_reused=True, new_model_inference_only=True,
                  evaluation=dict(num_tokens=None, native_size_wh=[640, 400], spatial_tokens=1000),
                  original_cosine=selected['spatial']['value'], independent_selections=selected,
                  coordinates=old['coordinates'], projection=old['projection'])
    report['view'] = dict(
        title='640×400 · 混合训练前后 × MoGe-2', subtitle='同样20个场景 · 同步三维视角 · 原生640×400',
        badge_label='640×400验证 spatial cosine', checkpoint_label='第94,367步 · 两种视图cos均最佳',
        weights='左侧为加入640×400后训练3,000步的C-RADIOv4-H + Attention适配器 + 冻结MoGe-2解码器。按1,268张验证图的spatial cosine选择第94,367步；原尺寸和640×400的cos均为本阶段最佳。中间为训练前91,367步，右侧为固定版本原始MoGe-2。640×400的NYU验证AbsRel最低的是93,367步，已另行保留；本页展示cos最佳权重。',
        protocol='三者使用完全相同的640×400中心裁剪RGB与传感器深度，RGB不缩放或拉伸。RADIO输入640×400，网格25×40，MoGe-2沿用原始DINO预处理与同一网格。训练前及MoGe-2的完整深度、相机与掩码直接复用上次结果；本次只对新权重运行PyTorch推理。各自相机、掩码和米制尺度，不做配准或尺度对齐。',
        caveat='以下是此前已查看的20个固定演示场景，不属于训练集或1,268张选权重验证集，也不代表完整测试集。最终116张NYU验证图上，640×400 AbsRel改善12.7%，但原尺寸AbsRel退步3.0%；不能声称整体精度全面提高。')
    atomic_json({k: v for k, v in report.items() if k != 'samples'}, args.output/'status.json')
    # Same manifests as the initial comparison, established by the verified fork.
    startup = json.loads((args.training_root/'startup-verification.json').read_text())
    assert startup['stage'] == 'passed'
    system, adapter = build_system(config, 'cuda')
    adapter.load_state_dict(state['adapter'], strict=True)
    adapter.eval().requires_grad_(False)
    del state
    gc.collect()
    for i, original in enumerate(old['samples']):
        saved = cached_sample(args.parent_export, original)
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
        cameras = [camera, *original['cameras']]
        metrics = {}
        for name, d, valid in zip(names, depths, masks, strict=True):
            assert valid.any() and np.isfinite(d[valid]).all() and (d[valid] > 0).all()
            metrics[name] = dict(sensor=depth_metrics(torch.from_numpy(d), gt, gt > 0),
                                 teacher=depth_metrics(torch.from_numpy(d), torch.from_numpy(depths[2]), torch.from_numpy(masks[2])))
            assert all(np.isfinite(v) for ref in metrics[name].values() for v in ref.values())
        for new_name, old_name in [('parent', 'student'), ('teacher', 'teacher')]:
            for ref in ('sensor', 'teacher'):
                for key, value in original['metrics'][old_name][ref].items():
                    assert metrics[new_name][ref][key] == value
        stride = original['stride']
        blob = b''.join(np.asarray(d[::stride, ::stride], dtype='<f4').tobytes() for d in depths)
        blob += b''.join(np.asarray(m[::stride, ::stride], dtype=np.uint8).tobytes() for m in masks)
        valid_depths = np.concatenate([d[m] for d, m in zip(depths, masks)])
        sample = dict(original, cameras=cameras, full_valid_points=[int(m.sum()) for m in masks],
                      range=np.quantile(valid_depths, [.02, .98]).tolist(), metrics=metrics,
                      projection_error_m=max(error, original['projection_error_m']),
                      payload=base64.b64encode(gzip.compress(blob, compresslevel=6, mtime=0)).decode(),
                      cached_parent_teacher_verified=True)
        np.savez_compressed(args.output/f"{sample['id']}.npz", rgb=rgb, sensor_depth=gt.numpy(),
                            student_depth=depths[0], parent_depth=depths[1], teacher_depth=depths[2],
                            student_mask=masks[0], parent_mask=masks[1], teacher_mask=masks[2],
                            cameras=np.asarray(cameras))
        report['samples'].append(sample)
        atomic_json(dict(status='exporting', completed=i+1, total=20, updated_at=time.time()), args.output/'status.json')
        print(f'New model exported {i+1}/20; parent and teacher cached geometry verified', flush=True)
        del features, encoded, pred
    report['summary'] = {name: {ref: {metric: float(np.mean([s['metrics'][name][ref][metric] for s in report['samples']]))
                                    for metric in ('abs_rel', 'rmse', 'delta1')}
                               for ref in ('sensor', 'teacher')} for name in names}
    report.update(status='complete', finished_at=time.time())
    atomic_json(report, args.output/'data.json')
    atomic_json(dict(status='complete', completed=20, total=20, checkpoint_step=94367,
                     finished_at=report['finished_at'], summary=report['summary'], backend='pytorch',
                     max_projection_error_m=max(s['projection_error_m'] for s in report['samples']),
                     parent_teacher_reused=True, cached_outputs_exact=True, new_model_inference_only=True), args.output/'status.json')
    print(json.dumps(dict(status='complete', summary=report['summary'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'training-root', 'parent-export', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    export(parser.parse_args())
