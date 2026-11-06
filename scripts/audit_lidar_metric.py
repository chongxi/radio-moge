"""Read-only, independent audit of the frozen iPhone LiDAR evaluation.

This publishes diagnostic evidence, never edits an input or replaces a metric.
The model phase has no optimizer and stores arrays for independent CPU review.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def reference_metrics(predicted, target):
    """Independent float64 implementation; target support cannot depend on prediction."""
    p, t = np.asarray(predicted, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError('Prediction and target grids differ')
    valid = np.isfinite(t) & (t > 0)
    if not valid.any():
        raise ValueError('Empty sensor support')
    p, t = p[valid], t[valid]
    bad = ~np.isfinite(p) | (p <= 0)
    p = np.where(bad, 0., p)
    error = p-t
    result = dict(abs_rel=float(np.mean(np.abs(error)/t)),
        rmse=float(np.sqrt(np.mean(error**2))),
        delta1=float(np.mean((p/t > .8) & (p/t < 1.25))),
        invalid_fraction=float(bad.mean()), pixels=int(t.size),
        signed_relative_bias=float(np.mean(error/t)))
    positive = p > 0
    scale = float(np.median(t[positive]/p[positive])) if positive.any() else None
    result['diagnostic_median_scale'] = scale
    result['diagnostic_scaled_abs_rel'] = (
        float(np.mean(np.abs(scale*p-t)/t)) if scale is not None else None)
    return result


def independent_nearest(array, output_hw):
    h, w = array.shape
    oh, ow = output_hw
    y = np.minimum(h-1, np.floor((np.arange(oh)+.5)*h/oh).astype(int))
    x = np.minimum(w-1, np.floor((np.arange(ow)+.5)*w/ow).astype(int))
    return array[y[:, None], x[None, :]]


def ref(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(path):
    return json.loads(Path(path).read_text())


def publish(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(path.name+'.incoming-'+str(os.getpid()))
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')
    os.rename(tmp, path)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def common(args):
    root = args.root.resolve()
    config = read(root/'config.json')
    spec = config['evaluation']['dataset_variants']['scannetpp-lidar']
    manifest = Path(spec['manifest'])
    records = rows(manifest)
    parent = Path(read(root/'completed.json')['source']['path'])
    parent_root = parent.parent.parent
    paths = dict(parent=parent_root/'training/eval-0246249/dataset-scannetpp-lidar',
                 final=root/'training/eval-0258946/dataset-scannetpp-lidar')
    logs = {name: read(path/'per_image.json') for name, path in paths.items()}
    if len(records) != 120 or set(Counter(r['scene'] for r in records).values()) != {4}:
        raise ValueError('Unexpected frozen 30 scene / 120 image population')
    for name in logs:
        if [r['image'] for r in logs[name]] != [r['image'] for r in records]:
            raise ValueError('Frozen evaluation order differs')
    return root, config, spec, manifest, records, paths, logs


def cpu(args):
    from PIL import Image
    from radio_moge.data import evaluation_image
    from radio_moge.metrics import depth_metrics
    import torch
    torch.set_num_threads(1)
    root, config, spec, manifest, records, paths, logs = common(args)
    output = args.output/'cpu'
    output.mkdir(parents=True, exist_ok=False)
    evidence = []
    for record in records:
        for key, digest in [('image', 'sha256'), ('depth', 'depth_sha256')]:
            if ref(record[key])['sha256'] != record[digest]:
                raise ValueError('Frozen payload digest mismatch: '+record[key])
        depth = np.load(record['depth'], allow_pickle=False)
        with Image.open(record['image']) as im:
            if im.size != (768, 576) or im.getexif().get(274, 1) != 1:
                raise ValueError('Unexpected orientation or prepared image grid')
        if (depth.shape != (576, 768) or depth.dtype != np.float32
                or not np.isfinite(depth).all() or np.any(depth < 0)
                or record['depth_scale'] != 1.0):
            raise ValueError('Invalid frozen metric-depth payload')
        image, gt = evaluation_image(record, spec['max_side'])
        expected = independent_nearest(depth, gt.shape)
        unequal = expected != gt.numpy()
        if tuple(gt.shape) != (480, 640) or tuple(image.shape) != (1, 3, 480, 640):
            raise ValueError('Unexpected actual evaluation dimensions')
        values = gt.numpy()[gt.numpy() > 0]
        k = np.asarray(record['alignment']['output_intrinsic'])
        evidence.append(dict(id=record['id'], image=record['image'], scene=record['scene'],
            frame_index=record['frame_index'], depth_sha256=record['depth_sha256'],
            rgb_sha256=record['sha256'], actual_gt_sha256=hashlib.sha256(gt.numpy().tobytes()).hexdigest(),
            sensor_valid_pixels=int(values.size), sensor_support_fraction=float(values.size/gt.numel()),
            depth_quantiles_metres=np.quantile(values, [0,.01,.5,.99,1]).tolist(),
            independent_nearest_differing_pixels=int(unequal.sum()),
            independent_nearest_max_difference=float(np.max(np.abs(expected-gt.numpy()))),
            known_focal_normalized=(k[0,0]/768).item(),
            known_fov_x_degrees=float(np.degrees(2*np.arctan(768/(2*k[0,0])))),
            source_intrinsic_unused_in_prediction=True, has_confidence_payload='confidence' in record))
    # Deliberately exercise invalid predictions and invalid targets independently.
    p = np.array([1., 2., np.nan, -1., 0., 9., np.inf, 1.25], np.float32)
    t = np.array([1., 1., 2., 2., 0., np.nan, 4., 1.], np.float32)
    independent = reference_metrics(p, t)
    actual = depth_metrics(torch.from_numpy(p), torch.from_numpy(t), torch.from_numpy(t > 0))
    fixture_differences = {k: actual[k]-independent[k] for k in actual}
    if max(map(abs, fixture_differences.values())) > 2e-7:
        raise ValueError('Independent metric fixture differs')
    aggregate = {}
    for name, data in logs.items():
        metric = read(paths[name]/'metrics.json')
        values = np.asarray([d['gt_student/abs_rel'] for d in data], np.float64)
        mean = float(values.mean())
        if not np.isclose(mean, metric['gt_student/abs_rel'], rtol=0, atol=1e-15):
            raise ValueError('Saved aggregate differs from its 120 per-image values')
        aggregate[name] = dict(saved=metric['gt_student/abs_rel'], independent_image_mean=mean,
            count=len(values), postprocess_failures=sum(d['postprocess_failed'] for d in data),
            invalid_prediction_fraction=max(d['gt_student/invalid_fraction'] for d in data),
            per_image=ref(paths[name]/'per_image.json'), metrics=ref(paths[name]/'metrics.json'))
    publish(output/'images.json', evidence)
    report = dict(stage='cpu_lidar_metric_audit_complete', created_at=datetime.now(timezone.utc).isoformat(),
        manifest=ref(manifest), images=120, scenes=30, all_rgb_and_depth_payload_hashes_exact=True,
        depth_units='metres', evaluation_hw=[480,640], native_sensor_hw=[192,256],
        aggregation='equal image mean; four images per scene, hence also equal scene mean',
        independent_aggregate=aggregate, metric_fixture_differences=fixture_differences,
        nearest_disagreement_pixels=sum(r['independent_nearest_differing_pixels'] for r in evidence),
        details=ref(output/'images.json'), model_used=False, optimizer_updates=0,
        physical_sensor_registration_proven=False, sensor_confidence_not_available=True,
        notes=['No GT-dependent alignment or scale fitting in the published metric.',
               'Known sensor intrinsics are not supplied to MoGe focal recovery.',
               'No range, confidence or depth-edge mask beyond finite positive sensor support.',
               'No conclusion about physical accuracy or real-wall flatness follows from arithmetic checks.'],
        auditor=ref(__file__))
    publish(output/'report.json', report)
    print(json.dumps(dict(report=ref(output/'report.json'), nearest_disagreement_pixels=report['nearest_disagreement_pixels'], aggregate=aggregate)), flush=True)


def model(args):
    import torch
    from radio_moge.data import evaluation_image
    from radio_moge.decoder import postprocess
    from radio_moge.metrics import depth_metrics
    from radio_moge.system import build_system
    import copy
    torch.set_num_threads(1)
    torch.manual_seed(43)
    root, config, spec, manifest, records, paths, logs = common(args)
    output = args.output/'model'
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    completion = read(root/'completed.json')
    checkpoints = dict(parent=completion['source'], final=read(root/'finish-verification-01/report.json')['checkpoint'])
    system, first_adapter = build_system(config, 'cuda')
    adapters = dict(parent=first_adapter, final=copy.deepcopy(first_adapter))
    for name, adapter in adapters.items():
        state = torch.load(checkpoints[name]['path'], map_location='cpu', weights_only=True, mmap=True)
        adapter.load_state_dict(state['adapter'], strict=True)
        adapter.eval().requires_grad_(False)
        del state
    results = []
    for i, record in enumerate(records):
        if time.monotonic()-began > 1500:
            raise TimeoutError('Bounded metric audit exceeded 1500 seconds')
        image, gt = evaluation_image(record, spec['max_side'])
        image = image.to('cuda')
        autocast = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if config['train']['precision'] == 'bf16' else nullcontext()
        with torch.no_grad(), autocast():
            features, target_features = system.encode(image, first_adapter.indices, num_tokens=spec['num_tokens'])
        target = system.decoder(target_features, image.shape[-2:])
        teacher = postprocess(target, image, config['models'].get('moge_source'))
        sensor = gt.numpy()
        arrays = dict(sensor=sensor, teacher=teacher['depth'][0].cpu().numpy())
        item = dict(id=record['id'], image=record['image'], scene=record['scene'], models={})
        item['teacher'] = reference_metrics(arrays['teacher'], sensor)
        item['teacher_saved_abs_rel_difference'] = item['teacher']['abs_rel'] - logs['final'][i]['gt_teacher/abs_rel']
        for name, adapter in adapters.items():
            with torch.no_grad(), autocast():
                adapted = adapter(features)
            raw = system.decoder(adapted, image.shape[-2:])
            prediction = postprocess(raw, image, config['models'].get('moge_source'))
            arrays[name] = prediction['depth'][0].cpu().numpy()
            independent = reference_metrics(arrays[name], sensor)
            actual = depth_metrics(prediction['depth'], gt.to('cuda')[None], (gt.to('cuda')>0)[None])
            differences = {k: actual[k]-independent[k] for k in actual}
            valid=np.isfinite(sensor)&(sensor>0)
            ratio=arrays[name][valid].astype(np.float64)/sensor[valid]
            boundary_pixels=int(np.count_nonzero((np.abs(ratio-.8)<2e-7)|(np.abs(ratio-1.25)<2e-7)))
            tolerances=dict(abs_rel=2e-6, rmse=2e-6, invalid_fraction=1e-7,
                delta1=boundary_pixels/int(valid.sum())+1e-7)
            if any(abs(differences[k]) > tolerances[k] for k in differences):
                raise ValueError('Actual tensor metric differs from independent array formula')
            item['models'][name] = dict(independent=independent, tensor_metrics=actual,
                tensor_minus_float64=differences, saved_abs_rel=logs[name][i]['gt_student/abs_rel'],
                delta1_float32_boundary_pixels=boundary_pixels,
                rerun_minus_saved_abs_rel=actual['abs_rel']-logs[name][i]['gt_student/abs_rel'],
                metric_scale=float(raw['metric_scale'].item()),
                inferred_focal_normalized=float(prediction['intrinsics'][0,0,0].item()))
        destination=output/f'frame-{i:03d}.npz'
        with destination.with_suffix('.incoming').open('xb') as f:
            np.savez_compressed(f, **arrays)
        os.rename(destination.with_suffix('.incoming'), destination)
        item['arrays']=ref(destination)
        results.append(item)
        if (i+1)%20 == 0:
            print(json.dumps(dict(completed_images=i+1, elapsed_seconds=time.monotonic()-began)), flush=True)
    summary = {}
    for name in adapters:
        summary[name] = dict(
            independent_abs_rel=float(np.mean([r['models'][name]['independent']['abs_rel'] for r in results])),
            tensor_abs_rel=float(np.mean([r['models'][name]['tensor_metrics']['abs_rel'] for r in results])),
            saved_abs_rel=float(np.mean([r['gt_student/abs_rel'] for r in logs[name]])),
            max_rerun_saved_absrel_difference=max(abs(r['models'][name]['rerun_minus_saved_abs_rel']) for r in results),
            max_tensor_float64_absrel_difference=max(abs(r['models'][name]['tensor_minus_float64']['abs_rel']) for r in results),
            diagnostic_scaled_abs_rel=float(np.mean([r['models'][name]['independent']['diagnostic_scaled_abs_rel'] for r in results])),
            median_scale_correction=float(np.median([r['models'][name]['independent']['diagnostic_median_scale'] for r in results])))
    publish(output/'per-image.json', results)
    report=dict(stage='independent_actual_prediction_lidar_metric_audit_complete',
        created_at=datetime.now(timezone.utc).isoformat(), images=120, checkpoint_count=2,
        checkpoint_refs=checkpoints, checkpoint_sha_reused_from_existing_closure=True,
        summary=summary, per_image=ref(output/'per-image.json'), manifest=ref(manifest),
        absrel_rmse_tensor_vs_float64_tolerance=2e-6, tensor_formula_matches=True,
        mean_teacher_abs_rel=float(np.mean([r['teacher']['abs_rel'] for r in results])),
        max_teacher_saved_abs_rel_difference=max(abs(r['teacher_saved_abs_rel_difference']) for r in results),
        diagnostic_scale_alignment_is_not_replacement_metric=True,
        physical_sensor_registration_proven=False, geometry_training=False, optimizer_updates=0,
        elapsed_seconds=time.monotonic()-began, peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,
        auditor=ref(__file__))
    publish(output/'report.json', report)
    print(json.dumps(dict(report=ref(output/'report.json'), summary=summary)), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['cpu','model'],required=True)
    args=parser.parse_args()
    (cpu if args.mode=='cpu' else model)(args)
