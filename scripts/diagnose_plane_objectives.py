"""Read-only loss/gradient diagnosis on five prespecified images and saved adapters.

No optimizer is constructed. Frozen decoder/backbones and checkpoints are unchanged.
Gradients are single-image Euclidean gradients, not reconstructed Adam updates.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from radio_moge.checkpoint import atomic_json
from radio_moge.data import evaluation_image
from radio_moge.decoder import camera_points, geometry_points_with_grad, postprocess
from radio_moge.losses import FeatureStatistics
from radio_moge.planes import PlaneConfig, detect_planes, fit_plane, plane_losses
from radio_moge.system import build_system


def identity(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def variance_decomposition(points, normal):
    p = np.asarray(points, dtype=np.float64)
    x = p - p.mean(0)
    cov = x.T @ x / len(x)
    values, vectors = np.linalg.eigh(cov)
    total = np.trace(cov)
    fixed_axis = float(normal @ cov @ normal / total)
    intrinsic = float(max(0, values[0]) / total)
    orientation = float(sum((values[i] - values[0]) * (normal @ vectors[:, i]) ** 2
                            for i in (1, 2)) / total)
    assert abs(fixed_axis - intrinsic - orientation) < 1e-10
    return dict(flatness_loss=fixed_axis, intrinsic_variance=intrinsic,
                orientation_excess=orientation,
                orientation_fraction=orientation / max(fixed_axis, 1e-15),
                own_plane_normalized_rms=float(np.sqrt(max(0, values[0]) /
                                                       max(values[1:].sum(), 1e-15))),
                teacher_normal_angle_deg=float(np.rad2deg(np.arccos(np.clip(
                    abs(normal @ vectors[:, 0]), 0, 1)))))


def gram_report(vectors):
    names = list(vectors)
    norms = {name: float(vector.norm()) for name, vector in vectors.items()}
    cosines = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            denominator = norms[a] * norms[b]
            cosines[a + ':' + b] = (float(torch.dot(vectors[a], vectors[b])) / denominator
                                    if denominator > 1e-20 else None)
    total = sum(vectors.values())
    total_norm = float(total.norm())
    # Positive means descending the combined loss locally descends this term too.
    alignment = {name: float(torch.dot(v, total)) / max(norms[name] * total_norm, 1e-20)
                 for name, v in vectors.items()}
    return dict(weighted_gradient_l2=norms, pairwise_cosine=cosines,
                cosine_to_total=alignment, total_gradient_l2=total_norm)


def inspect(system, adapter, stats, image, gt, cfg, patches, target_features, target_geometry):
    with torch.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
        features, _ = system.encode(image, adapter.indices, num_tokens=1200)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        predicted = adapter(features)
    raw = system.decoder(predicted, image.shape[-2:])
    points = geometry_points_with_grad(raw, cfg['models'].get('moge_source'))
    official = postprocess(raw, image, cfg['models'].get('moge_source'))
    mismatch = float((points.detach() - official['points']).abs().max())
    assert torch.allclose(points.detach(), official['points'], atol=2e-5, rtol=2e-5)
    feature = stats.loss(predicted, target_features)
    plane = plane_losses(points, target_geometry['points'], [patches])
    active = {'feature': sum(feature[k] * cfg['loss'][k] for k in feature)}
    active.update({k: v * cfg['loss'][k] for k, v in plane.items()})
    names, parameters = zip(*[(n, p) for n, p in adapter.named_parameters() if p.requires_grad])
    vectors, details = {}, {}
    mask = torch.as_tensor(np.logical_or.reduce([p.mask for p in patches]), device='cuda')
    for name, loss in active.items():
        gradients = torch.autograd.grad(loss, (*parameters, predicted.spatial,
                                              predicted.summary, raw['points']),
                                        retain_graph=True, allow_unused=True)
        vectors[name] = torch.cat([(torch.zeros_like(p) if g is None else g).float().reshape(-1)
                                   for p, g in zip(parameters, gradients[:-3], strict=True)])
        modules = {}
        for n, g in zip(names, gradients[:-3], strict=True):
            group = n.split('.')[0]
            modules[group] = modules.get(group, 0.) + (float(g.float().square().sum()) if g is not None else 0)
        details[name] = dict(weighted_loss=float(loss.detach()),
                            module_gradient_l2={k: v ** .5 for k, v in modules.items()},
                            spatial_output_gradient_l2=float(gradients[-3].norm()) if gradients[-3] is not None else 0,
                            summary_output_gradient_l2=float(gradients[-2].norm()) if gradients[-2] is not None else 0,
                            raw_points_gradient_outside_masks_l2=float(gradients[-1][0][~mask].norm())
                            if gradients[-1] is not None else None)
    gradient = gram_report(vectors)
    del gradients
    rows = []
    for number, patch in enumerate(patches, 1):
        decomposition = variance_decomposition(points.detach()[0].cpu().numpy()[patch.mask], patch.normal)
        rows.append(dict(region=number, **decomposition))
    # A diagnostic-only differentiable own-best-plane objective at the target mask.
    # Eigenvectors of a detached covariance give the exact first derivative of its
    # distinct minimum eigenvalue. This does not add a training loss or update weights.
    region = patches[0]
    support = points[0][torch.as_tensor(region.mask, device='cuda')]
    x = support - support.mean(0)
    with torch.no_grad():
        _, v = torch.linalg.eigh((x.T @ x / len(x)).double())
        n = v[:, 0].float()
    eigen_min = (x @ n).square().mean()
    own_shape = eigen_min / (x.square().sum(-1).mean() - eigen_min).clamp_min(1e-12)
    shape_gradients = torch.autograd.grad(own_shape, parameters, allow_unused=True)
    shape_norm = float(torch.stack([g.float().square().sum() for g in shape_gradients
                                    if g is not None]).sum().sqrt())
    shape_vector = torch.cat([(torch.zeros_like(p) if g is None else g).float().reshape(-1)
                              for p, g in zip(parameters, shape_gradients, strict=True)])
    shape_alignment = {name: float(torch.dot(shape_vector, v)) /
                       max(shape_norm * float(v.norm()), 1e-20) for name, v in vectors.items()}
    total_vector = sum(vectors.values())
    shape_alignment['total'] = float(torch.dot(shape_vector, total_vector)) / max(
        shape_norm * float(total_vector.norm()), 1e-20)
    del vectors, shape_vector, shape_gradients, total_vector
    scales = {}
    with torch.no_grad():
        baseline = system.decoder.scale_head(predicted.summary.float()).squeeze(1).exp()
        for factor in (.8, 1., 1.2):
            scale = system.decoder.scale_head(predicted.summary.float() * factor).squeeze(1).exp()
            ratio = float(scale / baseline)
            metric = variance_decomposition(points.detach()[0].cpu().numpy()[region.mask] * ratio,
                                            region.normal)
            scales[str(factor)] = dict(metric_scale_ratio=ratio, **metric)
    return dict(losses={k: float(v.detach()) for k, v in {**feature, **plane}.items()},
                terms=details, adapter_gradients=gradient, regions=rows,
                target_region=1, own_shape_gradient_l2=shape_norm,
                target_shape_gradient_cosine=shape_alignment, summary_amplitude_counterfactual=scales,
                differentiable_points_vs_official_max_abs=mismatch)


def run(root, output):
    if (output / 'status.json').exists():
        raise FileExistsError('Diagnostic already launched; inspect status instead')
    output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((root / 'plane-config.json').read_text())
    selection = []
    for part, count in [('train-probe', 2), ('validation', 2), ('screenshot', 1)]:
        records = list(map(json.loads, (root / f'{part}.jsonl').read_text().splitlines()))[:count]
        selection += [dict(partition=part, record=r) for r in records]
    checkpoints = {'initial': root / 'plane/initial.pt', 'control': root / 'control/latest.pt',
                   'plane': root / 'plane/latest.pt'}
    tracked = [*checkpoints.values(), *[Path(s['record'][k]) for s in selection
                                      for k in ('image', 'depth') if k in s['record']]]
    identities = {str(p): identity(p) for p in tracked}
    started = time.time()
    state = dict(stage='starting', pid=__import__('os').getpid(), started_at=started,
                 optimizer_updates=0, selection=selection, completed=[])
    atomic_json(state, output / 'status.json')
    with (root.parent / 'attention-24h/.training.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                          '--format=csv,noheader'], text=True).strip()
        assert not active, 'GPU has an existing process; no diagnostic launched'
        torch.set_num_threads(2)
        system, adapter = build_system(cfg, 'cuda')
        stats = FeatureStatistics(adapter.config['target_dim'], adapter.config['target_summary_dim']).cuda()
        references = []
        for sample in selection:
            image, gt = evaluation_image(sample['record'], 640)
            image = image.cuda()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                _, target = system.encode(image, adapter.indices, num_tokens=1200)
            with torch.no_grad():
                teacher = postprocess(system.decoder(target, image.shape[-2:]), image)
            patches = detect_planes(teacher['points'][0].cpu().numpy(), teacher['mask'][0].cpu().numpy(),
                                     PlaneConfig(**cfg['planes']))
            assert patches
            labels = np.zeros(image.shape[-2:], dtype=np.uint8)
            reliability = []
            gt_points = camera_points(gt[None].cuda(), teacher['intrinsics'])[0].cpu().numpy() if gt is not None else None
            for number, patch in enumerate(patches, 1):
                labels[patch.mask] = number
                item = dict(region=number, **patch.metadata(), teacher_normalized_rms=patch.rms / patch.extent)
                if gt_points is not None:
                    valid = patch.mask & (gt.numpy() > 0)
                    n, _, ev = fit_plane(gt_points[valid])
                    item.update(gt_valid_fraction=float(valid.sum()/patch.mask.sum()),
                                gt_own_plane_rms=float(np.sqrt(ev[0]/ev[1:].sum())),
                                teacher_gt_normal_angle_deg=float(np.rad2deg(np.arccos(np.clip(abs(n@patch.normal),0,1)))))
                reliability.append(item)
            with (output / f"masks-{sample['record']['id']}.npz").open('wb') as handle:
                np.savez_compressed(handle, rgb=(image[0].cpu().permute(1, 2, 0).numpy()*255).astype('uint8'),
                                    labels=labels)
            references.append((image, gt, target, teacher, patches))
            sample['teacher_mask_sha256'] = hashlib.sha256(labels.tobytes()).hexdigest()
            sample['teacher_planes'] = reliability
        results = []
        for arm, path in checkpoints.items():
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            adapter.load_state_dict(checkpoint['adapter'])
            stats.load_state_dict(checkpoint['statistics'])
            adapter.eval()
            versions = [p._version for p in adapter.parameters()]
            del checkpoint
            for sample, reference in zip(selection, references, strict=True):
                if time.time()-started > 900:
                    raise TimeoutError('Diagnostic exceeded fixed 15-minute bound')
                image, gt, target, teacher, patches = reference
                result = inspect(system, adapter, stats, image, gt, cfg, patches, target, teacher)
                result.update(arm=arm, id=sample['record']['id'], partition=sample['partition'])
                results.append(result)
                atomic_json(result, output / f"{arm}-{result['id']}.json")
                state.update(stage='diagnosing', updated_at=time.time())
                state['completed'].append([arm, result['id']])
                atomic_json(state, output/'status.json')
                print(json.dumps(dict(arm=arm, id=result['id'], gradients=result['adapter_gradients'])), flush=True)
            assert versions == [p._version for p in adapter.parameters()], 'Adapter changed'
        assert all(identity(p)==v for p,v in identities.items())
        report = dict(stage='passed', selection=selection, results=results, source_identities=identities,
                      source_files_unchanged=True, model_parameters_unchanged=True, optimizer_updates=0,
                      elapsed_seconds=time.time()-started, limitations=[
                          'Five prespecified diagnostic images, not a population estimate.',
                          'Single-image gradients in eval mode; not actual accumulated Adam updates.',
                          'Camera recovery uses the existing detached focal/shift estimates.',
                          'GT-plane check uses teacher intrinsics and is corroborative, not independent calibration.'])
        atomic_json(report, output/'gradient-verification.json')
        state.update(stage='complete', finished_at=time.time())
        atomic_json(state, output/'status.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.root, args.output)
    except Exception as error:
        path = args.output / 'status.json'
        if path.exists():
            state = json.loads(path.read_text())
            state.update(stage='needs_inspection', error=str(error), updated_at=time.time())
            atomic_json(state, path)
        raise
