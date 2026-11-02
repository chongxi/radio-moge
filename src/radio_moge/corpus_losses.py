"""Provenance-aware MoGe-2 core losses at exact cached measurement rays.

The pinned global/local/edge/scale FUNCTIONS remain upstream. Camera rectification,
unique-observation sampling, conservative edge support and pseudo routing are our
adapter, NOT a claim to reproduce the complete paper's training pipeline.
"""
import torch
import torch.nn.functional as F

from .official_gt import OFFICIAL_GT_KEYS, official_functions


def target_tensors(target, device):
    return {key: torch.as_tensor(value, device=device)
            if key in ('points', 'query_uv', 'valid', 'intrinsics', 'native_index') else value
            for key, value in target.items()}


def sample_predictions(points, target):
    """Only the predicted field is interpolated; targets stay exact observations."""
    valid, uv = target['valid'], target['query_uv'].detach().float()
    if (points.ndim != 3 or points.shape[-1] != 3
            or tuple(points.shape[:2]) != tuple(target['output_shape'])
            or valid.dtype != torch.bool or uv.shape != (*valid.shape, 2)):
        raise ValueError('Prediction/query dimensions differ')
    height, width = points.shape[:2]
    low = uv.new_tensor([.5 / width, .5 / height])
    if (not torch.isfinite(uv[valid]).all()
            or not ((uv[valid] >= low - 1e-7) & (uv[valid] <= 1 - low + 1e-7)).all()):
        raise ValueError('Measured ray would require prediction extrapolation')
    uv = torch.where(valid[..., None], uv, .5)
    sampled = F.grid_sample(points.float().permute(2, 0, 1)[None],
                            (uv * 2 - 1)[None], align_corners=False,
                            mode='bilinear', padding_mode='zeros')[0].permute(1, 2, 0)
    if not torch.isfinite(sampled[valid]).all():
        raise FloatingPointError('Nonfinite prediction at a supervised ray')
    return torch.where(valid[..., None], sampled, 1.)


def stable_edge_support(truth, valid, absolute=.05, relative=.02):
    """Both endpoints must be valid and no incident valid edge may cross a jump."""
    support = valid.clone()
    depth = truth[..., 2]
    for axis in (0, 1):
        a, b = [slice(None)] * 2, [slice(None)] * 2
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        invalid_edge = (valid[a] & valid[b]
                        & ((depth[a] - depth[b]).abs()
                           > absolute + relative * torch.minimum(depth[a], depth[b])))
        support[a] &= ~invalid_edge
        support[b] &= ~invalid_edge
    return support


def corpus_geometry_losses(raw, targets, moge_source=None, seed=0):
    """Image-balanced losses; pseudo scale is discarded, not metrically trusted.

    A sparse/empty image (<32 observations) contributes zero, with an explicit
    counter. Final admission must separately bound this rate; it is not silently
    converted to dense GT. Invalid target/predicted query values are sanitized
    BEFORE calling upstream code, including before products with zero masks.
    """
    points = raw['points']
    if len(targets) != len(points):
        raise ValueError('One geometry target is required per RGB image')
    functions = official_functions(moge_source)
    devices = [points.device.index] if points.is_cuda else []
    values = {key: [] for key in OFFICIAL_GT_KEYS}
    used, metric_used, pixels, edges = 0, 0, 0, 0
    with torch.autocast(points.device.type, enabled=False), torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for index, source in enumerate(targets):
            target = target_tensors(source, points.device)
            mode = target['supervision']
            if (mode not in ('metric_native', 'relative_pseudo')
                    or (mode == 'relative_pseudo') != (target['label_kind'] == 'teacher_pseudo')):
                raise ValueError('Geometry loss/provenance routing mismatch')
            mask = target['valid'].bool()
            truth = target['points'].detach().float()
            if (truth.shape != (*mask.shape, 3) or not torch.isfinite(truth[mask]).all()
                    or not (truth[..., 2][mask] > 0).all()):
                raise ValueError('Invalid supported geometry')
            ids = target['native_index'][mask]
            if (ids < 0).any() or len(ids) != len(torch.unique(ids)):
                raise ValueError('Repeated or invalid native observation IDs')
            k = target['intrinsics'].detach().float()
            if (k.shape != (3, 3) or not torch.isfinite(k).all()
                    or not torch.allclose(k[:2, 2], k.new_tensor([.5, .5]), atol=1e-6, rtol=0)
                    or not (torch.diag(k) > 0).all()):
                raise ValueError('Loss requires the verified centered camera')
            # A differentiable zero with no dependence on invalid raw values.
            zero = points[index].reshape(-1)[:0].sum()
            if mask.sum() < 32:
                for key in values:
                    values[key].append(zero)
                continue
            used += 1
            pixels += int(mask.sum())
            truth = torch.where(mask[..., None], truth, 1.)
            prediction = sample_predictions(points[index], target)
            if mode == 'relative_pseudo':
                # Remove the teacher's nominal scale before any absolute clamp
                # in the official functions or our support calculation.
                truth = truth / truth[..., 2][mask].median()
                truth = torch.where(mask[..., None], truth, 1.)
            focal = (k[0, 0].reciprocal().square() + k[1, 1].reciprocal().square()).rsqrt()
            fn, params = functions['gt_global']
            global_loss, _, scale, _ = fn(prediction, truth, mask, **params)
            values['gt_global'].append(global_loss)
            for key in ('gt_local4', 'gt_local16'):
                fn, params = functions[key]
                values[key].append(fn(prediction, truth, mask, focal, scale.detach(), **params)[0])
            edge_mask = stable_edge_support(truth, mask)
            edges += int((edge_mask[1:] & edge_mask[:-1]).sum()
                         + (edge_mask[:, 1:] & edge_mask[:, :-1]).sum())
            fn, params = functions['gt_edge']
            values['gt_edge'].append(fn(prediction, truth, edge_mask, **params)[0])
            if mode == 'metric_native':
                scale_pred = raw['metric_scale'][index].float()
                if not torch.isfinite(scale_pred) or scale_pred <= 0:
                    raise FloatingPointError('Invalid predicted metric scale')
                fn, params = functions['gt_scale']
                values['gt_scale'].append(fn(scale_pred, scale.detach(), **params)[0])
                metric_used += 1
            else:
                values['gt_scale'].append(zero)
        result = {key: torch.stack(value).mean() for key, value in values.items()}
        if not all(torch.isfinite(value) for value in result.values()):
            raise FloatingPointError('Nonfinite corpus geometry loss')
        result.update(gt_supervised_images=points.new_tensor(float(used)),
                      gt_skipped_sparse_images=points.new_tensor(float(len(points) - used)),
                      gt_metric_images=points.new_tensor(float(metric_used)),
                      gt_observations=points.new_tensor(float(pixels)),
                      gt_edge_pairs=points.new_tensor(float(edges)))
    return result
