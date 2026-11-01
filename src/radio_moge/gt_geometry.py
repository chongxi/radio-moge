"""Measured, calibrated geometry supervision (never teacher pseudo geometry).

The camera is known during supervision. A differentiable least-squares z shift
is inferred from the decoder's own XYZ and those camera rays, NOT from GT depth.
Metric scale remains predicted by the decoder. GT alone chooses valid support;
the predicted mask cannot hide errors. No per-image GT scale/shift alignment.
"""
import torch
import torch.nn.functional as F

from .decoder import camera_points


GT_LOSS_KEYS = ('gt_depth', 'gt_points', 'gt_gradient', 'gt_normal', 'gt_plane')


def image_mean(value, mask):
    """Mean over supported pixels, then images; an empty image is a data error."""
    mask = torch.broadcast_to(mask, value.shape)
    dims = tuple(range(1, value.ndim))
    count = mask.sum(dims)
    supported = count > 0
    means = torch.where(mask, value, 0).sum(dims) / count.clamp_min(1)
    return means.sum() / supported.sum().clamp_min(1)


def calibrated_geometry(raw, intrinsics, valid):
    """Project raw decoder outputs through known normalized pixel-center K.

    This is a supervised-camera training path, not the unknown-camera inference
    protocol. That existing inference/evaluation protocol remains unchanged.
    """
    with torch.autocast(raw['points'].device.type, enabled=False):
        points = raw['points'].float()
        if points.shape[:-1] != valid.shape or points.shape[-1] != 3:
            raise ValueError('Expected matching B,H,W,3 points and B,H,W support')
        if not torch.isfinite(points).all():
            raise FloatingPointError('Nonfinite predicted geometry')
        scale = raw['metric_scale'].float()
        if scale.shape != (len(points),) or not (torch.isfinite(scale) & (scale > 0)).all():
            raise FloatingPointError('Invalid predicted metric scale')
        rays = camera_points(torch.ones_like(points[..., 2]), intrinsics.float())
        xy = rays[..., :2]
        radius = xy.square().sum(-1)
        denominator = torch.where(valid, radius, 0).sum((1, 2))
        if (denominator <= 1e-8).any():
            raise ValueError('Insufficient camera support for z-shift recovery')
        numerator = (xy * points[..., :2]).sum(-1) - radius * points[..., 2]
        shift = torch.where(valid, numerator, 0).sum((1, 2)) / denominator
        depth = (points[..., 2] + shift[:, None, None]) * scale[:, None, None]
        # Supervise raw XY too: otherwise arbitrary XY fields could recover a
        # wrong focal at inference while matching depth under the supplied K.
        metric_points = torch.cat((points[..., :2] * scale[:, None, None, None],
                                   depth[..., None]), dim=-1)
        return {'depth': depth, 'points': metric_points,
                'projected_points': rays * depth[..., None], 'shift': shift}


def ground_truth_losses(raw, target, edge_ratio=.03, strides=(1, 2, 4)):
    """Dimensionless, per-image balanced robust errors on metric GT geometry.

    `gt_plane` is distance to fixed GT-fitted planes, so it constrains flatness,
    orientation AND position. Points/depth must remain active against collapse.
    Derivative/normal stencils never cross GT holes or GT depth discontinuities.
    Normals are derived from predicted points, not an independent normal head.
    """
    depth = target['depth'].detach().float()
    valid = target['valid'].bool() & torch.isfinite(depth) & (depth > 0)
    if depth.ndim != 3 or not valid.flatten(1).any(1).all():
        raise ValueError('Every GT image requires valid positive metric depth')
    depth = torch.where(valid, depth, 0)
    predicted = calibrated_geometry(raw, target['intrinsics'], valid)
    truth = camera_points(depth, target['intrinsics'].detach().float())
    p = torch.where(valid[..., None], predicted['projected_points'], 0)
    raw_points = torch.where(valid[..., None], predicted['points'], 0)
    scales = torch.stack([d[m].median() for d, m in zip(depth, valid, strict=True)]).clamp_min(.01)
    norm = scales[:, None, None, None]
    error = F.smooth_l1_loss(raw_points / norm, truth / norm, reduction='none', beta=.05)
    result = {'gt_points': image_mean(error, valid[..., None])}
    result['gt_depth'] = image_mean(F.smooth_l1_loss(
        predicted['depth'] / scales[:, None, None], depth / scales[:, None, None],
        reduction='none', beta=.05), valid)

    gradients, normals = [], []
    for stride in strides:
        if min(depth.shape[-2:]) <= 2 * stride:
            continue
        for dim in (1, 2):
            left = [slice(None)] * 3
            right = [slice(None)] * 3
            left[dim], right[dim] = slice(None, -stride), slice(stride, None)
            left, right = tuple(left), tuple(right)
            z0, z1 = depth[left], depth[right]
            support = valid[left] & valid[right]
            support &= (z0 - z1).abs() <= edge_ratio * torch.minimum(z0, z1)
            difference = (p[right] - p[left] - truth[right] + truth[left]) / norm
            gradients.append(image_mean(difference.abs(), support[..., None]))
        center = (slice(None), slice(stride, -stride), slice(stride, -stride))
        offsets = [(slice(None), slice(stride, -stride), slice(2 * stride, None)),
                   (slice(None), slice(stride, -stride), slice(None, -2 * stride)),
                   (slice(None), slice(2 * stride, None), slice(stride, -stride)),
                   (slice(None), slice(None, -2 * stride), slice(stride, -stride))]
        support = valid[center].clone()
        for offset in offsets:
            support &= valid[offset] & ((depth[offset] - depth[center]).abs()
                                       <= edge_ratio * depth[center])
        def cross(points):
            dx = (points[offsets[0]] - points[offsets[1]]) / norm
            dy = (points[offsets[2]] - points[offsets[3]]) / norm
            return torch.linalg.cross(dx, dy, dim=-1)
        gt_cross, pred_cross = cross(truth), cross(p)
        support &= gt_cross.norm(dim=-1) > 1e-10
        cosine = F.cosine_similarity(pred_cross, gt_cross, dim=-1, eps=1e-10)
        normals.append(image_mean(1 - cosine.clamp(-1, 1), support))
    zero = p.sum() * 0
    result['gt_gradient'] = torch.stack(gradients).mean() if gradients else zero
    result['gt_normal'] = torch.stack(normals).mean() if normals else zero
    labels, planes = target['plane_labels'].long(), target['planes'].detach().float()
    if labels.shape != depth.shape or planes.shape != (len(depth), 6, 4):
        raise ValueError('Expected B,H,W plane labels and B,6,4 GT plane equations')
    if ((labels < 0) | (labels > 6)).any() or not torch.isfinite(planes).all():
        raise ValueError('Invalid GT plane metadata')
    equations = torch.gather(planes, 1, (labels.flatten(1).clamp_min(1) - 1)[..., None]
                             .expand(-1, -1, 4)).reshape(*depth.shape, 4)
    support = valid & (labels > 0)
    residual = (predicted['projected_points'] * equations[..., :3]).sum(-1) + equations[..., 3]
    result['gt_plane'] = image_mean(F.smooth_l1_loss(
        residual / scales[:, None, None], torch.zeros_like(depth),
        reduction='none', beta=.01), support)
    result['gt_valid_fraction'] = valid.float().mean()
    result['gt_plane_fraction'] = support.float().mean()
    return result
