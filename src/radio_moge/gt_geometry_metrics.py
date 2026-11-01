"""Evaluation-only physical geometry metrics on fixed measured GT support.

No prediction mask or fitted GT scale/shift changes the evaluated pixels.
Self-fitted plane RMS measures flatness only; fixed-GT-plane RMS also measures
position, orientation and scale. Neither is a replacement for whole-image XYZ.
"""
import numpy as np


def metric_camera_points(depth, intrinsics):
    """Metric XYZ from camera-Z and normalized pixel-center pinhole intrinsics."""
    depth = np.asarray(depth, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if depth.ndim != 2 or intrinsics.shape != (3, 3):
        raise ValueError('Expected H,W depth and 3,3 camera intrinsics')
    if (not np.isfinite(intrinsics).all() or np.any(np.diag(intrinsics)[:2] <= 0)
            or not np.allclose(intrinsics[2], [0, 0, 1])
            or intrinsics[0, 1] != 0 or intrinsics[1, 0] != 0):
        raise ValueError('Expected finite normalized pinhole intrinsics without skew')
    height, width = depth.shape
    u, v = np.meshgrid((np.arange(width) + .5) / width,
                       (np.arange(height) + .5) / height)
    rays = np.stack(((u - intrinsics[0, 2]) / intrinsics[0, 0],
                     (v - intrinsics[1, 2]) / intrinsics[1, 1], np.ones_like(u)), -1)
    return rays * depth[..., None]


def _region(depth, raw_xyz, projected_xyz, truth_depth, truth_xyz, support):
    count = int(support.sum())
    if not count:
        return {'pixels': 0}
    error = depth[support] - truth_depth[support]
    raw_error = raw_xyz[support] - truth_xyz[support]
    projected_error = projected_xyz[support] - truth_xyz[support]
    return {
        'pixels': count,
        'abs_rel': float(np.mean(np.abs(error) / truth_depth[support])),
        'depth_rmse_m': float(np.sqrt(np.mean(error ** 2))),
        # Euclidean 3D RMSE, not coordinate-wise RMSE divided by sqrt(3).
        'raw_xyz_rmse_m': float(np.sqrt(np.mean(np.sum(raw_error ** 2, axis=-1)))),
        'projected_xyz_rmse_m': float(np.sqrt(np.mean(np.sum(projected_error ** 2, axis=-1)))),
        'nonpositive_depth_fraction': float(np.mean(depth[support] <= 0)),
    }


def _normals(points):
    dx = points[1:-1, 2:] - points[1:-1, :-2]
    dy = points[2:, 1:-1] - points[:-2, 1:-1]
    return np.cross(dx, dy)


def _normal_metrics(predicted_xyz, truth_xyz, truth_depth, valid, labels, edge_ratio):
    # Stride one: every point in the five-point stencil must be measured and
    # lie on the same side of a GT depth edge. No interpolation through holes.
    center = truth_depth[1:-1, 1:-1]
    support = valid[1:-1, 1:-1].copy()
    offsets = [(slice(1, -1), slice(2, None)), (slice(1, -1), slice(None, -2)),
               (slice(2, None), slice(1, -1)), (slice(None, -2), slice(1, -1))]
    for offset in offsets:
        support &= valid[offset] & (np.abs(truth_depth[offset] - center)
                                    <= edge_ratio * np.minimum(truth_depth[offset], center))
    gt_normal, pred_normal = _normals(truth_xyz), _normals(predicted_xyz)
    gt_length, pred_length = np.linalg.norm(gt_normal, axis=-1), np.linalg.norm(pred_normal, axis=-1)
    support &= gt_length > 1e-12
    results = {}
    for name, region in [('all', np.ones_like(support)),
                         ('plane_region', labels[1:-1, 1:-1] > 0),
                         ('nonplane_region', labels[1:-1, 1:-1] == 0)]:
        selected = support & region
        if not selected.any():
            results[name] = {'stencils': 0}
            continue
        # A collapsed predicted stencil receives the maximum angular penalty,
        # rather than disappearing from evaluation or being reported as zero.
        good = pred_length[selected] > 1e-12
        angles = np.full(int(selected.sum()), 180., dtype=np.float64)
        numerator = np.sum(gt_normal[selected][good] * pred_normal[selected][good], axis=-1)
        cosine = numerator / (gt_length[selected][good] * pred_length[selected][good])
        angles[good] = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        results[name] = {'stencils': int(selected.sum()),
                         'mean_angle_deg': float(angles.mean()),
                         'median_angle_deg': float(np.median(angles)),
                         'p90_angle_deg': float(np.quantile(angles, .9)),
                         'degenerate_prediction_fraction': float(np.mean(~good))}
    return results


def _fit_plane(points):
    center = points.mean(axis=0)
    centered = points - center
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / len(points))
    normal = eigenvectors[:, 0]
    return {'self_fit_rmse_m': float(np.sqrt(max(0., eigenvalues[0]))),
            'normal': normal, 'offset': -float(center @ normal),
            'in_plane_variance_m2': float(eigenvalues[1])}


def physical_geometry_metrics(depth, raw_xyz, target, edge_ratio=.03):
    """Evaluate one calibrated prediction; input XYZ and depth already in metres.

    `raw_xyz` preserves decoder XY; projection through measured K is evaluated
    separately. Nonfinite predictions on GT support raise an explicit error.
    GT plane masks/equations must be fixed before any predictions are examined.
    """
    if not 0 < edge_ratio < 1:
        raise ValueError('Expected a relative GT edge threshold in (0,1)')
    depth, raw_xyz = np.asarray(depth, dtype=np.float64), np.asarray(raw_xyz, dtype=np.float64)
    truth_depth = np.asarray(target['depth'], dtype=np.float64)
    valid = np.asarray(target['valid'], dtype=bool)
    labels, planes = np.asarray(target['plane_labels']), np.asarray(target['planes'], dtype=np.float64)
    if (depth.ndim != 2 or min(depth.shape) < 3 or truth_depth.shape != depth.shape
            or valid.shape != depth.shape or labels.shape != depth.shape
            or raw_xyz.shape != (*depth.shape, 3) or planes.shape != (6, 4)):
        raise ValueError('Mismatched geometry dimensions')
    if not valid.any() or not (np.isfinite(truth_depth[valid]) & (truth_depth[valid] > 0)).all():
        raise ValueError('Fixed GT support must contain positive finite measured depth')
    if not (np.isfinite(depth[valid]).all() and np.isfinite(raw_xyz[valid]).all()):
        raise FloatingPointError('Nonfinite prediction on fixed GT support')
    if (not np.isfinite(planes).all() or not np.isin(labels, np.arange(7)).all()
            or np.any((labels > 0) & ~valid)):
        raise ValueError('Invalid fixed GT plane annotation')
    # Holes are irrelevant and never gain support. Zeroing prevents NaNs in
    # unsupported locations from contaminating neighbouring arithmetic.
    truth_depth = np.where(valid, truth_depth, 0)
    depth = np.where(valid, depth, 0)
    raw_xyz = np.where(valid[..., None], raw_xyz, 0)
    truth_xyz = metric_camera_points(truth_depth, target['intrinsics'])
    projected_xyz = metric_camera_points(depth, target['intrinsics'])
    result = {name: _region(depth, raw_xyz, projected_xyz, truth_depth, truth_xyz, support)
              for name, support in [('all', valid), ('plane_region', valid & (labels > 0)),
                                    ('nonplane_region', valid & (labels == 0))]}
    result['projected_normals'] = _normal_metrics(
        projected_xyz, truth_xyz, truth_depth, valid, labels, edge_ratio)
    result['planes'] = []
    for label in np.unique(labels[valid]):
        if not label:
            continue
        mask = valid & (labels == label)
        equation = planes[int(label) - 1]
        if not np.isclose(np.linalg.norm(equation[:3]), 1., atol=1e-4):
            raise ValueError('Fixed GT plane normal must have unit length')
        predicted, measured = projected_xyz[mask], truth_xyz[mask]
        residual = predicted @ equation[:3] + equation[3]
        gt_residual = measured @ equation[:3] + equation[3]
        pred_fit, gt_fit = _fit_plane(predicted), _fit_plane(measured)
        cosine = float(pred_fit['normal'] @ equation[:3])
        sign = 1 if cosine >= 0 else -1
        identifiable = len(predicted) >= 3 and pred_fit['in_plane_variance_m2'] > 1e-12
        result['planes'].append({
            'label': int(label), 'pixels': int(mask.sum()),
            'fixed_gt_plane_rmse_m': float(np.sqrt(np.mean(residual ** 2))),
            'gt_to_fixed_plane_rmse_m': float(np.sqrt(np.mean(gt_residual ** 2))),
            'self_fit_rmse_m': pred_fit['self_fit_rmse_m'],
            'gt_self_fit_rmse_m': gt_fit['self_fit_rmse_m'],
            'self_fit_rmse_over_gt_median_depth': pred_fit['self_fit_rmse_m']
            / float(np.median(truth_depth[mask])),
            'fit_identifiable': bool(identifiable),
            'fit_normal_error_deg': float(np.degrees(np.arccos(np.clip(abs(cosine), 0, 1))))
            if identifiable else None,
            'fit_offset_error_m': abs(sign * pred_fit['offset'] - equation[3])
            if identifiable else None,
        })
    return result
