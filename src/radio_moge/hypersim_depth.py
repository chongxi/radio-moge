"""Hypersim range conversion with independent per-pixel world-position checks."""
import numpy as np

from .geometry_sources import hypersim_pinhole_frame, hypersim_pose_rotation, pinhole_rays, range_to_z


def half_ulp(values):
    """Bound storage quantization, rather than demanding impossible FP16 accuracy."""
    values = np.asarray(values)
    if values.dtype.kind != 'f':
        raise ValueError('Expected floating-point source geometry')
    with np.errstate(over='ignore', invalid='ignore'):
        high = np.nextafter(values, np.array(np.inf, dtype=values.dtype)).astype(np.float64)
        low = np.nextafter(values, np.array(-np.inf, dtype=values.dtype)).astype(np.float64)
        value = values.astype(np.float64)
        return .5 * np.maximum(high - value, value - low)


def convert_verified_range(radial, world, orientation, position, meters_per_asset_unit, ray_matrix):
    radial, world = np.asarray(radial), np.asarray(world)
    if world.shape != (*radial.shape, 3) or radial.ndim != 2 or meters_per_asset_unit <= 0:
        raise ValueError('Mismatched Hypersim world positions/range or invalid scale')
    k, frame = hypersim_pinhole_frame(ray_matrix)
    rotation, inherited_scale = hypersim_pose_rotation(orientation)
    cam = (world.astype(np.float64) - np.asarray(position)) @ rotation * meters_per_asset_unit
    cam *= np.array([1, -1, -1])
    truth = cam @ frame.T
    rays = pinhole_rays(k, *radial.shape)
    depth, support = range_to_z(radial, rays,
                               np.isfinite(truth).all(-1) & (truth[..., 2] > 0))
    predicted_xyz = depth[..., None] * rays
    error = np.linalg.norm(predicted_xyz - truth, axis=-1)
    # Position and range are independently stored as FP16 by the publisher.
    # Use their actual dtype ULPs. No GT-fitted calibration/scale, no interpolation.
    tolerance = 2 * (np.linalg.norm(half_ulp(world), axis=-1) * meters_per_asset_unit
                      + half_ulp(radial)) + 1e-4
    valid = support & np.isfinite(tolerance) & (error <= tolerance)
    stats = dict(original_finite_positive_fraction=float(support.mean()),
                 valid_fraction=float(valid.mean()),
                 inconsistent_supported_pixels=int((support & ~valid).sum()),
                 supported_pixels=int(support.sum()),
                 inherited_pose_uniform_scale=inherited_scale,
                 frame_rotation=frame.tolist())
    if support.any():
        stats['xyz_error_quantiles_m'] = np.quantile(error[support], [0, .5, .99, 1]).tolist()
        bounded = support & np.isfinite(tolerance)
        stats['unbounded_storage_uncertainty_pixels'] = int((support & ~bounded).sum())
        stats['quantization_bound_quantiles_m'] = (np.quantile(tolerance[bounded],
            [0, .5, .99, 1]).tolist() if bounded.any() else None)
    depth[~valid] = 0
    return depth, valid, k, stats
