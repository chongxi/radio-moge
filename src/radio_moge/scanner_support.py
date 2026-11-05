"""Frozen conditional scanner corroboration, not independent physical truth.

The centimeter residual budget is an operational filter, not a calibration
bound. Sparse development/confirmation probes never grant dense image support.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ScannerSupportContract:
    schema_version: int = 1
    label_kind: str = 'laser_mesh_derived'
    selected_station_count: int = 2
    maximum_nominal_range_residual_m: float = .01
    exact_target_numerical_tolerance_m: float = 1e-5
    required_raw_pixels_per_station: int = 9
    minimum_dense_image_support: float = .05
    minimum_dense_image_points: int = 32
    physical_accuracy_bound_claimed: bool = False

    def __post_init__(self):
        if asdict(self) != dict(schema_version=1, label_kind='laser_mesh_derived',
                selected_station_count=2, maximum_nominal_range_residual_m=.01,
                exact_target_numerical_tolerance_m=1e-5,
                required_raw_pixels_per_station=9, minimum_dense_image_support=.05,
                minimum_dense_image_points=32, physical_accuracy_bound_claimed=False):
            raise ValueError('Frozen conditional support policy required; no runtime tuning')


def exact_target_visibility(target, pose, hit, contract=ScannerSupportContract()):
    """Classify a new unshifted whole-mesh ray to the exact retained target.

    Pose validation and the actual closest query belong to the bound producer.
    Different coincident faces and target triangle boundaries stay unresolved.
    """
    pose = np.asarray(pose, np.float64)
    point = np.asarray(target['point_world'], np.float64)
    vertices = np.asarray(target['vertices_world'], np.float64)
    if pose.shape != (4, 4) or point.shape != (3,) or vertices.shape != (3, 3):
        raise ValueError('Exact target/pose shapes required')
    if not all(np.isfinite(x).all() for x in (pose, point, vertices)):
        raise ValueError('Finite exact target geometry required')
    local = np.linalg.solve(pose[:3, :3], point-pose[:3, 3])
    radius = float(np.linalg.norm(local))
    if radius <= 0 or not math.isfinite(radius):
        raise ValueError('Target cannot coincide with scanner origin')
    normal = np.cross(vertices[1]-vertices[0], vertices[2]-vertices[0])
    length = float(np.linalg.norm(normal))
    if length <= 0:
        raise ValueError('Nondegenerate original target triangle required')
    normal /= length
    tol = contract.exact_target_numerical_tolerance_m
    if abs((point-vertices[0]) @ normal) > tol:
        raise ValueError('Target not on its retained triangle')
    edge_distances = []
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        edge_distances.append(float(np.cross(b-a, point-a) @ normal/np.linalg.norm(b-a)))
    if min(edge_distances) < -tol:
        raise ValueError('Target outside its retained triangle')
    result = dict(target_local_radial_m=radius, first_hit=hit,
        triangle_boundary_clearance_m=min(edge_distances),
        exact_target_first_surface=False, physical_accuracy_accepted=False,
        physical_pose_uncertainty_bounded=False, continuous_footprint_visibility_proven=False,
        unique_first_face_proven=False, coincident_unreturned_faces_exhaustively_checked=False,
        measured_range_at_exact_target_direction=False)
    if hit is None:
        return dict(result, status='exact_target_ray_no_hit')
    t = float(hit['radial_range_m'])
    q = np.asarray(hit['point_world'], np.float64)
    if not math.isfinite(t) or t <= 0 or q.shape != (3,) or not np.isfinite(q).all():
        raise ValueError('Invalid exact target first hit')
    expected = pose[:3, 3]+(pose[:3, :3] @ (local/radius))*t
    if np.max(abs(q-expected)) > 1e-7:
        raise ValueError('First hit not on unnormalized published world ray')
    delta = t-radius
    result.update(first_minus_target_radial_m=delta, first_minus_target_world_m=float(np.linalg.norm(q-point)))
    if delta < -tol:
        return dict(result, status='exact_target_occluded')
    if delta > tol:
        return dict(result, status='exact_target_first_hit_disagrees')
    if hit['face_id'] != target['face_id']:
        return dict(result, status='exact_target_coincident_other_face')
    if min(edge_distances) <= tol:
        return dict(result, status='exact_target_triangle_boundary')
    return dict(result, status='exact_target_is_first_surface', exact_target_first_surface=True)


def station_support(row, contract=ScannerSupportContract()):
    reasons = []
    if not row['projected_footprint'].get('sampled_C0_correspondence', False):
        reasons.append('structural_footprint_unresolved')
    if not row.get('exact_target_visibility', {}).get('exact_target_first_surface', False):
        reasons.append('exact_target_visibility_unresolved')
    matches = row['angle_matches']
    pixels = matches.get('pixels', [])
    if len(pixels) != 9 or not matches.get('all_nine_residuals_available', False):
        reasons.append('raw_patch_or_direction_or_hit_incomplete')
    residuals = []
    for pixel in pixels:
        if 'signed_range_residual_m' not in pixel:
            continue
        if (pixel.get('status') != 'conditional_same_stored_angle_residual'
                or pixel.get('nominal_angle_matched') is not True or pixel.get('raw_supported') is not True):
            raise ValueError('Unsupported raw pixel cannot claim a conditional residual')
        value = pixel['signed_range_residual_m']
        if not math.isfinite(value):
            raise ValueError('Nonfinite conditional range residual')
        residuals.append(value)
    if len(residuals) != 9 and 'raw_patch_or_direction_or_hit_incomplete' not in reasons:
        raise ValueError('Complete patch claim disagrees with residuals')
    if any(abs(v) > contract.maximum_nominal_range_residual_m for v in residuals):
        reasons.append('nominal_range_budget_exceeded')
    return dict(conditional_station_support=not reasons, reasons=reasons,
        residuals_preserved=residuals, physical_accuracy_accepted=False)


def paired_support(rows, selected_scans, numerical_support, excluded,
                   contract=ScannerSupportContract()):
    if type(numerical_support) is not bool or type(excluded) is not bool:
        raise ValueError('Explicit boolean support and exclusion decisions required')
    if len(selected_scans) != 2 or len(set(selected_scans)) != 2:
        raise ValueError('Two distinct prospectively selected scanners required')
    if len(rows) != 2 or {r['scan'] for r in rows} != set(selected_scans):
        raise ValueError('Keep both selected scanner slots; no favorable replacement')
    identities = {(r['scene'], r['corpus_ordinal'], r['slot'], tuple(r['pixel_yx'])) for r in rows}
    if len(identities) != 1:
        raise ValueError('Station observations must address the same exact target')
    decisions = [station_support(row, contract) for row in rows]
    reasons = []
    if excluded:
        reasons.append('exact_prior_geometry_exclusion')
    if not numerical_support:
        reasons.append('numerical_support_absent')
    if not all(d['conditional_station_support'] for d in decisions):
        reasons.append('both_selected_stations_not_supported')
    return dict(conditional_sample_candidate=not reasons, reasons=reasons, stations=decisions,
        label_kind=contract.label_kind, independent_ground_truth=False,
        physical_accuracy_accepted=False, training_admission=False)


def dense_image_support(numerical_mask, conditional_mask, full_grid_evaluated,
                        contract=ScannerSupportContract()):
    a, b = np.asarray(numerical_mask), np.asarray(conditional_mask)
    if a.ndim != 2 or a.shape != b.shape or a.dtype != np.bool_ or b.dtype != np.bool_ or not a.size:
        raise ValueError('Same nonempty boolean image grids required')
    if np.any(b & ~a):
        raise ValueError('Source support cannot create numerical support')
    count = int(b.sum())
    return dict(points=count, full_image_pixels=a.size, fraction=count/a.size,
        conditional_dense_coverage_sufficient=(full_grid_evaluated is True and count >= 32 and count/a.size >= .05),
        sparse_grid_extrapolation_allowed=False, training_admission=False)
