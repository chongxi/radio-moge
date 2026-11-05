"""Finite-sample C0 mesh correspondence, not smoothness or physical accuracy.

The selected indexed patch must embed in a scanner-local gnomonic chart and
cover the convex hull of the nine saved ray directions. This says nothing about
unselected occluders between those rays, pose covariance, or physical truth.
"""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np

from radio_moge.source_footprint import _normal, _point_on_triangle, validate_saved_hits


@dataclass(frozen=True)
class ProjectedFootprintContract:
    schema_version: int = 1
    minimum_incidence_cos: float = .1
    point_tolerance_m: float = 1e-5
    predicate_epsilon_multiplier: int = 128
    maximum_chart_epsilon: float = 1e-8
    bridge_expansion_rings: int = 0

    def __post_init__(self):
        if asdict(self) != dict(schema_version=1, minimum_incidence_cos=.1,
                point_tolerance_m=1e-5, predicate_epsilon_multiplier=128,
                maximum_chart_epsilon=1e-8, bridge_expansion_rings=0):
            raise ValueError('Frozen projected diagnostic contract required')


def cross2(a, b):
    return a[0]*b[1]-a[1]*b[0]


def area(poly):
    if len(poly) < 3:
        return 0.
    p = np.asarray(poly)-poly[0]
    return abs(sum(cross2(p[i], p[i+1]) for i in range(1, len(p)-1)))/2


def ccw(poly):
    p = np.asarray(poly, np.float64)
    signed = sum(cross2(p[i]-p[0], p[i+1]-p[0]) for i in range(1, len(p)-1))
    return p if signed > 0 else p[::-1]


def convex_hull(points):
    p = sorted(set(map(tuple, points)))
    if len(p) < 3:
        return np.asarray(p)
    def half(values):
        result = []
        for x in values:
            while len(result) >= 2 and cross2(np.subtract(result[-1], result[-2]),
                                               np.subtract(x, result[-1])) <= 0:
                result.pop()
            result.append(x)
        return result
    return np.asarray(half(p)[:-1]+half(p[::-1])[:-1])


def clip_polygon(subject, boundary, epsilon):
    """Convex polygon intersection; tolerance only encloses numerical contacts."""
    output = list(np.asarray(subject, np.float64))
    boundary = ccw(boundary)
    for a, b in zip(boundary, np.roll(boundary, -1, axis=0)):
        if not output:
            break
        length = np.linalg.norm(b-a)
        if length <= epsilon:
            raise ValueError('Degenerate clipping boundary')
        incoming, output = output, []
        previous = incoming[-1]
        dp = cross2(b-a, previous-a)/length
        for current in incoming:
            dc = cross2(b-a, current-a)/length
            old_in, new_in = dp >= -epsilon, dc >= -epsilon
            if old_in != new_in:
                denominator = dp-dc
                if abs(denominator) <= np.finfo(float).tiny:
                    raise ValueError('Unresolved parallel clipping transition')
                t = np.clip(dp/denominator, 0., 1.)
                output.append(previous+t*(current-previous))
            if new_in:
                output.append(current)
            previous, dp = current, dc
    return np.asarray(output).reshape(-1, 2)


def connected(graph, start):
    seen, pending = set(), [start]
    while pending:
        v = pending.pop()
        if v not in seen:
            seen.add(v)
            pending.extend(graph[v]-seen)
    return seen


def vertex_links_are_manifold(faces):
    links = defaultdict(lambda: defaultdict(set))
    edge_counts = Counter()
    for ids in faces:
        for i, v in enumerate(ids):
            a, b = ids[(i+1) % 3], ids[(i+2) % 3]
            links[v][a].add(b)
            links[v][b].add(a)
            edge_counts[(v, min(a, b), max(a, b))] += 1
    if max(edge_counts.values(), default=0) > 1:
        return False
    for graph in links.values():
        if (connected(graph, next(iter(graph))) != set(graph)
                or any(len(n) > 2 for n in graph.values())
                or sum(len(n) == 1 for n in graph.values()) not in (0, 2)):
            return False
    return True


def point_segment_distance(point, a, b):
    t = np.clip((point-a) @ (b-a)/((b-a) @ (b-a)), 0., 1.)
    return float(np.linalg.norm(point-a-t*(b-a)))


def separated_triangles(a, b, epsilon):
    """Strict separating-axis gap; near contacts stay unresolved."""
    for triangle in (a, b):
        for p, q in zip(triangle, np.roll(triangle, -1, axis=0)):
            axis = np.array([-(q-p)[1], (q-p)[0]])/np.linalg.norm(q-p)
            x, y = a @ axis, b @ axis
            if min(y)-max(x) > epsilon or min(x)-max(y) > epsilon:
                return True
    return False


def vertex_cones_separated(a, b, common, epsilon):
    """A supporting line through the shared vertex must strictly separate cones."""
    for own, other in ((a, b), (b, a)):
        own_other = [v for v in own if not np.array_equal(v, common)]
        other_other = [v for v in other if not np.array_equal(v, common)]
        if len(own_other) != 2 or len(other_other) != 2:
            return False
        for j in (0, 1):
            edge = own_other[j]-common
            length = np.linalg.norm(edge)
            inside = cross2(edge, own_other[1-j]-common)/length
            values = [cross2(edge, v-common)/length for v in other_other]
            if abs(inside) > epsilon and all(np.sign(inside)*v < -epsilon for v in values):
                return True
    av = np.asarray([p-common for p in a if not np.array_equal(p, common)])
    bv = np.asarray([p-common for p in b if not np.array_equal(p, common)])
    for u in av/np.linalg.norm(av, axis=1, keepdims=True):
        for v in bv/np.linalg.norm(bv, axis=1, keepdims=True):
            normal = u-v
            length = np.linalg.norm(normal)
            if length > epsilon:
                normal /= length
                if min(av @ normal) > epsilon and max(bv @ normal) < -epsilon:
                    return True
    return False


def segment_may_touch_hull(a, b, hull, epsilon):
    """Conservative intersection with the closed, numerically expanded hull."""
    lower, upper = 0., 1.
    for p, q in zip(hull, np.roll(hull, -1, axis=0)):
        length = np.linalg.norm(q-p)
        start = cross2(q-p, a-p)/length
        end = cross2(q-p, b-p)/length
        if max(start, end) < -epsilon:
            return False
        slope = end-start
        if abs(slope) <= np.finfo(float).tiny:
            continue
        t = (-epsilon-start)/slope
        if slope > 0:
            lower = max(lower, t)
        else:
            upper = min(upper, t)
        if lower > upper:
            return False
    return lower <= upper


def projected_footprint(target, hits, pose, local_rays, topology,
                        contract=ProjectedFootprintContract()):
    result = dict(sampled_C0_correspondence=False, physical_accuracy_accepted=False,
        numerical_predicates_only=True, exact_arithmetic_embedding_proof=False,
        smoothness_certified=False, semantic_correspondence_certified=False,
        whole_mesh_visibility_between_samples_proven=False, exact_target_visibility_proven=False,
        continuous_angle_or_pose_uncertainty_bound=False)
    if len(hits) != 9 or np.asarray(local_rays).shape != (9, 3):
        raise ValueError('All nine frozen samples required')
    if any(h is None for h in hits):
        return dict(result, status='first_hit_missing')
    pose, local_rays = np.asarray(pose, np.float64), np.asarray(local_rays, np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or abs(np.linalg.det(pose[:3, :3])) < 1e-12:
        raise ValueError('Nonsingular published pose required')
    validate_saved_hits(hits, pose[:3, 3], local_rays @ pose[:3, :3].T)
    records = {r['face_id']: r for r in topology['faces']}
    wanted = {target['face_id'], *(h['face_id'] for h in hits)}
    if not wanted <= records.keys():
        return dict(result, status='topology_unavailable')
    records = {i: records[i] for i in sorted(wanted)}
    vertices, by_edge, graph = {}, defaultdict(list), {i: set() for i in wanted}
    triangles, normals = {}, {}
    for i, record in records.items():
        ids, v = record['vertex_ids'], np.asarray(record['vertices_world'], np.float64)
        if len(set(ids)) != 3 or v.shape != (3, 3) or not np.isfinite(v).all():
            raise ValueError('Finite nondegenerate indexed triangle required')
        for index, point in zip(ids, v):
            if index in vertices and not np.array_equal(vertices[index], point):
                raise ValueError('Same source vertex has inconsistent coordinates')
            vertices[index] = point
        triangles[i], normals[i] = v, _normal(v)
        for a, b in ((0, 1), (1, 2), (2, 0)):
            by_edge[tuple(sorted((ids[a], ids[b])))].append(i)
    if any(n is None for n in normals.values()):
        return dict(result, status='degenerate_triangle')
    if not np.array_equal(triangles[target['face_id']], target['vertices_world']):
        raise ValueError('Frozen target triangle differs')
    points = [(target['face_id'], np.asarray(target['point_world']))]+[
        (h['face_id'], np.asarray(h['point_world'])) for h in hits]
    if any(not _point_on_triangle(p, triangles[i], contract.point_tolerance_m) for i, p in points):
        raise ValueError('Saved point not on its recorded triangle')
    for edge, ids in by_edge.items():
        count = topology['edge_incidence'][str((edge[0] << 32) | edge[1])]
        if count > 2 or len(ids) > 2 or count < len(ids):
            return dict(result, status='nonmanifold_edge_unresolved')
        if len(ids) == 2:
            graph[ids[0]].add(ids[1])
            graph[ids[1]].add(ids[0])
    if connected(graph, target['face_id']) != wanted:
        return dict(result, status='connectivity_unproven_with_retained_faces')
    if not vertex_links_are_manifold([r['vertex_ids'] for r in records.values()]):
        return dict(result, status='nonmanifold_vertex_link_unresolved')
    center = local_rays[4]/np.linalg.norm(local_rays[4])
    axis = np.eye(3)[int(np.argmin(abs(center)))]
    right = np.cross(axis, center)
    right /= np.linalg.norm(right)
    up = np.cross(center, right)
    inverse = np.linalg.inv(pose[:3, :3])
    def world_to_local(p):
        return (np.asarray(p)-pose[:3, 3]) @ inverse.T
    def project(local):
        z = local @ center
        if np.any(z <= 128*np.finfo(float).eps*np.linalg.norm(local, axis=-1)):
            raise ValueError('chart_horizon_or_backwards_vertex')
        return np.stack([local @ right, local @ up], axis=-1)/z[..., None]
    try:
        raw = {i: project(world_to_local(v)) for i, v in vertices.items()}
        ray2 = project(local_rays)
        target2 = project(world_to_local(target['point_world']))
    except ValueError:
        return dict(result, status='chart_horizon_unresolved')
    offset, scale = ray2[4], float(np.max(np.ptp(ray2, axis=0)))
    if not np.isfinite(scale) or scale <= 1e-12:
        return dict(result, status='angular_footprint_degenerate')
    projected = {i: (v-offset)/scale for i, v in raw.items()}
    ray2, target2 = (ray2-offset)/scale, (target2-offset)/scale
    bound = max(1., *(float(np.max(abs(v))) for v in projected.values()))
    eps = contract.predicate_epsilon_multiplier*np.finfo(float).eps*bound**2
    result.update(chart_epsilon=eps, selected_faces=len(wanted))
    if eps > contract.maximum_chart_epsilon:
        return dict(result, status='chart_ill_conditioned')
    projected_tri = {i: np.array([projected[v] for v in r['vertex_ids']]) for i, r in records.items()}
    if any(area(v) <= eps for v in projected_tri.values()):
        return dict(result, status='projected_triangle_degenerate')
    incidence = []
    for i, v in triangles.items():
        rays = v-pose[:3, 3]
        incidence.extend(abs(rays @ normals[i])/np.linalg.norm(rays, axis=1))
    result['minimum_full_face_incidence_cos'] = float(min(incidence))
    if min(incidence) < contract.minimum_incidence_cos:
        return dict(result, status='grazing_patch_unresolved')
    dihedral = []
    for edge, ids in by_edge.items():
        if len(ids) == 2:
            a, b = (projected[v] for v in edge)
            sides = [cross2(b-a, projected_tri[i].mean(axis=0)-a) for i in ids]
            if sides[0]*sides[1] >= 0 or min(map(abs, sides)) <= eps:
                return dict(result, status='projected_fold_or_edge_uncertainty')
            oriented = [normals[i]*np.sign(normals[i] @ (triangles[i].mean(axis=0)-pose[:3, 3])) for i in ids]
            dihedral.append(float(np.rad2deg(np.arccos(np.clip(oriented[0] @ oriented[1], -1, 1)))))
    result['maximum_view_oriented_dihedral_degrees'] = max(dihedral, default=0.)
    for i, j in combinations(records, 2):
        shared = set(records[i]['vertex_ids']) & set(records[j]['vertex_ids'])
        if len(shared) == 2:
            # Opposite-side predicates above certify no common interior.
            continue
        if not shared:
            if not separated_triangles(projected_tri[i], projected_tri[j], eps):
                return dict(result, status='nonindexed_projected_overlap_or_contact')
        elif len(shared) == 1:
            if not vertex_cones_separated(projected_tri[i], projected_tri[j],
                                           projected[next(iter(shared))], eps):
                return dict(result, status='shared_vertex_contact_unresolved')
        else:
            return dict(result, status='duplicate_indexed_face')
    hull = convex_hull(ray2)
    hull_area = area(hull)
    if hull_area <= eps:
        return dict(result, status='angular_footprint_degenerate')
    if any(cross2(b-a, target2-a)/np.linalg.norm(b-a) < -eps
           for a, b in zip(hull, np.roll(hull, -1, axis=0))):
        return dict(result, status='target_outside_sampled_ray_hull')
    # Area alone can miss an arbitrarily thin hole/slit. Any exposed patch
    # boundary crossing or touching the ray hull makes coverage unresolved.
    for edge, ids in by_edge.items():
        if len(ids) == 1 and segment_may_touch_hull(projected[edge[0]], projected[edge[1]], hull, eps):
            return dict(result, status='exposed_patch_boundary_touches_ray_hull')
    covered = sum(area(clip_polygon(v, hull, eps)) for v in projected_tri.values())
    result.update(ray_hull_area=hull_area, selected_surface_coverage_area=covered)
    if abs(covered-hull_area) > 8*eps*len(wanted):
        return dict(result, status='ray_hull_not_covered_by_selected_surface')
    # Do not certify an exact sample on an exposed or missing mesh seam.
    for i, point in points:
        for edge, ids in by_edge.items():
            if i in ids and len(ids) == 1:
                distance = point_segment_distance(point, vertices[edge[0]], vertices[edge[1]])
                if distance <= contract.point_tolerance_m:
                    return dict(result, status='sample_on_exposed_or_unretained_edge')
    return dict(result, status='embedded_C0_patch_over_saved_ray_hull', sampled_C0_correspondence=True)
