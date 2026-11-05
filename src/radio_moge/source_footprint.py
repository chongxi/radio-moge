"""Conservative mesh-footprint diagnostics, never a physical accuracy gate.

Uses original indexed topology and saved first hits. No welding, fitted plane,
new ray query, replacement hit, or label mutation. Missing bridge triangles are
unresolved rather than evidence of a disconnected physical object.
"""
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class FootprintContract:
    schema_version: int = 1
    numerical_plane_distance_m: float = 1e-5
    numerical_normal_sine: float = 1e-5
    numerical_point_distance_m: float = 1e-5
    minimum_incidence_cos: float = .1
    bridge_expansion_rings: int = 0

    def __post_init__(self):
        if asdict(self) != dict(schema_version=1, numerical_plane_distance_m=1e-5,
                numerical_normal_sine=1e-5, numerical_point_distance_m=1e-5,
                minimum_incidence_cos=.1, bridge_expansion_rings=0):
            raise ValueError('Versioned diagnostic contract required; no runtime tuning')


def edge_keys(faces):
    f = np.asarray(faces)
    if (f.ndim != 2 or f.shape[1] != 3 or f.dtype.kind not in 'iu'
            or (f.size and (f.min() < 0 or f.max() >= 2**32))):
        raise ValueError('Original uint32-addressable triangular indices required')
    edges = np.sort(f[:, [[0, 1], [1, 2], [2, 0]]], axis=-1).astype(np.uint64)
    return (edges[..., 0] << np.uint64(32)) | edges[..., 1]


def extract_topology(vertices, faces, selected_ids, chunk=65536):
    """Read selected triangles and exact *global* incidence of their edges.

    No ray queries or adjacency expansion. Chunking bounds temporary memory;
    an unselected third face must prevent a false manifold-edge certificate.
    """
    ids = np.asarray(sorted(set(selected_ids)), dtype=np.int64)
    if not len(ids) or ids.min() < 0 or ids.max() >= len(faces):
        raise ValueError('Nonempty valid selected face identities required')
    chosen = np.asarray(faces)[ids]
    keys = np.unique(edge_keys(chosen))
    counts = np.zeros(len(keys), np.int64)
    for start in range(0, len(faces), chunk):
        values = edge_keys(np.asarray(faces)[start:start+chunk]).ravel()
        at = np.searchsorted(keys, values)
        valid = at < len(keys)
        where = np.flatnonzero(valid)
        where = where[keys[at[where]] == values[where]]
        np.add.at(counts, at[where], 1)
    return dict(faces=[dict(face_id=int(i), vertex_ids=f.tolist(),
                           vertices_world=np.asarray(vertices)[f].tolist())
                       for i, f in zip(ids, chosen)],
                edge_incidence={str(int(k)): int(n) for k, n in zip(keys, counts)},
                global_faces=len(faces), selected_faces=len(ids), bridge_expansion_rings=0)


def _normal(v):
    edges = v[1:]-v[0]
    n = np.cross(*edges)
    length = np.linalg.norm(n)
    if length <= 64*np.finfo(np.float64).eps*np.linalg.norm(edges[0])*np.linalg.norm(edges[1]):
        return None
    return n/length


def _point_on_triangle(point, v, tolerance):
    a, b = v[1:]-v[0]
    q = point-v[0]
    normal = _normal(v)
    if normal is None or abs(q @ normal) > tolerance:
        return False
    weights = np.linalg.lstsq(np.stack([a, b], axis=1), q, rcond=None)[0]
    # Each barycentric coordinate uses its own opposite-edge altitude. Using
    # the smallest altitude for all three admits distant points on skinny faces.
    area = np.linalg.norm(np.cross(a, b))
    eps = tolerance*np.array([np.linalg.norm(b), np.linalg.norm(a), np.linalg.norm(a-b)])/area
    return bool((np.array([*weights, 1-weights.sum()]) >= -eps).all())


def validate_saved_hits(hits, origin, world_rays):
    """Preserve the published scanner's local-radial parameter (no normalization)."""
    if np.asarray(world_rays).shape != (9, 3) or len(hits) != 9:
        raise ValueError('Nine saved angular rays required')
    for hit, ray in zip(hits, world_rays):
        if hit is None:
            continue
        t = hit['radial_range_m']
        point = np.asarray(hit['point_world'])
        if (not np.isfinite(t) or t < 0 or not np.isfinite(point).all()
                or not np.isfinite(ray).all() or np.linalg.norm(ray) == 0
                or np.max(abs(point-origin-t*ray)) > 1e-7):
            raise ValueError('Saved hit/radial parameter differs from published world ray')


def planar_footprint(target, hits, scanner_origin, topology, contract=FootprintContract()):
    """Classify the saved nine first hits without requiring identical face IDs."""
    if len(hits) != 9:
        raise ValueError('All nine fixed angular samples required')
    result = dict(diagnostic_planar_footprint=False, physical_accuracy_accepted=False,
                  all_first_hits_preserved=True, missing_hits=sum(h is None for h in hits))
    if result['missing_hits']:
        return dict(result, status='first_hit_missing')
    records = {x['face_id']: x for x in topology['faces']}
    seed = target['face_id']
    requested = {seed, *(h['face_id'] for h in hits)}
    if not requested <= records.keys():
        return dict(result, status='topology_unavailable')
    vertices = {i: np.asarray(records[i]['vertices_world'], np.float64) for i in requested}
    if any(v.shape != (3, 3) or not np.isfinite(v).all() for v in vertices.values()):
        raise ValueError('Finite original triangles required')
    if not np.array_equal(vertices[seed], np.asarray(target['vertices_world'])):
        raise ValueError('Target/source triangle changed')
    if not _point_on_triangle(np.asarray(target['point_world']), vertices[seed],
                              contract.numerical_point_distance_m):
        raise ValueError('Saved target not on its original triangle')
    for h in hits:
        if not _point_on_triangle(np.asarray(h['point_world']), vertices[h['face_id']],
                                  contract.numerical_point_distance_m):
            raise ValueError('Saved first hit not on its original triangle')
    normals = {i: _normal(v) for i, v in vertices.items()}
    if any(n is None for n in normals.values()):
        return dict(result, status='degenerate_triangle')
    n, origin = normals[seed], vertices[seed][0]
    max_distance = max(float(np.max(abs((v-origin) @ n))) for v in vertices.values())
    max_sine = max(float(np.linalg.norm(np.cross(n, m))) for m in normals.values())
    incidence = []
    for h in hits:
        ray = np.asarray(h['point_world'])-scanner_origin
        if not np.isfinite(ray).all() or np.linalg.norm(ray) <= 0:
            raise ValueError('Finite nonzero first-hit ray required')
        incidence.append(float(abs(ray @ normals[h['face_id']])/np.linalg.norm(ray)))
    result.update(max_seed_plane_distance_m=max_distance, max_seed_normal_sine=max_sine,
                  minimum_incidence_cos=min(incidence), distinct_hit_faces=len(requested))
    if (max_distance > contract.numerical_plane_distance_m
            or max_sine > contract.numerical_normal_sine):
        return dict(result, status='nonplanar_or_depth_discontinuous_unresolved')
    if min(incidence) < contract.minimum_incidence_cos:
        return dict(result, status='grazing_footprint_unresolved')
    by_edge = defaultdict(list)
    for i in requested:
        f = records[i]['vertex_ids']
        for a, b in ((0, 1), (1, 2), (2, 0)):
            by_edge[tuple(sorted((f[a], f[b])))].append(i)
    graph = {i: set() for i in requested}
    for edge, faces in by_edge.items():
        if len(faces) < 2:
            continue
        key = str((edge[0] << 32) | edge[1])
        if topology['edge_incidence'][key] != 2 or len(faces) != 2:
            return dict(result, status='nonmanifold_shared_edge_unresolved')
        i, j = faces
        fi = records[i]['vertex_ids']
        v0, v1 = (vertices[i][fi.index(x)] for x in edge)
        side = [float(np.cross(v1-v0, vertices[k].mean(axis=0)-v0) @ n) for k in (i, j)]
        if side[0]*side[1] >= 0:
            return dict(result, status='overlapping_same_side_faces_unresolved')
        graph[i].add(j)
        graph[j].add(i)
    reached, pending = set(), [seed]
    while pending:
        current = pending.pop()
        if current not in reached:
            reached.add(current)
            pending.extend(graph[current]-reached)
    if reached != requested:
        return dict(result, status='connectivity_unproven_with_retained_faces')
    # A point on an exposed/missing edge is not certified by a planar neighbor
    # elsewhere. Internal seams are allowed only with both incident faces here.
    for h in hits:
        i, point = h['face_id'], np.asarray(h['point_world'])
        f, v = records[i]['vertex_ids'], vertices[i]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            distance = np.linalg.norm(np.cross(v[b]-v[a], point-v[a]))/np.linalg.norm(v[b]-v[a])
            if distance <= contract.numerical_point_distance_m:
                edge = tuple(sorted((f[a], f[b])))
                key = str((edge[0] << 32) | edge[1])
                if topology['edge_incidence'][key] != 2 or len(by_edge[edge]) != 2:
                    return dict(result, status='exposed_or_unretained_edge_unresolved')
    return dict(result, status='connected_planar_first_surface_footprint',
                diagnostic_planar_footprint=True)


def sampled_range_evidence(hits, raw_patch):
    """Always retain source gaps, even when the topology/footprint is unresolved.

    Envelopes describe finite samples, not continuous angular/pose uncertainty.
    Overlap is not a physical pass; disjoint intervals are not a calibrated
    physical-error lower bound. No best-station averaging or fit.
    """
    values = raw_patch['values']
    ranges, mask = np.asarray(values['range']), np.asarray(values['mask'])
    if ranges.shape != mask.shape or ranges.ndim != 2:
        raise ValueError('Matching original range/mask patch required')
    supported = (mask == 255) & (ranges > 50) & (ranges < 65535)
    first = [h['radial_range_m'] for h in hits if h is not None]
    if any(not np.isfinite(x) or x < 0 for x in first):
        raise ValueError('Finite nonnegative saved first ranges required')
    result = dict(raw_patch_pixels=int(ranges.size), raw_supported_pixels=int(supported.sum()),
                  saved_first_hits=len(first), physical_accuracy_accepted=False,
                  continuous_uncertainty_bound=False)
    if first:
        result['first_hit_interval_m'] = [float(min(first)), float(max(first))]
    if ranges.shape != (3, 3) or not supported.all() or not first or len(first) != 9:
        return dict(result, status='source_patch_or_first_hits_incomplete')
    measured = ranges.astype(np.float64)/1000
    lower, upper = float(measured.min()-max(first)), float(measured.max()-min(first))
    gap = lower if lower > 0 else upper if upper < 0 else 0.
    return dict(result, status='sampled_intervals_disjoint' if gap else 'sampled_intervals_overlap',
                measured_interval_m=[float(measured.min()), float(measured.max())],
                first_hit_interval_m=[float(min(first)), float(max(first))],
                signed_sampled_envelope_m=[lower, upper],
                signed_interval_separation_m=gap)
