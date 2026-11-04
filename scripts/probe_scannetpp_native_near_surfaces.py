"""Bounded CPU diagnosis of reviewed DSLR near surfaces; never repair labels.

Compare original cached Z and Embree hits against an independent float64,
double-sided all-face ray/triangle calculation. Keep raw positive intersections
before the producer's 1 cm policy filter. No rasterizer, fitting or optimizer.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np


def analytic_hits(vertices, faces, origin, direction, chunk=16384):
    """Moller-Trumbore with bounded workspace; t follows the supplied direction.

    Determinant tolerance is relative to edge/direction norms. Barycentric
    tolerance is dimensionless and is reported, not the cache's 2 mm tolerance.
    Ties sort by original face ID. Boundary/ill-conditioned hits remain visible.
    """
    vertices, origin, direction = (np.asarray(x, dtype=np.float64)
                                   for x in (vertices, origin, direction))
    faces = np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3
            or faces.dtype.kind not in 'iu' or origin.shape != (3,) or direction.shape != (3,)
            or not all(np.isfinite(x).all() for x in (vertices, origin, direction))
            or not np.linalg.norm(direction) > 0 or type(chunk) is not int or chunk < 1
            or (faces.size and (faces.min() < 0 or faces.max() >= len(vertices)))):
        raise ValueError('Finite geometry, nonzero ray and valid triangle indices required')
    tolerance = 64*np.finfo(np.float64).eps
    hits, count = [], 0
    beyond = {x: None for x in (.01, .1, 1.)}
    for start in range(0, len(faces), chunk):
        triangles = vertices[faces[start:start+chunk]]
        e1, e2 = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
        p = np.cross(direction, e2)
        det = np.einsum('ij,ij->i', e1, p)
        scale = np.linalg.norm(e1, axis=1)*np.linalg.norm(e2, axis=1)*np.linalg.norm(direction)
        usable = np.abs(det) > tolerance*scale
        inv = np.divide(1., det, out=np.zeros_like(det), where=usable)
        offset = origin-triangles[:, 0]
        u = np.einsum('ij,ij->i', offset, p)*inv
        q = np.cross(offset, e1)
        v = (q @ direction)*inv
        t = np.einsum('ij,ij->i', e2, q)*inv
        good = usable & np.isfinite(t) & (t > 0) & (u >= -tolerance) & (v >= -tolerance)
        good &= u+v <= 1+tolerance
        for j in np.flatnonzero(good):
            count += 1
            bary = np.array([1-u[j]-v[j], u[j], v[j]])
            point = origin+t[j]*direction
            hit = dict(face_id=int(start+j), t=float(t[j]), barycentrics=bary.tolist(),
                boundary_hit=bool(bary.min() <= 1e-10), relative_determinant=float(abs(det[j])/scale[j]),
                ray_triangle_residual_m=float(np.linalg.norm(point-bary @ triangles[j])))
            hits.append(hit)
            for threshold, previous in beyond.items():
                if t[j] > threshold and (previous is None or (hit['t'], hit['face_id']) < (previous['t'], previous['face_id'])):
                    beyond[threshold] = hit
        hits = sorted(hits, key=lambda x: (x['t'], x['face_id']))[:8]
    return dict(first_hits=hits, positive_triangle_intersections=count,
        first_beyond_t={str(k): v for k, v in beyond.items()},
        determinant_relative_tolerance=tolerance, barycentric_tolerance=tolerance,
        direction_normalized=False, nearest_hit_selected_before_policy_filter=True)


def sample_pixels(depth, valid):
    """At most six deterministic centers; select geometry quantiles and controls."""
    flat = np.flatnonzero(valid)
    if not len(flat):
        raise ValueError('Reviewed candidate must contain valid support')
    result = {}
    for q in (.1, .5, .9):
        target = np.quantile(depth[valid], q)
        i = int(flat[np.argmin(abs(depth.ravel()[flat]-target))])
        result.setdefault(i, []).append(f'supported_depth_quantile_{q}')
    h, w = depth.shape
    for y, x, name in ((h//2, w//2, 'image_center'), (h//4, w//4, 'upper_left_quarter')):
        result.setdefault(y*w+x, []).append(name)
    invalid = np.flatnonzero(~valid)
    if len(invalid):
        distance = (invalid//w-h/2)**2+(invalid % w-w/2)**2
        result.setdefault(int(invalid[np.argmin(distance)]), []).append('invalid_nearest_center')
    return [(i//w, i % w, reasons) for i, reasons in result.items()]


def run(review_path, output, ordinals):
    # Imports requiring source-reader dependencies are unnecessary for math tests.
    import trimesh
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    from cache_scannetpp_geometry import camera_for_row, colmap_poses
    from radio_moge.corpus_admission import checked_json, checked_path, file_ref
    from radio_moge.geometry_cache import publish_json, read_geometry_member
    from radio_moge.geometry_sources import pinhole_rays

    review_ref = file_ref(review_path)
    review = checked_json(review_ref)
    if (review['stage'] != 'visual_review_complete_source_quality_unresolved'
            or review['training_admission'] is not False or review['optimizer_updates'] != 0
            or not 1 <= len(set(ordinals)) == len(ordinals) <= 10):
        raise ValueError('Completed bounded visual review and unique explicit ordinals required')
    selected = [r for r in review['records'] if r['corpus_ordinal'] in ordinals]
    if len(selected) != len(ordinals) or any(r['decision'] != 'quarantine_candidate' for r in selected):
        raise ValueError('Only exact reviewed quarantined candidates may be diagnosed')
    output.mkdir(exist_ok=False)
    results, sources, active = [], {}, None
    mesh = intersector = None
    for chosen in sorted(selected, key=lambda r: (r['group'], r['corpus_ordinal'])):
        receipt = checked_json(chosen['receipt'])
        entries = [e for e in receipt['entries'] if e['key'] == chosen['entry_key']]
        if len(entries) != 1:
            raise ValueError('Selected immutable entry missing or repeated')
        entry = entries[0]
        row, meta = entry['image'], entry['geometry_metadata']
        if (entry['sha256'] != chosen['member_sha256'] or row['image'] != chosen['image']['path']
                or row['sha256'] != chosen['image']['sha256'] or row['group'] != chosen['group']
                or meta['policy_sha256'] != chosen['policy_sha256']
                or meta['depth_convention'] != 'camera_z' or meta['units'] != 'meters'
                or meta['camera_pixel_convention'] != 'normalized_half_center'):
            raise ValueError('Visual/member/RGB/camera identity differs')
        checked_path(chosen['page'])
        scene = Path(row['image']).parents[2]
        paths = [scene/'scans/mesh_aligned_0.05.ply', scene/'dslr/colmap/images.txt',
                 scene/'dslr/nerfstudio/transforms_undistorted.json']
        proofs = {p['source']: p['sha256'] for p in meta['source_proofs']}
        refs = [dict(path=str(p), sha256=proofs[str(p)]) for p in paths]
        if active != row['scene']:
            del mesh, intersector
            gc.collect()
            for ref in refs:
                checked_path(ref)
            mesh = trimesh.load_mesh(paths[0], process=False)
            intersector = RayMeshIntersector(mesh)
            calibration = json.loads(paths[2].read_text())
            frames = {Path(f['file_path']).name: f for f in calibration['frames']}
            poses = colmap_poses(paths[1])
            active = row['scene']
            sources[active] = dict(refs=refs,vertices=len(mesh.vertices), faces=len(mesh.faces))
        elif sources[active]['refs'] != refs:
            raise ValueError('Same scene candidates have different source versions')
        kd, c2w = camera_for_row(row, calibration, frames, poses)
        cached = read_geometry_member(Path(chosen['receipt']['path']).parent/receipt['tar'], entry)
        depth, valid, cached_k, cached_meta = cached
        if cached_meta != meta or not np.array_equal(cached_k, kd):
            raise ValueError('Cached metadata or ray calibration changed')
        rays = pinhole_rays(kd, *depth.shape)
        origin, rotation = c2w[:3, 3], c2w[:3, :3]
        records = []
        for y, x, reasons in sample_pixels(depth, valid):
            direction = rays[y, x] @ rotation.T
            locations, ids, triangles = intersector.intersects_location(
                origin[None].copy(), direction[None].copy(), multiple_hits=False)
            embree = None
            if len(ids):
                if len(ids) != 1 or ids[0] != 0:
                    raise ValueError('Unexpected single-ray nearest hit result')
                z = float(((locations[0]-origin) @ rotation)[2])
                embree = dict(face_id=int(triangles[0]), point_world=locations[0].tolist(), camera_z_m=z)
            raw = analytic_hits(mesh.vertices, mesh.faces, origin, direction)
            unique = {h['face_id']: h for h in raw['first_hits']}
            unique.update({h['face_id']: h for h in raw['first_beyond_t'].values() if h is not None})
            for h in unique.values():
                point = origin+h['t']*direction
                verts = np.asarray(mesh.vertices[mesh.faces[h['face_id']]], dtype=np.float64)
                h.update(point_world=point.tolist(), camera_z_m=float(((point-origin) @ rotation)[2]),
                    vertices_world=verts.tolist(), vertices_camera=((verts-origin) @ rotation).tolist())
            first = raw['first_hits'][0] if raw['first_hits'] else None
            records.append(dict(pixel_yx=[y,x],selection=reasons,cached_valid=bool(valid[y,x]),
                cached_z_m=float(depth[y,x]) if valid[y,x] else None, ray_origin_world=origin.tolist(),
                ray_direction_world=direction.tolist(), camera_ray=rays[y,x].tolist(), embree=embree,
                analytical=raw, analytical_minus_cached_z_m=(first['camera_z_m']-float(depth[y,x])) if first and valid[y,x] else None,
                analytical_minus_embree_z_m=(first['camera_z_m']-embree['camera_z_m']) if first and embree else None))
        case = dict(corpus_ordinal=chosen['corpus_ordinal'],group=row['group'],image=chosen['image'],
            entry_key=chosen['entry_key'],member_sha256=entry['sha256'],receipt=chosen['receipt'],page=chosen['page'],
            source_refs=refs,grid=list(depth.shape),intrinsics=kd.tolist(),c2w=c2w.tolist(),rays=records)
        target=output/f'{chosen["corpus_ordinal"]}.json'
        publish_json(target,case)
        results.append(file_ref(target))
        publish_json(output/'status.json',dict(stage='probing',completed=len(results),expected=len(selected),gpu_used=False))
        print(json.dumps(dict(completed=len(results),corpus_ordinal=chosen['corpus_ordinal'],rays=len(records))),flush=True)
    checked_path(review_ref)
    for value in sources.values():
        for ref in value['refs']:
            checked_path(ref)
    report=dict(stage='bounded_raw_mesh_ray_diagnosis_complete',review=review_ref,cases=results,sources=sources,
        gpu_used=False,optimizer_updates=0,training_admission=False,source_quality_accepted=False,
        physical_accuracy_established=False,geometry_edited=False,source_flags_cleared=False,
        near_hits_skipped_or_repaired=False,whole_scene_coverage=False,code=file_ref(Path(__file__)))
    publish_json(output/'report.json',report)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('review',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--ordinals',required=True,type=int,nargs='+')
    args=parser.parse_args()
    run(args.review.resolve(),args.output.resolve(),args.ordinals)
