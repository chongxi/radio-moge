"""Bounded caches for the frozen source policy; no rendering or new thresholds.

The scalar correspondence arithmetic and tie order are retained. Only immutable
angle windows, whole-mesh rays and global edge incidence are reused. Target
visibility and C0 decisions are always evaluated for each exact target.
"""
from collections import OrderedDict

import numpy as np

from probe_scannetpp_exact_target_visibility import query_targets
from probe_scannetpp_triangle_scan_support import directions, scanner_local_target
from probe_scannetpp_v2_source_pilot import nearest_world_hits, raw_patch
from radio_moge.projected_footprint import projected_footprint
from radio_moge.saved_angle_match import OFFSETS, match_saved_angles
from radio_moge.scanner_support import station_support
from radio_moge.source_footprint import edge_keys


class AngleLookup:
    """Same bounded search as nearest_angle_pixel, including all exact ties."""
    def __init__(self, azimuth, polar, stride=32, window_limit=64):
        if (azimuth.shape != polar.shape or azimuth.ndim != 2 or not azimuth.size
                or stride < 1 or window_limit < 1):
            raise ValueError('Matching nonempty angle grids and bounded cache required')
        self.az, self.pol = azimuth, polar
        self.stride, self.limit = stride, window_limit
        self.sampled = directions(azimuth[::stride, ::stride], polar[::stride, ::stride])
        self.windows = OrderedDict()

    def __call__(self, target):
        target = np.asarray(target, np.float64)
        if target.shape != (3,) or not np.isfinite(target).all() or np.linalg.norm(target) == 0:
            raise ValueError('Nonzero finite target required')
        target = target / np.linalg.norm(target)
        error = np.linalg.norm(self.sampled-target, axis=-1)
        count = min(8, error.size)
        # The cutoff includes EVERY tie; stable row-major selection matches the
        # original full stable argsort. No dot-product or approximate NN search.
        flat = error.ravel()
        cutoff = np.partition(flat, count-1)[count-1]
        choices = np.flatnonzero(flat <= cutoff)
        selected = choices[np.argsort(flat[choices], kind='stable')[:count]]
        best = None
        for i in selected:
            key = int(i)
            if key not in self.windows:
                cy, cx = np.unravel_index(key, error.shape)
                s = self.stride
                y0, y1 = max(0, cy*s-s), min(self.az.shape[0], cy*s+s+1)
                x0, x1 = max(0, cx*s-s), min(self.az.shape[1], cx*s+s+1)
                self.windows[key] = (y0, x0, directions(self.az[y0:y1, x0:x1], self.pol[y0:y1, x0:x1]))
                if len(self.windows) > self.limit:
                    self.windows.popitem(last=False)
            y0, x0, local = self.windows[key]
            self.windows.move_to_end(key)
            distance = np.linalg.norm(local-target, axis=-1)
            y, x = np.unravel_index(int(np.argmin(distance)), distance.shape)
            candidate = (float(distance[y, x]), y+y0, x+x0)
            if best is None or candidate < best:
                best = candidate
        chord, y, x = best
        return int(y), int(x), directions(self.az[y, x], self.pol[y, x]), float(2*np.arcsin(min(1, chord/2)))


class TopologyIndex:
    """One exact global edge census; bounded per-target face dictionaries."""
    def __init__(self, vertices, faces, maximum_faces=20_000_000, chunk=65536):
        if len(faces) > maximum_faces or not len(faces):
            raise ValueError('Declared global topology memory bound exceeded')
        self.vertices, self.faces = np.asarray(vertices), np.asarray(faces)
        values = np.empty(3*len(faces), np.uint64)
        for start in range(0, len(faces), chunk):
            stop = min(start+chunk, len(faces))
            values[3*start:3*stop] = edge_keys(self.faces[start:stop]).ravel()
        self.keys, self.counts = np.unique(values, return_counts=True)

    def selected(self, ids):
        ids = np.asarray(sorted(set(ids)), np.int64)
        if not len(ids) or ids.min() < 0 or ids.max() >= len(self.faces):
            raise ValueError('Valid nonempty original face IDs required')
        chosen = self.faces[ids]
        keys = np.unique(edge_keys(chosen))
        at = np.searchsorted(self.keys, keys)
        if np.any(at >= len(self.keys)) or not np.array_equal(self.keys[at], keys):
            raise ValueError('Selected edge absent from global topology')
        return dict(faces=[dict(face_id=int(i), vertex_ids=f.tolist(),
            vertices_world=self.vertices[f].tolist()) for i, f in zip(ids, chosen)],
            edge_incidence={str(int(k)): int(n) for k, n in zip(keys, self.counts[at])},
            global_faces=len(self.faces), selected_faces=len(ids), bridge_expansion_rings=0)


class StationSession:
    """One original scanner at a time; no final target decision is cached."""
    def __init__(self, raw, pose, engine, topology, hit_cache_limit=4096):
        if (set(raw) != {'range', 'mask', 'azim', 'elev'}
                or len({a.shape for a in raw.values()}) != 1
                or any(a.ndim != 2 or a.dtype.kind not in 'iu' for a in raw.values())
                or hit_cache_limit < 1):
            raise ValueError('All original integer scanner grids required')
        self.raw, self.pose, self.engine, self.topology = raw, np.asarray(pose, np.float64), engine, topology
        self.lookup = AngleLookup(raw['azim'], raw['elev'])
        self.hits, self.limit = OrderedDict(), hit_cache_limit
        self.new_nine_ray_queries = 0

    def measurement(self, target):
        local = scanner_local_target(self.pose, target['point_world'])
        y, x, _, angle = self.lookup(local)
        az, pol = int(self.raw['azim'][y, x]), int(self.raw['elev'][y, x])
        key = (az, pol)
        if key not in self.hits:
            rays = directions(np.array([az+a for a, _ in OFFSETS]), np.array([pol+p for _, p in OFFSETS]))
            self.hits[key] = (rays, nearest_world_hits(self.engine, self.pose, rays))
            self.new_nine_ray_queries += 1
            if len(self.hits) > self.limit:
                self.hits.popitem(last=False)
        self.hits.move_to_end(key)
        rays, hits = self.hits[key]
        # Raw pixels are NOT cached by angle: multiple original pixels can have
        # the same quantized directions but different masks/ranges.
        measurement = dict(slot=target['slot'], scanner_pixel_yx=[y, x], angular_search_error_rad=angle,
            stored_azimuth_millirad=az, stored_polar_millirad=pol,
            target_scanner_radial_m=float(np.linalg.norm(local)), raw_patch=raw_patch(self.raw, y, x),
            discrete_angle_sensitivity=dict(samples=[dict(azimuth_offset_millirad=a,
                polar_offset_millirad=p, nearest_whole_mesh_hit=h) for (a, p), h in zip(OFFSETS, hits)]))
        return measurement, rays, hits

    def evaluate(self, targets, ordinal, scene, scan):
        exact = query_targets(targets, self.pose, self.engine)
        rows = []
        for target, visibility in zip(targets, exact):
            row = dict(corpus_ordinal=ordinal, scene=scene, scan=scan,
                       slot=target['slot'], pixel_yx=target['pixel_yx'])
            if target['numerical_support']:
                measurement, rays, hits = self.measurement(target)
                ids = {target['face_id'], *(h['face_id'] for h in hits if h is not None)}
                footprint = projected_footprint(target, hits, self.pose, rays, self.topology.selected(ids))
                matches = match_saved_angles(measurement)
                row['measurement'] = measurement
            else:
                footprint = dict(status='DSLR_numerically_unsupported', sampled_C0_correspondence=False,
                                 physical_accuracy_accepted=False)
                matches = dict(status='DSLR_numerically_unsupported', physical_accuracy_accepted=False)
            row.update(projected_footprint=footprint, angle_matches=matches,
                       exact_target_visibility=visibility['visibility'])
            row['decision'] = station_support(row)
            rows.append(row)
        return rows


def merge_masks(numerical, station_masks, evaluated, excluded):
    """Unvisited pixels stay unknown/false; never interpolate sparse evidence."""
    a = np.asarray(numerical)
    if (a.ndim != 2 or a.dtype != bool or len(station_masks) != 2 or len(evaluated) != 2
            or type(excluded) is not bool):
        raise ValueError('Two fixed station grids and explicit exclusion required')
    for mask, done in zip(station_masks, evaluated):
        if (mask.shape != a.shape or done.shape != a.shape or mask.dtype != bool or done.dtype != bool
                or (mask & (~done | ~a)).any()):
            raise ValueError('Station support must be evaluated and numerical')
    complete = evaluated[0] & evaluated[1]
    mask = a & station_masks[0] & station_masks[1] & complete
    if excluded:
        mask[:] = False
    return mask, complete
