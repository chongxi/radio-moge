"""New curved-surface diagnostic using only immutable saved JSON evidence."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from probe_scannetpp_triangle_scan_support import directions, scanner_pose_diagnostics
from probe_scannetpp_v2_source_pilot import read
from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import publish_json
from radio_moge.projected_footprint import ProjectedFootprintContract, projected_footprint


PRIOR_SHA = '03b49d95f3184501dd7ad99659b96026019e1f3be7dc580d5151bbfb6f9c3971'


def summarize(rows):
    if len(rows) != 768 or len({(r['corpus_ordinal'], r['scan'], r['slot']) for r in rows}) != 768:
        raise ValueError('All 768 unique station slots required')
    by_image = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_image[r['corpus_ordinal']][r['slot']].append(r)
    if len(by_image) != 4 or any(len(slots) != 96 for slots in by_image.values()):
        raise ValueError('Four frozen images with96slots required')
    def eligible(row):
        return row['projected_footprint']['sampled_C0_correspondence']
    def overlapping(row):
        return eligible(row) and row.get('range_evidence', {}).get('status') == 'sampled_intervals_overlap'
    per_image = []
    for image, slots in sorted(by_image.items()):
        if any(len(v) != 2 or {r['scan'] for r in v} != {'0', '1'} for v in slots.values()):
            raise ValueError('Both original stations required for each slot')
        record = dict(corpus_ordinal=image, DSLR_slots=96)
        for name, key in [('structural', eligible), ('structural_and_sampled_overlap', overlapping)]:
            record[name+'_at_least_one_station'] = sum(any(key(r) for r in v) for v in slots.values())
            record[name+'_both_stations'] = sum(all(key(r) for r in v) for v in slots.values())
        per_image.append(record)
    return dict(DSLR_slots=384, station_slots=768, per_image=per_image,
        structural_observations=sum(eligible(r) for r in rows),
        multiface_structural_observations=sum(eligible(r) and r['projected_footprint']['selected_faces'] > 1 for r in rows),
        structural_and_sampled_overlap_observations=sum(overlapping(r) for r in rows),
        status_counts=dict(Counter(r['projected_footprint']['status'] for r in rows)),
        physical_accepted_pixels=0, source_quality_accepted=False)


def run(root, output):
    prior_ref = file_ref(root/'scannetpp-native-v2-source-footprint-02/report.json')
    if prior_ref['sha256'] != PRIOR_SHA:
        raise ValueError('Frozen preceding evidence required')
    prior = read(prior_ref, root)
    output.mkdir(exist_ok=False)
    contract = ProjectedFootprintContract()
    publish_json(output/'contract.json', dict(parameters=asdict(contract), prior=prior_ref,
        topology='original indexed edges and manifold vertex links; no bridge expansion or welding',
        projection='scanner-local gnomonic chart from exact inverse published3x3; normalize chart by nine-ray extent',
        injectivity='nondegenerate forward triangles; intersections exactly indexed shared edge/vertex; no fold/overlap',
        footprint='selected patch covers convex hull of nine saved rays and contains target direction',
        coverage_guard='every exposed selected-patch edge must be strictly outside expanded ray hull; area only secondary check',
        predicate_scope='conservative floating-point diagnostics, not exact-arithmetic proof',
        visible_samples='reuse unshifted whole-mesh first hits, never replace or normalize world ray',
        source_evidence='retained independently even for rejected structural correspondence',
        crease='visible C0 crease can pass structure; smoothness and semantic correspondence explicitly unproven',
        unknown_visibility='unselected foreground between samples and at exact target untested',
        numerical_predicates_are_physical=False, continuous_uncertainty_bound=False,
        independent_physical_calibration=False, training_admission=False))
    rows, output_refs = [], []
    cache = {}
    def saved(ref):
        key = (ref['path'], ref['sha256'])
        if key not in cache:
            cache[key] = read(ref, root)
        return cache[key]
    for ref in prior['artifacts']:
        if not ref['path'].endswith('-observations.json'):
            continue
        values = saved(ref)
        new_rows = []
        for row in values:
            result = dict(row, preceding_observations=ref)
            station = saved(row['station'])
            targets = saved(station['targets'])
            target, m = targets['slots'][row['slot']], station['measurements'][row['slot']]
            if (target['slot'] != row['slot'] or m['slot'] != row['slot']
                    or targets['corpus_ordinal'] != row['corpus_ordinal'] or station['scan'] != row['scan']
                    or target['pixel_yx'] != row['pixel_yx']):
                raise ValueError('Original identity/slot changed')
            if not target['numerical_support']:
                result['projected_footprint'] = dict(status='DSLR_numerically_unsupported',
                    sampled_C0_correspondence=False, physical_accuracy_accepted=False)
            else:
                samples = m['discrete_angle_sensitivity']['samples']
                offsets = [(s['azimuth_offset_millirad'], s['polar_offset_millirad']) for s in samples]
                if offsets != [(a, p) for a in (-1, 0, 1) for p in (-1, 0, 1)]:
                    raise ValueError('Original nine-ray sampling required')
                rays = directions(np.array([m['stored_azimuth_millirad']+a for a, _ in offsets]),
                                  np.array([m['stored_polar_millirad']+p for _, p in offsets]))
                pose = np.asarray(station['scanner_pose'])
                scanner_pose_diagnostics(pose)
                result['projected_footprint'] = projected_footprint(target,
                    [s['nearest_whole_mesh_hit'] for s in samples], pose, rays, saved(row['topology']), contract)
            new_rows.append(result)
        name = Path(ref['path']).name
        publish_json(output/name, new_rows)
        output_refs.append(file_ref(output/name))
        rows.extend(new_rows)
        print(name, dict(Counter(r['projected_footprint']['status'] for r in new_rows)), flush=True)
    # Bind all prior counterexamples; structure must never suppress source evidence.
    examples = []
    for before in prior['counterexamples_preserved']:
        key = (before['corpus_ordinal'], before['scan'], before['slot'])
        after = next(r for r in rows if (r['corpus_ordinal'], r['scan'], r['slot']) == key)
        if before['range_evidence'] != after['range_evidence']:
            raise ValueError('Prior source gap modified')
        examples.append(after)
    result = dict(stage='saved_projected_surface_diagnostic_complete_not_accepted',
        prior=prior_ref, contract=file_ref(output/'contract.json'), observations=output_refs,
        coverage=summarize(rows), counterexamples_preserved=examples,
        prior_counterexample=prior['prior_counterexample'],
        numerical_renderer_tolerance_m=.002, existing_image_support_floor=.05,
        whole_source_acceptance_contract_complete=False, confirmation_images_read=0,
        new_mesh_reads=0, new_range_reads=0, new_RGB_reads=0, new_rays=0,
        gpu_used=False, optimizer_updates=0, source_quality_accepted=False,
        training_admission=False, labels_changed=False,
        code=[file_ref(Path(__file__)), file_ref(Path(__file__).parents[1]/'src/radio_moge/projected_footprint.py')])
    publish_json(output/'report.json', result)
    print(file_ref(output/'report.json'))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
