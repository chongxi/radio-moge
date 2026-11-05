"""Apply the frozen two-station policy to saved pilot evidence and NEW target rays."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path

from probe_scannetpp_v2_source_pilot import bound_path, read
from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import publish_json
from radio_moge.scanner_support import ScannerSupportContract, paired_support


def coverage(points):
    keys = {(p['corpus_ordinal'], p['slot']) for p in points}
    if len(points) != 384 or len(keys) != 384 or len({p['corpus_ordinal'] for p in points}) != 4:
        raise ValueError('All384unique pilot slots required')
    images = []
    for ordinal in sorted({p['corpus_ordinal'] for p in points}):
        rows = [p for p in points if p['corpus_ordinal'] == ordinal]
        if {p['slot'] for p in rows} != set(range(96)):
            raise ValueError('All96slots in each original image required')
        candidate = sum(p['decision']['conditional_sample_candidate'] for p in rows)
        images.append(dict(corpus_ordinal=ordinal, sampled_slots=96,
            conditional_sample_candidates=candidate, sampled_fraction=candidate/96,
            full_image_source_support_known=False, image_eligible_for_training=False))
    return dict(DSLR_slots=384, station_slots=768,
        conditional_sample_candidates=sum(p['decision']['conditional_sample_candidate'] for p in points),
        station_reason_counts=dict(Counter(reason for p in points for station in p['decision']['stations']
                                          for reason in station['reasons'])),
        point_reason_counts=dict(Counter(reason for p in points for reason in p['decision']['reasons'])),
        per_image=images, source_accepted_images=0, physical_accepted_pixels=0,
        dense_source_mask_constructed=False, sparse_coverage_extrapolated=False)


def run(root, exact_path, output):
    exact_ref = file_ref(exact_path)
    exact = read(exact_ref, root)
    plan = read(exact['plan'], root)
    policy = ScannerSupportContract()
    implementation_ref = file_ref(Path(__file__).parents[1]/'src/radio_moge/scanner_support.py')
    if (exact['stage'] != 'new_exact_target_visibility_complete_not_accepted'
            or exact['station_slots'] != 768 or exact['new_rays'] != 598
            or exact['training_admission'] is not False or plan['policy'] != asdict(policy)
            or implementation_ref['sha256'] != plan['code'][1]['sha256']):
        raise ValueError('Completed new target evidence and exact frozen policy implementation required')
    excluded = [json.loads(line)['corpus_ordinal'] for line in bound_path(plan['exclusions'], root).read_text().splitlines()]
    if len(excluded) != 4190 or len(set(excluded)) != 4190:
        raise ValueError('All4190prior exact geometry dispositions retained')
    excluded = set(excluded)
    saved_exact = {}
    for ref in exact['observations']:
        for row in read(ref, root):
            key = (row['corpus_ordinal'], row['scan'], row['slot'])
            if key in saved_exact:
                raise ValueError('Duplicate exact-target identity')
            saved_exact[key] = (row, ref)
    if len(saved_exact) != 768:
        raise ValueError('All768exact-target station slots required')
    output.mkdir(exist_ok=False)
    points, refs, used = [], [], set()
    for image in plan['images']:
        target_rows = read(image['targets'], root)['slots']
        stations = [read(ref, root) for ref in image['stations']]
        selected_scans = [station['scan'] for station in stations]
        by_slot = defaultdict(list)
        for row in read(image['observations'], root):
            key = (row['corpus_ordinal'], row['scan'], row['slot'])
            item, source = saved_exact[key]
            if (row['pixel_yx'] != item['pixel_yx'] or row['scene'] != item['scene']
                    or row['station']['sha256'] != item['station']['sha256']
                    or item['targets']['sha256'] != image['targets']['sha256']):
                raise ValueError('Nominal and exact-target evidence identity mismatch')
            if key in used:
                raise ValueError('Duplicate source observation')
            used.add(key)
            by_slot[row['slot']].append(dict(row, exact_target_visibility=item['visibility'],
                                           exact_target_observations=source))
        batch = []
        for slot in range(96):
            target = target_rows[slot]
            if target['slot'] != slot:
                raise ValueError('Target slot ordering changed')
            decision = paired_support(by_slot[slot], selected_scans,
                target['numerical_support'], image['corpus_ordinal'] in excluded, policy)
            batch.append(dict(corpus_ordinal=image['corpus_ordinal'], scene=image['scene'], slot=slot,
                pixel_yx=target['pixel_yx'], decision=decision, station_evidence=by_slot[slot]))
        path = output/f"{image['corpus_ordinal']}-decisions.json"
        publish_json(path, batch)
        refs.append(file_ref(path))
        points.extend(batch)
    if used != set(saved_exact):
        raise ValueError('Exact-target denominator dropped')
    nominal = read(plan['nominal_report'], root)
    examples = []
    for old in nominal['counterexamples_preserved']:
        point = next(p for p in points if (p['corpus_ordinal'], p['slot']) == (old['corpus_ordinal'], old['slot']))
        station = next(r for r in point['station_evidence'] if r['scan'] == old['scan'])
        if old['range_evidence'] != station['range_evidence']:
            raise ValueError('Prior source conflict suppressed or changed')
        examples.append(point)
    result = dict(stage='frozen_conditional_source_policy_pilot_complete_not_admitted',
        plan=exact['plan'], new_target_visibility=exact_ref, nominal_report=plan['nominal_report'],
        exclusions=plan['exclusions'], policy=asdict(policy), decisions=refs,
        coverage=coverage(points), counterexamples_preserved=examples,
        prior_counterexample=nominal['prior_counterexample'],
        label_role='conditional released-source corroborated laser_mesh_derived candidate only',
        engineering_budget_not_physical_bound=True, independent_physical_calibration=False,
        source_quality_accepted=False, training_admission=False, labels_changed=False,
        new_exclusions=0, new_mesh_reads=0, new_range_reads=0, new_RGB_reads=0, new_rays=0,
        confirmation_images_read=0, GPU_launch=False, optimizer_updates=0,
        code=[file_ref(Path(__file__)), implementation_ref])
    publish_json(output/'report.json', result)
    print(result['coverage'], flush=True)
    print(file_ref(output/'report.json'), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--exact', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.root.resolve(), args.exact.resolve(), args.output.resolve())
