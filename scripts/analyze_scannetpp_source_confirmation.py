"""Describe saved confirmation coverage without reopening source data or tuning."""
import argparse
from collections import Counter
from pathlib import Path

from probe_scannetpp_v2_source_pilot import bound_path, read
from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import publish_json
from run_scannetpp_source_confirmation import coverage, validate_scope

REPORT_SHA = 'c754bfe88efddae6c7cfcfef7921265e9416f421022c083b315bbb89636fa506'


def stages(point):
    rows = point['station_evidence']
    numerical = point['numerical_support']
    exact = numerical and all(r['exact_target_visibility'].get('exact_target_first_surface', False) for r in rows)
    structural = exact and all(r['projected_footprint'].get('sampled_C0_correspondence', False) for r in rows)
    complete = structural and all(r['angle_matches'].get('all_nine_residuals_available', False) for r in rows)
    within = complete and all(abs(p['signed_range_residual_m']) <= .01 for r in rows for p in r['angle_matches']['pixels'])
    return dict(numerical=numerical, both_exact_target=exact, both_exact_and_C0=structural,
        both_exact_C0_complete9=complete, both_exact_C0_complete9_within_budget=within,
        final_conditional=point['decision']['conditional_sample_candidate'])


def run(root, output):
    report_ref = file_ref(root/'scannetpp-native-v2-source-confirmation-01/report.json')
    if report_ref['sha256'] != REPORT_SHA:
        raise ValueError('Exact complete fixed confirmation report required')
    report = read(report_ref, root)
    plan = read(report['plan'], root)
    rows = read(plan['selection'], root)
    original = read(plan['original_selection'], root)
    validate_scope(rows, read(original['selection'], root))
    points, summaries, records = [], [], []
    for row, ref in zip(rows, report['records']):
        record = read(ref, root)
        if record['corpus_ordinal'] != row['corpus_ordinal']:
            raise ValueError('Confirmation record identity changed')
        bound_path(record['page'], root)
        batch = read(record['source_evidence']['decisions'], root)
        counts = dict(Counter(k for p in batch for k, v in stages(p).items() if v))
        for p in batch:
            s = stages(p)
            excluded = 'exact_prior_geometry_exclusion' in p['decision']['reasons']
            if s['final_conditional'] != (s['both_exact_C0_complete9_within_budget'] and not excluded):
                raise ValueError('Saved fixed rule decision and evidence disagree')
        summaries.append(dict(corpus_ordinal=row['corpus_ordinal'], stage_counts=counts,
            numerical_dense_fraction=record['public_consensus']['retained_pixels']/(row['grid'][0]*row['grid'][1]),
            minimum_numerical_support_passed=record['minimum_numerical_support_passed'],
            consensus_rejected_pixels=record['public_consensus']['rejected_pixels'],
            geometry_excluded=False, source_admitted=False))
        points.extend(batch)
        records.append(record)
    recomputed = coverage(points, rows)
    if recomputed != report['coverage']:
        raise ValueError('Saved all-slot coverage differs')
    stage_counts = {k: sum(stages(p)[k] for p in points) for k in stages(points[0])}
    observations = [r for p in points for r in p['station_evidence']]
    ranges = [d for p in points for s in p['decision']['stations'] for d in s['residuals_preserved']]
    result = dict(stage='saved_confirmation_coverage_review_complete_not_admitted',
        confirmation=report_ref, policy=report['policy'], coverage=recomputed,
        cumulative_stage_counts=stage_counts, cumulative_order_is_descriptive=True,
        per_image=summaries,
        images_with_zero_conditional_samples=sum(s['stage_counts'].get('final_conditional', 0) == 0 for s in summaries),
        conditional_fraction_of_all_slots=stage_counts['final_conditional']/1536,
        conditional_fraction_of_numerical_slots=stage_counts['final_conditional']/stage_counts['numerical'],
        station_reason_counts=dict(Counter(reason for p in points for s in p['decision']['stations'] for reason in s['reasons'])),
        exact_target_statuses=dict(Counter(o['exact_target_visibility']['status'] for o in observations)),
        projected_footprint_statuses=dict(Counter(o['projected_footprint']['status'] for o in observations)),
        complete_nine_match_station_slots=sum(o['angle_matches'].get('all_nine_residuals_available', False) for o in observations),
        preserved_nominal_pixel_residuals=len(ranges),
        residual_pixels_over_engineering_budget=sum(abs(v) > .01 for v in ranges),
        all_images_numerical_floor_passed=all(s['minimum_numerical_support_passed'] for s in summaries),
        public_consensus_rejected_pixels=sum(s['consensus_rejected_pixels'] for s in summaries),
        source_reads=0, RGB_reads=0, mesh_reads=0, ray_queries=0, GPU_launch=False,
        labels_changed=False, thresholds_changed=False, new_exclusions=0,
        optimizer_updates=0, source_quality_accepted=False, training_admission=False,
        caveats=['No failed-slot count is a bad-depth rate: missing support, ambiguity and occlusion remain counted.',
                 'Fixed first two scanner IDs are not an optimized coverage choice; do not substitute using these results.',
                 'Sparse confirmation fractions do not prove dense coverage, physical accuracy or population error.',
                 'Sequential diagnostic stage counts are order dependent, not causal attributions.',
                 'Raw residual observations share acquisition and quantized directions; not independent physical trials.'],
        code=file_ref(Path(__file__)))
    output.mkdir(exist_ok=False)
    publish_json(output/'report.json', result)
    print(result['cumulative_stage_counts'])
    print(result['exact_target_statuses'])
    print(file_ref(output/'report.json'))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
